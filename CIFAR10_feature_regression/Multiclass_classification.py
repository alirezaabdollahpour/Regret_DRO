import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import math
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Tuple, List

import matplotlib.pyplot as plt

from corrected_cross_entropy import CrossEntropyLoss, cross_entropy, LOSS_IMPLEMENTATION


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(data_loader, model, device):
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)

def pgd_attack(model, features, labels, epsilon, alpha=None, num_iter=10,
               return_details=False):
    """Clean-start L2 PGD. Track correctness at the clean point and every step."""
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("PGD radius must be finite and nonnegative.")
    if num_iter < 1:
        raise ValueError("PGD needs at least one step.")
    alpha = epsilon / 8 if alpha is None else alpha
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("PGD step size must be finite and nonnegative.")
    device = next(model.parameters()).device
    original = features.detach().to(device)
    labels = labels.to(device)
    perturbed = original.clone()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(original)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite logits during PGD evaluation.")
            robust_correct = logits.argmax(1).eq(labels)
        if epsilon > 0:
            for _ in range(num_iter):
                with torch.enable_grad():
                    perturbed = perturbed.detach().requires_grad_(True)
                    loss = cross_entropy(model(perturbed), labels)
                    grad, = torch.autograd.grad(loss, perturbed)
                with torch.no_grad():
                    if not torch.isfinite(grad).all():
                        raise ValueError("Nonfinite feature gradient during PGD.")
                    grad_norm = torch.linalg.vector_norm(grad, dim=1, keepdim=True)
                    direction = grad / grad_norm.clamp_min(torch.finfo(grad.dtype).tiny)
                    delta = perturbed.detach() + alpha * direction - original
                    delta_norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
                    scale = (epsilon / delta_norm.clamp_min(torch.finfo(delta.dtype).tiny)).clamp(max=1)
                    perturbed = (original + delta * scale).detach()
                    logits = model(perturbed)
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite logits during PGD evaluation.")
                    robust_correct &= logits.argmax(1).eq(labels)
        if return_details:
            return perturbed.detach(), robust_correct
        return perturbed.detach()
    finally:
        model.train(was_training)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x): return self.linear(x)

class DROError(Exception): pass

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.linear(x)

class BaseLinearDRO:
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool, sample_level: int = 6):
        self.input_dim, self.num_classes, self.fit_intercept = input_dim, num_classes, fit_intercept
        self.model: nn.Module
        self.sample_level = sample_level
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor: return torch.as_tensor(data, dtype=torch.float32)
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1: X = X.reshape(-1, self.input_dim)
        if y.ndim > 1: y = y.flatten()
        if X.shape[0] != y.shape[0]: raise DROError(f"Shapes mismatch: X {X.shape[0]}, y {y.shape[0]}")
        if X.shape[1] != self.input_dim: raise DROError(f"Input dim mismatch: expected {self.input_dim}, got {X.shape[1]}")
        return X, y
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        dataset = TensorDataset(self._to_tensor(X), torch.as_tensor(y, dtype=torch.long))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_val, _ = self._validate_inputs(X, np.zeros(X.shape[0]))
        self.model.eval()
        with torch.no_grad():
            model_device = next(self.model.parameters()).device
            inputs = self._to_tensor(X_val).to(model_device)
            return self.model(inputs).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(y.flatten() == np.argmax(self.predict(X), axis=1)))
    
    def _compute_loss(self, predictions, targets, m, lambda_reg):
        criterion = CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg
    
    def _compute_dual_loss(self, data, targets, lam, epsilon):
        m = 2 ** 6
        expanded_data = data.repeat_interleave(m, dim=0)
        noise = torch.randn_like(expanded_data) * math.sqrt(epsilon)
        noisy_data = expanded_data + noise
        repeated_target = targets.repeat_interleave(m, dim=0)
        
        predictions = self.model(noisy_data)
        criterion = CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, repeated_target) / max(lam * epsilon, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lam * epsilon
    
class SDRO_Dual(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 1e-3,
                 lambda_param: float = 1e2, max_itr: int = 100, learning_rate: float = 1e-2,
                 sample_level: int = 6, batch_size: int = 64, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.max_itr, self.learning_rate, self.sample_level, self.batch_size = epsilon, lambda_param, max_itr, learning_rate, sample_level, batch_size
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(input_dim, output_dim=num_classes, bias=fit_intercept).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        loss_history = []
        self.model.train()
        for epoch in range(self.max_itr):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training SinkhornLinearDRO Epoch {epoch+1}/{self.max_itr}", leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                levels = np.arange(self.sample_level + 1)
                numerators = 2.0**(-levels)
                denominator = 2.0 - 2.0**(-self.sample_level)
    
                probabilities = numerators / denominator
                sampled_level = np.random.choice(levels, p=probabilities)
                m = 2 ** sampled_level
                
                expanded_data = data.repeat_interleave(m, dim=0)
                noise = torch.randn_like(expanded_data) * math.sqrt(self.epsilon)
                noisy_data = expanded_data + noise
                repeated_target = target.repeat_interleave(m, dim=0)
                
                predictions = self.model(noisy_data)
                loss = self._compute_loss(predictions, repeated_target, m, lambda_reg)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_epoch_loss)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_Dual_run{run_id}_epoch_{epoch+1}.pth')

        return loss_history


class SDRO_RGO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 20, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.rgo_inner_lr, self.rgo_inner_steps = rgo_inner_lr, rgo_inner_steps
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.rgo_vectorized_max_trials = 100 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _rgo_sampler_vectorized(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor, current_model_state: nn.Module, num_samples_to_generate: int, epoch: int) -> torch.Tensor:
        batch_size = x_original_batch.size(0)
        x_orig_detached_batch = x_original_batch.detach()
        x_pert_batch = x_orig_detached_batch.clone()
        lr_inner = self.rgo_inner_lr
        inner_steps = int(min(5, self.rgo_inner_steps * (epoch + 1) / self.max_itr))
        for _ in range(inner_steps):
            x_pert_batch.requires_grad_(True)
            per_sample_losses = self._get_model_loss_value_batched(x_pert_batch, y_original_batch, current_model_state)
            per_sample_grads, = torch.autograd.grad(outputs=per_sample_losses, inputs=x_pert_batch, grad_outputs=torch.ones_like(per_sample_losses))
            x_pert_batch = x_pert_batch.detach()
            grad_total = -per_sample_grads / self.lambda_param + 2 * (x_pert_batch - x_orig_detached_batch)
            x_pert_batch -= lr_inner * grad_total
        x_opt_star_batch = x_pert_batch
        var_rgo = self.epsilon
        if var_rgo <= 1e-12: return x_opt_star_batch.repeat_interleave(num_samples_to_generate, dim=0)
        std_rgo = math.sqrt(var_rgo)
        f_model_loss_opt_star = self._get_model_loss_value_batched(x_opt_star_batch, y_original_batch, current_model_state)
        norm_sq_opt_star = torch.sum((x_opt_star_batch - x_orig_detached_batch) ** 2, dim=1)
        f_L_xi_opt_star = (-f_model_loss_opt_star / (self.lambda_param * self.epsilon)) + (norm_sq_opt_star / self.epsilon)
        x_opt_star_3d, x_original_3d, f_L_xi_opt_star_3d = x_opt_star_batch.unsqueeze(1), x_orig_detached_batch.unsqueeze(1), f_L_xi_opt_star.unsqueeze(1)
        final_accepted_perturbations = torch.zeros((batch_size, num_samples_to_generate, self.input_dim), device=self.device)
        active_flags = torch.ones((batch_size, num_samples_to_generate), dtype=torch.bool, device=self.device)
        for _ in range(self.rgo_vectorized_max_trials):
            if not active_flags.any(): break
            pert_proposals = torch.randn_like(final_accepted_perturbations) * std_rgo
            x_candidates = x_opt_star_3d + pert_proposals
            x_candidates_flat = x_candidates.view(-1, self.input_dim)
            y_repeated = y_original_batch.repeat_interleave(num_samples_to_generate, dim=0)
            f_model_loss_candidates = self._get_model_loss_value_batched(x_candidates_flat, y_repeated, current_model_state).view(batch_size, num_samples_to_generate)
            norm_sq_candidates = torch.sum((x_candidates - x_original_3d) ** 2, dim=2)
            f_L_xi_candidates = (-f_model_loss_candidates / (self.lambda_param * self.epsilon)) + (norm_sq_candidates / self.epsilon)
            diff_cand_opt_norm_sq = torch.sum(pert_proposals**2, dim=2)
            exponent_term3 = diff_cand_opt_norm_sq / (2 * var_rgo)
            acceptance_probs = torch.exp(torch.clamp(-f_L_xi_candidates + f_L_xi_opt_star_3d + exponent_term3, max=10))
            newly_accepted_mask = (torch.rand_like(acceptance_probs) < acceptance_probs) & active_flags
            final_accepted_perturbations[newly_accepted_mask] = pert_proposals[newly_accepted_mask]
            active_flags[newly_accepted_mask] = False
        return (x_opt_star_3d + final_accepted_perturbations).view(-1, self.input_dim)
    
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training RGO Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_rgo_batch = self._rgo_sampler_vectorized(x_original_batch_dev, y_original_batch_dev, self.model, self.num_samples, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_rgo_batch)
                optimization_loss = CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_RGO_run{run_id}_epoch_{epoch+1}.pth')

        return loss_history
    
class SDRO_WFR(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _WFR_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        batch_size = x_orig.shape[0]

        weights = torch.ones((batch_size, self.num_samples), dtype=torch.float32, device=self.device) / self.num_samples

        x_clone = x_orig.clone().detach().unsqueeze(1).expand(-1, self.num_samples, -1).contiguous().view(-1, self.input_dim)
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = x_orig.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, self.input_dim)

        wt_lr = self.inner_lr
        weight_exponent = 1 - self.lambda_param * self.epsilon * wt_lr

        num_steps = int(max(5, self.inner_itr * (epoch + 1) / self.max_itr))
        
        for _ in range(num_steps):
            x_clone.requires_grad_(True)
            
            loss_values = CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach()
            
            mean = x_clone + self.inner_lr * (grads - 2 * self.lambda_param * (x_clone - x_original_expanded))
            std_dev = torch.sqrt(torch.tensor(2 * self.inner_lr * self.lambda_param * self.epsilon, device=self.device))
            
            noise = torch.randn_like(mean) * std_dev
            x_clone = mean + noise

            with torch.no_grad():
                current_loss = CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated).view(batch_size, -1)
                dist_sq = torch.sum((x_clone - x_original_expanded) ** 2, dim=1).view(batch_size, -1)
                
                energy_term = current_loss - 2 * self.lambda_param * dist_sq

                weights = weights ** weight_exponent * torch.exp(energy_term * wt_lr)

                weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

                low_weight_mask = weights < 1e-4
                rows_with_low_weights = torch.any(low_weight_mask, dim=1)

                if torch.any(rows_with_low_weights):
                    x_reshaped = x_clone.view(batch_size, self.num_samples, -1)

                    max_weight_vals, max_weight_indices = torch.max(weights, dim=1, keepdim=True)
                    highest_weight_point_data = torch.gather(x_reshaped, 1, max_weight_indices.unsqueeze(-1).expand(-1, -1, self.input_dim))

                    low_weights_sum = torch.sum(weights * low_weight_mask, dim=1, keepdim=True)
                    num_low_weights = torch.sum(low_weight_mask, dim=1, keepdim=True, dtype=weights.dtype)
                    avg_weight = (max_weight_vals + low_weights_sum) / (num_low_weights + 1.0 + 1e-9)
                    avg_weight_expanded = avg_weight.expand_as(weights)

                    max_weight_mask = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, max_weight_indices, True)
                    update_mask = (low_weight_mask | max_weight_mask) & rows_with_low_weights.unsqueeze(1)
                    
                    weights = torch.where(update_mask, avg_weight_expanded, weights)

                    x_update_mask = low_weight_mask & rows_with_low_weights.unsqueeze(1)
                    replacement_data = highest_weight_point_data.expand_as(x_reshaped)
                    x_reshaped = torch.where(x_update_mask.unsqueeze(-1), replacement_data, x_reshaped)
                    x_clone = x_reshaped.view(-1, self.input_dim)
                    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

        return x_clone.detach(), weights.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WFR Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WFR_batch, weights = self._WFR_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_WFR_batch)
                loss_values = CrossEntropyLoss(reduction='none')(predictions_logits_batch, y_repeated_batch)
                optimization_loss = (loss_values.view(-1, self.num_samples) * weights).sum(dim=1).mean()
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_WFR_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history
    
class SDRO_WGF(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _WGF_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        x_clone = x_clone.unsqueeze(1).expand(-1, self.num_samples, -1).contiguous().view(-1, self.input_dim)
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = x_orig.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, self.input_dim)
        num_steps = int(max(5, self.inner_itr * (epoch + 1) / self.max_itr))
        for _ in range(num_steps):
            x_clone.requires_grad_(True)
            loss_values = CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach()
            
            mean = x_clone + self.inner_lr * (grads - 2*self.lambda_param*(x_clone - x_original_expanded))
            std_dev = torch.sqrt(torch.tensor(2*self.inner_lr*self.lambda_param*self.epsilon, device=self.device))
            
            noise = torch.randn_like(mean) * std_dev
            x_clone = mean + noise

        return x_clone.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WGF Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WGF_batch = self._WGF_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_WGF_batch)
                optimization_loss = CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_WGF_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history
    
class SDRO_SVG(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, 
                 lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, 
                 learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, 
                 inner_itr: int = 200, adagrad_hist_decay: float = 0.9, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr, self.adagrad_hist_decay = inner_lr, inner_itr, adagrad_hist_decay
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)

    def _rbf_kernel_batched_torch(self, particles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = particles.shape
        device = particles.device

        sq_dist = torch.cdist(particles, particles, p=2).pow(2)

        median_sq_dist = torch.median(sq_dist.view(B, -1), dim=1, keepdim=True)[0]
        median_sq_dist = median_sq_dist.view(B, 1, 1)
        h_squared = median_sq_dist / (torch.log(torch.tensor(S, dtype=torch.float32, device=device)) + 1e-8)

        K = torch.exp(-sq_dist / (2 * h_squared))

        diff = particles.unsqueeze(2) - particles.unsqueeze(1)
        grad_K_x = -diff / h_squared.unsqueeze(-1) * K.unsqueeze(-1)

        return K, grad_K_x

    def _svg_sampler(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        if x_original_batch.shape[0] == 0:
            return torch.empty(0, x_original_batch.shape[1], device=x_original_batch.device)

        B, D = x_original_batch.shape
        S = self.num_samples

        x_orig_repeated = x_original_batch.unsqueeze(1).repeat(1, S, 1)
        y_repeated = y_original_batch.repeat_interleave(S)

        particles = x_orig_repeated.clone().detach()
        particles += torch.randn_like(particles) * 0.1
        hist_grad = torch.zeros_like(particles)

        for _ in range(int(self.inner_itr * (epoch + 1)/self.max_itr)):
            x_tensor = particles.view(B * S, D).clone().requires_grad_(True)
            x_orig_repeated_flat = x_orig_repeated.view(B * S, D)

            neg_log_likelihood = CrossEntropyLoss(reduction='sum')(model(x_tensor), y_repeated)
            grad_log_py_x, = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_tensor)
            
            grad_log_px = -2 * self.lambda_param * (x_tensor - x_orig_repeated_flat)

            total_grad_flat = (grad_log_py_x + grad_log_px) / (self.lambda_param * self.epsilon)
            total_grad = total_grad_flat.view(B, S, D)

            K, grad_K_x = self._rbf_kernel_batched_torch(particles)
            
            K_grad_prod = torch.bmm(K, total_grad)
            
            sum_grad_K = torch.sum(grad_K_x, dim=2)

            svg_grad = (K_grad_prod + sum_grad_K) / S

            with torch.no_grad():
                # Adagrad optimizer for stable updates
                hist_grad = self.adagrad_hist_decay * hist_grad + (1 - self.adagrad_hist_decay) * (svg_grad**2)
                adj_grad = svg_grad / (1e-6 + torch.sqrt(hist_grad))
                particles += self.inner_lr * adj_grad

        return particles.view(B * S, D).detach()

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training SVG Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                
                self.model.eval()
                x_svg_batch = self._svg_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                
                self.model.train()
                predictions_logits_batch = self.model(x_svg_batch)
                optimization_loss = CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_SVG_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history

class WRM(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, lambda_param: float = 1.0, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.lambda_param = lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.max_itr, self.learning_rate, self.batch_size = max_itr, learning_rate, batch_size 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _sinha_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        num_steps = int(max(5, self.inner_itr * (epoch + 1) / self.max_itr))
        for _ in range(num_steps):
            loss_values = CrossEntropyLoss(reduction='none')(model(x_clone), y_orig)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach() + self.inner_lr * (grads - 2*self.lambda_param*(x_clone - x_orig))
            x_clone.requires_grad_(True)
        return x_clone.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WRM Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WRM_batch = self._sinha_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                self.model.train()
                predictions_logits_batch = self.model(x_WRM_batch)
                optimization_loss = CrossEntropyLoss()(predictions_logits_batch, y_original_batch_dev)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                with torch.no_grad():
                    comparable_loss = CrossEntropyLoss()(self.model(x_original_batch_dev), y_original_batch_dev)
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/WRM_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history


def _train_clean_head(model, train_loader, epochs, learning_rate, checkpoint_dir,
                      run_id, device, method, snr_lambda=0.0, snrfrob_gamma=0.0):
    if not math.isfinite(snr_lambda) or snr_lambda < 0:
        raise ValueError("snr_lambda must be finite and nonnegative.")
    if not math.isfinite(snrfrob_gamma) or snrfrob_gamma < 0:
        raise ValueError("snrfrob_gamma must be finite and nonnegative.")
    device = DEVICE if device is None else torch.device(device)
    model.to(device).train()
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    loss_history = []
    for epoch in range(epochs):
        epoch_loss, count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Seed {run_id} {method} Epoch {epoch+1}/{epochs}", leave=False)
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            clean_ce = cross_entropy(model(features), labels)
            loss = clean_ce
            if method == "SNR":
                W = model.linear.weight
                op_norm_sq = torch.linalg.eigvalsh(W @ W.T)[-1]
                loss = clean_ce + snr_lambda * op_norm_sq
            elif method == "SNRFrob":
                frobenius_norm_sq = model.linear.weight.square().sum()
                loss = clean_ce + snrfrob_gamma * frobenius_norm_sq
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {method} training objective.")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            count += labels.size(0)
            pbar.set_postfix(loss=loss.item())
        loss_history.append(epoch_loss / count)
        torch.save(model.state_dict(), Path(checkpoint_dir) / f"{method}_run{run_id}_epoch_{epoch+1}.pth")
    return loss_history


def train_saa(model, train_loader, epochs=30, learning_rate=0.001,
              checkpoint_dir='checkpoints', run_id=0, device=None):
    return _train_clean_head(model, train_loader, epochs, learning_rate,
                             checkpoint_dir, run_id, device, "SAA")


def train_snr(model, train_loader, epochs=30, learning_rate=0.001,
              checkpoint_dir='checkpoints', run_id=0, device=None, snr_lambda=0.01):
    return _train_clean_head(model, train_loader, epochs, learning_rate,
                             checkpoint_dir, run_id, device, "SNR", snr_lambda)


def train_snrfrob(model, train_loader, epochs=30, learning_rate=0.001,
                  checkpoint_dir='checkpoints', run_id=0, device=None, snrfrob_gamma=0.01):
    return _train_clean_head(model, train_loader, epochs, learning_rate,
                             checkpoint_dir, run_id, device, "SNRFrob",
                             snrfrob_gamma=snrfrob_gamma)


def evaluate_model(model, test_features, test_labels, epsilon_list, pgd_steps=10,
                   batch_size=128):
    """Return feature-space PGD accuracy (%) at each absolute radius."""
    if not isinstance(model, nn.Module):
        model = model.model
    features = torch.as_tensor(test_features, dtype=next(model.parameters()).dtype)
    labels = torch.as_tensor(test_labels, dtype=torch.long)
    if len(labels) == 0:
        raise ValueError("Evaluation data must not be empty.")
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
    results = {}
    for radius in dict.fromkeys([0.0] + list(epsilon_list)):
        correct = 0
        for batch_features, batch_labels in loader:
            _, robust_correct = pgd_attack(model, batch_features, batch_labels,
                                           epsilon=radius, num_iter=pgd_steps,
                                           return_details=True)
            correct += robust_correct.sum().item()
        results[radius] = 100.0 * correct / len(labels)
        print(f"Feature-space PGD accuracy (radius={radius:.6g}): {results[radius]:.2f}%")
    return results


def clean_objective_metrics(model, features, labels, snr_lambda, batch_size=128,
                            snrfrob_gamma=None):
    """Measure the final clean objective on the embeddings used to fit the head."""
    device = next(model.parameters()).device
    model.eval()
    ce_sum = 0.0
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(features, labels), batch_size=batch_size):
            ce_sum += cross_entropy(model(x.to(device)), y.to(device),
                                                  reduction="sum").item()
        W = model.linear.weight
        penalty = torch.linalg.eigvalsh(W @ W.T)[-1].item()
        frobenius_penalty = W.square().sum().item()
    clean_ce = ce_sum / len(labels)
    if snrfrob_gamma is not None:
        return {"clean_ce": clean_ce, "frobenius_penalty": frobenius_penalty,
                "weighted_frobenius_penalty": snrfrob_gamma * frobenius_penalty,
                "total_objective": clean_ce + snrfrob_gamma * frobenius_penalty,
                "final_op_norm": math.sqrt(max(penalty, 0.0)),
                "final_frobenius_norm": math.sqrt(frobenius_penalty)}
    return {"clean_ce": clean_ce, "spectral_penalty": penalty,
            "weighted_spectral_penalty": snr_lambda * penalty,
            "total_objective": clean_ce + snr_lambda * penalty,
            "final_op_norm": math.sqrt(max(penalty, 0.0))}


def stratified_split_indices(labels, validation_fraction, seed):
    """Split training indices with a local RNG and at least one item per class in each set."""
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Training labels must be a nonempty vector.")
    labels = labels.cpu()
    generator = torch.Generator().manual_seed(seed)
    fit_parts, validation_parts = [], []
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=True)[0]
        if len(indices) < 2:
            raise ValueError("Each class needs at least two training examples for a holdout.")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = max(1, min(len(indices) - 1, round(len(indices) * validation_fraction)))
        validation_parts.append(indices[:count])
        fit_parts.append(indices[count:])
    fit_indices, validation_indices = torch.cat(fit_parts), torch.cat(validation_parts)
    fit_indices = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
    validation_indices = validation_indices[torch.randperm(len(validation_indices), generator=generator)]
    return fit_indices, validation_indices


def select_matched_gamma(reference_accuracies, candidate_accuracies):
    """Select by the absolute gap of mean validation clean accuracies; break ties by gamma."""
    if not reference_accuracies or not candidate_accuracies:
        raise ValueError("Reference and candidate validation accuracies must not be empty.")
    for gamma, accuracies in candidate_accuracies.items():
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("Gamma values must be finite and nonnegative.")
        if set(accuracies) != set(reference_accuracies):
            raise ValueError("SNR and every gamma candidate must use the same seeds.")
    for accuracies in [reference_accuracies, *candidate_accuracies.values()]:
        if any(not math.isfinite(value) or not 0 <= value <= 100 for value in accuracies.values()):
            raise ValueError("Validation clean accuracies must be finite percentages in [0, 100].")
    reference_mean = float(np.mean(list(reference_accuracies.values())))
    return min(candidate_accuracies, key=lambda gamma: (
        abs(float(np.mean(list(candidate_accuracies[gamma].values()))) - reference_mean), gamma))


FEATURE_METADATA = {
    "schema_version": 1,
    "dataset": "CIFAR10",
    "backbone": "resnet18",
    "weights": "ResNet18_Weights.IMAGENET1K_V1",
    "feature_dim": 512,
    "pooling": "native_global_average_pool_fc_identity",
    "preprocessing": {"resize": [224, 224], "interpolation": "bilinear",
                      "antialias": True, "to_tensor": True,
                      "mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225]},
}


def validate_feature_cache(data):
    if not isinstance(data, dict) or data.get("metadata") != FEATURE_METADATA:
        raise ValueError("Cache metadata does not match the frozen ResNet-18 feature space.")
    for split, size in (("train", 50000), ("test", 10000)):
        features, labels = data.get(f"{split}_features"), data.get(f"{split}_labels")
        if not isinstance(features, torch.Tensor) or features.shape != (size, 512):
            raise ValueError(f"Invalid {split} feature shape in cache.")
        if not isinstance(labels, torch.Tensor) or labels.shape != (size,):
            raise ValueError(f"Invalid {split} label shape in cache.")
        if features.dtype != torch.float32 or labels.dtype != torch.long:
            raise ValueError(f"Invalid {split} tensor dtypes in cache.")
        if not torch.isfinite(features).all() or not ((labels >= 0) & (labels < 10)).all():
            raise ValueError(f"Invalid {split} tensor values in cache.")


def get_cifar10_dataloaders(batch_size=128, data_dir='./data', num_workers=2):
    import torchvision
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    # Cache in dataset order. Training shuffles these fixed embeddings later.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def load_or_extract_features(cache_path, data_dir, device, feature_batch_size=128,
                             num_workers=2):
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        # Reject without overwriting a cache whose provenance or contents differ.
        validate_feature_cache(data)
        print(f"Loaded ResNet-18 features from {cache_path}")
        return data

    from torchvision.models import resnet18, ResNet18_Weights

    print("Extracting frozen ResNet-18 features (512 dimensions).")
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.requires_grad_(False)
    backbone.eval()
    train_loader, test_loader = get_cifar10_dataloaders(feature_batch_size, data_dir, num_workers)
    train_features, train_labels = extract_features(train_loader, backbone, device)
    test_features, test_labels = extract_features(test_loader, backbone, device)
    data = {"metadata": FEATURE_METADATA, "train_features": train_features,
            "train_labels": train_labels, "test_features": test_features,
            "test_labels": test_labels}
    validate_feature_cache(data)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(f"Saved ResNet-18 features to {cache_path}")
    return data


def set_plot_style():
    style = Path("jz.mplstyle")
    if not style.is_file():
        style = Path(__file__).with_name("jz.mplstyle")
    plt.style.use(str(style) if style.is_file() else "default")


def plot_comparison(records, methods, levels, radii, output_dir):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for method in methods:
        runs = [record for record in records if record["method"] == method]
        if not runs:
            raise ValueError(f"Missing evaluation results for {method}.")
        for run in runs:
            missing = [radius for radius in radii if radius not in run["results"]]
            if missing:
                raise ValueError(f"Missing {method} radii: {missing}")
        accuracies = np.array([[run["results"][radius] for radius in radii] for run in runs])
        if not np.isfinite(accuracies).all():
            raise ValueError(f"Nonfinite evaluation results for {method}.")
        mean, std = accuracies.mean(axis=0), accuracies.std(axis=0)
        line, = ax.plot(levels, mean, label=method)
        ax.fill_between(levels, mean - std, mean + std, color=line.get_color(), alpha=0.2)
    ax.set(xlabel="Relative radius (rho / mean test feature norm)",
           ylabel="Feature-space PGD accuracy (%)", ylim=(0, 100))
    ax.legend()
    fig.savefig(Path(output_dir) / "feature_space_pgd_accuracy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(constrained_layout=True)
    for method in methods:
        losses = np.array([r["losses"] for r in records if r["method"] == method])
        if losses.size == 0 or not np.isfinite(losses).all():
            raise ValueError(f"Missing or nonfinite training losses for {method}.")
        epochs = np.arange(1, losses.shape[1] + 1)
        mean, std = losses.mean(axis=0), losses.std(axis=0)
        line, = ax.plot(epochs, mean, label=method)
        ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.2)
    ax.set(xlabel="Epoch", ylabel="Recorded training loss (method specific)")
    ax.legend()
    fig.savefig(Path(output_dir) / "training_loss.pdf")
    plt.close(fig)


CLEAN_METHODS = ("SAA", "SNR", "SNRFrob")
METHODS = (*CLEAN_METHODS, "Dual", "WGF", "WRM", "WFR", "SVG", "RGO")
CSV_FIELDS = ["method", "configuration", "seed", "mean_test_norm", "relative_level",
              "absolute_radius", "clean_accuracy", "pgd_accuracy", "metric",
              "cross_entropy",
              "snr_lambda", "snrfrob_gamma", "lambda_param", "epsilon", "learning_rate", "batch_size",
              "epochs", "weight_decay", "pgd_steps", "pgd_step_size", "clean_ce",
              "spectral_penalty", "weighted_spectral_penalty", "total_objective",
              "final_op_norm", "frobenius_penalty", "weighted_frobenius_penalty",
              "final_frobenius_norm", "hyperparameters", "checkpoint"]


def configuration_id(method, hyperparameters):
    encoded = json.dumps(hyperparameters, sort_keys=True).encode()
    return method + "_" + hashlib.sha256(encoded).hexdigest()[:12]


def train_clean_configuration(method, features, labels, seed, hp, directory, device):
    """Give each clean configuration the same initial head and minibatch order per seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = LogisticRegression()
    loader = DataLoader(TensorDataset(features, labels), batch_size=hp["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    trainers = {"SAA": train_saa, "SNR": train_snr, "SNRFrob": train_snrfrob}
    extra = {}
    if method == "SNR":
        extra["snr_lambda"] = hp["snr_lambda"]
    elif method == "SNRFrob":
        extra["snrfrob_gamma"] = hp["snrfrob_gamma"]
    losses = trainers[method](head, loader, epochs=hp["epochs"], learning_rate=hp["learning_rate"],
                              checkpoint_dir=directory, run_id=seed, device=device, **extra)
    return head, losses


def run_validation_sweep(args, fit_features, fit_labels, validation_features,
                          validation_labels, run_dir, device, clean_hp):
    """Select gamma on training holdout data and retain the same heads for final evaluation."""
    mean_validation_norm = torch.linalg.vector_norm(validation_features, dim=1).mean().item()
    levels = [k * 0.008 for k in range(11)]
    radii = [q * mean_validation_norm for q in levels]
    fields = ["mean_validation_norm" if name == "mean_test_norm" else name for name in CSV_FIELDS]
    candidates = []
    with (run_dir / "validation_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        configurations = [("SNR", None)] + [("SNRFrob", gamma) for gamma in sorted(args.snrfrob_gamma_grid)]
        for seed in args.seeds:
            for method, gamma in configurations:
                hp = {**clean_hp, **({"snr_lambda": args.snr_lambda} if method == "SNR"
                                     else {"snrfrob_gamma": gamma})}
                config = configuration_id(method, hp)
                directory = run_dir / "validation" / config / f"seed_{seed}"
                directory.mkdir(parents=True)
                head, losses = train_clean_configuration(
                    method, fit_features, fit_labels, seed, hp, directory, device)
                diagnostics = clean_objective_metrics(
                    head, fit_features, fit_labels, args.snr_lambda, args.batch_size,
                    snrfrob_gamma=gamma)
                results = evaluate_model(head, validation_features, validation_labels, radii,
                                         args.pgd_steps, args.batch_size)
                checkpoint = directory / "validation.pth"
                torch.save({"state_dict": head.state_dict(), "method": method, "seed": seed,
                            "configuration": config, "hyperparameters": hp,
                            "feature_metadata": FEATURE_METADATA, "loss_history": losses,
                            "diagnostics": diagnostics, "mean_validation_norm": mean_validation_norm,
                            "pgd_steps": args.pgd_steps, "validation_results": results}, checkpoint)
                for q, radius in zip(levels, radii):
                    row = {"method": method, "configuration": config, "seed": seed,
                           "mean_validation_norm": mean_validation_norm, "relative_level": q,
                           "absolute_radius": radius, "clean_accuracy": results[0.0],
                           "pgd_accuracy": results[radius], "metric": "feature-space PGD accuracy",
                           "pgd_steps": args.pgd_steps, "pgd_step_size": radius / 8,
                           "hyperparameters": json.dumps(hp, sort_keys=True),
                           "checkpoint": str(checkpoint.relative_to(run_dir)), **diagnostics}
                    row.update({key: value for key, value in hp.items() if key in fields})
                    writer.writerow(row)
                stream.flush()
                candidates.append({"method": method, "seed": seed, "hp": hp, "head": head,
                                   "losses": losses, "directory": directory, "clean_accuracy": results[0.0]})
    reference = {r["seed"]: r["clean_accuracy"] for r in candidates if r["method"] == "SNR"}
    accuracies = {gamma: {r["seed"]: r["clean_accuracy"] for r in candidates
                         if r["method"] == "SNRFrob" and r["hp"]["snrfrob_gamma"] == gamma}
                  for gamma in args.snrfrob_gamma_grid}
    selected_gamma = select_matched_gamma(reference, accuracies)
    reference_mean = float(np.mean(list(reference.values())))
    candidate_means = {gamma: float(np.mean(list(values.values()))) for gamma, values in accuracies.items()}
    signed_gap = candidate_means[selected_gamma] - reference_mean
    selection = {
        "criterion": "minimum_absolute_gap_in_mean_validation_clean_accuracy",
        "tie_breaker": "smaller_gamma", "selected_gamma": selected_gamma,
        "snr_lambda": args.snr_lambda, "seeds": args.seeds,
        "reference_mean_clean_accuracy": reference_mean,
        "candidate_mean_clean_accuracies": candidate_means,
        "selected_mean_clean_accuracy": candidate_means[selected_gamma],
        "signed_clean_accuracy_gap_pp": signed_gap, "absolute_clean_accuracy_gap_pp": abs(signed_gap),
        "match_tolerance_pp": args.match_clean_tolerance,
        "matched_within_tolerance": abs(signed_gap) <= args.match_clean_tolerance,
        "mean_validation_norm": mean_validation_norm,
        "validation_fraction": args.validation_fraction, "validation_seed": args.validation_seed,
        "fit_size": len(fit_labels), "validation_size": len(validation_labels),
        "selection_uses_test_data": False, "refit_on_full_training_data": False,
    }
    with (run_dir / "validation_selection.json").open("w") as stream:
        json.dump(selection, stream, indent=2)
    print(f"Selected SNRFrob gamma={selected_gamma:g}; validation clean gap={signed_gap:+.4f} percentage points.")
    if not selection["matched_within_tolerance"]:
        print("The closest candidate is outside the clean-accuracy tolerance. Expand the gamma grid for a closer match.")
    retained = {(r["method"], r["seed"]): r for r in candidates if r["method"] == "SNR" or
                r["hp"]["snrfrob_gamma"] == selected_gamma}
    return retained, selection


def run_experiment(args):
    if args.snrfrob_gamma_grid and not {"SNR", "SNRFrob"}.issubset(args.methods):
        raise ValueError("A validation gamma sweep requires both SNR and SNRFrob.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    device = DEVICE if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")
    data = load_or_extract_features(args.cache, args.data_dir, device,
                                    args.feature_batch_size, args.num_workers)
    train_features, train_labels = data["train_features"], data["train_labels"]
    run_dir = Path(args.output_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Outputs: {run_dir.resolve()}")
    split_hp = {}
    retained, selection = {}, None
    if args.snrfrob_gamma_grid:
        fit_indices, validation_indices = stratified_split_indices(
            train_labels, args.validation_fraction, args.validation_seed)
        validation_features, validation_labels = train_features[validation_indices], train_labels[validation_indices]
        train_features, train_labels = train_features[fit_indices], train_labels[fit_indices]
        split_hp = {"training_split": "stratified_fit", "validation_fraction": args.validation_fraction,
                    "validation_seed": args.validation_seed, "fit_size": len(fit_indices),
                    "validation_size": len(validation_indices), "split_file": "validation_split.pth"}
        torch.save({"fit_indices": fit_indices, "validation_indices": validation_indices,
                    "feature_metadata": FEATURE_METADATA, **split_hp}, run_dir / "validation_split.pth")
    clean_hp = {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": args.batch_size,
                "epochs": args.epochs, "weight_decay": 0.0, "betas": [0.9, 0.999],
                "adam_eps": 1e-8, "bias": True, "feature_dim": 512, "num_classes": 10,
                "cross_entropy": LOSS_IMPLEMENTATION, **split_hp}
    print(f"Paired clean-head settings: {json.dumps(clean_hp, sort_keys=True)}")
    if args.snrfrob_gamma_grid:
        retained, selection = run_validation_sweep(
            args, train_features, train_labels, validation_features, validation_labels,
            run_dir, device, clean_hp)

    # Test data enter the computation only after the validation choice is fixed.
    test_features, test_labels = data["test_features"], data["test_labels"]
    mean_test_norm = torch.linalg.vector_norm(test_features, dim=1).mean().item()
    levels = [k * 0.008 for k in range(11)]
    radii = [q * mean_test_norm for q in levels]
    print(f"Full clean test mean feature norm: {mean_test_norm:.9g}")
    with (run_dir / "experiment.json").open("w") as stream:
        json.dump({"arguments": vars(args), "feature_metadata": FEATURE_METADATA,
                   "cross_entropy": LOSS_IMPLEMENTATION,
                   "mean_test_norm": mean_test_norm, "levels": levels,
                   "validation_selection": selection,
                   "metric": "feature-space PGD accuracy",
                   "clean_head_diagnostics": "final fit-data CE; penalties are unweighted squared weight norms"},
                  stream, indent=2)

    records = []
    with (run_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()

        def finish_head(method, head, hp, seed, losses, checkpoint_dir):
            if not all(torch.isfinite(p).all() for p in head.parameters()):
                raise ValueError(f"Nonfinite trained parameters for {method}, seed {seed}.")
            config = configuration_id(method, hp)
            diagnostics = (clean_objective_metrics(head, train_features, train_labels,
                                                   args.snr_lambda, args.batch_size,
                                                   snrfrob_gamma=hp.get("snrfrob_gamma"))
                           if method in ("SNR", "SNRFrob") else {})
            results = evaluate_model(head, test_features, test_labels, radii,
                                     args.pgd_steps, args.batch_size)
            checkpoint = checkpoint_dir / "final.pth"
            torch.save({"state_dict": head.state_dict(), "method": method, "seed": seed,
                        "configuration": config, "hyperparameters": hp,
                        "feature_metadata": FEATURE_METADATA, "mean_test_norm": mean_test_norm,
                        "loss_history": losses, "diagnostics": diagnostics,
                        "validation_selection": selection,
                        "pgd_steps": args.pgd_steps, "results": results}, checkpoint)
            for q, radius in zip(levels, radii):
                row = {"method": method, "configuration": config, "seed": seed,
                       "mean_test_norm": mean_test_norm, "relative_level": q,
                       "absolute_radius": radius, "clean_accuracy": results[0.0],
                       "pgd_accuracy": results[radius], "metric": "feature-space PGD accuracy",
                       "pgd_steps": args.pgd_steps, "pgd_step_size": radius / 8,
                       "hyperparameters": json.dumps(hp, sort_keys=True),
                       "checkpoint": str(checkpoint.relative_to(run_dir)), **diagnostics}
                row.update({key: value for key, value in hp.items() if key in CSV_FIELDS})
                writer.writerow(row)
            stream.flush()
            records.append({"method": method, "hyperparameters": hp, "seed": seed,
                            "losses": losses, "results": results})

        for seed in args.seeds:
            for method in (m for m in args.methods if m in CLEAN_METHODS):
                if (method, seed) in retained:
                    fitted = retained[(method, seed)]
                    finish_head(method, fitted["head"], fitted["hp"], seed,
                                fitted["losses"], fitted["directory"])
                    continue
                hp = dict(clean_hp)
                if method == "SNR":
                    hp["snr_lambda"] = args.snr_lambda
                elif method == "SNRFrob":
                    hp["snrfrob_gamma"] = args.snrfrob_gamma
                directory = run_dir / configuration_id(method, hp) / f"seed_{seed}"
                directory.mkdir(parents=True)
                head, losses = train_clean_configuration(
                    method, train_features, train_labels, seed, hp, directory, device)
                finish_head(method, head, hp, seed, losses, directory)

        classes = {"Dual": SDRO_Dual, "WGF": SDRO_WGF, "WRM": WRM,
                   "WFR": SDRO_WFR, "SVG": SDRO_SVG, "RGO": SDRO_RGO}
        for method in (m for m in args.methods if m in classes):
            for lam in args.sdro_lambdas:
                # WRM has no epsilon parameter; reuse it across epsilon plots too.
                for eps in ([None] if method == "WRM" else args.sdro_epsilons):
                    params = {"lambda_param": lam, "max_itr": args.epochs,
                              "learning_rate": 0.005, "batch_size": args.batch_size}
                    if eps is not None:
                        params["epsilon"] = eps
                    if method == "Dual":
                        params["sample_level"] = 5
                    elif method == "RGO":
                        params.update(num_samples=16, rgo_inner_lr=0.01, rgo_inner_steps=20)
                    else:
                        params.update(inner_lr=0.01, inner_itr=200)
                        if method != "WRM":
                            params["num_samples"] = 16
                        if method == "SVG":
                            params["adagrad_hist_decay"] = 0.9
                    hp = {**params, "epochs": args.epochs, "optimizer": "Adam",
                          "weight_decay": 0.0 if method == "Dual" else 1e-4,
                          "betas": [0.9, 0.999], "adam_eps": 1e-8,
                          "bias": True, "feature_dim": 512, "num_classes": 10,
                          "cross_entropy": LOSS_IMPLEMENTATION, **split_hp}
                    for seed in args.seeds:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        directory = run_dir / configuration_id(method, hp) / f"seed_{seed}"
                        directory.mkdir(parents=True)
                        wrapper = classes[method](512, 10, device=device.type, **params)
                        losses = wrapper.fit(train_features.numpy(), train_labels.numpy(),
                                             checkpoint_dir=str(directory), run_id=seed)
                        finish_head(method, wrapper.model, hp, seed, losses, directory)

    has_dro = any(m not in CLEAN_METHODS for m in args.methods)
    plot_sweeps = [(lam, eps) for lam in args.sdro_lambdas for eps in args.sdro_epsilons] if has_dro else [(None, None)]
    for lam, eps in plot_sweeps:
        selected = [r for r in records if r["method"] in CLEAN_METHODS or
                    (r["hyperparameters"]["lambda_param"] == lam and
                     (r["method"] == "WRM" or r["hyperparameters"]["epsilon"] == eps))]
        directory = run_dir / "plots" / f"lambda_{lam}_epsilon_{eps}"
        directory.mkdir(parents=True)
        plot_comparison(selected, args.methods, levels, radii, directory)
    print(f"Results saved to {run_dir / 'results.csv'}")
    return run_dir


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def positive_float(value):
    parsed = nonnegative_float(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def fraction_float(value):
    parsed = positive_float(value)
    if parsed >= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 linear heads in one frozen ResNet-18 feature space.")
    parser.add_argument("--methods", default="all", help="Comma-separated methods, or all (default).")
    parser.add_argument("--snr-lambda", type=nonnegative_float, default=0.01,
                        help="Fixed SNR coefficient; no automatic selection (default: 0.01).")
    frobenius = parser.add_mutually_exclusive_group()
    frobenius.add_argument("--snrfrob-gamma", type=nonnegative_float, default=0.01,
                           help="Fixed squared Frobenius penalty coefficient (default: 0.01).")
    frobenius.add_argument("--snrfrob-gamma-grid", type=nonnegative_float, nargs="+",
                           help="Match SNR validation clean accuracy over this gamma grid; requires SNR,SNRFrob.")
    parser.add_argument("--validation-fraction", type=fraction_float, default=0.2,
                        help="Training holdout fraction used only for a gamma grid (default: 0.2).")
    parser.add_argument("--validation-seed", type=int, default=2022)
    parser.add_argument("--match-clean-tolerance", type=nonnegative_float, default=0.5,
                        help="Report a clean-accuracy match within this many percentage points (default: 0.5).")
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2022])
    parser.add_argument("--sdro-lambdas", type=positive_float, nargs="+", default=[10.0])
    parser.add_argument("--sdro-epsilons", type=positive_float, nargs="+", default=[0.2, 0.02, 0.002])
    parser.add_argument("--pgd-steps", type=positive_int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cache", default="cifar10_resnet18_imagenet1k_v1_features.pth")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/cifar10_resnet18")
    parser.add_argument("--feature-batch-size", type=positive_int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic tests without downloads; check CUDA if available.")
    args = parser.parse_args()
    if args.smoke_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent),
                                                   pattern="test_*.py")
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if outcome.wasSuccessful() else 1)
    args.methods = list(METHODS) if args.methods.lower() == "all" else args.methods.split(",")
    args.methods = [m.strip() for m in args.methods]
    if not args.methods or any(m not in METHODS for m in args.methods):
        parser.error(f"--methods must contain members of {METHODS}, or all")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must not contain duplicates")
    if args.snrfrob_gamma_grid and not {"SNR", "SNRFrob"}.issubset(args.methods):
        parser.error("--snrfrob-gamma-grid requires both SNR and SNRFrob in --methods")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if any(seed < 0 or seed >= 2**32 for seed in [*args.seeds, args.validation_seed]):
        parser.error("Training and validation seeds must be in [0, 2**32)")
    for values in (args.seeds, args.sdro_lambdas, args.sdro_epsilons, args.snrfrob_gamma_grid or []):
        if len(set(values)) != len(values):
            parser.error("Seed and sweep lists must not contain duplicates")
    run_experiment(args)


if __name__ == '__main__':
    main()
