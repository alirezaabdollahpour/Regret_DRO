"""Numerical corrections from Barron, Gnanapragasam, and Tschannen (2026).

Algorithms: https://arxiv.org/html/2610.31642v3, Sections 2 and 3.
The class axis is last. Integer targets have the other logit dimensions.
"""

import torch
from torch import nn


LOSS_IMPLEMENTATION = "arxiv_2610.31642v3"


def _reduce(loss, reduction):
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'.")


def log_softmax(logits):
    """Compute log probabilities with the Section 2 collapse correction."""
    index = logits.argmax(dim=-1, keepdim=True)
    maximum = logits.gather(-1, index)
    mask = torch.arange(logits.shape[-1], device=logits.device) == index
    # A constant at the selected class keeps its gradient in the other terms.
    relative = (logits - maximum).masked_fill(mask, 0.0)
    other_mass = relative.exp().masked_fill(mask, 0.0).sum(dim=-1, keepdim=True)
    return relative - torch.log1p(other_mass)


def softmax(logits):
    """Compute probabilities through the corrected log-softmax graph."""
    return log_softmax(logits).exp()


def cross_entropy(logits, targets, reduction="mean"):
    """Compute hard-target CE with the stable Section 2 algorithm (Eq. 6)."""
    if targets.shape != logits.shape[:-1] or targets.dtype != torch.long:
        raise ValueError("Integer targets must have shape logits.shape[:-1] and dtype torch.long.")
    target_index = targets.unsqueeze(-1)
    target_logit = logits.gather(-1, target_index)
    mask = torch.arange(logits.shape[-1], device=logits.device) == target_index
    relative = (logits - target_logit).masked_fill(mask, 0.0)
    # Include the constant zero in the maximum to handle wrong targets safely.
    offset = relative.amax(dim=-1, keepdim=True)
    other_mass = (relative - offset).exp().masked_fill(mask, 0.0).sum(dim=-1)
    offset = offset.squeeze(-1)
    loss = offset + torch.log1p(other_mass + torch.expm1(-offset))
    return _reduce(loss, reduction)


class CrossEntropyLoss(nn.Module):
    """Use the corrected hard-target loss with the existing criterion calls."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        return cross_entropy(logits, targets, self.reduction)


def soft_target_cross_entropy(student_logits, teacher_logits, reduction="mean"):
    """Compute fixed-teacher CE with the Section 3 student gradient (Eq. 9).

    Targets are raw teacher logits, not probabilities. The gradient replacement
    is for first-order differentiation, as in the paper's implementation.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have the same shape.")
    if student_logits.dtype != teacher_logits.dtype or student_logits.device != teacher_logits.device:
        raise ValueError("Student and teacher logits must have the same dtype and device.")
    # Form the replacement gradient without a graph, including the far-logit fallback.
    with torch.no_grad():
        student_log_probability = log_softmax(student_logits)
        student_probability = student_log_probability.exp()
        teacher_probability = softmax(teacher_logits)
        difference = student_logits - teacher_logits
        centered = difference - (teacher_probability * difference).sum(dim=-1, keepdim=True)
        log_ratio = centered - torch.log1p(
            (teacher_probability * torch.expm1(centered)).sum(dim=-1, keepdim=True)
        )
        probability_difference = torch.where(
            log_ratio.abs() < 1,
            teacher_probability * torch.expm1(log_ratio),
            student_probability - teacher_probability,
        )
        gradient = probability_difference - student_probability * probability_difference.sum(dim=-1, keepdim=True)
        value = -(teacher_probability * student_log_probability).sum(dim=-1)
    loss = value + (gradient * (student_logits - student_logits.detach())).sum(dim=-1)
    return _reduce(loss, reduction)
