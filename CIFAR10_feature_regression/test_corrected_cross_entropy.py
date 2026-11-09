"""Numerical checks for Sections 2 and 3 of arXiv:2610.31642v3."""

import math
import unittest

import torch
import torch.nn.functional as F

from corrected_cross_entropy import (
    cross_entropy,
    log_softmax,
    softmax,
    soft_target_cross_entropy,
)


class CorrectedCrossEntropyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(37)

    def assert_zero_gradient_sum(self, gradient, rtol=2e-6):
        residual = gradient.sum(dim=-1).abs()
        scale = gradient.abs().sum(dim=-1)
        self.assertTrue((residual <= rtol * scale).all())

    def check_hard_target_tails(self, device):
        for margin in (18.0, 40.0, 80.0):
            with self.subTest(device=device, margin=margin):
                logits = torch.tensor([[margin / 2, -margin / 2, -margin / 2]],
                                      dtype=torch.float32, device=device, requires_grad=True)
                target = torch.zeros(1, dtype=torch.long, device=device)
                loss = cross_entropy(logits, target)
                gradient, = torch.autograd.grad(loss, logits)
                tail = math.exp(-margin)
                expected_loss = torch.tensor(math.log1p(2 * tail), dtype=logits.dtype,
                                             device=device)
                expected_gradient = torch.tensor(
                    [[-2 * tail, tail, tail]], dtype=logits.dtype, device=device,
                ) / (1 + 2 * tail)
                torch.testing.assert_close(loss, expected_loss, rtol=3e-6, atol=0)
                torch.testing.assert_close(gradient, expected_gradient, rtol=3e-6, atol=0)
                self.assertGreater(loss.item(), 0)
                self.assertLess(gradient[0, 0].item(), 0)
                self.assert_zero_gradient_sum(gradient)

    def test_hard_target_preserves_loss_and_gradients_past_precision_cliff(self):
        self.check_hard_target_tails("cpu")

    def test_log_softmax_and_softmax_preserve_target_derivatives(self):
        logits = torch.tensor([9.0, -9.0, -9.0], requires_grad=True)
        log_probabilities = log_softmax(logits)
        probabilities = softmax(logits)
        tail = math.exp(-18)
        expected_loss = math.log1p(2 * tail)
        self.assertAlmostEqual(log_probabilities[0].item(), -expected_loss, delta=1e-13)
        expected_log_gradient = torch.tensor([2 * tail, -tail, -tail]) / (1 + 2 * tail)
        log_gradient, = torch.autograd.grad(log_probabilities[0], logits)
        probability_gradient, = torch.autograd.grad(probabilities[0], logits)
        torch.testing.assert_close(log_gradient, expected_log_gradient, rtol=2e-6, atol=0)
        torch.testing.assert_close(probability_gradient, expected_log_gradient / (1 + 2 * tail),
                                   rtol=2e-6, atol=0)
        self.assertGreater(probability_gradient[0].item(), 0)
        self.assert_zero_gradient_sum(log_gradient)
        self.assert_zero_gradient_sum(probability_gradient)

    def test_values_and_gradients_match_normal_cross_entropy_for_all_reductions(self):
        values = torch.randn(6, 5, dtype=torch.float64)
        targets = torch.tensor([0, 4, 1, 3, 2, 0])
        for reduction in ("none", "sum", "mean"):
            with self.subTest(reduction=reduction):
                actual_logits = values.clone().requires_grad_(True)
                reference_logits = values.clone().requires_grad_(True)
                actual = cross_entropy(actual_logits, targets, reduction=reduction)
                reference = F.cross_entropy(reference_logits, targets, reduction=reduction)
                torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
                weights = torch.randn_like(actual)
                actual_gradient, = torch.autograd.grad(actual, actual_logits, grad_outputs=weights)
                reference_gradient, = torch.autograd.grad(reference, reference_logits, grad_outputs=weights)
                torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-11, atol=1e-12)
                self.assert_zero_gradient_sum(actual_gradient, rtol=1e-12)

    def test_class_axis_is_last_and_targets_can_be_scalar(self):
        values = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0, 1, 2], [3, 0, 2]])
        actual = cross_entropy(values, targets, reduction="none")
        reference = F.cross_entropy(values.reshape(-1, 4), targets.flatten(), reduction="none")
        self.assertEqual(actual.shape, targets.shape)
        torch.testing.assert_close(actual, reference.reshape_as(targets), rtol=1e-12, atol=1e-12)
        scalar = cross_entropy(values[0, 0], targets[0, 0], reduction="none")
        self.assertEqual(scalar.ndim, 0)
        torch.testing.assert_close(scalar, actual[0, 0])

    def test_ties_and_large_incorrect_logits_are_finite(self):
        logits = torch.tensor([[0.0, 0.0, 0.0], [1000.0, -1000.0, 0.0],
                               [1000.0, 1000.0, -1000.0], [-1000.0, 0.0, 1000.0]],
                              dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1, 1, 0, 0])
        actual = cross_entropy(logits, targets, reduction="none")
        reference = F.cross_entropy(logits, targets, reduction="none")
        actual_gradient, = torch.autograd.grad(actual.sum(), logits, retain_graph=True)
        reference_gradient, = torch.autograd.grad(reference.sum(), logits)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.isfinite(actual_gradient).all())
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-12, atol=1e-12)

    def test_shift_and_class_permutation_preserve_loss_and_gradients(self):
        values = torch.tensor([[9.0, -9.0, -9.0], [-3.0, 2.0, 1.0]], dtype=torch.float64)
        targets = torch.tensor([0, 2])
        permutation = torch.tensor([2, 0, 1])
        inverse = permutation.argsort()
        original = values.clone().requires_grad_(True)
        transformed = (values[:, permutation] + 200.0).requires_grad_(True)
        loss = cross_entropy(original, targets)
        changed_loss = cross_entropy(transformed, inverse[targets])
        gradient, = torch.autograd.grad(loss, original)
        changed_gradient, = torch.autograd.grad(changed_loss, transformed)
        torch.testing.assert_close(loss, changed_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gradient[:, permutation], changed_gradient, rtol=1e-12, atol=1e-12)

    def test_autograd_first_and_second_derivatives(self):
        values = torch.tensor([[0.2, -0.9, 1.4], [1.2, 0.3, -0.7]],
                              dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([2, 1])
        objective = lambda logits: cross_entropy(logits, targets, reduction="sum")
        self.assertTrue(torch.autograd.gradcheck(objective, (values,)))
        self.assertTrue(torch.autograd.gradgradcheck(objective, (values,)))
        self.assertTrue(torch.autograd.gradcheck(log_softmax, (values,)))
        self.assertTrue(torch.autograd.gradcheck(softmax, (values,)))

    def check_soft_target_cancellation(self, device):
        teacher = torch.tensor([[2.0, 1.0, 0.0]], device=device)
        perturbation = torch.tensor([[-1e-7, 1e-7, 0.0]], device=device)
        student = (teacher + perturbation).requires_grad_(True)
        loss = soft_target_cross_entropy(student, teacher)
        gradient, = torch.autograd.grad(loss, student)
        expected = student.detach().double().softmax(-1) - teacher.double().softmax(-1)
        torch.testing.assert_close(gradient.double(), expected, rtol=2e-5, atol=1e-14)
        self.assertGreater(gradient[0, 2].item(), 0)
        self.assert_zero_gradient_sum(gradient)

    def test_soft_targets_near_teacher_match_float64_reference(self):
        self.check_soft_target_cancellation("cpu")

    def test_equal_soft_targets_have_exact_zero_student_gradient(self):
        teacher = torch.tensor([[2.0, 1.0, 0.0], [9.0, -9.0, -9.0],
                                [-3.0, 0.5, 1.25]], requires_grad=True)
        student = teacher.detach().clone().requires_grad_(True)
        loss = soft_target_cross_entropy(student, teacher)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))
        self.assertIsNone(teacher.grad)
        expected = -(teacher.detach().double().softmax(-1) *
                     teacher.detach().double().log_softmax(-1)).sum(-1).mean()
        torch.testing.assert_close(loss.double(), expected, rtol=2e-6, atol=1e-8)

    def test_soft_targets_normal_values_gradients_and_reductions(self):
        teacher = torch.randn(4, 5, dtype=torch.float64)
        values = torch.randn(4, 5, dtype=torch.float64)
        for reduction in ("none", "sum", "mean"):
            with self.subTest(reduction=reduction):
                student = values.clone().requires_grad_(True)
                actual = soft_target_cross_entropy(student, teacher, reduction=reduction)
                per_item = -(teacher.softmax(-1) * values.log_softmax(-1)).sum(-1)
                reference = per_item if reduction == "none" else getattr(per_item, reduction)()
                torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
                gradient, = torch.autograd.grad(actual.sum(), student)
                expected = values.softmax(-1) - teacher.softmax(-1)
                if reduction == "mean":
                    expected /= len(values)
                torch.testing.assert_close(gradient, expected, rtol=1e-11, atol=1e-12)

    def test_far_soft_targets_use_finite_fallback(self):
        student = torch.tensor([[-1000.0, 1000.0, 0.0], [1000.0, -1000.0, 0.0]],
                               requires_grad=True)
        teacher = -student.detach()
        loss = soft_target_cross_entropy(student, teacher, reduction="sum")
        gradient, = torch.autograd.grad(loss, student)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(loss, torch.tensor(4000.0))
        torch.testing.assert_close(gradient, torch.tensor([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_precision_cliff_and_cancellation(self):
        self.check_hard_target_tails("cuda")
        self.check_soft_target_cancellation("cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
