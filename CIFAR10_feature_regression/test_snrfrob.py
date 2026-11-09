"""Small CPU checks for Frobenius regularization and validation matching."""

import copy
import io
import math
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

import Multiclass_classification as experiment


class SNRFrobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(61)

    def test_zero_gamma_matches_saa_at_each_epoch(self):
        features = torch.randn(17, 512)
        labels = torch.randint(0, 10, (17,))
        initial = experiment.LogisticRegression()
        saa, frobenius = copy.deepcopy(initial), copy.deepcopy(initial)

        def loader():
            return DataLoader(
                TensorDataset(features, labels), batch_size=6, shuffle=True,
                generator=torch.Generator().manual_seed(2022),
            )

        with tempfile.TemporaryDirectory() as directory:
            saa_dir = Path(directory) / "saa"
            frobenius_dir = Path(directory) / "frobenius"
            saa_losses = experiment.train_saa(
                saa, loader(), epochs=2, checkpoint_dir=saa_dir, device="cpu",
            )
            frobenius_losses = experiment.train_snrfrob(
                frobenius, loader(), epochs=2, checkpoint_dir=frobenius_dir,
                device="cpu", snrfrob_gamma=0.0,
            )
            self.assertEqual(saa_losses, frobenius_losses)
            saa_files = sorted(saa_dir.glob("*.pth"))
            frobenius_files = sorted(frobenius_dir.glob("*.pth"))
            self.assertEqual(len(saa_files), 2)
            self.assertEqual(len(frobenius_files), 2)
            for saa_file, frobenius_file in zip(saa_files, frobenius_files):
                saa_state = torch.load(saa_file, map_location="cpu", weights_only=True)
                frobenius_state = torch.load(frobenius_file, map_location="cpu", weights_only=True)
                for name in saa_state:
                    torch.testing.assert_close(saa_state[name], frobenius_state[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(frobenius.linear.weight, initial.linear.weight))

    def test_positive_gamma_matches_exact_weight_only_objective(self):
        features = torch.randn(7, 512, dtype=torch.float64)
        labels = torch.randint(0, 10, (7,))
        actual = experiment.LogisticRegression().double()
        with torch.no_grad():
            actual.linear.bias.copy_(torch.linspace(-2, 2, 10, dtype=torch.float64))
        expected = copy.deepcopy(actual)
        gamma = 0.7
        optimizer = torch.optim.Adam(expected.parameters(), lr=0.001, weight_decay=0.0)
        clean_ce = experiment.cross_entropy(expected(features), labels)
        loss = clean_ce + gamma * expected.linear.weight.square().sum()
        loss.backward()
        expected_gradients = [parameter.grad.clone() for parameter in expected.parameters()]
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            history = experiment.train_snrfrob(
                actual, DataLoader(TensorDataset(features, labels), batch_size=7),
                epochs=1, checkpoint_dir=directory, device="cpu", snrfrob_gamma=gamma,
            )
        self.assertEqual(len(history), 1)
        self.assertTrue(math.isfinite(history[0]))
        self.assertEqual(actual(features).shape, (7, 10))
        for actual_parameter, expected_parameter, expected_gradient in zip(
                actual.parameters(), expected.parameters(), expected_gradients):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
            torch.testing.assert_close(actual_parameter.grad, expected_gradient, rtol=0, atol=0)

    def test_invalid_gamma_is_rejected(self):
        features = torch.randn(2, 512)
        labels = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(features, labels), batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            for gamma in (-0.1, float("nan"), float("inf")):
                with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                    experiment.train_snrfrob(
                        experiment.LogisticRegression(), loader, epochs=1,
                        checkpoint_dir=directory, device="cpu", snrfrob_gamma=gamma,
                    )

    def test_diagnostics_use_squared_frobenius_norm(self):
        head = experiment.LogisticRegression().double()
        features = torch.randn(9, 512, dtype=torch.float64)
        labels = torch.randint(0, 10, (9,))
        gamma = 0.3
        metrics = experiment.clean_objective_metrics(
            head, features, labels, snr_lambda=7.0, batch_size=4, snrfrob_gamma=gamma,
        )
        with torch.no_grad():
            penalty = head.linear.weight.square().sum().item()
            clean_ce = experiment.cross_entropy(head(features), labels).item()
        self.assertAlmostEqual(metrics["clean_ce"], clean_ce, places=12)
        self.assertAlmostEqual(metrics["frobenius_penalty"], penalty, places=12)
        self.assertAlmostEqual(metrics["weighted_frobenius_penalty"], gamma * penalty, places=12)
        self.assertAlmostEqual(metrics["final_frobenius_norm"] ** 2, penalty, places=12)
        self.assertAlmostEqual(metrics["total_objective"], clean_ce + gamma * penalty, places=12)

    def test_stratified_split_is_complete_disjoint_and_deterministic(self):
        labels = torch.tensor([0] * 9 + [1] * 15 + [2] * 6)
        state = torch.random.get_rng_state().clone()
        fit, validation = experiment.stratified_split_indices(labels, 0.2, 2022)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        repeated_fit, repeated_validation = experiment.stratified_split_indices(labels, 0.2, 2022)
        self.assertTrue(torch.equal(fit, repeated_fit))
        self.assertTrue(torch.equal(validation, repeated_validation))
        self.assertEqual(fit.dtype, torch.long)
        self.assertEqual(validation.dtype, torch.long)
        self.assertEqual(fit.device.type, "cpu")
        self.assertEqual(validation.device.type, "cpu")
        self.assertFalse(set(fit.tolist()) & set(validation.tolist()))
        self.assertEqual(sorted(fit.tolist() + validation.tolist()), list(range(len(labels))))
        for label in labels.unique():
            total = labels.eq(label).sum().item()
            val_count = labels[validation].eq(label).sum().item()
            self.assertGreater(val_count, 0)
            self.assertLess(val_count, total)
            self.assertLessEqual(abs(val_count - 0.2 * total), 1)

    def test_invalid_split_is_rejected(self):
        labels = torch.tensor([0, 0, 1, 1])
        for fraction in (-0.1, 0.0, 1.0, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                experiment.stratified_split_indices(labels, fraction, 2022)
        with self.assertRaises(ValueError):
            experiment.stratified_split_indices(torch.tensor([0, 0, 1]), 0.2, 2022)

    def test_selection_matches_mean_clean_accuracy_and_breaks_ties_by_gamma(self):
        reference = {2022: 80.0, 2023: 84.0}
        candidates = {
            2.0: {2022: 79.0, 2023: 83.0},
            0.2: {2022: 78.0, 2023: 86.0},
            0.1: {2022: 81.0, 2023: 83.0},
        }
        self.assertEqual(experiment.select_matched_gamma(reference, candidates), 0.1)
        # Selection compares aggregate accuracy, not the mean per-seed gap.
        candidates = {0.2: {2022: 78.0, 2023: 86.0},
                      0.1: {2022: 80.1, 2023: 84.1}}
        self.assertEqual(experiment.select_matched_gamma(reference, candidates), 0.2)

    def test_selection_rejects_invalid_or_unpaired_validation_results(self):
        reference = {2022: 80.0, 2023: 84.0}
        invalid_candidates = [
            {},
            {0.1: {2022: 80.0}},
            {0.1: {2022: 80.0, 2024: 84.0}},
            {0.1: {2022: float("nan"), 2023: 84.0}},
            {-0.1: {2022: 80.0, 2023: 84.0}},
            {float("inf"): {2022: 80.0, 2023: 84.0}},
        ]
        for candidates in invalid_candidates:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                experiment.select_matched_gamma(reference, candidates)
        for invalid_reference in ({}, {2022: float("inf"), 2023: 84.0}):
            with self.subTest(reference=invalid_reference), self.assertRaises(ValueError):
                experiment.select_matched_gamma(invalid_reference, {0.1: reference})

    def test_cli_fixed_gamma_dispatches_requested_experiment(self):
        for gamma in (0.0, 0.1, 2.0):
            argv = ["Multiclass_classification.py", "--methods", "SAA,SNR,SNRFrob",
                    "--epochs", "10", "--snr-lambda", "2.0", "--snrfrob-gamma", str(gamma),
                    "--seeds", "2022", "--pgd-steps", "10"]
            with self.subTest(gamma=gamma), mock.patch("sys.argv", argv), \
                    mock.patch.object(experiment, "run_experiment") as run:
                experiment.main()
            run.assert_called_once()
            args = run.call_args.args[0]
            self.assertEqual(args.methods, ["SAA", "SNR", "SNRFrob"])
            self.assertEqual(args.epochs, 10)
            self.assertEqual(args.snr_lambda, 2.0)
            self.assertEqual(args.snrfrob_gamma, gamma)
            self.assertIsNone(args.snrfrob_gamma_grid)
            self.assertEqual(args.seeds, [2022])
            self.assertEqual(args.pgd_steps, 10)

    def test_cli_gamma_grid_dispatches_validation_settings(self):
        argv = ["Multiclass_classification.py", "--methods", "SNR,SNRFrob",
                "--snrfrob-gamma-grid", "0", "0.01", "0.1", "2",
                "--validation-fraction", "0.25", "--validation-seed", "17",
                "--match-clean-tolerance", "0.2"]
        with mock.patch("sys.argv", argv), mock.patch.object(experiment, "run_experiment") as run:
            experiment.main()
        run.assert_called_once()
        args = run.call_args.args[0]
        self.assertEqual(args.methods, ["SNR", "SNRFrob"])
        self.assertEqual(args.snrfrob_gamma_grid, [0.0, 0.01, 0.1, 2.0])
        self.assertEqual(args.validation_fraction, 0.25)
        self.assertEqual(args.validation_seed, 17)
        self.assertEqual(args.match_clean_tolerance, 0.2)

    def test_cli_rejects_invalid_gamma_sweep_before_training(self):
        invalid_options = [
            ["--snrfrob-gamma", "-0.1"],
            ["--snrfrob-gamma", "nan"],
            ["--snrfrob-gamma", "inf"],
            ["--snrfrob-gamma-grid", "0", "nan"],
            ["--snrfrob-gamma-grid", "-0.1", "0.1"],
            ["--snrfrob-gamma-grid", "0.1", "0.1"],
            ["--snrfrob-gamma-grid", "0.1", "1", "--snrfrob-gamma", "0.1"],
            ["--snrfrob-gamma-grid", "0.1", "1", "--validation-fraction", "0"],
            ["--snrfrob-gamma-grid", "0.1", "1", "--validation-fraction", "1"],
            ["--snrfrob-gamma-grid", "0.1", "1", "--methods", "SAA,SNR"],
            ["--snrfrob-gamma-grid", "0.1", "1", "--methods", "SNRFrob"],
        ]
        for options in invalid_options:
            argv = ["Multiclass_classification.py", "--methods", "SNR,SNRFrob", *options]
            with self.subTest(options=options), mock.patch("sys.argv", argv), \
                    mock.patch.object(experiment, "run_experiment") as run, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                experiment.main()
            self.assertEqual(caught.exception.code, 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
