"""Exercise the all-method runner on synthetic embeddings, without downloads."""

import argparse
import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import torch

import Multiclass_classification as experiment


class ExperimentIntegrationTests(unittest.TestCase):
    def test_all_methods_share_radii_and_reuse_clean_heads(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        generator = torch.Generator().manual_seed(79)
        data = {
            "train_features": torch.randn(4, 512, generator=generator),
            "train_labels": torch.tensor([0, 1, 2, 3]),
            "test_features": torch.randn(4, 512, generator=generator),
            "test_labels": torch.tensor([0, 1, 2, 3]),
        }
        expected_mean = torch.linalg.vector_norm(data["test_features"], dim=1).mean().item()
        log = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="multiclass_integration_") as temp:
            args = argparse.Namespace(
                device="cpu", cache="unused.pth", data_dir="unused", feature_batch_size=4,
                num_workers=0, output_dir=temp, methods=list(experiment.METHODS),
                epochs=1, batch_size=4, seeds=[2022], sdro_lambdas=[10.0],
                sdro_epsilons=[0.02, 0.002], snr_lambda=0.01, pgd_steps=10,
                snrfrob_gamma=0.1, snrfrob_gamma_grid=None, validation_fraction=0.2,
                validation_seed=2022, match_clean_tolerance=0.5,
            )
            with mock.patch.object(experiment, "load_or_extract_features", return_value=data), \
                    mock.patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("Standard CE was called")), \
                    mock.patch.object(experiment, "train_saa", wraps=experiment.train_saa) as saa, \
                    mock.patch.object(experiment, "train_snr", wraps=experiment.train_snr) as snr, \
                    mock.patch.object(experiment, "train_snrfrob", wraps=experiment.train_snrfrob) as frobenius, \
                    redirect_stdout(log), redirect_stderr(log):
                result_dir = experiment.run_experiment(args)
            self.assertEqual(saa.call_count, 1)
            self.assertEqual(snr.call_count, 1)
            self.assertEqual(frobenius.call_count, 1)
            with (result_dir / "results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 14 * 11)
            counts = Counter(row["method"] for row in rows)
            self.assertEqual(counts, {"SAA": 11, "SNR": 11, "SNRFrob": 11, "WRM": 11,
                                      "Dual": 22, "WGF": 22, "WFR": 22,
                                      "SVG": 22, "RGO": 22})
            checkpoints = {row["checkpoint"] for row in rows}
            self.assertEqual(len(checkpoints), 14)
            self.assertEqual(len(list(result_dir.rglob("final.pth"))), 14)
            self.assertEqual(len(list(result_dir.rglob("*_epoch_1.pth"))), 14)
            self.assertEqual(len(list(result_dir.rglob("*.pdf"))), 4)
            unique_rows = {(row["method"], row["configuration"], row["seed"],
                            row["relative_level"]) for row in rows}
            self.assertEqual(len(unique_rows), len(rows))
            for row in rows:
                self.assertEqual(float(row["mean_test_norm"]), expected_mean)
                level = float(row["relative_level"])
                radius = float(row["absolute_radius"])
                self.assertAlmostEqual(radius, level * expected_mean)
                self.assertAlmostEqual(float(row["pgd_step_size"]), radius / 8)
                self.assertEqual(row["metric"], "feature-space PGD accuracy")
                self.assertEqual(row["cross_entropy"], experiment.LOSS_IMPLEMENTATION)
                self.assertLessEqual(float(row["pgd_accuracy"]), float(row["clean_accuracy"]))
                if level == 0:
                    self.assertEqual(row["pgd_accuracy"], row["clean_accuracy"])
                self.assertTrue((result_dir / row["checkpoint"]).is_file())
                if row["method"] in experiment.CLEAN_METHODS:
                    self.assertEqual(float(row["learning_rate"]), 0.001)
                    self.assertEqual(float(row["weight_decay"]), 0.0)
                    self.assertEqual(row["epsilon"], "")
                    self.assertEqual(row["lambda_param"], "")
                if row["method"] == "SNR":
                    penalty = float(row["spectral_penalty"])
                    self.assertAlmostEqual(float(row["final_op_norm"]) ** 2, penalty)
                    self.assertAlmostEqual(float(row["weighted_spectral_penalty"]), 0.01 * penalty)
                    self.assertAlmostEqual(float(row["total_objective"]),
                                           float(row["clean_ce"]) + 0.01 * penalty)
                if row["method"] == "SNRFrob":
                    penalty = float(row["frobenius_penalty"])
                    self.assertEqual(float(row["snrfrob_gamma"]), 0.1)
                    self.assertEqual(row["snr_lambda"], "")
                    self.assertAlmostEqual(float(row["final_frobenius_norm"]) ** 2, penalty)
                    self.assertAlmostEqual(float(row["weighted_frobenius_penalty"]), 0.1 * penalty)
                    self.assertAlmostEqual(float(row["total_objective"]), float(row["clean_ce"]) + 0.1 * penalty)
            snr_row = next(row for row in rows if row["method"] == "SNR")
            saved = torch.load(result_dir / snr_row["checkpoint"], weights_only=True)
            head = experiment.LogisticRegression()
            head.load_state_dict(saved["state_dict"])
            metrics = experiment.clean_objective_metrics(
                head, data["train_features"], data["train_labels"], 0.01, batch_size=4)
            self.assertEqual(saved["diagnostics"], metrics)
            with (result_dir / "experiment.json").open() as stream:
                metadata = json.load(stream)
            self.assertEqual(metadata["feature_metadata"], experiment.FEATURE_METADATA)
            self.assertEqual(metadata["cross_entropy"], experiment.LOSS_IMPLEMENTATION)
            self.assertEqual(metadata["mean_test_norm"], expected_mean)


if __name__ == "__main__":
    unittest.main(verbosity=2)
