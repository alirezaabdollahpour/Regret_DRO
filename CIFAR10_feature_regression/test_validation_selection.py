"""Check the validation-only SNRFrob selection and final head reuse."""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import Multiclass_classification as experiment


class ValidationSelectionTests(unittest.TestCase):
    def test_gamma_selection_is_paired_and_precedes_test_access(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        generator = torch.Generator().manual_seed(409)
        training_features = torch.randn(20, 512, generator=generator)
        training_labels = torch.arange(20).remainder(2)
        test_features = 50 + 5 * torch.randn(5, 512, generator=generator)
        test_labels = torch.tensor([0, 1, 0, 1, 0])
        fit_indices, validation_indices = experiment.stratified_split_indices(training_labels, 0.2, 2022)
        fit_features, fit_labels = training_features[fit_indices], training_labels[fit_indices]
        validation_features = training_features[validation_indices]
        validation_norm = torch.linalg.vector_norm(validation_features, dim=1).mean().item()
        test_norm = torch.linalg.vector_norm(test_features, dim=1).mean().item()
        selected = {"ready": False}

        class GuardedData(dict):
            def __getitem__(self, key):
                if key in ("test_features", "test_labels") and not selected["ready"]:
                    raise AssertionError("Test data were read before gamma selection.")
                return super().__getitem__(key)

        data = GuardedData(train_features=training_features, train_labels=training_labels,
                           test_features=test_features, test_labels=test_labels)
        captured_training = defaultdict(list)
        trained_heads, evaluated_test_heads = {}, []
        actual_train = experiment._train_clean_head
        actual_configuration = experiment.train_clean_configuration
        actual_evaluate = experiment.evaluate_model
        actual_select = experiment.select_matched_gamma

        def check_train(model, loader, epochs, learning_rate, checkpoint_dir,
                        run_id, device, method, snr_lambda=0.0, snrfrob_gamma=0.0):
            self.assertTrue(torch.equal(loader.dataset.tensors[0], fit_features))
            self.assertTrue(torch.equal(loader.dataset.tensors[1], fit_labels))
            state = {key: value.clone() for key, value in model.state_dict().items()}
            generator_state = loader.generator.get_state()
            order = torch.cat([features for features, _ in loader])
            loader.generator.set_state(generator_state)
            captured_training[run_id].append((method, snrfrob_gamma, state, order))
            self.assertEqual(learning_rate, 0.001)
            return actual_train(model, loader, epochs, learning_rate, checkpoint_dir,
                                run_id, device, method, snr_lambda, snrfrob_gamma)

        def check_configuration(method, features, labels, seed, hp, directory, device):
            head, history = actual_configuration(method, features, labels, seed, hp, directory, device)
            trained_heads[id(head)] = (method, seed, hp.get("snrfrob_gamma"))
            return head, history

        def check_selection(reference, candidates):
            self.assertFalse(selected["ready"])
            self.assertEqual(set(reference), {2022, 2023})
            self.assertEqual(set(candidates), {0.0, 0.1, 1.0})
            self.assertTrue(all(set(values) == set(reference) for values in candidates.values()))
            gamma = actual_select(reference, candidates)
            selected.update(ready=True, gamma=gamma)
            return gamma

        def check_evaluation(head, features, labels, radii, *args, **kwargs):
            if features is test_features:
                self.assertTrue(selected["ready"])
                method, seed, gamma = trained_heads[id(head)]
                if method == "SNRFrob":
                    self.assertEqual(gamma, selected["gamma"])
                evaluated_test_heads.append((method, seed, gamma))
                scale = test_norm
            else:
                self.assertFalse(selected["ready"])
                self.assertTrue(torch.equal(features, validation_features))
                scale = validation_norm
            self.assertEqual(radii, [k * 0.008 * scale for k in range(11)])
            return actual_evaluate(head, features, labels, radii, *args, **kwargs)

        log = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="snrfrob_validation_") as temporary:
            args = argparse.Namespace(
                device="cpu", cache="unused.pth", data_dir="unused", feature_batch_size=4,
                num_workers=0, output_dir=temporary, methods=["SAA", "SNR", "SNRFrob"],
                epochs=1, batch_size=8, seeds=[2022, 2023], sdro_lambdas=[10.0],
                sdro_epsilons=[0.02], snr_lambda=2.0, snrfrob_gamma=0.01,
                snrfrob_gamma_grid=[0.0, 0.1, 1.0], validation_fraction=0.2,
                validation_seed=2022, match_clean_tolerance=0.5, pgd_steps=2,
            )
            with mock.patch.object(experiment, "load_or_extract_features", return_value=data), \
                    mock.patch.object(experiment, "_train_clean_head", side_effect=check_train), \
                    mock.patch.object(experiment, "train_clean_configuration", side_effect=check_configuration), \
                    mock.patch.object(experiment, "select_matched_gamma", side_effect=check_selection), \
                    mock.patch.object(experiment, "evaluate_model", side_effect=check_evaluation), \
                    redirect_stdout(log), redirect_stderr(log):
                output_dir = experiment.run_experiment(args)

            for seed, runs in captured_training.items():
                self.assertEqual(len(runs), 5)
                self.assertEqual(Counter(run[0] for run in runs), {"SNR": 1, "SNRFrob": 3, "SAA": 1})
                reference_state, reference_order = runs[0][2:]
                for _, _, state, order in runs:
                    self.assertTrue(torch.equal(order, reference_order))
                    for key in reference_state:
                        torch.testing.assert_close(state[key], reference_state[key], rtol=0, atol=0)
            self.assertEqual(len(evaluated_test_heads), 6)
            self.assertEqual(Counter(item[0] for item in evaluated_test_heads), {"SAA": 2, "SNR": 2, "SNRFrob": 2})

            selection = json.loads((output_dir / "validation_selection.json").read_text())
            self.assertEqual(selection["selected_gamma"], selected["gamma"])
            self.assertFalse(selection["selection_uses_test_data"])
            self.assertFalse(selection["refit_on_full_training_data"])
            self.assertEqual(selection["fit_size"], len(fit_indices))
            self.assertEqual(selection["validation_size"], len(validation_indices))
            self.assertEqual(selection["mean_validation_norm"], validation_norm)
            means = {float(key): value for key, value in selection["candidate_mean_clean_accuracies"].items()}
            reference_mean = selection["reference_mean_clean_accuracy"]
            expected_gamma = min(means, key=lambda gamma: (abs(means[gamma] - reference_mean), gamma))
            self.assertEqual(selected["gamma"], expected_gamma)
            self.assertEqual(selection["signed_clean_accuracy_gap_pp"], means[expected_gamma] - reference_mean)
            self.assertEqual(selection["matched_within_tolerance"], selection["absolute_clean_accuracy_gap_pp"] <= 0.5)

            split = torch.load(output_dir / "validation_split.pth", weights_only=True)
            self.assertTrue(torch.equal(split["fit_indices"], fit_indices))
            self.assertTrue(torch.equal(split["validation_indices"], validation_indices))
            with (output_dir / "validation_results.csv").open(newline="") as stream:
                validation_rows = list(csv.DictReader(stream))
            with (output_dir / "results.csv").open(newline="") as stream:
                final_rows = list(csv.DictReader(stream))
            self.assertEqual(len(validation_rows), 8 * 11)
            self.assertEqual(len(final_rows), 6 * 11)
            self.assertEqual(len(list(output_dir.rglob("validation.pth"))), 8)
            self.assertEqual(len(list(output_dir.rglob("final.pth"))), 6)
            for row in validation_rows:
                self.assertEqual(float(row["mean_validation_norm"]), validation_norm)
                self.assertNotIn("mean_test_norm", row)
            for row in final_rows:
                self.assertEqual(float(row["mean_test_norm"]), test_norm)
                hp = json.loads(row["hyperparameters"])
                self.assertEqual(hp["training_split"], "stratified_fit")
                self.assertEqual(hp["fit_size"], len(fit_indices))
                if row["method"] == "SNRFrob":
                    self.assertEqual(float(row["snrfrob_gamma"]), expected_gamma)
                final_path = output_dir / row["checkpoint"]
                final_checkpoint = torch.load(final_path, weights_only=True)
                if row["method"] != "SAA":
                    validation_checkpoint = torch.load(final_path.with_name("validation.pth"), weights_only=True)
                    for key, value in final_checkpoint["state_dict"].items():
                        torch.testing.assert_close(value, validation_checkpoint["state_dict"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
