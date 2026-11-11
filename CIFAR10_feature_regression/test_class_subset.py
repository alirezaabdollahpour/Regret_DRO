"""Check class-subset training and attacks with small synthetic embeddings."""

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import math
import tempfile
import unittest
from unittest import mock

import torch

import Multiclass_classification as experiment


class ClassSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_args(self, output_dir, classes):
        return argparse.Namespace(
            device="cpu", cache="unused.pth", data_dir="unused", feature_batch_size=8,
            num_workers=0, output_dir=output_dir, methods=list(experiment.METHODS),
            classes=classes, epochs=1, batch_size=8, seeds=[2022], sdro_lambdas=[10.0],
            sdro_epsilons=[0.02], snr_lambda=0.01, pgd_steps=2,
            snrfrob_gamma=0.1, snrfrob_gamma_grid=None, validation_fraction=0.25,
            validation_seed=2022, match_clean_tolerance=0.5,
        )

    def assert_class_metadata(self, metadata, classes):
        self.assertEqual(metadata["selected_classes"], classes)
        self.assertEqual(metadata["num_classes"], len(classes))
        self.assertEqual(metadata["label_mapping"],
                         {str(label): index for index, label in enumerate(classes)})

    def test_noncontiguous_labels_preserve_rows_and_source_tensors(self):
        features = torch.arange(21, dtype=torch.float32).reshape(7, 3)
        labels = torch.tensor([9, 7, 2, 0, 7, 2, 4])
        original_features, original_labels = features.clone(), labels.clone()
        selected_features, selected_labels, indices = experiment.select_class_subset(
            features, labels, [7, 2])
        self.assertEqual(indices.tolist(), [1, 2, 4, 5])
        self.assertEqual(selected_labels.tolist(), [1, 0, 1, 0])
        torch.testing.assert_close(selected_features, features[[1, 2, 4, 5]], rtol=0, atol=0)
        torch.testing.assert_close(features, original_features, rtol=0, atol=0)
        torch.testing.assert_close(labels, original_labels, rtol=0, atol=0)
        self.assertEqual(selected_features.dtype, features.dtype)
        self.assertEqual(selected_features.device, features.device)
        self.assertEqual(selected_labels.dtype, torch.long)
        self.assertEqual(indices.dtype, torch.long)

    def test_invalid_class_lists_and_missing_selected_classes_are_rejected(self):
        for classes in ([], [1], [1, 1], [-1, 2], [1, 10]):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                experiment.normalize_class_ids(classes)
        for labels in (torch.tensor([0, 1, 9]), torch.tensor([0, 9]),
                       torch.empty(0, dtype=torch.long)):
            with self.subTest(labels=labels.tolist()), self.assertRaises(ValueError):
                experiment.select_class_subset(torch.zeros(len(labels), 512), labels, [1, 7])

    def test_cli_class_examples_and_default(self):
        for options, expected in (([], list(range(10))),
                                  (["--classes", "1", "2"], [1, 2]),
                                  (["--classes", "3", "1", "2"], [1, 2, 3])):
            with self.subTest(options=options), \
                    mock.patch("sys.argv", ["experiment", "--methods", "all", *options]), \
                    mock.patch.object(experiment, "run_experiment") as run:
                experiment.main()
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0].classes, expected)
                self.assertEqual(run.call_args.args[0].methods, list(experiment.METHODS))

    def test_cli_rejects_invalid_classes_before_loading_data(self):
        for values in ([], ["1"], ["1", "1"], ["-1", "2"],
                       ["1", "10"], ["1", "2.5"], ["1,2"]):
            with self.subTest(values=values), \
                    mock.patch("sys.argv", ["experiment", "--classes", *values]), \
                    mock.patch.object(experiment, "run_experiment") as run, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                experiment.main()
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_all_methods_train_and_attack_only_two_or_three_selected_classes(self):
        generator = torch.Generator().manual_seed(83)
        data = {
            "train_features": 0.2 * torch.randn(8, 512, generator=generator),
            "train_labels": torch.tensor([0, 1, 7, 4, 1, 7, 4, 9]),
            "test_features": 0.2 * torch.randn(7, 512, generator=generator),
            "test_labels": torch.tensor([9, 7, 0, 4, 1, 7, 4]),
        }
        data["train_features"][[0, 7]] = 1000
        data["test_features"][[0, 2]] = 1000
        originals = {key: tensor.clone() for key, tensor in data.items()}
        for classes in ([1, 7], [1, 4, 7]):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory(prefix="class_subset_") as temporary:
                args = self.make_args(temporary, list(reversed(classes)))
                train_indices = [i for i, label in enumerate(data["train_labels"].tolist()) if label in classes]
                test_indices = [i for i, label in enumerate(data["test_labels"].tolist()) if label in classes]
                train_features = data["train_features"][train_indices]
                test_features = data["test_features"][test_indices]
                train_labels = torch.tensor([classes.index(data["train_labels"][i].item()) for i in train_indices])
                test_labels = torch.tensor([classes.index(data["test_labels"][i].item()) for i in test_indices])
                expected_mean = torch.linalg.vector_norm(test_features, dim=1).mean().item()
                expected_radii = [k * 0.008 * expected_mean for k in range(11)]
                trained, evaluated, attacked_radii = [], [], []
                actual_clean_train = experiment._train_clean_head
                actual_dro_loader = experiment.BaseLinearDRO._create_dataloader
                actual_evaluate, actual_attack = experiment.evaluate_model, experiment.pgd_attack

                def check_clean_training(model, loader, *positional, **keywords):
                    torch.testing.assert_close(loader.dataset.tensors[0], train_features, rtol=0, atol=0)
                    torch.testing.assert_close(loader.dataset.tensors[1], train_labels, rtol=0, atol=0)
                    self.assertEqual(model.linear.out_features, len(classes))
                    trained.append("clean")
                    return actual_clean_train(model, loader, *positional, **keywords)

                def check_dro_loader(wrapper, features, labels, batch_size):
                    torch.testing.assert_close(torch.from_numpy(features), train_features, rtol=0, atol=0)
                    torch.testing.assert_close(torch.from_numpy(labels), train_labels, rtol=0, atol=0)
                    self.assertEqual(wrapper.num_classes, len(classes))
                    trained.append(type(wrapper).__name__)
                    return actual_dro_loader(wrapper, features, labels, batch_size)

                def check_evaluation(head, features, labels, radii, *positional, **keywords):
                    torch.testing.assert_close(features, test_features, rtol=0, atol=0)
                    torch.testing.assert_close(labels, test_labels, rtol=0, atol=0)
                    self.assertEqual(radii, expected_radii)
                    self.assertEqual(tuple(head(features).shape), (len(test_indices), len(classes)))
                    self.assertTrue(torch.isfinite(experiment.cross_entropy(head(features), labels)))
                    for parameter in head.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(head.linear.weight.grad.norm().item(), 0)
                    evaluated.append(head)
                    return actual_evaluate(head, features, labels, radii, *positional, **keywords)

                def check_attack(head, features, labels, epsilon, **keywords):
                    torch.testing.assert_close(features, test_features, rtol=0, atol=0)
                    torch.testing.assert_close(labels, test_labels, rtol=0, atol=0)
                    attacked_radii.append(epsilon)
                    perturbed, robust_correct = actual_attack(head, features, labels, epsilon, **keywords)
                    self.assertTrue(torch.isfinite(perturbed).all())
                    self.assertTrue((torch.linalg.vector_norm(perturbed - features, dim=1) <= epsilon + 1e-6).all())
                    self.assertEqual(tuple(robust_correct.shape), tuple(labels.shape))
                    return perturbed, robust_correct

                log = io.StringIO()
                with mock.patch.object(experiment, "load_or_extract_features", return_value=data), \
                        mock.patch.object(experiment, "_train_clean_head", side_effect=check_clean_training), \
                        mock.patch.object(experiment.BaseLinearDRO, "_create_dataloader", new=check_dro_loader), \
                        mock.patch.object(experiment, "evaluate_model", side_effect=check_evaluation), \
                        mock.patch.object(experiment, "pgd_attack", side_effect=check_attack), \
                        mock.patch.object(experiment, "plot_comparison"), \
                        redirect_stdout(log), redirect_stderr(log):
                    output = experiment.run_experiment(args)
                self.assertEqual(len(trained), len(experiment.METHODS))
                self.assertEqual(Counter(trained)["clean"], 3)
                self.assertEqual(len(evaluated), len(experiment.METHODS))
                self.assertEqual(Counter(attacked_radii), {radius: len(experiment.METHODS) for radius in expected_radii})
                for key, tensor in originals.items():
                    torch.testing.assert_close(data[key], tensor, rtol=0, atol=0)
                with (output / "results.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(Counter(row["method"] for row in rows), {method: 11 for method in experiment.METHODS})
                for row in rows:
                    self.assertEqual(json.loads(row["selected_classes"]), classes)
                    self.assertEqual(int(row["num_classes"]), len(classes))
                    self.assertEqual(int(row["train_size"]), len(train_indices))
                    self.assertEqual(int(row["fit_size"]), len(train_indices))
                    self.assertEqual(int(row["validation_size"]), 0)
                    self.assertEqual(int(row["test_size"]), len(test_indices))
                    self.assertEqual(float(row["mean_test_norm"]), expected_mean)
                    self.assertAlmostEqual(float(row["absolute_radius"]), float(row["relative_level"]) * expected_mean)
                    self.assertLessEqual(float(row["pgd_accuracy"]), float(row["clean_accuracy"]))
                    self.assert_class_metadata(json.loads(row["hyperparameters"]), classes)
                for relative_path in {row["checkpoint"] for row in rows}:
                    saved = torch.load(output / relative_path, weights_only=True)
                    self.assert_class_metadata(saved["hyperparameters"], classes)
                    self.assertEqual(tuple(saved["state_dict"]["linear.weight"].shape), (len(classes), 512))
                    self.assertEqual(tuple(saved["state_dict"]["linear.bias"].shape), (len(classes),))
                    self.assertEqual(len(saved["loss_history"]), 1)
                    self.assertTrue(all(math.isfinite(value) for value in saved["loss_history"]))
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(2022)
                        initial = experiment.LogisticRegression(num_classes=len(classes))
                    self.assertFalse(torch.equal(saved["state_dict"]["linear.weight"], initial.linear.weight))
                    initial.load_state_dict(saved["state_dict"])
                    self.assertTrue(torch.isfinite(initial(test_features)).all())
                metadata = json.loads((output / "experiment.json").read_text())
                self.assert_class_metadata(metadata, classes)
                self.assertEqual(metadata["feature_metadata"], experiment.FEATURE_METADATA)
                self.assertEqual(metadata["mean_test_norm"], expected_mean)

    def test_subset_validation_uses_original_cache_indices_and_no_test_data_for_selection(self):
        classes = [2, 7]
        generator = torch.Generator().manual_seed(419)
        training_features = torch.randn(12, 512, generator=generator)
        training_labels = torch.tensor([0, 7, 2, 9, 2, 7, 0, 2, 7, 9, 7, 2])
        test_features = 10 + torch.randn(5, 512, generator=generator)
        test_labels = torch.tensor([0, 7, 2, 9, 7])
        selected_indices = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11])
        remapped_labels = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0])
        fit, validation = experiment.stratified_split_indices(remapped_labels, 0.25, 2022)
        fit_features, fit_labels = training_features[selected_indices[fit]], remapped_labels[fit]
        validation_features = training_features[selected_indices[validation]]
        validation_labels = remapped_labels[validation]
        selected_test_features, selected_test_labels = test_features[[1, 2, 4]], torch.tensor([1, 0, 1])
        selected = {"ready": False}

        class GuardedData(dict):
            def __getitem__(self, key):
                if key.startswith("test_") and not selected["ready"]:
                    raise AssertionError("Test data were accessed before validation selection.")
                return super().__getitem__(key)

        data = GuardedData(train_features=training_features, train_labels=training_labels,
                           test_features=test_features, test_labels=test_labels)
        actual_train, actual_select = experiment._train_clean_head, experiment.select_matched_gamma
        actual_evaluate = experiment.evaluate_model
        initial_states, orders, evaluation_splits = [], [], []

        def check_training(model, loader, *positional, **keywords):
            torch.testing.assert_close(loader.dataset.tensors[0], fit_features, rtol=0, atol=0)
            torch.testing.assert_close(loader.dataset.tensors[1], fit_labels, rtol=0, atol=0)
            self.assertEqual(model.linear.out_features, 2)
            initial_states.append({key: value.clone() for key, value in model.state_dict().items()})
            generator_state = loader.generator.get_state()
            orders.append(torch.cat([features for features, _ in loader]))
            loader.generator.set_state(generator_state)
            return actual_train(model, loader, *positional, **keywords)

        def check_selection(reference, candidates):
            self.assertFalse(selected["ready"])
            gamma = actual_select(reference, candidates)
            selected.update(ready=True, gamma=gamma)
            return gamma

        def check_evaluation(head, features, labels, radii, *positional, **keywords):
            expected_features = selected_test_features if selected["ready"] else validation_features
            expected_labels = selected_test_labels if selected["ready"] else validation_labels
            torch.testing.assert_close(features, expected_features, rtol=0, atol=0)
            torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
            mean_norm = torch.linalg.vector_norm(expected_features, dim=1).mean().item()
            self.assertEqual(radii, [k * 0.008 * mean_norm for k in range(11)])
            evaluation_splits.append("test" if selected["ready"] else "validation")
            return actual_evaluate(head, features, labels, radii, *positional, **keywords)

        with tempfile.TemporaryDirectory(prefix="subset_validation_") as temporary:
            args = self.make_args(temporary, classes)
            args.methods = list(experiment.CLEAN_METHODS)
            args.snrfrob_gamma_grid = [0.0, 0.1]
            log = io.StringIO()
            with mock.patch.object(experiment, "load_or_extract_features", return_value=data), \
                    mock.patch.object(experiment, "_train_clean_head", side_effect=check_training), \
                    mock.patch.object(experiment, "select_matched_gamma", side_effect=check_selection), \
                    mock.patch.object(experiment, "evaluate_model", side_effect=check_evaluation), \
                    mock.patch.object(experiment, "plot_comparison"), \
                    redirect_stdout(log), redirect_stderr(log):
                output = experiment.run_experiment(args)
            self.assertEqual(len(initial_states), 4)
            self.assertEqual(evaluation_splits, ["validation"] * 3 + ["test"] * 3)
            for state, order in zip(initial_states[1:], orders[1:]):
                for key in initial_states[0]:
                    torch.testing.assert_close(state[key], initial_states[0][key], rtol=0, atol=0)
                torch.testing.assert_close(order, orders[0], rtol=0, atol=0)
            split = torch.load(output / "validation_split.pth", weights_only=True)
            self.assert_class_metadata(split, classes)
            self.assertEqual(split["index_space"], "full_training_feature_cache")
            self.assertEqual(split["fit_indices"].tolist(), selected_indices[fit].tolist())
            self.assertEqual(split["validation_indices"].tolist(), selected_indices[validation].tolist())
            self.assertFalse(set(split["fit_indices"].tolist()) & set(split["validation_indices"].tolist()))
            selection = json.loads((output / "validation_selection.json").read_text())
            self.assertFalse(selection["selection_uses_test_data"])
            self.assertEqual(selection["selected_gamma"], selected["gamma"])
            self.assertEqual(selection["fit_size"], len(fit))
            self.assertEqual(selection["validation_size"], len(validation))
            for filename in ("validation_results.csv", "results.csv"):
                with (output / filename).open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 33)
                for row in rows:
                    self.assertEqual(json.loads(row["selected_classes"]), classes)
                    self.assertEqual(int(row["num_classes"]), 2)
                    self.assertEqual(int(row["train_size"]), len(selected_indices))
                    self.assertEqual(int(row["fit_size"]), len(fit))
                    self.assertEqual(int(row["validation_size"]), len(validation))
                    if filename == "validation_results.csv":
                        self.assertNotIn("test_size", row)
                    else:
                        self.assertEqual(int(row["test_size"]), len(selected_test_labels))
                    saved = torch.load(output / row["checkpoint"], weights_only=True)
                    self.assert_class_metadata(saved["hyperparameters"], classes)
                    self.assertEqual(tuple(saved["state_dict"]["linear.weight"].shape), (2, 512))


if __name__ == "__main__":
    unittest.main(verbosity=2)
