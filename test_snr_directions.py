"""Small analytic checks for affine-head direction diagnostics."""

import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import snr_directions as directions


class DirectionGeometryTests(unittest.TestCase):
    def test_feature_metadata_matches_all_heads(self):
        metadata = {"backbone": "resnet18", "feature_dim": 512,
                    "preprocessing": {"resize": [224, 224]}}
        data = {"metadata": metadata, "test_features": np.zeros((2, 512))}
        checkpoints = {"SAA": {"feature_metadata": copy.deepcopy(metadata)},
                       "SNR": {"feature_metadata": copy.deepcopy(metadata)}}
        directions.validate_feature_metadata(data, checkpoints)
        checkpoints["SNR"]["feature_metadata"]["backbone"] = "other_backbone"
        with self.assertRaisesRegex(ValueError, "do not match"):
            directions.validate_feature_metadata(data, checkpoints)

    def test_feature_metadata_cannot_be_missing_from_one_input(self):
        metadata = {"backbone": "resnet18", "feature_dim": 512}
        with self.assertRaisesRegex(ValueError, "absent or invalid"):
            directions.validate_feature_metadata({}, {"SAA": {"feature_metadata": metadata}})
        with self.assertRaisesRegex(ValueError, "do not match"):
            directions.validate_feature_metadata({"metadata": metadata}, {"SAA": {}})
        with self.assertRaisesRegex(ValueError, "absent or invalid"):
            directions.validate_feature_metadata({"metadata": "invalid"}, {"SAA": {}})

    def test_plain_numpy_inputs_without_metadata_remain_supported(self):
        directions.validate_feature_metadata(
            {"test_features": np.ones((3, 2))}, {"head": {"weight": np.eye(2)}},
        )

    def test_reference_scale_preserves_torch_float32_expression(self):
        import torch
        features = torch.randn(41, 512, generator=torch.Generator().manual_seed(37),
                               dtype=torch.float32)
        expected = torch.linalg.vector_norm(features, dim=1).mean().item()
        self.assertEqual(directions.mean_feature_norm(features), expected)

    def test_numpy_cli_saves_provenance_and_rejects_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.npz"
            head_path = root / "head.npz"
            output_path = root / "diagnostics"
            test_features = np.array([[1.7, .2, .5], [.2, 1.7, -.6], [-1.8, -2.2, .4]])
            np.savez(feature_path,
                     train_features=np.array([[2., .1, .2], [2., -.1, -.2],
                                              [.1, 2., .3], [-.1, 2., -.3],
                                              [-2., -2., .1], [-2.1, -1.9, -.1]]),
                     train_labels=np.repeat([0, 1, 2], 2),
                     test_features=test_features, test_labels=np.array([0, 1, 2]))
            np.savez(head_path, weight=np.array([[2., 0., 0.], [0., 2., 0.], [-2., -2., 0.]]),
                     bias=np.array([.2, -.1, 0.]))
            command = [sys.executable, str(Path(directions.__file__).resolve()),
                       "--features", str(feature_path), "--head", f"toy={head_path}",
                       "--level", "0.04", "--draws", "3", "--seed", "19",
                       "--out", str(output_path)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            files = [output_path / name for name in ("directions.png", "directions.csv", "provenance.json")]
            self.assertTrue(all(path.stat().st_size > 0 for path in files))
            with (output_path / "directions.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({int(row["rank"]) for row in rows}, {2})
            self.assertAlmostEqual(float(rows[-1]["mean_adversarial_energy"]), 1.)
            self.assertEqual(float(rows[-1]["head_subspace_accuracy"]),
                             float(rows[-1]["unrestricted_accuracy"]))
            provenance = json.loads((output_path / "provenance.json").read_text(encoding="utf-8"))
            self.assertFalse(provenance["retrained"])
            self.assertEqual(provenance["training_examples"], 6)
            self.assertEqual(provenance["test_examples"], 3)
            self.assertEqual(provenance["random_draws"], 3)
            self.assertEqual(provenance["random_seed"], 19)
            self.assertEqual(provenance["heads"]["toy"]["sha256"], directions.file_sha256(head_path))
            self.assertEqual(provenance["feature_cache"]["sha256"], directions.file_sha256(feature_path))
            expected_mean = float(np.linalg.norm(test_features, axis=1).mean())
            self.assertEqual(provenance["mean_test_norm"], expected_mean)
            self.assertEqual(provenance["radius"], .04 * expected_mean)
            hashes = {path: directions.file_sha256(path) for path in files}
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Output already exists", repeated.stderr)
            self.assertEqual(hashes, {path: directions.file_sha256(path) for path in files})

    def test_affine_distances_include_bias_and_clean_errors(self):
        weight = np.array([[2., 0.], [-1., 0.]])
        bias = np.array([-.5, .25])
        features = np.array([[.7, 1.], [-.1, -2.], [.1, 3.]])
        labels = np.array([0, 1, 0])
        correct, margins, normals = directions.classifier_geometry(
            weight, bias, features, labels,
        )
        lengths = np.linalg.norm(normals, axis=2)
        distances = margins[np.arange(2), 1 - labels[:2]] / 3.
        np.testing.assert_allclose(distances, [.45, .35])
        self.assertEqual(directions.exact_accuracy(margins, labels, lengths, 0.),
                         100 * correct.mean())
        for radius, expected in ((.2, 200 / 3), (.4, 100 / 3), (.5, 0.)):
            self.assertAlmostEqual(
                directions.exact_accuracy(margins, labels, lengths, radius), expected,
            )

    def test_restricted_boundary_uses_projected_normal_length(self):
        weight = np.array([[1., 2.], [-2., -2.]])
        bias = np.array([.3, -.7])
        features = np.array([[1., 1.]])
        labels = np.array([0])
        _, margins, normals = directions.classifier_geometry(weight, bias, features, labels)
        full_lengths = np.linalg.norm(normals, axis=2)
        x_lengths = np.linalg.norm(normals @ np.array([[1.], [0.]]), axis=2)
        y_lengths = np.linalg.norm(normals @ np.array([[0.], [1.]]), axis=2)
        self.assertAlmostEqual(margins[0, 1] / full_lengths[0, 1], 1.6)
        self.assertAlmostEqual(margins[0, 1] / x_lengths[0, 1], 8 / 3)
        self.assertAlmostEqual(margins[0, 1] / y_lengths[0, 1], 2.)
        self.assertEqual(directions.exact_accuracy(margins, labels, full_lengths, 1.5), 100.)
        self.assertEqual(directions.exact_accuracy(margins, labels, full_lengths, 1.7), 0.)
        self.assertEqual(directions.exact_accuracy(margins, labels, x_lengths, 2.1), 100.)
        self.assertEqual(directions.exact_accuracy(margins, labels, y_lengths, 2.1), 0.)
        for radius, expected in ((1.5, 0), (1.7, 1)):
            attacked = features + radius * normals[0, 1] / full_lengths[0, 1]
            self.assertEqual(int((attacked @ weight.T + bias).argmax(1)[0]), expected)

    def test_null_subspace_keeps_clean_accuracy(self):
        weight = np.array([[2., 0.], [-1., 0.]])
        bias = np.array([-.5, .25])
        features = np.array([[.7, 1.], [-.1, -2.], [.1, 3.]])
        labels = np.array([0, 1, 0])
        correct, margins, normals = directions.classifier_geometry(
            weight, bias, features, labels,
        )
        lengths = np.linalg.norm(normals @ np.array([[0.], [1.]]), axis=2)
        self.assertEqual(directions.exact_accuracy(margins, labels, lengths, 100.),
                         100 * correct.mean())

    def test_closed_ball_ties_follow_argmax_class_order(self):
        weight = np.array([[1., 0.], [-1., 0.]])
        features = np.array([[.5, 0.], [-.5, 0.]])
        labels = np.array([0, 1])
        _, margins, normals = directions.classifier_geometry(
            weight, np.zeros(2), features, labels,
        )
        lengths = np.linalg.norm(normals, axis=2)
        self.assertEqual(directions.exact_accuracy(margins, labels, lengths, .5), 50.)

    def test_centering_preserves_scores_differences_and_robustness(self):
        rng = np.random.default_rng(6)
        weight = rng.normal(size=(3, 5))
        bias = np.array([.4, -.2, .8])
        features = rng.normal(size=(17, 5))
        labels = (features @ weight.T + bias).argmax(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.npz"
            np.savez(path, weight=weight, bias=bias)
            centered_weight, centered_bias = directions.load_head(path)
        np.testing.assert_allclose(centered_weight.mean(0), 0., atol=1e-15)
        np.testing.assert_allclose(centered_bias.mean(), 0., atol=1e-15)
        correct, margins, normals = directions.classifier_geometry(weight, bias, features, labels)
        new_correct, new_margins, new_normals = directions.classifier_geometry(
            centered_weight, centered_bias, features, labels,
        )
        np.testing.assert_array_equal(new_correct, correct)
        np.testing.assert_allclose(new_margins, margins, atol=1e-14)
        np.testing.assert_allclose(new_normals, normals, atol=1e-14)
        for radius in (0., .1, .7, 2.):
            self.assertEqual(
                directions.exact_accuracy(margins, labels, np.linalg.norm(normals, axis=2), radius),
                directions.exact_accuracy(new_margins, labels,
                                          np.linalg.norm(new_normals, axis=2), radius),
            )

    def test_positive_logit_scaling_keeps_boundary_distances(self):
        rng = np.random.default_rng(8)
        weight = rng.normal(size=(4, 6))
        bias = rng.normal(size=4)
        features = rng.normal(size=(23, 6))
        labels = (features @ weight.T + bias).argmax(1)
        _, margins, normals = directions.classifier_geometry(weight, bias, features, labels)
        lengths = np.linalg.norm(normals, axis=2)
        competitors = lengths[labels] > 0
        for scale in (.003, 8.):
            _, scaled_margins, scaled_normals = directions.classifier_geometry(
                scale * weight, scale * bias, features, labels,
            )
            scaled_lengths = np.linalg.norm(scaled_normals, axis=2)
            np.testing.assert_allclose(
                margins[competitors] / lengths[labels][competitors],
                scaled_margins[competitors] / scaled_lengths[labels][competitors],
                rtol=1e-12, atol=1e-12,
            )
            for radius in (.05, .5, 1.):
                self.assertEqual(
                    directions.exact_accuracy(margins, labels, lengths, radius),
                    directions.exact_accuracy(scaled_margins, labels, scaled_lengths, radius),
                )

    def test_full_rowspace_captures_adversarial_energy_and_accuracy(self):
        rng = np.random.default_rng(9)
        weight = np.array([[2., 0., 0., 0.], [0., 1., 0., 0.], [-2., -1., 0., 0.]])
        bias = np.array([.1, -.2, .3])
        features = rng.normal(size=(31, 4))
        labels = (features @ weight.T + bias).argmax(1)
        _, margins, normals = directions.classifier_geometry(weight, bias, features, labels)
        lengths = np.linalg.norm(normals, axis=2)
        distances = np.divide(margins, lengths[labels], out=np.full_like(margins, np.inf),
                              where=lengths[labels] > 0)
        closest = distances.argmin(1)
        adversarial = directions.unit_rows(normals[labels, closest])
        _, spectrum, vh = np.linalg.svd(weight, full_matrices=False)
        rank = np.linalg.matrix_rank(weight)
        self.assertEqual(rank, 2)
        basis = vh[:rank].T
        energy = directions.direction_energy(adversarial, basis)
        np.testing.assert_allclose(energy[:, -1], 1., atol=1e-14)
        projected_lengths = np.linalg.norm(normals @ basis, axis=2)
        np.testing.assert_allclose(projected_lengths, lengths, atol=1e-14)
        for radius in (0., .2, .8, 2.):
            self.assertEqual(directions.exact_accuracy(margins, labels, lengths, radius),
                             directions.exact_accuracy(margins, labels, projected_lengths, radius))

    def test_clean_pca_uses_within_class_training_variation(self):
        residual = np.array([[-3., 0.], [3., 0.], [0., -1.], [0., 1.]])
        features = np.vstack((residual + [0., 100.], residual + [0., -100.]))
        labels = np.repeat([0, 1], 4)
        means, pcs = directions.clean_geometry(features, labels, 2)
        np.testing.assert_allclose(means, [[0., 100.], [0., -100.]])
        np.testing.assert_allclose(np.abs(pcs[:, 0]), [1., 0.], atol=1e-14)
        np.testing.assert_allclose(pcs.T @ pcs, np.eye(2), atol=1e-14)

    def test_tied_subspace_energy_is_invariant_to_basis_rotation(self):
        unit_vectors = directions.unit_rows(np.array([[1., 2., 3.], [-2., 1., 1.]]))
        basis = np.eye(3)[:, :2]
        angle = .61
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        energy = directions.direction_energy(unit_vectors, basis)
        rotated = directions.direction_energy(unit_vectors, basis @ rotation)
        np.testing.assert_allclose(energy[:, -1], rotated[:, -1], atol=1e-14)
        self.assertFalse(np.allclose(energy[:, 0], rotated[:, 0]))


if __name__ == "__main__":
    unittest.main()
