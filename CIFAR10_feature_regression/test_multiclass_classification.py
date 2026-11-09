"""Small CPU checks. No CIFAR-10 data or pretrained weights are needed."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

import Multiclass_classification as experiment


class ClassifierSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(13)

    def test_spectral_penalty_matches_svd_and_has_gradient(self):
        head = experiment.LogisticRegression().double()
        weight = head.linear.weight
        penalty = torch.linalg.eigvalsh(weight @ weight.T)[-1]
        expected = torch.linalg.svdvals(weight)[0].square()
        torch.testing.assert_close(penalty, expected, rtol=1e-10, atol=1e-12)
        penalty.backward()
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertGreater(weight.grad.norm().item(), 0.0)
        self.assertIsNone(head.linear.bias.grad)

    def test_zero_lambda_matches_saa_at_each_epoch(self):
        features = torch.randn(19, 512)
        labels = torch.randint(0, 10, (19,))
        initial = experiment.LogisticRegression()
        saa, snr = copy.deepcopy(initial), copy.deepcopy(initial)

        def loader():
            return DataLoader(
                TensorDataset(features, labels), batch_size=6, shuffle=True,
                generator=torch.Generator().manual_seed(42),
            )

        with tempfile.TemporaryDirectory() as directory:
            saa_dir, snr_dir = Path(directory) / "saa", Path(directory) / "snr"
            saa_dir.mkdir()
            snr_dir.mkdir()
            saa_history = experiment.train_saa(
                saa, loader(), epochs=2, checkpoint_dir=str(saa_dir), device="cpu",
            )
            snr_history = experiment.train_snr(
                snr, loader(), epochs=2, checkpoint_dir=str(snr_dir),
                device="cpu", snr_lambda=0.0,
            )
            self.assertEqual(saa_history, snr_history)
            saa_files = sorted(saa_dir.glob("*.pth"))
            snr_files = sorted(snr_dir.glob("*.pth"))
            self.assertEqual(len(saa_files), 2)
            self.assertEqual(len(snr_files), 2)
            for saa_file, snr_file in zip(saa_files, snr_files):
                saa_state = torch.load(saa_file, map_location="cpu", weights_only=True)
                snr_state = torch.load(snr_file, map_location="cpu", weights_only=True)
                for name in saa_state:
                    torch.testing.assert_close(saa_state[name], snr_state[name], rtol=0, atol=0)
        for actual, expected in zip(saa.parameters(), snr.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(saa.linear.weight, initial.linear.weight))

    def test_positive_lambda_matches_explicit_objective_update(self):
        features = torch.randn(7, 512)
        labels = torch.randint(0, 10, (7,))
        actual = experiment.LogisticRegression()
        expected = copy.deepcopy(actual)
        optimizer = torch.optim.Adam(expected.parameters(), lr=0.001, weight_decay=0.0)
        weight = expected.linear.weight
        loss = experiment.cross_entropy(expected(features), labels)
        loss = loss + 0.7 * torch.linalg.eigvalsh(weight @ weight.T)[-1]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        loader = DataLoader(TensorDataset(features, labels), batch_size=7)
        with tempfile.TemporaryDirectory() as directory:
            history = experiment.train_snr(
                actual, loader, epochs=1, checkpoint_dir=directory,
                device="cpu", snr_lambda=0.7,
            )
        self.assertEqual(len(history), 1)
        for got, want in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_pgd_zero_radius_projection_and_parameter_gradients(self):
        head = experiment.LogisticRegression(input_dim=5, num_classes=3).double()
        features = torch.randn(9, 5, dtype=torch.float64) * 3
        labels = head(features).argmax(1)
        labels[0] = (labels[0] + 1) % 3
        expected_clean = head(features).argmax(1).eq(labels)
        head.linear.weight.grad = torch.randn_like(head.linear.weight)
        head.linear.bias.grad = None
        saved_grad = head.linear.weight.grad.clone()
        evaluated_features = []
        hook = head.register_forward_pre_hook(
            lambda model, args: evaluated_features.append(args[0].detach().clone())
        )
        try:
            unchanged, clean_mask = experiment.pgd_attack(
                head, features, labels, epsilon=0.0, return_details=True,
            )
            torch.testing.assert_close(unchanged, features, rtol=0, atol=0)
            self.assertTrue(torch.equal(clean_mask, expected_clean))
            for radius in (0.07, 0.8, 2.0):
                evaluated_features.clear()
                perturbed, robust_mask = experiment.pgd_attack(
                    head, features, labels, epsilon=radius, return_details=True,
                )
                self.assertFalse(perturbed.requires_grad)
                self.assertTrue((torch.linalg.vector_norm(perturbed - features, dim=1) <= radius + 1e-10).all())
                self.assertFalse(robust_mask[0].item())
                for iterate in evaluated_features:
                    self.assertTrue((torch.linalg.vector_norm(iterate - features, dim=1) <= radius + 1e-10).all())
        finally:
            hook.remove()
        results = experiment.evaluate_model(
            head, features, labels, [0.0, 0.8], pgd_steps=10, batch_size=4,
        )
        self.assertAlmostEqual(results[0.0], expected_clean.double().mean().item() * 100)
        self.assertLessEqual(results[0.8], results[0.0])
        torch.testing.assert_close(head.linear.weight.grad, saved_grad, rtol=0, atol=0)
        self.assertIsNone(head.linear.bias.grad)

    def test_pgd_affine_boundary_distances(self):
        head = experiment.LogisticRegression(input_dim=2, num_classes=2).double()
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor([[3.0, -1.0], [-1.0, 2.0]], dtype=torch.float64))
            head.linear.bias.copy_(torch.tensor([0.6, -0.4], dtype=torch.float64))
        normal = torch.tensor([0.8, -0.6], dtype=torch.float64)
        tangent = torch.tensor([0.6, 0.8], dtype=torch.float64)
        signed_distances = torch.tensor([0.2, -0.2, 0.6, -0.6, 1.4, -1.4], dtype=torch.float64)
        offsets = torch.tensor([-3.0, 0.5, 2.0, -1.0, 0.0, 4.0], dtype=torch.float64)
        features = -0.2 * normal + signed_distances[:, None] * normal + offsets[:, None] * tangent
        labels = signed_distances.lt(0).long()
        logits = head(features)
        rows = torch.arange(len(features))
        distances = (logits[rows, labels] - logits[rows, 1 - labels]) / 5.0
        torch.testing.assert_close(distances, signed_distances.abs())
        for radius in (0.1, 0.35, 0.9, 1.7):
            perturbed, robust_mask = experiment.pgd_attack(
                head, features, labels, epsilon=radius, num_iter=10, return_details=True,
            )
            self.assertTrue(torch.equal(robust_mask, distances.gt(radius)))
            displacement = -signed_distances.sign()[:, None] * normal * radius
            torch.testing.assert_close(perturbed - features, displacement, rtol=1e-7, atol=1e-9)

    def test_pgd_remembers_an_incorrect_intermediate_iterate(self):
        class CurvedHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

            def forward(self, features):
                margin = self.scale * ((features[:, 0] - 0.5).square() - 0.1)
                return torch.stack((margin, torch.zeros_like(margin)), dim=1)

        head = CurvedHead()
        features = torch.zeros(1, 2, dtype=torch.float64)
        labels = torch.zeros(1, dtype=torch.long)
        perturbed, robust_mask = experiment.pgd_attack(
            head, features, labels, epsilon=0.9, alpha=0.75,
            num_iter=2, return_details=True,
        )
        self.assertEqual(head(perturbed).argmax(1).item(), 0)
        self.assertFalse(robust_mask.item())

    def test_extraction_freezes_and_uses_eval_without_gradients(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 512), torch.nn.Dropout())
        features = torch.randn(5, 3)
        labels = torch.tensor([4, 1, 0, 3, 2])
        observations = []
        hook = model.register_forward_pre_hook(
            lambda module, args: observations.append(
                (module.training, torch.is_grad_enabled(), any(p.requires_grad for p in module.parameters()))
            )
        )
        try:
            extracted, extracted_labels = experiment.extract_features(
                DataLoader(TensorDataset(features, labels), batch_size=2), model, "cpu",
            )
        finally:
            hook.remove()
        self.assertEqual(observations, [(False, False, False)] * 3)
        self.assertEqual(tuple(extracted.shape), (5, 512))
        self.assertEqual(extracted.device.type, "cpu")
        self.assertFalse(extracted.requires_grad)
        self.assertTrue(torch.equal(extracted_labels, labels))
        torch.testing.assert_close(extracted, model(features))

    def test_cache_rejects_wrong_provenance_with_512_columns(self):
        data = {
            "metadata": copy.deepcopy(experiment.FEATURE_METADATA),
            "train_features": torch.zeros(3, 512), "train_labels": torch.zeros(3, dtype=torch.long),
            "test_features": torch.zeros(2, 512), "test_labels": torch.zeros(2, dtype=torch.long),
        }
        for key in experiment.FEATURE_METADATA:
            invalid = copy.deepcopy(data)
            invalid["metadata"][key] = "incompatible"
            with self.subTest(metadata_field=key), self.assertRaisesRegex(ValueError, "metadata"):
                experiment.validate_feature_cache(invalid)
        del data["metadata"]
        with self.assertRaisesRegex(ValueError, "metadata"):
            experiment.validate_feature_cache(data)

    def test_cache_miss_extracts_once_then_hit_skips_backbone_and_data(self):
        import torchvision.models

        train = (torch.randn(3, 512), torch.tensor([0, 1, 2]))
        test = (torch.randn(2, 512), torch.tensor([1, 0]))
        backbone = torch.nn.Linear(3, 512)
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "resnet18_test.pth"
            with mock.patch.object(experiment, "validate_feature_cache"), \
                    mock.patch.object(experiment, "get_cifar10_dataloaders", return_value=("train", "test")) as loaders, \
                    mock.patch.object(experiment, "extract_features", side_effect=(train, test)) as extract, \
                    mock.patch.object(torchvision.models, "resnet18", return_value=backbone) as resnet:
                fresh = experiment.load_or_extract_features(cache_path, directory, "cpu")
                self.assertEqual(extract.call_count, 2)
                self.assertEqual(loaders.call_count, 1)
                resnet.assert_called_once_with(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                self.assertIsInstance(backbone.fc, torch.nn.Identity)
                self.assertIs(fresh["train_features"], train[0])
                self.assertIs(fresh["test_features"], test[0])
                self.assertEqual(fresh["metadata"], experiment.FEATURE_METADATA)
                self.assertTrue(cache_path.is_file())

            with mock.patch.object(experiment, "validate_feature_cache"), \
                    mock.patch.object(experiment, "get_cifar10_dataloaders", side_effect=AssertionError("cache hit loaded images")), \
                    mock.patch.object(experiment, "extract_features", side_effect=AssertionError("cache hit extracted features")), \
                    mock.patch.object(torchvision.models, "resnet18", side_effect=AssertionError("cache hit built backbone")):
                cached = experiment.load_or_extract_features(cache_path, directory, "cpu")
            for name in ("train_features", "train_labels", "test_features", "test_labels"):
                torch.testing.assert_close(cached[name], fresh[name], rtol=0, atol=0)
            self.assertEqual(cached["metadata"], experiment.FEATURE_METADATA)

    def test_plot_uses_default_style_and_rejects_missing_radius(self):
        with mock.patch.object(experiment.Path, "is_file", return_value=False), \
                mock.patch.object(experiment.plt.style, "use") as use_style:
            experiment.set_plot_style()
            use_style.assert_called_once_with("default")
        records = [{"method": "SNR", "results": {0.0: 80.0}, "losses": [0.5]}]
        try:
            with tempfile.TemporaryDirectory() as directory, \
                    self.assertRaisesRegex(ValueError, "Missing SNR radii"):
                experiment.plot_comparison(records, ["SNR"], [0.0, 0.008], [0.0, 0.4], directory)
        finally:
            experiment.plt.close("all")


if __name__ == "__main__":
    unittest.main(verbosity=2)
