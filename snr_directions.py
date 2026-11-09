"""Compare frozen affine classification heads on the same feature cache.

Example:
  python snr_directions.py --features cifar10_resnet18_features.pth \
      --head SAA=saa.pth --head SNR_2=snr.pth --level 0.08 --out directions

The cache must contain train_features, train_labels, test_features, test_labels.
Use PyTorch tensor dictionaries (.pt/.pth) or NumPy archives (.npz). Head files
can be ordinary linear/fc state dicts or .npz files with weight and bias arrays.
Use checkpoints from the same feature extractor and preprocessing pipeline.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_mapping(path):
    if Path(path).suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return dict(archive)
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_feature_metadata(data, checkpoints):
    """Require matching metadata when any input records its feature space."""
    metadata = data.get("metadata")
    if metadata is None and all(h.get("feature_metadata") is None for h in checkpoints.values()):
        return
    if not isinstance(metadata, dict):
        raise ValueError("Feature cache metadata is absent or invalid")
    for name, checkpoint in checkpoints.items():
        if checkpoint.get("feature_metadata") != metadata:
            raise ValueError(f"{name}: checkpoint and cache feature metadata do not match")


def mean_feature_norm(features):
    """Use the cache tensor dtype for the experiment's reference scale."""
    if isinstance(features, np.ndarray):
        return float(np.linalg.norm(features, axis=1).mean())
    import torch
    return torch.linalg.vector_norm(features, dim=1).mean().item()


def load_head(path):
    state = load_mapping(path)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    keys = [k for k in state if k == "weight" or k.endswith(
        ("linear.weight", "fc.weight", "classifier.weight"))]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected one affine head, found {keys}. "
                         "Supply an isolated head state dict or weight/bias NPZ.")
    key = keys[0]
    w = np.asarray(state[key], dtype=np.float64)
    if w.ndim != 2 or not np.isfinite(w).all():
        raise ValueError(f"{path}: invalid head weight")
    b = np.asarray(state.get(key[:-6] + "bias", np.zeros(w.shape[0])),
                   dtype=np.float64)
    if b.shape != (len(w),) or not np.isfinite(b).all():
        raise ValueError(f"{path}: invalid head bias")
    return w - w.mean(axis=0), b - b.mean()


def labels_for(data, key, n, classes):
    y = np.asarray(data[key]).reshape(-1)
    if len(y) != n or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= classes)):
        raise ValueError(f"Invalid {key}; expected one integer class index per sample")
    return y.astype(int, copy=False)


def clean_geometry(train, y, classes):
    # PCA of pooled within-class variation. Fit exclusively on training data.
    if np.any(np.bincount(y, minlength=classes) == 0):
        raise ValueError("At least one training class is absent")
    means = np.stack([train[y == c].mean(0, dtype=np.float64) for c in range(classes)])
    scatter = np.zeros((train.shape[1], train.shape[1]), dtype=np.float64)
    for i in range(0, len(y), 2048):
        residual = train[i:i+2048] - means[y[i:i+2048]]
        scatter += residual.T @ residual
    _, vectors = np.linalg.eigh(scatter)
    return means, vectors[:, ::-1]


def classifier_geometry(w, b, x, y):
    scores = x @ w.T + b
    margins = scores[np.arange(len(y)), y, None] - scores
    # normals[a,b] points from class a toward class b.
    normals = w[None, :, :] - w[:, None, :]
    return scores.argmax(1) == y, margins, normals


def exact_accuracy(margins, y, pair_lengths, radius):
    # At equality, numpy/torch argmax selects the smaller class index.
    slack = margins - radius * pair_lengths[y]
    rivals = np.arange(margins.shape[1])[None, :]
    loses = (slack < 0) | ((slack == 0) & (rivals < y[:, None]))
    return 100 * np.mean(~loses.any(axis=1))


def direction_energy(unit_vectors, basis):
    return np.cumsum((unit_vectors @ basis) ** 2, axis=1)


def unit_rows(rows):
    norms = np.linalg.norm(rows, axis=1)
    return rows[norms > 0] / norms[norms > 0, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True)
    parser.add_argument("--head", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument("--level", type=float, default=0.08,
                        help="radius divided by mean test feature norm")
    parser.add_argument("--draws", type=int, default=50,
                        help="number of random subspaces for the control")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--out", type=Path, default=Path("directions"))
    args = parser.parse_args()
    if args.level < 0 or not np.isfinite(args.level) or args.draws < 1:
        parser.error("Use a finite nonnegative level and at least one random draw.")
    for filename in ("directions.png", "directions.csv", "provenance.json"):
        if (args.out / filename).exists():
            parser.error(f"Output already exists: {args.out / filename}. Use a new --out directory.")
    heads = {}
    checkpoints = {}
    head_paths = {}
    for spec in args.head:
        name, sep, path = spec.partition("=")
        if not (name and sep and path) or name in heads:
            parser.error("Supply a unique NAME=FILE for each head.")
        checkpoints[name] = load_mapping(path)
        head_paths[name] = Path(path)
        heads[name] = load_head(path)
    classes, d = next(iter(heads.values()))[0].shape
    data = load_mapping(args.features)
    validate_feature_metadata(data, checkpoints)
    train = np.asarray(data["train_features"])
    x = np.asarray(data["test_features"], dtype=np.float64)
    for features in (train, x):
        if features.ndim != 2 or features.shape[1] != d or not len(features):
            raise ValueError("Empty feature cache or dimensions inconsistent with head")
        if not np.isfinite(features).all():
            raise ValueError("Feature cache contains nonfinite values")
    ytrain = labels_for(data, "train_labels", len(train), classes)
    y = labels_for(data, "test_labels", len(x), classes)
    means, clean_pcs = clean_geometry(train, ytrain, classes)
    mean_test_norm = mean_feature_norm(data["test_features"])
    radius = args.level * mean_test_norm
    for name, checkpoint in checkpoints.items():
        old_mean = checkpoint.get("mean_test_norm")
        if old_mean is not None and not np.isclose(old_mean, mean_test_norm, rtol=1e-7, atol=0):
            warnings.warn(f"{name}: saved test mean {old_mean:.10g} differs from the current cache "
                          f"({mean_test_norm:.10g}). Use the current cache for this diagnostic.")
    common = np.ones(len(y), dtype=bool)
    for w, b in heads.values():
        if w.shape != (classes, d):
            raise ValueError("All heads must use the same features and class ordering")
        common &= ((x @ w.T + b).argmax(1) == y)
    if not common.any():
        raise ValueError("No test example is correct under every supplied head")
    clean_dirs = unit_rows(x[common] - means[y[common]])
    if not len(clean_dirs):
        raise ValueError("All selected within-class residuals are zero")

    fig, axes = plt.subplots(len(heads), 3, figsize=(13.3, 3.65*len(heads)),
                             squeeze=False, constrained_layout=True)
    rows = []
    for (name, (w, b)), (sax, dax, rax) in zip(heads.items(), axes):
        correct, margins, normals = classifier_geometry(w, b, x, y)
        lengths = np.linalg.norm(normals, axis=2)
        full_acc = exact_accuracy(margins, y, lengths, radius)
        _, spectrum, vh = np.linalg.svd(w, full_matrices=False)
        rank = int(np.sum(spectrum > max(w.shape)*np.finfo(float).eps*spectrum[0]))
        if rank == 0:
            raise ValueError(f"{name}: constant head has no directional subspace")
        spectrum = spectrum[:rank]
        basis = vh[:rank].T
        ks = np.arange(1, rank+1)
        # Keep tied/nearly tied singular values together in subspace plots.
        gaps = -np.diff(spectrum, append=0.) / spectrum[0]
        cut = gaps > 1e-3
        cut[-1] = True

        lengths_for_y = lengths[y[common]]
        radii = np.divide(margins[common], lengths_for_y,
                          out=np.full_like(lengths_for_y, np.inf), where=lengths_for_y > 0)
        nearest = radii.argmin(axis=1)
        adv_dirs = unit_rows(normals[y[common], nearest])
        adv = direction_energy(adv_dirs, basis).mean(0)
        clean_energy = direction_energy(clean_dirs, basis)
        clean = clean_energy.mean(0)
        has_projection = clean_energy[:, -1] > 1e-12
        conditional = (clean_energy[has_projection] /
                       clean_energy[has_projection, -1:]).mean(0) if has_projection.any() else np.full(rank, np.nan)

        def restricted_curve(vectors):
            projections = np.sqrt(np.cumsum((normals @ vectors) ** 2, axis=-1))
            return np.array([exact_accuracy(margins, y, projections[:, :, k-1], radius)
                             for k in ks])

        singular_acc = restricted_curve(basis)
        natural_acc = restricted_curve(clean_pcs[:, :rank])
        rng = np.random.default_rng(args.seed)
        random_curves = []
        for _ in range(args.draws):
            rotation, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
            random_curves.append(restricted_curve(basis @ rotation))
        low, median, high = np.percentile(random_curves, [10, 50, 90], axis=0)
        stable_rank = np.sum((spectrum / spectrum[0])**2)
        sax.plot(ks, spectrum/spectrum[0], "o-", color="#294b71")
        sax.set_title(f"{name}   |   singular values")
        sax.set_ylabel("Fraction of largest singular value")
        sax.text(.98, .05, f"largest = {spectrum[0]:.3g}\nstable rank = {stable_rank:.2f}",
                 transform=sax.transAxes, ha="right", va="bottom")
        dax.plot(ks[cut], adv[cut], "o-", label="Nearest adversarial")
        dax.plot(ks[cut], clean[cut], "s-", label="Clean variation")
        dax.plot(ks[cut], conditional[cut], ".--", label="Clean, within row space")
        dax.plot(ks, ks/rank, ":", color="0.3", label="Isotropic in row space")
        dax.set_title("Fraction of direction captured")
        dax.set_ylabel("Mean squared projection")
        dax.legend(fontsize=8)
        rax.plot(ks[cut], singular_acc[cut], "o-", label="Head singular directions")
        rax.plot(ks, natural_acc, "s-", label="Clean train PCs")
        rax.plot(ks, median, "--", color="0.4", label="Random in row space")
        rax.fill_between(ks, low, high, color="0.5", alpha=.16)
        rax.axhline(full_acc, ls=":", color="black", label="Unrestricted")
        rax.set_title(f"Exact robustness   |   radius {radius:.3g}")
        rax.set_ylabel("All-test robust accuracy (%)")
        rax.legend(fontsize=8)
        for ax in (sax, dax, rax):
            ax.set_xticks(ks)
            ax.set_xlabel("Singular index" if ax is sax else "Subspace dimension k")
            ax.grid(alpha=.18)
        sax.set_ylim(-.04, 1.05)
        dax.set_ylim(-.04, 1.05)
        rax.set_ylim(0, 103)
        for i, k in enumerate(ks):
            rows.append(dict(head=name, k=k, rank=rank, sigma=spectrum[i],
                             relative_sigma=spectrum[i]/spectrum[0], stable_rank=stable_rank,
                             relative_gap_after=gaps[i], identifiable_cut=bool(cut[i]),
                             mean_clean_energy=clean[i], mean_clean_conditional=conditional[i],
                             mean_adversarial_energy=adv[i], common_correct_n=int(common.sum()),
                             clean_direction_n=len(clean_dirs), conditional_n=int(has_projection.sum()),
                             evaluation_n=len(y), level=args.level, radius=radius,
                             mean_test_norm=mean_test_norm, random_draws=args.draws,
                             random_seed=args.seed,
                             clean_accuracy=100*correct.mean(), unrestricted_accuracy=full_acc,
                             head_subspace_accuracy=singular_acc[i], clean_pc_accuracy=natural_acc[i],
                             random_rowspace_p10=low[i], random_rowspace_median=median[i],
                             random_rowspace_p90=high[i]))
        print(f"{name}: clean {100*correct.mean():.2f}%, unrestricted robust {full_acc:.2f}%, "
              f"rank {rank}, common clean-correct n={common.sum()}")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "directions.png", dpi=180)
    plt.close(fig)
    with (args.out / "directions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "feature_cache": {"path": str(Path(args.features).resolve()),
                          "sha256": file_sha256(args.features), "metadata": data.get("metadata")},
        "script_sha256": file_sha256(__file__),
        "heads": {name: {"path": str(head_paths[name].resolve()),
                          "sha256": file_sha256(head_paths[name]),
                          "method": checkpoint.get("method"), "seed": checkpoint.get("seed"),
                          "hyperparameters": checkpoint.get("hyperparameters"),
                          "saved_mean_test_norm": checkpoint.get("mean_test_norm")}
                  for name, checkpoint in checkpoints.items()},
        "mean_test_norm": mean_test_norm, "level": args.level, "radius": radius,
        "training_examples": len(train), "test_examples": len(x),
        "common_correct_n": int(common.sum()), "random_draws": args.draws, "random_seed": args.seed,
        "geometry_dtype": "float64", "spectrum": "class-row-centered head",
        "clean_pca": "training within-class residuals", "near_tie_relative_gap": 1e-3,
        "metric": "exact feature-space robust accuracy", "retrained": False,
        "numpy_version": np.__version__,
    }
    with (args.out / "provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)


if __name__ == "__main__":
    main()
