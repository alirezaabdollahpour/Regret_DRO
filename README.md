# Regret_DRO

Compare eight linear classifiers on CIFAR-10: `SAA`, `SNR`, `Dual`, `WGF`, `WRM`, `WFR`, `SVG`, and `RGO`.

All methods use the same 512-dimensional features from a frozen ResNet-18 model.
The model uses ImageNet weights.
SNR adds a squared spectral-norm penalty to the clean cross-entropy loss.
SAA and SNR use the same initial weights, minibatch order, and Adam settings for each seed.

Use Python 3.12.
The CPU tests passed with Python 3.12.3 and the package versions in `requirements.txt`.

1. Get the code.

   ```text
   git clone https://github.com/alirezaabdollahpour/Regret_DRO.git
   cd Regret_DRO
   ```

2. Create and activate a Python environment. These commands use Windows PowerShell.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   On Linux or macOS, use `source .venv/bin/activate` to activate the environment.

3. Run the CPU smoke tests.

   ```text
   python CIFAR10_feature_regression/Multiclass_classification.py --smoke-test
   ```

   The tests use small synthetic inputs. They do not download data or model weights.
   They check SNR, PGD, feature extraction, cache reuse, plots, and the full experiment procedure.
   These tests do not measure classifier quality on CIFAR-10.

4. Run a quick SAA/SNR comparison.

   ```text
   python CIFAR10_feature_regression/Multiclass_classification.py --methods "SAA,SNR" --epochs 1 --snr-lambda 0.01 --seeds 2022 --pgd-steps 10
   ```

   This command trains each head for one epoch.
   The first run downloads CIFAR-10 and the ResNet-18 weights if they are absent.
   It extracts features for the full training and test sets.
   This first extraction can take a long time on a CPU.
   Later runs load the feature cache.

5. Run the full comparison.

   ```text
   python CIFAR10_feature_regression/Multiclass_classification.py --methods all --epochs 10 --snr-lambda 0.01 --seeds 2022 --pgd-steps 10 --sdro-lambdas 10 --sdro-epsilons 0.2 0.02 0.002
   ```

   Replace `0.01` with your fixed SNR coefficient. It must be nonnegative.
   The script does not select this coefficient from the data.
   SAA and SNR train once per seed. The script reuses their results for each SDRO configuration.

Run all commands from the repository directory.
The script uses CUDA when it is available. Otherwise, it uses the CPU.
Add `--device cpu` to select the CPU.
Use `--seeds 2022 2023 2024` to run three seeds.
Use `--help` to show all options.

The metric is **feature-space PGD accuracy**, in percent.
PGD changes the extracted features. It does not change image pixels.
Each radius equals a relative level times the mean norm of all clean test features.
All methods use the same radii. The default attack uses 10 steps and a clean start.

Each run creates a separate directory under `results/cifar10_resnet18/`.
The directory contains `results.csv`, `experiment.json`, head checkpoints, and PDF plots.
The CSV has one row per method, configuration, seed, and perturbation level.
It records clean accuracy, PGD accuracy, radii, and training settings.
For SNR, it also records the final clean training CE, spectral penalty, total objective, and operator norm.
The `spectral_penalty` column is the squared operator norm before multiplication by `snr_lambda`.

The default feature cache is `cifar10_resnet18_imagenet1k_v1_features.pth`.
The script rejects a cache with incompatible metadata or tensor contents.
Use `--cache PATH` to select another cache file.
Data, feature caches, checkpoints, and experiment results are not included in Git.
