# Regret_DRO

Compare nine linear classifiers on CIFAR-10: `SAA`, `SNR`, `SNRFrob`, `Dual`, `WGF`, `WRM`, `WFR`, `SVG`, and `RGO`.

All methods use the same 512-dimensional features from a frozen ResNet-18 model.
The model uses ImageNet weights.
SNR adds a squared spectral-norm penalty to the clean cross-entropy loss.
SNRFrob uses $\mathrm{CE} + \gamma\|W\|_F^2$, where $\|W\|_F^2$ is the sum of the squared weight entries.
Both penalties apply only to the head weight matrix. They do not apply to its bias.
SAA, SNR, and SNRFrob use the same initial weights and minibatch order for each seed.
They use Adam with learning rate `0.001` and zero optimizer weight decay.

All methods use the corrected hard-target cross-entropy from
[A Fix for Softmax and Cross-Entropy, v3, Section 2](https://arxiv.org/html/2610.31642v3).
This includes the inner samplers, PGD attacks, and loss measurements.
The loss keeps small target gradients when the classifier has high confidence.
The equations, labels, and loss reductions are unchanged.
The loss module also has the paper's corrected softmax and log-softmax.
Its Section 3 loss accepts raw teacher logits and fixes soft-target cancellation.
This separate helper treats the teacher as fixed and supports first-order gradients.
The CIFAR-10 experiment uses integer labels and does not call this helper.

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

3. Run the smoke tests.

   ```text
   python CIFAR10_feature_regression/Multiclass_classification.py --smoke-test
   ```

   The tests use small synthetic inputs. They do not download data or model weights.
   They run on the CPU and include a small CUDA check when CUDA is available.
   They check the clean-head penalties, validation selection, PGD, feature extraction, cache reuse, and plots.
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
   python CIFAR10_feature_regression/Multiclass_classification.py --methods all --epochs 10 --snr-lambda 0.01 --snrfrob-gamma 0.01 --seeds 2022 --pgd-steps 10 --sdro-lambdas 10 --sdro-epsilons 0.2 0.02 0.002
   ```

   Replace `0.01` with your fixed SNR coefficient. It must be nonnegative.
   The script does not select this coefficient from the data.
   Each clean-head configuration trains once per seed. The script reuses its results for each SDRO configuration.

6. Compare SNR with a fixed Frobenius penalty. Use this command in a notebook cell.

   ```python
   !python CIFAR10_feature_regression/Multiclass_classification.py --methods "SAA,SNR,SNRFrob" --epochs 10 --snr-lambda 2.0 --snrfrob-gamma 0.1 --seeds 2022 --pgd-steps 10
   ```

   Change `--snrfrob-gamma` to test another fixed coefficient. It must be nonnegative.
   Its default value is `0.01`. A value of zero gives the SAA objective.
   This mode trains on the full training set. It does not select gamma from validation data.
   Remove `!` when you use the command in a terminal.

7. Select gamma to match SNR's validation clean accuracy.

   ```python
   !python CIFAR10_feature_regression/Multiclass_classification.py --methods "SAA,SNR,SNRFrob" --epochs 10 --snr-lambda 2.0 --snrfrob-gamma-grid 0 0.001 0.01 0.1 0.3 1 2 --validation-fraction 0.2 --validation-seed 2022 --match-clean-tolerance 0.5 --seeds 2022 --pgd-steps 10
   ```

   Grid mode requires both `SNR` and `SNRFrob`. Do not combine the grid with `--snrfrob-gamma`.
   The script reserves 20% of the training data for validation, with the same class proportions.
   All methods train on the same remaining 80%. The split is fixed by `--validation-seed`.
   SNR and each gamma candidate use the same initial head and minibatch order for each seed.
   The script selects one gamma with the smallest absolute gap between the mean validation clean accuracies.
   It uses the same seeds for both means. If candidates have equal gaps, it selects the smaller gamma.
   Test accuracy and PGD accuracy do not affect this choice.

   The report gives the signed and absolute clean-accuracy gaps in percentage points.
   `--match-clean-tolerance 0.5` marks a gap of at most 0.5 percentage points as a close match.
   The script still selects the closest candidate if the gap exceeds this tolerance.
   In that case, it reports the failed tolerance check. Expand the grid before claiming a close match.
   The script evaluates the selected heads on test data without further training.
   It does not refit them on the full training set.
   Compare these heads within the same run, because earlier fixed-coefficient runs used more training data.

Run all commands from the repository directory.
The script uses CUDA when it is available. Otherwise, it uses the CPU.
Add `--device cpu` to select the CPU.
Use `--seeds 2022 2023 2024` to run three seeds.
Use `--help` to show all options.

The metric is **feature-space PGD accuracy**, in percent.
PGD changes the extracted features. It does not change image pixels.
Each radius equals a relative level times the mean norm of all clean test features.
All methods use the same radii. The default attack uses 10 steps and a clean start.
In grid mode, validation attacks use the mean norm of the validation features.
Final test attacks use the mean norm of the full clean test set.

Each run creates a separate directory under `results/cifar10_resnet18/`.
The directory contains `results.csv`, `experiment.json`, head checkpoints, and PDF plots.
The CSV has one row per method, configuration, seed, and perturbation level.
It records clean accuracy, PGD accuracy, radii, and training settings.
The `cross_entropy` field records the correction version in the CSV and run metadata.
For SNR, it also records the final clean training CE, spectral penalty, total objective, and operator norm.
The `spectral_penalty` column is the squared operator norm before multiplication by `snr_lambda`.
For SNRFrob, `snrfrob_gamma` records the coefficient.
Its diagnostics include `frobenius_penalty`, `weighted_frobenius_penalty`, `final_frobenius_norm`, and `final_op_norm`.
The Frobenius penalty is the sum of the squared weight entries before multiplication by gamma.
The CSV also records its clean CE and total objective on the data used to train the head.

Grid mode also saves these files:

- `validation_results.csv`: SNR and all gamma candidates, for each seed and validation attack level.
- `validation_selection.json`: the selected gamma, selection rule, accuracy gaps, and tolerance check.
- `validation_split.pth`: the training and validation indices.
- `validation/`: the candidate checkpoints.

The final `results.csv` contains the selected SNRFrob model, SNR, and the other requested methods.

The default feature cache is `cifar10_resnet18_imagenet1k_v1_features.pth`.
The script rejects a cache with incompatible metadata or tensor contents.
Use `--cache PATH` to select another cache file.
Data, feature caches, checkpoints, and experiment results are not included in Git.
