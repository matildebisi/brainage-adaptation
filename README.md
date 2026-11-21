# Brain-age transfer learning

Code supporting the master's thesis experiments on cross-population transfer of
3D DenseNet brain-age models. It includes conventional fine-tuning, LoRA, DoRA,
custom 3D PiSSA and FiltLoRA, subject-level cross-validation, oversampling,
ensemble experiments and selected statistical analyses.


## Code provenance and acknowledgements

This repository builds on the original brain-age prediction code developed by **Michela Gravina**, available [here](https://github.com/MichiUnina/Assessing-Demographic-Bias-in-Brain-Age-Prediction-Models-Using-Multiple-Deep-Learning-Paradigms).

The codebase was subsequently extended by **Georgia Kenyon** and **Sophie Martin** to investigate transfer learning for brain-age prediction.

Building on their work, the code has been further adapted, extended, and reorganized for the experiments reported in this master's thesis.



## Contributions of this thesis

Building on the previous work, this master's thesis extends the transfer learning framework through:

- **Cross-validation:** Extension to four subject-level folds to assess the consistency of adaptation results.
- **Parameter-efficient fine-tuning (PEFT):** Integration of LoRA and DoRA as alternatives to conventional fine-tuning.
- **Custom 3D adapters:** Implementation of PiSSA and FiltLoRA for 3D convolutional layers.
- **Ensemble learning:** Investigation of frozen ensembles combining pretrained brain-age models.
- **Statistical analyses:** Evaluation of adaptation improvements, age-related prediction bias, and acquisition-site effects.



## Layout

```text
brainage/
  data/                 # MRI reader, transforms, datasets, saved folds
  models/               # 3D DenseNet, custom adapters, trainable-layer policy
  config.py             # CLI configuration
  main.py               # Train, select checkpoint and evaluate
  training.py
  inference.py          # Test inference after training
  infer_all_subjects.py # Development inference from selected checkpoints
  zero_shot.py            # Inference from an unadapted pretrained checkpoint
scripts/
  run_traditional.sh
  run_peft.sh
  run_oversampling.sh
analysis/
  merge_predictions.py      # Merge predictions across runs and seeds
  prepare_subject_errors.py # Build the subject-level error table 
  select_models.py          # Validation-only model selection
  summarize_runs.py         # Summarise completed training runs
  age_bias.R                # Mixed-model age bias comparisons
  ensembles.py
  improvement_analysis.R
  site_effects.R
  age_bias_plot.py
```

## Data and paths

All environment-specific directories were replaced by placeholders. **No scans,
subject-level CSVs, pretrained weights, predictions or experiment outputs are
included.**

Expected inputs:

- `<DEVELOPMENT_CSV>`: development metadata, containing `SUBJ_ID`, `SESSION_ID`,
  `filename`, `AGE`, `Sex`, and site information needed by split generation.
- `<TEST_CSV>`: held-out evaluation metadata (also `CDR`, `SES_NUM`).
- `<IMAGE_ROOT>`: root containing preprocessed NIfTI volumes; `filename`
  is a path relative to this root.
- `<PRETRAINED_WEIGHTS_DIR>`: checkpoint directory for one pretrained paradigm.
- `<PRETRAINED_WEIGHTS_ROOT>`: parent directory containing one subfolder per paradigm
  (e.g. `TP1_White/best_weights.pth`, `TP2_Asian/best_weights.pth`, etc.).
- `<RUNS_DIR>`: directory containing training outputs (`TL_*/seedN/ft1p0/`).
- `<SPLIT_DIR>`: subject-level folds produced by the split generator.

## Installation

Install a PyTorch build appropriate to your hardware; then:

```bash
pip install -r requirements.txt
```

For statistical analyses install the R packages `ARTool`, `emmeans`, `dplyr`,
`readr`, `tidyr`, `purrr`, `tibble`, `lme4`, and `lmerTest`.


## Workflow overview

The experimental workflow consists of the following steps:

### 1. Prepare the development folds

- **Script:** `brainage/data/splits.py`
- **Inputs:** Development metadata (`<DEVELOPMENT_CSV>`) and preprocessed MRI volumes (`<IMAGE_ROOT>`).
- **Outputs:** Four subject-level training/validation splits saved in `<SPLIT_DIR>`.

These splits are used consistently throughout training and model selection.

### 2. Fine-tune the models

- **Scripts:** `brainage/main.py`, launched through `scripts/run_traditional.sh`, `scripts/run_peft.sh`, or `scripts/run_oversampling.sh`.
- **Inputs:** Pretrained checkpoints, development data, and the previously generated folds.
- **Outputs:** Trained checkpoints, `TL_summary.json`, and individual test prediction CSVs.

Each experiment uses three folds for training and the remaining fold for validation. The best checkpoint is selected based on validation performance. At the end of training, `brainage/inference.py` automatically generates predictions on the held-out test set.

### 3. Generate development predictions

- **Script:** `brainage/infer_all_subjects.py`
- **Inputs:** Saved checkpoints (`TL_summary.json`), development metadata, and preprocessed MRI volumes.
- **Outputs:** `development_predictions.csv` for each experiment and seed.

The selected checkpoints are used to predict ages across the development cohort. These predictions are subsequently evaluated only on the validation subjects corresponding to each fold.

### 4. Merge development predictions

- **Script:** `analysis/merge_predictions.py` (`--mode development`)
- **Inputs:** Individual `development_predictions.csv` files and `<DEVELOPMENT_CSV>`.
- **Output:** `<VALIDATION_PREDICTIONS_CSV>`.

The predictions from all model configurations and seeds are combined into a single subject-level table.

### 5. Select the best model configurations using validation data

- **Script:** `analysis/select_models.py`
- **Inputs:** `<VALIDATION_PREDICTIONS_CSV>` and `<SPLIT_DIR>`.
- **Outputs:** Validation performance summaries and `selected_models_validation_all_TPs.csv`.

For each pretrained model and adaptation strategy, the best configuration is selected based on its mean validation MAE across the four folds. **The held-out test set is not used for model selection.**

### 6. Merge test predictions

- **Script:** `analysis/merge_predictions.py` (`--mode test`)
- **Inputs:** Individual test prediction CSVs generated during training and `<TEST_CSV>`.
- **Output:** `<TEST_PREDICTIONS_CSV>`.

Predictions from all experiments and seeds are combined into a single table for final evaluation.

### 7. Compute subject-level prediction errors

- **Script:** `analysis/prepare_subject_errors.py`
- **Inputs:** `<TEST_PREDICTIONS_CSV>` and `selected_models_validation_all_TPs.csv`.
- **Output:** `ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv`.

For each selected configuration, predictions from the four fold-specific models are used to compute subject-level signed and absolute errors, following the aggregation and subject-exclusion criteria used in the thesis.

### 8. Perform statistical analyses

- **Scripts:** `analysis/improvement_analysis.R`, `analysis/age_bias.R`, `analysis/site_effects.R`, and `analysis/age_bias_plot.py`.
- **Inputs:** Subject-level error table and, where required, the development fold definitions.
- **Outputs:** Statistical comparisons and plots evaluating adaptation improvements, age-related bias, and acquisition-site effects.

**Note:** Zero-shot predictions used in the subject-level statistical analyses are included as inference-only runs (`freeze_mode=none`). The separate `brainage/zero_shot.py` script provides standalone inference from an unadapted pretrained checkpoint.

The sections below provide the commands and configuration details for each step.


## Training on an SGE cluster

Training experiments were run using Sun Grid Engine (SGE).
Three standalone Bash scripts are provided:

- `scripts/run_traditional.sh`: conventional fine-tuning.
- `scripts/run_peft.sh`: LoRA, DoRA, PiSSA, or FiltLoRA.
- `scripts/run_oversampling.sh`: fine-tuning with age-bin and sex oversampling.

Each script contains its own resource requests, paths,
environment setup, and training parameters. No additional
configuration files are required.

### Configuration

Before submission, edit the selected script to specify:

- Input paths: development and test CSVs, MRI directory,
  saved splits, and pretrained checkpoint.
- Output directory and Python environment.
- Training parameters such as learning rate, batch size,
  number of epochs, and fine-tuning configuration.

For PEFT experiments, set `METHOD` to `LoRA`, `DoRA`,
`PiSSA`, or `FiltLoRA`. Adapter parameters such as rank,
alpha, dropout, and target layers can be configured
directly in the script.

The oversampling script supports both conventional
fine-tuning and PEFT methods. The `MIN_SAMPLES_PER_BIN`
parameter controls the oversampling target.

### Running experiments

Submit jobs from the repository root:

```bash
qsub scripts/run_traditional.sh
qsub scripts/run_peft.sh
qsub scripts/run_oversampling.sh
```

The scripts use SGE job arrays (`-t 1-4`) to run four
precomputed cross-validation folds, with seeds 1–4.

The PEFT method can also be selected at submission time:

```bash
qsub -v METHOD=PiSSA scripts/run_peft.sh
qsub -v METHOD=FiltLoRA scripts/run_peft.sh
```

Resource requests, including GPU type, memory, and
wall time, should be adjusted to the available cluster.

These scripts illustrate the experimental workflow.
The default hyperparameters are examples and do not
represent every configuration evaluated in the thesis.


### Adapter choices

| `METHOD` | Implemented using | Notes |
| --- | --- | --- |
| `LoRA` | PEFT | Standard low-rank update of selected layers |
| `DoRA` | PEFT `use_dora` | Magnitude/direction variant |
| `PiSSA` | Custom `brainage/models/adapters.py` | SVD initializes A/B, subtracts their exact contribution from frozen W |
| `FiltLoRA` | Custom `brainage/models/adapters.py` | 3D trigonometric filter between low-rank projections A and B |

`TARGET_SCOPE` supports `head`, `dense4`, `dense3_4`, `dense2_4`,
`dense1_4`, `encoder`, and `all`. It selects MONAI DenseNet module names, not
an arbitrary comma-separated module list. Custom PiSSA and FiltLoRA are
saved as full state dictionaries, while PEFT LoRA/DoRA save adapter directories.

**PiSSA:** `initialize_pissa_wrapper()` reshapes the pretrained Conv3d kernel to
2D, takes its leading singular vectors and values, initializes the low-rank
branch, and stores the residual in the frozen base kernel.

**FiltLoRA:** `FilteredLoRAConv3d` applies `CoLoRATrigFilter3d` between the
low-rank projections. This filter uses a fixed spatial Gaussian kernel and
per-sample/channel trigonometric weights derived from the activation range.

## Analyses and ensembles

`analysis/ensembles.py` contains the TP1-only versus TP1+TP3 frozen
ensemble experiment (mean/learned weights, gate and ridge stacking). Run
`python analysis/ensembles.py --help` for its required prediction CSV, split
directory and output path. Model-selection input must come from the development
folds, not the held-out test set.

### 1. Generate development predictions

After fine-tuning, each run already contains `TL_summary.json`, the selected
checkpoint and a test-prediction CSV produced by `brainage.main`/
`brainage.inference`. To obtain the **development** predictions, run:

```bash
python -m brainage.infer_all_subjects \
  --runs-dir '<RUNS_DIR>' \
  --development-csv '<DEVELOPMENT_CSV>' \
  --image-root '<IMAGE_ROOT>' \
  --pretrained-root '<PRETRAINED_WEIGHTS_ROOT>'
```

The script reads each `TL_summary.json`, loads the corresponding pretrained
backbone and fine-tuned checkpoint, and writes `development_predictions.csv`
next to the summary (`<RUNS_DIR>/TL_*/seedN/ft1p0/`). It uses the same validation
image transform as the test inference. To process a single run, pass
`--summary '<PATH_TO_TL_summary.json>'`; `--dry-run` checks available paths and
configuration without running MRI inference.

For historical **FiltLoRA** runs that do not record `lora_alpha` in the summary,
pass `--filter-alpha` with the **actual training value**. For example, add
`--filter-alpha 16` only if that run used alpha 16. If values differ between
runs, process them individually with `--summary` and the corresponding alpha.
The script does not guess this value because it changes model predictions.

### 2. Merge predictions across runs

Combine the existing per-run CSVs into wide tables (one prediction column per
configuration and seed, matched on `SUBJ_ID`):

```bash
# Development predictions: input from infer_all_subjects.py
python analysis/merge_predictions.py \
  --runs-dir '<RUNS_DIR>' --base-csv '<DEVELOPMENT_CSV>' \
  --mode development --output '<VALIDATION_PREDICTIONS_CSV>'

# Test predictions: input from brainage.inference during training
python analysis/merge_predictions.py \
  --runs-dir '<RUNS_DIR>' --base-csv '<TEST_CSV>' \
  --mode test --output '<TEST_PREDICTIONS_CSV>'
```

Both merged tables use the same model/seed naming convention, with the
fine-tuning tag (e.g. `ft1p0`) appearing exactly once. The merger does not run
inference; it combines previously generated CSVs.

### 3. Select models on validation folds

```bash
python analysis/select_models.py --validation_results '<VALIDATION_PREDICTIONS_CSV>' \
  --split_dir '<SPLIT_DIR>' --out_dir '<VALIDATION_REPORT_DIR>' \
  --split_prefix dataset

# Optional overview of completed training runs
python analysis/summarize_runs.py --runs-dir '<RUNS_DIR>' \
  --output '<SUMMARY_CSV>'
```

Model selection uses only validation-fold subjects from the development cohort,
not test predictions. Its output includes
`<VALIDATION_REPORT_DIR>/selected_models_validation_all_TPs.csv`.

### 4. Prepare subject-level errors on the held-out test set

Use the merged test predictions and **validation-selected** configurations:

```bash
python analysis/prepare_subject_errors.py \
  --predictions '<TEST_PREDICTIONS_CSV>' \
  --selected-models '<VALIDATION_REPORT_DIR>/selected_models_validation_all_TPs.csv' \
  --out-dir '<RESULTS_DIR>' \
  --exclude-qc-notes
```

The output `ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv`
contains `Subject`, `Pretraining`, `Adaptation`, `Age`, `Sex`, `SiteGroup`,
`abs_error` and `signed_error`, among other columns. The `--exclude-qc-notes`
flag applies the subject exclusion rule used in the thesis. The absolute error
is computed by averaging the **four individual absolute errors** per subject,
not by taking the absolute error of the mean prediction.

### 5. Run statistical analyses

```bash
Rscript analysis/improvement_analysis.R '<RESULTS_DIR>' '<SPLIT_DIR>'
Rscript analysis/site_effects.R '<RESULTS_DIR>'
Rscript analysis/age_bias.R '<RESULTS_DIR>'
python analysis/age_bias_plot.py \
  --input '<RESULTS_DIR>/ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv' \
  --output '<PLOT_PATH>'
```

The improvement analysis additionally requires the named fold CSVs in
`<SPLIT_DIR>`. These steps describe the analysis workflow after preprocessing;
model training, development inference and data preparation require access to
the preprocessed images and pretrained weights, which are not distributed here.
