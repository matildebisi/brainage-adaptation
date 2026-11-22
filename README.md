# Brain-age transfer learning

This repository contains code developed and extended as part of a master's thesis investigating cross-population adaptation of 3D DenseNet brain-age prediction models. It includes conventional fine-tuning, parameter-efficient methods (LoRA, DoRA, PiSSA, and FiltLoRA), subject-level cross-validation, oversampling, ensemble experiments, and statistical analyses of predictive performance and model biases.


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
  zero_shot.py          # Inference from an unadapted pretrained checkpoint
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


## Training on an SGE cluster

Training experiments were run using Sun Grid Engine (SGE).
Submission commands are provided in Step 2 of the experimental workflow below.
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


## Experimental workflow

The steps below describe the progression from subject-level cross-validation to
final statistical analyses. Inputs, outputs, and commands are given for each step.
Model configuration selection is performed **only on development validation
folds**, never on the held-out test cohort.

### 1. Prepare development folds

- **Script:** `brainage/data/splits.py`
- **Inputs:** `<DEVELOPMENT_CSV>`, `<IMAGE_ROOT>`.
- **Output:** Four subject-level training/validation splits in `<SPLIT_DIR>`.

```bash
python -m brainage.data.splits \
  --csv '<DEVELOPMENT_CSV>' --img-root '<IMAGE_ROOT>' \
  --output-dir '<SPLIT_DIR>' --prefix dataset \
  --n-splits 4 --split-seed 42
```

All observations from the same subject are kept in the same fold. The saved
folds must correspond to the development cohort used for model training and
validation-based selection.

### 2. Fine-tune models and generate test predictions

- **Scripts:** `brainage/main.py`, `brainage/training.py`, and `brainage/inference.py`, launched via `scripts/run_traditional.sh`, `scripts/run_peft.sh`, or `scripts/run_oversampling.sh`.
- **Inputs:** Pretrained checkpoint, `<DEVELOPMENT_CSV>`, `<TEST_CSV>`, `<IMAGE_ROOT>`, and `<SPLIT_DIR>`.
- **Outputs:** Trained checkpoints, per-run `TL_summary.json`, and individual test prediction CSVs in `<RUNS_DIR>`.

After configuring the SGE scripts (see **Training on an SGE cluster** above),
submit the relevant jobs from the repository root:

```bash
qsub scripts/run_traditional.sh
qsub scripts/run_peft.sh
qsub scripts/run_oversampling.sh
```

Run only the scripts corresponding to the experiments you intend to perform.
The job arrays (`-t 1-4`) run four precomputed folds (seeds 1–4). Each run
trains on three folds and validates on the remaining fold, selecting its best
checkpoint using validation performance. At the end of each training run,
`brainage/inference.py` automatically generates predictions for the held-out
test cohort. These test predictions are **not** used to choose configurations.

The PEFT method can be specified at submission time:

```bash
qsub -v METHOD=PiSSA scripts/run_peft.sh
qsub -v METHOD=FiltLoRA scripts/run_peft.sh
```

Resource requests should be adjusted to the target cluster. The provided
default hyperparameters illustrate the workflow rather than every setting
investigated in the thesis.

### 3. Generate development predictions

- **Script:** `brainage/infer_all_subjects.py`
- **Inputs:** Run summaries (`TL_summary.json`), saved checkpoints, `<DEVELOPMENT_CSV>`, `<IMAGE_ROOT>`, and `<PRETRAINED_WEIGHTS_ROOT>`.
- **Output:** A `development_predictions.csv` file for each run and seed.

```bash
python -m brainage.infer_all_subjects \
  --runs-dir '<RUNS_DIR>' \
  --development-csv '<DEVELOPMENT_CSV>' \
  --image-root '<IMAGE_ROOT>' \
  --pretrained-root '<PRETRAINED_WEIGHTS_ROOT>'
```

The script loads the selected checkpoints, predicts ages for the development
cohort, and writes `development_predictions.csv` beside each `TL_summary.json`
(under `<RUNS_DIR>/TL_*/seedN/ft1p0/`). Later, only the corresponding
validation-fold subjects are used for model selection.


### 4. Merge development predictions

- **Script:** `analysis/merge_predictions.py` (`--mode development`)
- **Inputs:** Per-run `development_predictions.csv` files and `<DEVELOPMENT_CSV>`.
- **Output:** `<VALIDATION_PREDICTIONS_CSV>`.

```bash
python analysis/merge_predictions.py \
  --runs-dir '<RUNS_DIR>' --base-csv '<DEVELOPMENT_CSV>' \
  --mode development --output '<VALIDATION_PREDICTIONS_CSV>'
```

The script matches rows on `SUBJ_ID` and combines predictions into a single
subject-level table with one column per model configuration and seed. It
merges existing predictions; it does not perform inference.

### 5. Select model configurations using validation folds

- **Script:** `analysis/select_models.py`
- **Inputs:** `<VALIDATION_PREDICTIONS_CSV>` and `<SPLIT_DIR>`.
- **Outputs:** Validation performance summaries and `<VALIDATION_REPORT_DIR>/selected_models_validation_all_TPs.csv`.

```bash
python analysis/select_models.py \
  --validation_results '<VALIDATION_PREDICTIONS_CSV>' \
  --split_dir '<SPLIT_DIR>' \
  --out_dir '<VALIDATION_REPORT_DIR>' \
  --split_prefix dataset
```

For each pretrained model and adaptation strategy, configurations are compared
using their mean MAE across the four validation folds. Only each model's
corresponding validation subjects are evaluated; the held-out test set is
not involved in this selection.


### Summarise training runs (optional)

`analysis/summarize_runs.py` aggregates the results of completed experiments across seeds, reporting the mean and standard deviation of:

- **Validation performance:** best validation loss, R², and correlation.
- **Training efficiency:** percentage of trainable parameters, total training duration (minutes), and average time per epoch (seconds).
- **Training details:** epoch corresponding to the best validation performance.

The resulting CSV contains one row per model configuration and records the number of seeds included in the summary. This facilitates comparisons of predictive performance and computational efficiency across adaptation strategies.

```bash
python analysis/summarize_runs.py --runs-dir '<RUNS_DIR>' \
  --output '<SUMMARY_CSV>'
```


### 6. Merge test predictions

- **Script:** `analysis/merge_predictions.py` (`--mode test`)
- **Inputs:** Individual test prediction CSVs generated in Step 2 and `<TEST_CSV>`.
- **Output:** `<TEST_PREDICTIONS_CSV>`.

```bash
python analysis/merge_predictions.py \
  --runs-dir '<RUNS_DIR>' --base-csv '<TEST_CSV>' \
  --mode test --output '<TEST_PREDICTIONS_CSV>'
```

This combines all available test predictions by `SUBJ_ID`.

### 7. Compute subject-level prediction errors

- **Script:** `analysis/prepare_subject_errors.py`
- **Inputs:** `<TEST_PREDICTIONS_CSV>` and `<VALIDATION_REPORT_DIR>/selected_models_validation_all_TPs.csv`.
- **Output:** `<RESULTS_DIR>/ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv`.

```bash
python analysis/prepare_subject_errors.py \
  --predictions '<TEST_PREDICTIONS_CSV>' \
  --selected-models '<VALIDATION_REPORT_DIR>/selected_models_validation_all_TPs.csv' \
  --out-dir '<RESULTS_DIR>' \
  --exclude-qc-notes
```

The resulting table includes `Subject`, `Pretraining`, `Adaptation`, `Age`,
`Sex`, `SiteGroup`, `abs_error`, and `signed_error`. It combines test predictions
for **validation-selected** configurations across four fold-specific models.
The absolute error is the mean of the four individual absolute errors. The `--exclude-qc-notes` option
applies the subject-exclusion criterion used in the thesis.

### 8. Run statistical analyses

- **Scripts:** `analysis/improvement_analysis.R`, `analysis/site_effects.R`, `analysis/age_bias.R`, and `analysis/age_bias_plot.py`.
- **Inputs:** The subject-level error table and, for improvement analysis, `<SPLIT_DIR>`.
- **Outputs:** Statistical comparisons and plots of adaptation improvements, age-related bias, and acquisition-site effects.

```bash
Rscript analysis/improvement_analysis.R '<RESULTS_DIR>'
Rscript analysis/site_effects.R '<RESULTS_DIR>'
Rscript analysis/age_bias.R '<RESULTS_DIR>'
python analysis/age_bias_plot.py \
  --input '<RESULTS_DIR>/ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv' \
  --output '<PLOT_PATH>'
```



## Ensemble experiments

`analysis/ensembles.py` contains the TP1-only versus TP1+TP3 frozen ensemble
experiments (mean/learned weights, gate, and ridge stacking). Run
`python analysis/ensembles.py --help` for its required prediction CSV, split
directory, and output arguments. Any model selection for ensemble weights must
use development folds rather than held-out test data.
