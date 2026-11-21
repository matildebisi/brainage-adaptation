#!/bin/bash
#$ -S /bin/bash
#$ -l h_rt=5:00:00
#$ -l tmem=16G
#$ -pe gpu 1
#$ -l gpu=true,gpu_type=(rtx3090)
#$ -cwd
#$ -j y
#$ -R y
#$ -N brainage_traditional
#$ -t 1-4
#$ -tc 4

set -euo pipefail

# Submit with qsub from the repository root (the -cwd directive uses it).
PROJECT_ROOT="$PWD"
DEVELOPMENT_CSV="<DEVELOPMENT_CSV>"
TEST_CSV="<TEST_CSV>"
IMAGE_ROOT="<IMAGE_ROOT>"
SPLIT_DIR="<SPLIT_DIR>"
SPLIT_PREFIX="dataset"
PRETRAINED_WEIGHTS_DIR="<PRETRAINED_WEIGHTS_DIR>"
PRETRAINED_WEIGHTS="<PRETRAINED_WEIGHTS_FILE>"
OUTPUT_DIR="<OUTPUT_DIR>"

VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"

BATCH_SIZE=4
EPOCHS=60
LR=0.0001
FT_RATIO=1.0

# One array task per fold; seeds are matched to the fold numbers.
TASK_ID="${SGE_TASK_ID:-1}"
if [[ ! "$TASK_ID" =~ ^[1-4]$ ]]; then
    echo "Expected SGE_TASK_ID between 1 and 4 (got $TASK_ID)." >&2
    exit 2
fi
FOLD="$TASK_ID"
SEED="$TASK_ID"

cd "$PROJECT_ROOT"
if [[ -n "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
fi
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Fail early if an input path was not configured.
for path in "$DEVELOPMENT_CSV" "$TEST_CSV" "$IMAGE_ROOT" "$SPLIT_DIR" \
            "$PRETRAINED_WEIGHTS_DIR" "$PRETRAINED_WEIGHTS" "$OUTPUT_DIR"; do
    if [[ -z "$path" || "$path" == \<*\> ]]; then
        echo "Replace the path placeholders at the top of this script." >&2
        exit 2
    fi
done
for file in "$DEVELOPMENT_CSV" "$TEST_CSV" "$PRETRAINED_WEIGHTS"; do
    [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 2; }
done
for dir in "$IMAGE_ROOT" "$SPLIT_DIR" "$PRETRAINED_WEIGHTS_DIR"; do
    [[ -d "$dir" ]] || { echo "Missing directory: $dir" >&2; exit 2; }
done
for part in train validation; do
    file="$SPLIT_DIR/${SPLIT_PREFIX}_fold_${FOLD}_base_${part}_rows.csv"
    [[ -f "$file" ]] || { echo "Missing fold file: $file" >&2; exit 2; }
done
mkdir -p "$OUTPUT_DIR"

echo "Starting ${JOB_ID:-local} / task $TASK_ID: fold $FOLD, seed $SEED"

# Full fine-tuning; change to head/dense4/etc. for partial fine-tuning.
FREEZE_MODE="all"

python -m brainage.main \
    --loadModel \
    --weightsFolder "$PRETRAINED_WEIGHTS_DIR" \
    --weights "$PRETRAINED_WEIGHTS" \
    --csv_path "$DEVELOPMENT_CSV" \
    --test "$TEST_CSV" \
    --imgPath "$IMAGE_ROOT" \
    --split_dir "$SPLIT_DIR" \
    --split_prefix "$SPLIT_PREFIX" \
    --fold "$FOLD" \
    --manualSeed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --freeze_mode "$FREEZE_MODE" \
    --ft_ratio "$FT_RATIO" \
    --batchSize "$BATCH_SIZE" \
    --nEpoch "$EPOCHS" \
    --lr "$LR"
