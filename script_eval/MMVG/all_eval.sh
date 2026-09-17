#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

DATASETS="${DATASETS:-rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad}"
MODALITIES="${MODALITIES:-rgbt}"
EVAL_SETS="${EVAL_SETS:-test}"
IMGSIZE="${IMGSIZE:-224}"
BATCHSIZE="${BATCHSIZE:-36}"
CUDADEVICES="${CUDADEVICES:-0}"

LOG_ROOT="$PROJECT_ROOT/logs/eval/MMVG"
mkdir -p "$LOG_ROOT"

for dataset in $DATASETS; do
    for modality in $MODALITIES; do
        echo "==================== MMVG: $dataset / $modality ===================="
        LOG_FILE="$LOG_ROOT/${IMGSIZE}_${modality}_${dataset}.log"
        DATASET="$dataset" \
        MODALITY="$modality" \
        IMGSIZE="$IMGSIZE" \
        BATCHSIZE="$BATCHSIZE" \
        CUDADEVICES="$CUDADEVICES" \
        EVAL_SETS="$EVAL_SETS" \
        stdbuf -oL -eL bash "$SCRIPT_DIR/single_eval.sh" 2>&1 | tee "$LOG_FILE"
    done
done
