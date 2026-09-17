#!/usr/bin/env bash
set -euo pipefail

# Evaluate the released RGBT-VGNet checkpoints on all benchmark datasets.
DATASETS=${DATASETS:-"rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad"}
EVAL_SETS=${EVAL_SETS:-"val test testA testB testC"}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-16}
CUDADEVICES=${CUDADEVICES:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
EVAL_SCRIPT="$SCRIPT_DIR/evaluate.sh"
LOG_ROOT=${LOG_ROOT:-"$REPO_ROOT/logs/eval/RGBT_VGNet"}
mkdir -p "$LOG_ROOT"

for ds in $DATASETS; do
  echo -e "\n==================== [RGBT-VGNet/IAFv3] DATASET: $ds ===================="
  LOG_FILE="$LOG_ROOT/${IMGSIZE}_rgbt_${ds}.log"
  DATASET="$ds" IMGSIZE="$IMGSIZE" BATCHSIZE="$BATCHSIZE" \
    CUDADEVICES="$CUDADEVICES" EVAL_SETS="$EVAL_SETS" \
    bash "$EVAL_SCRIPT" 2>&1 | tee "$LOG_FILE"
done
