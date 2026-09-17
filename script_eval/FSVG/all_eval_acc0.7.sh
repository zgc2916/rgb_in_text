#!/bin/bash

# FSVG eval script (acc 0.7 version placeholder)
# - 3 datasets
# - only evaluate on: val, test
# NOTE: 0.7 相关的具体逻辑你可以在模型或 single_eval 里自行修改

DATASETS=${DATASETS:-"rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad"}
MODALITIES=${MODALITIES:-"rgbt"}
EVAL_SETS=${EVAL_SETS:-"val test"}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-64}
CUDADEVICES=${CUDADEVICES:-4,5}
CLIPMODEL=${CLIPMODEL:-"ViT-B/16"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"

# Keep eval logs separate from training logs (training uses logs/fsvg/*)
LOG_ROOT="./logs/eval/FSVG_acc0.7"
mkdir -p "$LOG_ROOT"

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [FSVG acc0.7] DATASET: $ds, MODALITY: $m ==========================="
    LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"
    DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES CLIPMODEL="$CLIPMODEL" EVAL_SETS="$EVAL_SETS" \
      stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
  done
done

