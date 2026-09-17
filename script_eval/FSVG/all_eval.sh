#!/bin/bash

# Batch evaluation script: iterate over datasets × modalities for FSVG

DATASETS=${DATASETS:-"rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad"}
MODALITIES=${MODALITIES:-"rgb ir rgbt"}
EVAL_SETS=${EVAL_SETS:-"test testA testB testC val \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-64}
CUDADEVICES=${CUDADEVICES:-0}
CLIPMODEL=${CLIPMODEL:-"ViT-B/16"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"

# Keep eval logs separate from training logs (training uses logs/fsvg/*)
LOG_ROOT="./logs/eval/FSVG"
mkdir -p "$LOG_ROOT"

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [FSVG] DATASET: $ds, MODALITY: $m ==========================="
    LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"
    DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES CLIPMODEL="$CLIPMODEL" EVAL_SETS="$EVAL_SETS" \
      stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
  done
done
