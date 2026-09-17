#!/bin/bash

# Only evaluate the requested 4 (modality, dataset) pairs:
# - ir   on flir
# - rgb  on flir
# - rgb  on mfad
# - rgbt on mfad
#
# You can still override by exporting PAIRS manually, e.g.
#   PAIRS="rgb:rgbtvg_flir ir:rgbtvg_flir" bash all_eval.sh
PAIRS=${PAIRS:-"ir:rgbtvg_m3fd"}
EVAL_SETS=${EVAL_SETS:-"test testA testB testC val \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-64}
CUDADEVICES=${CUDADEVICES:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"
LOG_ROOT="./logs/eval/AttBalance"
mkdir -p "$LOG_ROOT"

for pair in $PAIRS; do
  m="${pair%%:*}"
  ds="${pair#*:}"
  if [ -z "$m" ] || [ -z "$ds" ] || [ "$m" = "$ds" ]; then
    echo "Invalid PAIRS entry: '$pair' (expected MODALITY:DATASET)"
    exit 2
  fi

  echo -e "\n==================== [AttBalance] DATASET: $ds, MODALITY: $m ==========================="
  LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"
  DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES EVAL_SETS="$EVAL_SETS" \
    stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
done
