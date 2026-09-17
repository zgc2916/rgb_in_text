#!/bin/bash

# AttBalance eval script (acc 0.7 version placeholder)
# - 3 datasets
# - only evaluate on: val, test
# NOTE: 0.7 相关的具体逻辑你可以在模型或 single_eval 里自行修改

# 默认 3 个数据集，单一模态 ir，你也可以在外面通过 PAIRS 覆盖
PAIRS=${PAIRS:-"rgbt:rgbtvg_flir rgbt:rgbtvg_m3fd rgbt:rgbtvg_mfad"}
EVAL_SETS=${EVAL_SETS:-"test val"}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-64}
CUDADEVICES=${CUDADEVICES:-4,5}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"
LOG_ROOT="./logs/eval/AttBalance_acc0.7"
mkdir -p "$LOG_ROOT"

for pair in $PAIRS; do
  m="${pair%%:*}"
  ds="${pair#*:}"
  if [ -z "$m" ] || [ -z "$ds" ] || [ "$m" = "$ds" ]; then
    echo "Invalid PAIRS entry: '$pair' (expected MODALITY:DATASET)"
    exit 2
  fi

  echo -e "\n==================== [AttBalance acc0.7] DATASET: $ds, MODALITY: $m ==========================="
  LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"
  rm -f "$LOG_FILE"
  DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES EVAL_SETS="$EVAL_SETS" \
    stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
done

