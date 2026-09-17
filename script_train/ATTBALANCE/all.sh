#!/bin/bash

IMGSIZE=${IMGSIZE:-${1:-224}}
BATCHSIZE=${BATCHSIZE:-${2:-64}}
CUDADEVICES=${CUDADEVICES:-${3:-2,3}}
EPOCHS=${EPOCHS:-${4:-120}}

export IMGSIZE
export BATCHSIZE
export CUDADEVICES
export EPOCHS

DATASETS=("rgbtvg_mfad")
MODALITIES=("rgb")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="$REPO_ROOT/logs/attbalance"

mkdir -p "$LOG_ROOT"

for m in "${MODALITIES[@]}"; do

  mkdir -p "$LOG_ROOT/$m"
  echo "Start AttBalance training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$m"

  for ds in "${DATASETS[@]}"; do
    export DATASET=$ds
    export MODALITY=$m
    # rgbt 模态跳过 flir 数据集
    if [[ "$m" == "rgbt" && "$ds" == "rgbtvg_flir" ]]; then
      continue
    fi
    ds_name=${ds#rgbtvg_}
    echo "===== Start AttBalance ${ds_name^^} training (MODALITY=$m) ====="
    stdbuf -oL -eL bash "$SCRIPT_DIR/single.sh" 2>&1 | tee "$LOG_ROOT/$m/${IMGSIZE}_${BATCHSIZE}_${ds_name}.log"
  done
done
