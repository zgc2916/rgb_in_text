#!/bin/bash

# ================= Global hyper-parameters =================
IMGSIZE=${1:-224}
BATCHSIZE=${2:-32}
CUDADEVICES=${3:-0,1,2,3}

export IMGSIZE
export BATCHSIZE
export CUDADEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLIPVG_FIXED="${REPO_ROOT}/../dataset_and_pretrain_model/pretrain_model/pretrained_weights/clipvg/best_checkpoint.pth"

DATASETS=("rgbtvg_m3fd" )
MODALITIES=("rgbt" )


for MODALITY in "${MODALITIES[@]}"; do
  export MODALITY
  echo "Start FSVG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

  # Log directory
  mkdir -p logs/fsvg/$MODALITY

  for DATASET in "${DATASETS[@]}"; do
    export DATASET
    ds_name=${DATASET#rgbtvg_}
    export CLIPVG_PRETRAINED="$CLIPVG_FIXED"

    # Only run:
    # - rgbt modality on m3fd / mfad
    # - ir modality on mfad
    if [ "$MODALITY" = "rgbt" ] && [ "$DATASET" != "rgbtvg_m3fd" ] && [ "$DATASET" != "rgbtvg_mfad" ]; then
      echo "Skipping FSVG training for DATASET=$DATASET and MODALITY=$MODALITY (only m3fd/mfad for rgbt)"
      continue
    fi
    if [ "$MODALITY" = "ir" ] && [ "$DATASET" != "rgbtvg_mfad" ]; then
      echo "Skipping FSVG training for DATASET=$DATASET and MODALITY=$MODALITY (only mfad for ir)"
      continue
    fi
    if [ -f "$CLIPVG_PRETRAINED" ]; then
      echo "[FSVG all] pretrained for ${ds_name}/${MODALITY}: $CLIPVG_PRETRAINED"
    else
      echo "[FSVG all] warning: no pretrained found for ${ds_name}/${MODALITY}"
    fi

    echo "===== Start FSVG ${ds_name^^} training for MODALITY=$MODALITY ====="

    stdbuf -oL -eL bash "${REPO_ROOT}/script_train/FSVG/single.sh" 2>&1 | tee logs/fsvg/$MODALITY/${IMGSIZE}_${BATCHSIZE}_${ds_name}.log
  done
done
