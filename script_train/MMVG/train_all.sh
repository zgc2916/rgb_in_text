#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export IMGSIZE="${IMGSIZE:-224}"
export BATCHSIZE="${BATCHSIZE:-32}"
export MODALITY="${MODALITY:-rgbt}"
export LAVS_MODE="lavs"
export CUDADEVICES="${CUDADEVICES:-0}"

for DATA_SET in rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad; do
    echo "==================== starting $DATA_SET ===================="
    DATA_SET="$DATA_SET" bash "$SCRIPT_DIR/train_single.sh"
done
