#!/bin/bash
export DATASET=rgbtvg_m3fd
export MODALITY=rgbt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export CLIPVG_PRETRAINED="${REPO_ROOT}/../dataset_and_pretrain_model/pretrain_model/pretrained_weights/clipvg/best_checkpoint.pth"
bash "${SCRIPT_DIR}/single.sh"
