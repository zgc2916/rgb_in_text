#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export IMGSIZE="${IMGSIZE:-224}"
export LAVS_MODE="${LAVS_MODE:-lavs}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VL_BACKBONE="${VL_BACKBONE:-siglip2}"
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/pretrain_model}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${MODEL_ROOT}/pretrained_weights/SiLP2}"
export MAX_QUERY_LEN="${MAX_QUERY_LEN:-64}"
export CONTRASTIVE_LOSS="${CONTRASTIVE_LOSS:-siglip}"
export IMAGE_NORM="${IMAGE_NORM:-siglip2}"

if [[ "${VL_BACKBONE}" == "siglip2" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys
import torch
import transformers
import peft
import timm

from transformers import SiglipModel

print(
    "SigLIP2 runtime:"
    f" python={sys.executable}"
    f" torch={torch.__version__}"
    f" transformers={transformers.__version__}"
    f" peft={peft.__version__}"
    f" timm={timm.__version__}"
)
PY
fi

"${SCRIPT_DIR}/prepare_pretrained.sh"
bash "${SCRIPT_DIR}/ref_flir/train.sh"
bash "${SCRIPT_DIR}/ref_m3fd/train.sh"
bash "${SCRIPT_DIR}/ref_mfad/train.sh"
