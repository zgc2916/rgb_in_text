#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/pretrain_model}"
PRETRAINED_DIR="${MODEL_ROOT}/pretrained_weights"
MMVG_CKPT="${PRETRAINED_DIR}/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth"
CLIP_DIR="${PRETRAINED_DIR}/CLIP/clip-vit-base-patch16"
SIGLIP2_DIR="${PRETRAINED_MODEL_PATH:-${PRETRAINED_DIR}/SiLP2}"

if [[ "${VL_BACKBONE:-clip}" == "siglip2" ]]; then
  if [[ -f "${SIGLIP2_DIR}/model.safetensors" \
        && -f "${SIGLIP2_DIR}/config.json" \
        && -f "${SIGLIP2_DIR}/tokenizer.json" \
        && -f "${SIGLIP2_DIR}/tokenizer.model" \
        && -f "${SIGLIP2_DIR}/tokenizer_config.json" \
        && -f "${SIGLIP2_DIR}/preprocessor_config.json" ]]; then
    exit 0
  fi

  if [[ "${AUTO_DOWNLOAD:-1}" != "1" ]]; then
    echo "Missing SigLIP2 files under ${SIGLIP2_DIR}." >&2
    exit 1
  fi

  if command -v hf >/dev/null 2>&1; then
    HF_CLI=(hf download)
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=(huggingface-cli download)
  else
    echo "Missing Hugging Face CLI. Install it in the rgbtvg_siglip2 environment first." >&2
    exit 1
  fi

  mkdir -p "${SIGLIP2_DIR}"
  "${HF_CLI[@]}" google/siglip2-base-patch16-224 --local-dir "${SIGLIP2_DIR}"
  [[ -f "${SIGLIP2_DIR}/model.safetensors" ]] || { echo "SigLIP2 model download failed." >&2; exit 1; }
  exit 0
fi

if [[ -f "${MMVG_CKPT}" && -f "${CLIP_DIR}/pytorch_model.bin" ]]; then
  exit 0
fi

if [[ "${AUTO_DOWNLOAD:-1}" != "1" ]]; then
  echo "Missing released pretrained models under ${PRETRAINED_DIR}." >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}"
if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli download)
else
  echo "Missing Hugging Face CLI. Install requirements.txt first." >&2
  exit 1
fi

"${HF_CLI[@]}" JiawenXi/RGBT-Ground-Model \
  --include "pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth" \
  --include "pretrained_weights/CLIP/clip-vit-base-patch16/*" \
  --local-dir "${MODEL_ROOT}"

[[ -f "${MMVG_CKPT}" ]] || { echo "Pretrained MMVG checkpoint download failed." >&2; exit 1; }
[[ -f "${CLIP_DIR}/pytorch_model.bin" ]] || { echo "CLIP model download failed." >&2; exit 1; }
