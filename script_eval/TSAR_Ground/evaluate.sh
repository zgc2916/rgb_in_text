#!/usr/bin/env bash
set -euo pipefail

# Evaluate a phase-1 TSAR-Ground/GQR checkpoint.  This script intentionally
# reconstructs GQRv1, while model.eval() still returns MMVGFusion's normal
# five-value inference output.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-rgbtvg_flir}"
IMGSIZE="${IMGSIZE:-224}"
BATCHSIZE="${BATCHSIZE:-12}"
# Match the two-card training default while avoiding the currently busy GPU 1.
CUDADEVICES="${CUDADEVICES:-0,2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_SETS="${EVAL_SETS:-val test testA testB testC}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/pretrain_model}"
VL_BACKBONE="${VL_BACKBONE:-clip}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${MODEL_ROOT}/pretrained_weights/CLIP/clip-vit-base-patch16}"

if [[ "${VL_BACKBONE}" == "siglip2" ]]; then
  MAX_QUERY_LEN="${MAX_QUERY_LEN:-64}"
  CONTRASTIVE_LOSS="${CONTRASTIVE_LOSS:-siglip}"
  IMAGE_NORM="${IMAGE_NORM:-siglip2}"
else
  MAX_QUERY_LEN="${MAX_QUERY_LEN:-77}"
  CONTRASTIVE_LOSS="${CONTRASTIVE_LOSS:-clip}"
  IMAGE_NORM="${IMAGE_NORM:-dataset}"
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/datasets/VG/image_data}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:?Set EVAL_MODEL_PATH to a TSAR-GQR checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${EVAL_MODEL_PATH}")/evaluation}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${OUTPUT_DIR}}"
NPROC_PER_NODE="$(tr ',' '\n' <<<"${CUDADEVICES}" | sed '/^$/d' | wc -l | tr -d ' ')"
if (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > 2 )); then
  echo "TSAR-Ground evaluation accepts 1-2 GPUs; CUDADEVICES='${CUDADEVICES}' resolves to ${NPROC_PER_NODE}." >&2
  exit 2
fi

for EVAL_SET in ${EVAL_SETS}; do
  echo "[TSAR-Ground/GQR] evaluate dataset=${DATASET} split=${EVAL_SET} checkpoint=${EVAL_MODEL_PATH}"
  env CUDA_VISIBLE_DEVICES="${CUDADEVICES}" "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE}" --use_env --master_port "${MASTER_PORT:-34887}" \
    train_val/mmvg_eval.py \
    --model ViT-B/16 --model_name MMVGFusion --FusionMethod GQRv1 --enable_gqr --modality rgbt \
    --open_lora True --open_text_guided_fusion True --lavs_mode "${LAVS_MODE:-lavs}" \
    --lora_r_rgb 16 --lora_r_ir 48 --hi_lora_stage 3 --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCHSIZE}" --dataset "${DATASET}" --imsize "${IMGSIZE}" \
    --max_query_len "${MAX_QUERY_LEN}" --vl_hidden_dim 512 --normalize_before --mixup_pretrain \
    --vl_backbone "${VL_BACKBONE}" --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    --contrastive_loss "${CONTRASTIVE_LOSS}" --image_norm "${IMAGE_NORM}" --use_mask_loss \
    --report_acc07 \
    --data_root "${DATA_ROOT}" --split_root "${SPLIT_ROOT}" --eval_set "${EVAL_SET}" \
    --eval_model "${EVAL_MODEL_PATH}" --output_dir "${OUTPUT_DIR}" --eval_out_dir "${EVAL_OUT_DIR}"
done
