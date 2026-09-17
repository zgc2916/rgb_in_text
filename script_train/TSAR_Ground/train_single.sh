#!/usr/bin/env bash
set -euo pipefail

# TSAR-Ground phase 1: IAFv3 + Grounding-Quality Reliability Router (GQR).
# This is intentionally separate from script_train/RGBT_VGNet so official
# baseline recipes remain unchanged.  The default recipe uses no more than
# two GPUs and the released CLIP/MMVG initialization for a clean ablation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-rgbtvg_flir}"
case "${DATASET}" in
  rgbtvg_flir)
    DEFAULT_LR="0.0001"
    DEFAULT_PORT="31887"
    ;;
  rgbtvg_m3fd)
    DEFAULT_LR="0.0002"
    DEFAULT_PORT="32887"
    ;;
  rgbtvg_mfad)
    DEFAULT_LR="0.0001"
    DEFAULT_PORT="33887"
    ;;
  *)
    echo "Unsupported DATASET: ${DATASET}" >&2
    exit 2
    ;;
esac

IMGSIZE="${IMGSIZE:-224}"
BATCHSIZE="${BATCHSIZE:-12}"
# GPUs 0 and 2 were selected as the currently idle 24-GB cards.  Override
# explicitly when the local scheduler assigns different devices.
CUDADEVICES="${CUDADEVICES:-0,2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-120}"
LR="${LR:-${DEFAULT_LR}}"
MASTER_PORT="${MASTER_PORT:-${DEFAULT_PORT}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/pretrain_model}"
# Keep phase-1 comparable with the documented official MMVGFusion baseline.
VL_BACKBONE="${VL_BACKBONE:-clip}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${MODEL_ROOT}/pretrained_weights/CLIP/clip-vit-base-patch16}"
if [[ "${VL_BACKBONE}" == "siglip2" ]]; then
  MAX_QUERY_LEN="${MAX_QUERY_LEN:-64}"
  CONTRASTIVE_LOSS="${CONTRASTIVE_LOSS:-siglip}"
  IMAGE_NORM="${IMAGE_NORM:-siglip2}"
  RETRAIN="${RETRAIN:-}"
else
  MAX_QUERY_LEN="${MAX_QUERY_LEN:-77}"
  CONTRASTIVE_LOSS="${CONTRASTIVE_LOSS:-clip}"
  IMAGE_NORM="${IMAGE_NORM:-dataset}"
  RETRAIN="${RETRAIN:-${MODEL_ROOT}/pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth}"
fi

MODEL_ROOT="${MODEL_ROOT}" VL_BACKBONE="${VL_BACKBONE}" PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH}" \
  "${REPO_ROOT}/script_train/RGBT_VGNet/prepare_pretrained.sh"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/datasets/VG/image_data}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output_tsar_ground}"

NPROC_PER_NODE="$(tr ',' '\n' <<<"${CUDADEVICES}" | sed '/^$/d' | wc -l | tr -d ' ')"
if (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > 2 )); then
  echo "TSAR-Ground training accepts 1-2 GPUs; CUDADEVICES='${CUDADEVICES}' resolves to ${NPROC_PER_NODE}." >&2
  exit 2
fi

# The official FLIR recipe uses 8 GPUs x 12 samples = global batch 96.  With
# one or two GPUs, accumulate micro-batches so the optimizer sees a comparable
# effective batch instead of taking 4-8x more noisy updates per epoch.
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-96}"
MICRO_GLOBAL_BATCH=$(( BATCHSIZE * NPROC_PER_NODE ))
DEFAULT_GRAD_ACCUM_STEPS=$(( (TARGET_GLOBAL_BATCH + MICRO_GLOBAL_BATCH - 1) / MICRO_GLOBAL_BATCH ))
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${DEFAULT_GRAD_ACCUM_STEPS}}"
if (( GRAD_ACCUM_STEPS < 1 )); then
  echo "GRAD_ACCUM_STEPS must be positive; got ${GRAD_ACCUM_STEPS}." >&2
  exit 2
fi
EFFECTIVE_GLOBAL_BATCH=$(( MICRO_GLOBAL_BATCH * GRAD_ACCUM_STEPS ))

# This recovery profile keeps the released backbone learning rate, but makes
# the randomly initialized GQR branch conservative and waits for its two
# auxiliary localizers before allowing it to correct IAFv3.
NEW_MODULE_LR="${NEW_MODULE_LR:-0.0001}"
CLIP_MAX_NORM="${CLIP_MAX_NORM:-1.0}"
GQR_AUX_WEIGHT="${GQR_AUX_WEIGHT:-0.15}"
GQR_ROUTER_WEIGHT="${GQR_ROUTER_WEIGHT:-0.10}"
GQR_START_EPOCH="${GQR_START_EPOCH:-10}"
GQR_RAMP_EPOCHS="${GQR_RAMP_EPOCHS:-10}"
GQR_ETA_MAX="${GQR_ETA_MAX:-1.0}"
GQR_TEACHER_MARGIN="${GQR_TEACHER_MARGIN:-0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/TSAR_Ground_GQRv1_${IMGSIZE}/${DATASET}/seed_${SEED:-13}_stable}"

INIT_ARGS=()
if [[ -n "${RETRAIN}" ]]; then
  INIT_ARGS+=(--retrain "${RETRAIN}")
fi

echo "[TSAR-Ground/GQR stable] dataset=${DATASET} GPUs=${CUDADEVICES} per_gpu_batch=${BATCHSIZE} grad_accum=${GRAD_ACCUM_STEPS} effective_global_batch=${EFFECTIVE_GLOBAL_BATCH} output=${OUTPUT_DIR}"
env CUDA_VISIBLE_DEVICES="${CUDADEVICES}" "${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" --use_env --master_port "${MASTER_PORT}" \
  train_val/mmvg_train.py \
  --model ViT-B/16 --model_name MMVGFusion --FusionMethod GQRv1 --enable_gqr --modality rgbt \
  --open_lora True --open_text_guided_fusion True --lavs_mode "${LAVS_MODE:-lavs}" \
  --lora_r_rgb 16 --lora_r_ir 48 --num_workers "${NUM_WORKERS}" --epochs "${EPOCHS}" \
  --batch_size "${BATCHSIZE}" --lr "${LR}" --new_module_lr "${NEW_MODULE_LR}" --lr_scheduler cosine \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" --clip_max_norm "${CLIP_MAX_NORM}" \
  --gqr_tau "${GQR_TAU:-0.25}" --gqr_aux_weight "${GQR_AUX_WEIGHT}" \
  --gqr_router_weight "${GQR_ROUTER_WEIGHT}" --gqr_start_epoch "${GQR_START_EPOCH}" \
  --gqr_ramp_epochs "${GQR_RAMP_EPOCHS}" --gqr_eta_max "${GQR_ETA_MAX}" \
  --gqr_teacher_margin "${GQR_TEACHER_MARGIN}" \
  --report_acc07 \
  --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --imsize "${IMGSIZE}" \
  --max_query_len "${MAX_QUERY_LEN}" --normalize_before --mixup_pretrain --dataset "${DATASET}" \
  --vl_backbone "${VL_BACKBONE}" --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
  --contrastive_loss "${CONTRASTIVE_LOSS}" --image_norm "${IMAGE_NORM}" \
  --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss \
  --data_root "${DATA_ROOT}" --split_root "${SPLIT_ROOT}" --seed "${SEED:-13}" \
  --output_dir "${OUTPUT_DIR}" "${INIT_ARGS[@]}"
