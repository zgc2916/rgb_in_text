#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="rgbtvg_flir"
IMGSIZE="${IMGSIZE:-224}"
BATCHSIZE="${BATCHSIZE_FLIR:-${BATCHSIZE:-12}}"
CUDADEVICES="${CUDADEVICES_FLIR:-${CUDADEVICES:-0,1,2,3,4,5,6,7}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/pretrain_model}"
VL_BACKBONE="${VL_BACKBONE:-siglip2}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${MODEL_ROOT}/pretrained_weights/SiLP2}"
MAX_QUERY_LEN="${MAX_QUERY_LEN:-64}"
CONTRASTIVE_LOSS="${CONTRASTIVE_LOSS:-siglip}"
IMAGE_NORM="${IMAGE_NORM:-siglip2}"
MODEL_ROOT="${MODEL_ROOT}" VL_BACKBONE="${VL_BACKBONE}" PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH}" \
  "${REPO_ROOT}/script_train/RGBT_VGNet/prepare_pretrained.sh"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/datasets/VG/image_data}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled}"
if [[ "${VL_BACKBONE}" == "siglip2" ]]; then
  RETRAIN="${RETRAIN:-}"
else
  RETRAIN="${RETRAIN:-${MODEL_ROOT}/pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth}"
fi
INIT_ARGS=()
if [[ -n "${RETRAIN}" ]]; then
  INIT_ARGS+=(--retrain "${RETRAIN}")
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output_retraining}"
OUTPUT_DIR="${OUTPUT_ROOT}/MMVGFusion_${IMGSIZE}_rgbt_fusion_IAFv3/${DATASET}/rgbt_finetuning_base_${VL_BACKBONE}_weight/output"
NPROC_PER_NODE="$(tr ',' '\n' <<<"${CUDADEVICES}" | wc -l | tr -d ' ')"

env CUDA_VISIBLE_DEVICES="${CUDADEVICES}" "${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" --use_env --master_port "${MASTER_PORT_FLIR:-28887}" \
  train_val/mmvg_train.py --model_name MMVGFusion --FusionMethod IAFv3 --modality rgbt \
  --open_lora True --open_text_guided_fusion True --lavs_mode "${LAVS_MODE:-lavs}" \
  --lora_r_rgb 16 --lora_r_ir 48 --num_workers "${NUM_WORKERS:-4}" --epochs 120 \
  --batch_size "${BATCHSIZE}" --lr 0.0001 --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --imsize "${IMGSIZE}" \
  --max_query_len "${MAX_QUERY_LEN}" --normalize_before --mixup_pretrain --dataset "${DATASET}" \
  --vl_backbone "${VL_BACKBONE}" --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
  --contrastive_loss "${CONTRASTIVE_LOSS}" --image_norm "${IMAGE_NORM}" \
  --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss \
  --data_root "${DATA_ROOT}" --split_root "${SPLIT_ROOT}" \
  --output_dir "${OUTPUT_DIR}" "${INIT_ARGS[@]}"
