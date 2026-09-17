#!/usr/bin/env bash
set -euo pipefail

# Evaluate RGBT-VGNet on one dataset and modality.
DATA_SET=${DATASET:-rgbtvg_flir}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
RESOURCE_ROOT=${RESOURCE_ROOT:-"$(cd "$REPO_ROOT/.." && pwd)/dataset_and_pretrain_model"}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-16}
MODALITY="rgbt"
CUDADEVICES=${CUDADEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
VL_BACKBONE=${VL_BACKBONE:-siglip2}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"$RESOURCE_ROOT/pretrain_model/pretrained_weights/SiLP2"}
MAX_QUERY_LEN=${MAX_QUERY_LEN:-64}
CONTRASTIVE_LOSS=${CONTRASTIVE_LOSS:-siglip}
IMAGE_NORM=${IMAGE_NORM:-siglip2}
MASTER_PORT=${MASTER_PORT:-28773}

EVAL_SETS=${EVAL_SETS:-"val test testA testB testC"}

DATA_ROOT=${DATA_ROOT:-"$RESOURCE_ROOT/datasets/VG/image_data"}
SPLIT_ROOT=${SPLIT_ROOT:-"$RESOURCE_ROOT/datasets/VG/ref_data_shuffled"}
WEIGHT_ROOT=${WEIGHT_ROOT:-"$RESOURCE_ROOT/result/MMVG"}
case "$DATA_SET" in
  rgbtvg_flir) DEFAULT_WEIGHT="$WEIGHT_ROOT/MMVG_rgbt_flir_best.pth" ;;
  rgbtvg_m3fd) DEFAULT_WEIGHT="$WEIGHT_ROOT/MMVG_rgbt_m3fd_best.pth" ;;
  rgbtvg_mfad) DEFAULT_WEIGHT="$WEIGHT_ROOT/MMVG_rgbt_mfad_best.pth" ;;
  *) echo "Unsupported DATASET: $DATA_SET" >&2; exit 2 ;;
esac
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-$DEFAULT_WEIGHT}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/eval_official/RGBT_VGNet_${IMGSIZE}_${MODALITY}/$DATA_SET"}

if [[ ! -f "$EVAL_MODEL_PATH" ]]; then
  echo "Checkpoint not found: $EVAL_MODEL_PATH" >&2
  echo "Download the released model weights and place them in: $WEIGHT_ROOT" >&2
  exit 2
fi

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES "$PYTHON_BIN" -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

# Some model dependencies use paths relative to the repository root.
cd "$REPO_ROOT"

EVAL_ARGS=( \
  --model_name MMVGFusion \
  --FusionMethod IAFv3 \
  --open_lora True \
  --open_text_guided_fusion True \
  --modality $MODALITY \
  --batch_size $BATCHSIZE \
  --dataset $DATA_SET \
  --vl_hidden_dim 512 \
  --imsize $IMGSIZE \
  --max_query_len $MAX_QUERY_LEN \
  --vl_backbone $VL_BACKBONE \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --contrastive_loss $CONTRASTIVE_LOSS \
  --image_norm $IMAGE_NORM \
  --normalize_before \
  --mixup_pretrain \
  --use_mask_loss \
  --hi_lora_stage 3 \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --output_dir $OUTPUT_DIR \
  --model ViT-B/16 \
)


evaluate() {
  local eval_set=$1
  echo -e "\n>>>> [RGBT-VGNet] Eval set: $eval_set, model: $EVAL_MODEL_PATH"
  "${DIST_CMD[@]}" \
    --master_port "$MASTER_PORT" \
    "$REPO_ROOT/train_val/mmvg_eval.py" \
    "${EVAL_ARGS[@]}" \
    --eval_model "$EVAL_MODEL_PATH" \
    --eval_set "$eval_set"
}

for es in $EVAL_SETS; do
  evaluate "$es"
done
