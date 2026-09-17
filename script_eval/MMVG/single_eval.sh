#!/usr/bin/env bash
set -euo pipefail

# Original MMVG/LAVS evaluation entry point.  Do not change this to
# MMVGFusion/IAFv3: the checkpoints in result/MMVG use the MMVG model.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

DATA_SET="${DATASET:-rgbtvg_flir}"
IMGSIZE="${IMGSIZE:-224}"
BATCHSIZE="${BATCHSIZE:-16}"
MODALITY="${MODALITY:-rgbt}"
CUDADEVICES="${CUDADEVICES:-0}"
LAVS_MODE="${LAVS_MODE:-lavs}"
LORA_R_RGB="${LORA_R_RGB:-16}"
LORA_R_IR="${LORA_R_IR:-48}"
NUM_WORKERS="${NUM_WORKERS:-2}"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/../dataset_and_pretrain_model}"
DATA_ROOT="${DATA_ROOT:-$DATASET_ROOT/datasets/VG/image_data}"
SPLIT_ROOT="${SPLIT_ROOT:-$DATASET_ROOT/datasets/VG/ref_data_shuffled}"
MODEL_PATH="${EVAL_MODEL_PATH:-$DATASET_ROOT/result/MMVG/MMVG_${MODALITY}_${DATA_SET#rgbtvg_}_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/eval_official/MMVG_${IMGSIZE}_${MODALITY}/$DATA_SET}"
EVAL_SETS="${EVAL_SETS:-val test testA testB testC}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Missing MMVG checkpoint: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Missing data root: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -d "$SPLIT_ROOT/$DATA_SET" ]]; then
    echo "Missing split directory: $SPLIT_ROOT/$DATA_SET" >&2
    exit 1
fi

NPROC_PER_NODE=$(awk -F, '{print NF}' <<< "$CUDADEVICES")
DIST_CMD=(env "CUDA_VISIBLE_DEVICES=$CUDADEVICES" python -m torch.distributed.launch
    "--nproc_per_node=$NPROC_PER_NODE" --use_env)

EVAL_ARGS=(
    --model ViT-B/16
    --model_name MMVG
    --FusionMethod concat
    --modality "$MODALITY"
    --open_lora true
    --open_text_guided_fusion true
    --lavs_mode "$LAVS_MODE"
    --lora_r_rgb "$LORA_R_RGB"
    --lora_r_ir "$LORA_R_IR"
    --num_workers "$NUM_WORKERS"
    --batch_size "$BATCHSIZE"
    --dataset "$DATA_SET"
    --vl_hidden_dim 512
    --imsize "$IMGSIZE"
    --max_query_len 77
    --normalize_before
    --mixup_pretrain
    --use_mask_loss
    --data_root "$DATA_ROOT"
    --split_root "$SPLIT_ROOT"
    --eval_model "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --eval_out_dir "$OUTPUT_DIR"
)

echo "[MMVG/LAVS] dataset=$DATA_SET modality=$MODALITY gpus=$CUDADEVICES"
echo "[MMVG/LAVS] checkpoint=$MODEL_PATH"

for eval_set in $EVAL_SETS; do
    echo ">>>> evaluating split: $eval_set"
    "${DIST_CMD[@]}" --master_port "${EVAL_PORT:-28773}" \
        train_val/mmvg_eval.py \
        "${EVAL_ARGS[@]}" \
        --eval_set "$eval_set"
done
