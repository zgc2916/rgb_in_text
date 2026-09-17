#!/usr/bin/env bash
set -euo pipefail

# Original MMVG/LAVS training recipe.  This intentionally builds MMVG rather
# than the newer MMVGFusion/IAFv3 model.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

DATA_SET="${DATA_SET:-${DATASET:-rgbtvg_flir}}"
IMGSIZE="${IMGSIZE:-224}"
BATCHSIZE="${BATCHSIZE:-32}"
MODALITY="${MODALITY:-rgbt}"
CUDADEVICES="${CUDADEVICES:-0}"
LAVS_MODE="${LAVS_MODE:-lavs}"
LORA_R_RGB="${LORA_R_RGB:-16}"
LORA_R_IR="${LORA_R_IR:-48}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-60}"
STAGE_EPOCHS="${STAGE_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
EVAL_SETS="${EVAL_SETS:-val test}"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/../dataset_and_pretrain_model}"
DATA_ROOT="${DATA_ROOT:-$DATASET_ROOT/datasets/VG/image_data}"
SPLIT_ROOT="${SPLIT_ROOT:-$DATASET_ROOT/datasets/VG/ref_data_shuffled}"
RETRAIN="${RETRAIN:-$DATASET_ROOT/pretrain_model/pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output_retraining}"

# Keep the old model name fixed on this branch.  Changing this value would
# silently switch to the IAFv3 model in models/__init__.py.
MODEL_NAME="MMVG"
FUSION_METHOD="concat"
MODEL="ViT-B/16"

if [[ "$MODALITY" != "rgbt" ]]; then
    echo "Warning: the released MMVG result weights are rgbt; current modality is '$MODALITY'." >&2
fi
if [[ "$LAVS_MODE" != "lavs" ]]; then
    echo "Warning: this branch targets original LAVS; current lavs_mode is '$LAVS_MODE'." >&2
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Missing data root: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -d "$SPLIT_ROOT/$DATA_SET" ]]; then
    echo "Missing split directory: $SPLIT_ROOT/$DATA_SET" >&2
    exit 1
fi
if [[ ! -f "$RETRAIN" ]]; then
    echo "Missing retraining checkpoint: $RETRAIN" >&2
    exit 1
fi

NPROC_PER_NODE=$(awk -F, '{print NF}' <<< "$CUDADEVICES")
DIST_CMD=(env "CUDA_VISIBLE_DEVICES=$CUDADEVICES" python -m torch.distributed.launch
    "--nproc_per_node=$NPROC_PER_NODE" --use_env)

RUN_TAG="${LAVS_MODE}/lora_r${LORA_R_RGB}_${LORA_R_IR}"
RUN_ROOT="$OUTPUT_ROOT/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/$RUN_TAG"
OUTPUT_DIR_WARMUP="$RUN_ROOT/output_v100"
OUTPUT_DIR_STAGE1="$RUN_ROOT/output_v101"
OUTPUT_DIR_STAGE2="$RUN_ROOT/output_v102"
OUTPUT_DIR_STAGE3="$RUN_ROOT/output_v103"

TRAIN_COMMON_ARGS=(
    --model "$MODEL"
    --model_name "$MODEL_NAME"
    --FusionMethod "$FUSION_METHOD"
    --modality "$MODALITY"
    --open_lora true
    --open_text_guided_fusion true
    --lavs_mode "$LAVS_MODE"
    --lora_r_rgb "$LORA_R_RGB"
    --lora_r_ir "$LORA_R_IR"
    --num_workers "$NUM_WORKERS"
    --batch_size "$BATCHSIZE"
    --lr_scheduler cosine
    --aug_crop
    --aug_scale
    --aug_translate
    --vl_hidden_dim 512
    --imsize "$IMGSIZE"
    --max_query_len 77
    --normalize_before
    --mixup_pretrain
    --dataset "$DATA_SET"
    --use_contrastive_loss
    --use_rtcc_constrain_loss
    --use_mask_loss
    --data_root "$DATA_ROOT"
    --split_root "$SPLIT_ROOT"
    --sup_type full
)

EVAL_COMMON_ARGS=(
    --model "$MODEL"
    --model_name "$MODEL_NAME"
    --FusionMethod "$FUSION_METHOD"
    --modality "$MODALITY"
    --open_lora true
    --open_text_guided_fusion true
    --lavs_mode "$LAVS_MODE"
    --lora_r_rgb "$LORA_R_RGB"
    --lora_r_ir "$LORA_R_IR"
    --num_workers "$EVAL_WORKERS"
    --batch_size "$BATCHSIZE"
    --dataset "$DATA_SET"
    --imsize "$IMGSIZE"
    --max_query_len 77
    --normalize_before
    --mixup_pretrain
    --use_mask_loss
    --save_hilora_clip
    --data_root "$DATA_ROOT"
    --split_root "$SPLIT_ROOT"
)

run_train() {
    local output_dir="$1"
    local epochs="$2"
    local lr="$3"
    shift 3

    mkdir -p "$output_dir"
    echo "[MMVG/LAVS] train: dataset=$DATA_SET output=$output_dir epochs=$epochs lr=$lr"
    "${DIST_CMD[@]}" --master_port "${TRAIN_PORT:-28887}" \
        train_val/mmvg_train.py \
        "${TRAIN_COMMON_ARGS[@]}" \
        --epochs "$epochs" \
        --lr "$lr" \
        --output_dir "$output_dir" \
        --save_hilora_clip \
        "$@"
}

run_eval() {
    local output_dir="$1"
    local checkpoint="$2"
    local eval_set="$3"

    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing evaluation checkpoint: $checkpoint" >&2
        exit 1
    fi
    echo "[MMVG/LAVS] eval: dataset=$DATA_SET split=$eval_set checkpoint=$checkpoint"
    "${DIST_CMD[@]}" --master_port "${EVAL_PORT:-28888}" \
        train_val/mmvg_eval.py \
        "${EVAL_COMMON_ARGS[@]}" \
        --eval_model "$checkpoint" \
        --eval_set "$eval_set" \
        --output_dir "$output_dir" \
        --eval_out_dir "$output_dir"
}

eval_stage() {
    local output_dir="$1"
    local checkpoint="$2"
    if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
        return
    fi
    for eval_set in $EVAL_SETS; do
        run_eval "$output_dir" "$checkpoint" "$eval_set"
    done
}

echo "==================== MMVG/LAVS: $DATA_SET ===================="
echo "project=$PROJECT_ROOT"
echo "data=$DATA_ROOT"
echo "split=$SPLIT_ROOT"
echo "gpus=$CUDADEVICES nproc=$NPROC_PER_NODE batch_per_gpu=$BATCHSIZE"

# Warm-up from the released MMVG mixup-pretraining checkpoint.
run_train "$OUTPUT_DIR_WARMUP" "$WARMUP_EPOCHS" 0.0005 \
    --retrain "$RETRAIN"
eval_stage "$OUTPUT_DIR_WARMUP" "$OUTPUT_DIR_WARMUP/best_checkpoint.pth"

# HiLoRA stages: each stage starts from the previous stage's complete model
# and its saved CLIP adapter checkpoint.
run_train "$OUTPUT_DIR_STAGE1" "$STAGE_EPOCHS" 0.00010 \
    --hi_lora_stage 1 \
    --hi_lora_retrain "$OUTPUT_DIR_WARMUP/best_checkpoint.pth" \
    --hi_lora_clip "$OUTPUT_DIR_WARMUP/clip_lora_stage_with_bridge.pth"
eval_stage "$OUTPUT_DIR_STAGE1" "$OUTPUT_DIR_STAGE1/best_checkpoint.pth"

run_train "$OUTPUT_DIR_STAGE2" "$STAGE_EPOCHS" 0.00002 \
    --hi_lora_stage 2 \
    --hi_lora_retrain "$OUTPUT_DIR_STAGE1/best_checkpoint.pth" \
    --hi_lora_clip "$OUTPUT_DIR_STAGE1/clip_lora_stage_with_bridge.pth"
eval_stage "$OUTPUT_DIR_STAGE2" "$OUTPUT_DIR_STAGE2/best_checkpoint.pth"

run_train "$OUTPUT_DIR_STAGE3" "$STAGE_EPOCHS" 0.000005 \
    --hi_lora_stage 3 \
    --hi_lora_retrain "$OUTPUT_DIR_STAGE2/best_checkpoint.pth" \
    --hi_lora_clip "$OUTPUT_DIR_STAGE2/clip_lora_stage_with_bridge.pth"
eval_stage "$OUTPUT_DIR_STAGE3" "$OUTPUT_DIR_STAGE3/best_checkpoint.pth"

echo "MMVG/LAVS finished: $DATA_SET"
echo "final checkpoint: $OUTPUT_DIR_STAGE3/best_checkpoint.pth"
