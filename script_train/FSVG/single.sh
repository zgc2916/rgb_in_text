#!/bin/bash

DATA_SET=${DATASET:-rgbtvg_flir}
echo -e "\n\n\n\n\n\n\n==================== FSVG single dataset: $DATA_SET ==========================="

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-16}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}
EPOCHS=${EPOCHS:-120}
CLIPMODEL=${CLIPMODEL:-"ViT-B/16"}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES TORCH_USE_CUDA_DSA=1 python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
DATA_SET_SHORT="${DATA_SET#rgbtvg_}"

# Default to fixed CLIP-VG pretraining checkpoint. Users can override by setting CLIPVG_PRETRAINED.
DEFAULT_CLIPVG_PRETRAINED="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/clipvg/best_checkpoint.pth"
CLIPVG_PRETRAINED=${CLIPVG_PRETRAINED:-"$DEFAULT_CLIPVG_PRETRAINED"}

TRAIN_EXTRA_ARGS=()
if [ -f "$CLIPVG_PRETRAINED" ]; then
  echo "[FSVG] use CLIP-VG pretrained: $CLIPVG_PRETRAINED"
  TRAIN_EXTRA_ARGS+=(--clipvg_pretrained "$CLIPVG_PRETRAINED")
else
  echo "[FSVG] warning: pretrained not found at $CLIPVG_PRETRAINED, train from CLIP init only."
fi

OUTPUT_DIR="./output_training/FSVG_${IMGSIZE}_${MODALITY}/$DATA_SET"
EVAL_MODEL_PATH="$OUTPUT_DIR/best_checkpoint.pth"
mkdir -p $OUTPUT_DIR

# ==================== TRAIN ====================
"${DIST_CMD[@]}" \
  --master_port 28500 \
  train_val/fsvg_train.py \
  --model_name FSVG \
  --model "$CLIPMODEL" \
  --imsize $IMGSIZE \
  --batch_size $BATCHSIZE \
  --lr 0.00001 \
  --epochs $EPOCHS \
  --dataset $DATA_SET \
  --modality $MODALITY \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --max_query_len 77 \
  --aug_crop \
  --aug_scale \
  --aug_translate \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --output_dir $OUTPUT_DIR

# ==================== EVALUATE ====================
evaluate() {
  local eval_set=$1
  "${DIST_CMD[@]}" \
    --master_port 28501 \
    train_val/fsvg_eval.py \
    --model_name FSVG \
    --model "$CLIPMODEL" \
    --imsize $IMGSIZE \
    --batch_size $BATCHSIZE \
    --num_workers 4 \
    --dataset $DATA_SET \
    --modality $MODALITY \
    --max_query_len 77 \
    --eval_set "$eval_set" \
    --eval_model "$EVAL_MODEL_PATH" \
    --output_dir $OUTPUT_DIR \
    --data_root $DATA_ROOT \
    --split_root $SPLIT_ROOT
}

evaluate "val"
evaluate "test"
evaluate "testA"
evaluate "testB"
evaluate "testC"
