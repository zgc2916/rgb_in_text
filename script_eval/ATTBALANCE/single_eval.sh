#!/bin/bash

DATA_SET=${DATASET:-rgbtvg_flir}
echo -e "\n\n\n\n\n\n\n==================== AttBalance single eval dataset: $DATA_SET ==========================="

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-8}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}

EVAL_SETS=${EVAL_SETS:-"test testA testB testC val \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY testA testB testC val"}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES TORCH_USE_CUDA_DSA=1 python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-"./output_training/AttBalance_${IMGSIZE}_${MODALITY}/$DATA_SET/best_checkpoint.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output_evaluation/AttBalance_${IMGSIZE}_${MODALITY}/$DATA_SET"}

EVAL_ARGS=( \
  --model_name AttBalance \
  --imsize $IMGSIZE \
  --batch_size $BATCHSIZE \
  --modality $MODALITY \
  --num_workers 1 \
  --bert_enc_num 12 \
  --detr_enc_num 6 \
  --backbone resnet50 \
  --dataset $DATA_SET \
  --max_query_len 20 \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --output_dir $OUTPUT_DIR \
)

evaluate() {
  local eval_set=$1
  echo -e "\n>>>> [AttBalance] Eval set: $eval_set, model: $EVAL_MODEL_PATH"
  "${DIST_CMD[@]}" \
    --master_port 28611 \
    train_val/transvg_eval.py \
    "${EVAL_ARGS[@]}" \
    --eval_set "$eval_set" \
    --eval_model "$EVAL_MODEL_PATH"
}

for es in $EVAL_SETS; do
  evaluate "$es"
done
