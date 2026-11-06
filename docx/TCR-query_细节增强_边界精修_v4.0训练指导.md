# v4.0：TCR-query、红外细节增强与边界精修训练指南

本版本包含单模型 TCR-query + 红外细节 CNN + 两步边界精修。训练从公共 MMVG mixup 预训练权重初始化，不加载已在 RefFLIR 上微调的 C1/TCR 权重。主要入口：`script_train/clip_v4/train.py`、`script_train/clip_v4/evaluate.py`；结构见 `models/tcr_query_lavs.py`、`models/tir_detail_refine.py` 和 `script_train/clip_full_match/runtime.py`。v4 入口提供显式资源路径，其他历史控制器可能依赖本机旧实验及绝对路径，不应直接视为可移植入口。

## 环境与数据

实验使用 Python、PyTorch 2.2.2、Transformers 4.45.2、PEFT 0.12.0，CUDA 和两张 GPU。数据、公共权重和训练输出不包含在 Git 标签中。按已有项目格式准备：RGB/TIR 图像目录；`split-root/rgbtvg_m3fd/rgbtvg_m3fd_{train,val,test}.pth`（RefFLIR 同理）；CLIP ViT-B/16 目录；Bert 目录；公共 MMVG mixup checkpoint。授权和下载来源以相应数据集及预训练模型提供方为准。

```bash
git clone --branch v4.0 git@github.com:zgc2916/rgb_in_text.git rgb_in_text_v4
cd rgb_in_text_v4
export V4_PYTHON=/data/zhanggongcai/anaconda3/bin/python
export V4_RESOURCES="$PWD/dataset_and_pretrain_model"
export V4_IMAGES="$V4_RESOURCES/datasets/VG/image_data"
export V4_SPLITS="$V4_RESOURCES/datasets/VG/ref_data_shuffled"
export V4_CLIP="$V4_RESOURCES/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16"
export V4_BERT="$V4_RESOURCES/pretrain_model/pretrained_weights/Bert"
export V4_PUBLIC="$V4_RESOURCES/pretrain_model/pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth"
export V4_GPUS=2,3
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 USE_TF=0
```

根据机器修改路径和 GPU 编号。确认两个 GPU 可用，且 `V4_PUBLIC` 是公共预训练权重，不是目标数据集微调权重。

## RefM³FD 训练

以下是当前完整实验的主要配置：120 轮，seed 13，两卡各 batch 12、梯度累积 4（有效 batch 96），AdamW，学习率和 weight decay 1e-4，余弦调度，HiLoRA 阶段、数据增强及原 C1 损失来自 `recipe.json`，附加 TCR、红外区域和边界辅助损失。默认按验证集 Acc@0.5 保存最佳权重。新模型的 TCR、细节、边界模块随机初始化；公共 checkpoint 的可兼容参数和注意力按原实验迁移。

```bash
export V4_OUT="$PWD/output_v4/m3fd_joint_seed13"
mkdir -p "$V4_OUT"
nohup env CUDA_VISIBLE_DEVICES="$V4_GPUS" "$V4_PYTHON" -u -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m script_train.clip_v4.train \
  --dataset rgbtvg_m3fd --data-root "$V4_IMAGES" --split-root "$V4_SPLITS" \
  --clip-path "$V4_CLIP" --bert-path "$V4_BERT" --public-checkpoint "$V4_PUBLIC" \
  --output-dir "$V4_OUT" > "$V4_OUT/console.log" 2>&1 < /dev/null &
echo $!
tail -f "$V4_OUT/console.log"
```

`nohup` 和后台运行允许关闭 VS Code 后继续训练；不要关闭宿主机。`log.txt` 保存逐轮验证指标，`best_checkpoint.pth` 为验证集选出的最佳模型，`checkpoint.pth` 为最近断点。请保留启动日志和运行命令。`--epochs` 默认 120，不应以缩短轮次的结果冒充完整训练。

RefFLIR 将 `--dataset` 改为 `rgbtvg_flir` 并使用新的 `V4_OUT`；RefMFAD 使用 `rgbtvg_mfad`，但本次 v4 联合方案尚无完整 RefMFAD 实验结果。

## 断点续训

使用原来的目录、数据、GPU 数量及训练设置，把 `--public-checkpoint "$V4_PUBLIC"` 替换为 `--resume "$V4_OUT/checkpoint.pth"`。训练器恢复模型、优化器、学习率调度器和阶段；原训练器不保存各 rank 的完整随机数状态，中断恢复不保证逐位复现。不要与首次运行同时启动。

## 验证集和测试集

确认训练结束，先由验证集选择 `best_checkpoint.pth`，再固定该权重评估：

```bash
CUDA_VISIBLE_DEVICES="$V4_GPUS" "$V4_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m script_train.clip_v4.evaluate \
  --checkpoint "$V4_OUT/best_checkpoint.pth" --output-root "$V4_OUT/evaluation" \
  --splits val test --data-root "$V4_IMAGES" --split-root "$V4_SPLITS" \
  --clip-path "$V4_CLIP" --bert-path "$V4_BERT"
```

输出 `val.json`、`test.json`，包含 Acc@0.5、Acc@0.7、mIoU、分组结果和逐表达预测。程序先验证所选权重的验证集指标可复现。历史测试集已经观察过，不属于全新盲测；不要按测试集结果反复选权重。

## 已有实验结果

| 数据集 | 验证 Acc@0.5 | 测试 Acc@0.5 | 测试 Acc@0.7 | 测试 mIoU |
|---|---:|---:|---:|---:|
| RefFLIR | 78.62% | 74.10% | 55.04% | 62.77% |
| RefM³FD | 76.79% | 75.87% | 64.46% | 66.84% |

以上均为 seed13 单模型结果，不保证跨种子稳定。RefM³FD 匹配 TCR-query 对照测试 Acc@0.5 为 76.48%；RefFLIR 匹配对照测试尚未评估。RefM³FD 训练/验证有一张同名图像重叠；测试与训练、验证无同名文件，未排查跨文件名内容重复。
