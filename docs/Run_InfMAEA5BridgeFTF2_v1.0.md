# InfMAEA5BridgeFTF2 v1.0：运行教学

本说明对应 Git 标签 **`v1.0`**，用于在 RefFLIR (`rgbtvg_flir`) 上复现
`MMVGFusion + InfMAEA5BridgeFTF2` 的两卡训练与验证。

该版本的 TIR 分支是官方 InfMAE，而不是 TIR CLIP：InfMAE 原始权重冻结，
仅在 F2 细节层和 F3 后三层加入 LoRA；RGB 分支仍使用 CLIP ViT-B/16 与
RGB LoRA。模型、数据和预训练权重均不随 Git 仓库发布。

## 1. 获取 v1.0 源码

```bash
git clone git@github.com:zgc2916/rgb_in_text.git RGBT-GroundBench
cd RGBT-GroundBench
git checkout v1.0
```

后续命令均假设当前工作目录为仓库根目录。

## 2. 创建环境

推荐使用仓库的 Conda 环境定义：

```bash
conda env create -f environment_full.yml
conda activate rgbt
```

已存在兼容 PyTorch/CUDA 环境时，也可安装最小依赖：

```bash
pip install -r requirements.txt
```

本版本已在 Python 3.9、PyTorch 2.2.2、Transformers 4.30.0、PEFT 0.11.1
的组合上验证。确认 GPU 可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 3. 准备外部资源

这些目录被 `.gitignore` 排除，必须由使用者自行准备。令资源根目录为
`RGBT_ASSET_ROOT`，其结构应为：

```text
dataset_and_pretrain_model/
├── datasets/VG/
│   ├── image_data/
│   └── ref_data_shuffled/
└── pretrain_model/pretrained_weights/
    ├── CLIP/clip-vit-base-patch16/
    └── MMVG/mixup_pretraining_base/mixup/
        └── merged_fixed_best_checkpoint_peft0111.pth
```

默认代码布局使用仓库相邻目录 `../dataset_and_pretrain_model`。例如：

```bash
export RGBT_PROJECT_ROOT="$PWD"
export RGBT_ASSET_ROOT="$(cd "$RGBT_PROJECT_ROOT/.." && pwd)/dataset_and_pretrain_model"

test -d "$RGBT_ASSET_ROOT/datasets/VG/image_data"
test -d "$RGBT_ASSET_ROOT/datasets/VG/ref_data_shuffled"
test -d "$RGBT_ASSET_ROOT/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16"
test -f "$RGBT_ASSET_ROOT/pretrain_model/pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth"
```

还需要官方 InfMAE 源码和权重：

```bash
git clone https://github.com/liufangcen/InfMAE.git InfMAE
git -C InfMAE apply ../patches/infmae_runtime_compatibility.patch
# 将官方 InfMAE.pth 放到：InfMAE/InfMAE.pth
test -f "$RGBT_PROJECT_ROOT/InfMAE/InfMAE.pth"
```

`InfMAE/InfMAE.pth`、数据集、CLIP 权重、MMVG 初始化权重及训练输出都不应
加入 Git，也不应推送到仓库。

## 4. 可选：先做源码自检

```bash
python -m py_compile \
  models/tsar_modules.py models/mmvg_fusion.py utils/loss_utils.py \
  tests/test_infmae_modules.py tests/test_tsar_modules.py

pytest -q \
  tests/test_infmae_modules.py \
  tests/test_tsar_modules.py \
  tests/test_iafv3_equivalence.py
```

## 5. 两卡训练 RefFLIR

下面命令使用两张可见 GPU。将 `CUDA_VISIBLE_DEVICES=0,1` 改为机器实际可用的
物理卡号；每卡 batch size 为 6、梯度累积为 8，因此有效全局 batch size 为
`2 × 6 × 8 = 96`。

```bash
export RGBT_RUN_DIR="$RGBT_PROJECT_ROOT/output_infmae_a5_f2full/InfMAEA5BridgeFTF2_224/rgbtvg_flir/seed_13_full120"
mkdir -p "$RGBT_RUN_DIR"

CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false \
python -m torch.distributed.launch \
  --nproc_per_node=2 --use_env --master_port=29747 \
  train_val/mmvg_train.py \
  --model_name MMVGFusion \
  --FusionMethod InfMAEA5BridgeFTF2 \
  --modality rgbt \
  --model ViT-B/16 \
  --vl_backbone clip \
  --open_lora True \
  --open_text_guided_fusion True \
  --lavs_mode lavs \
  --lora_r_rgb 16 \
  --lora_r_ir 48 \
  --imsize 224 \
  --max_query_len 77 \
  --vl_hidden_dim 512 \
  --vl_enc_layers 6 \
  --vl_dec_layers 6 \
  --normalize_before \
  --mixup_pretrain \
  --dataset rgbtvg_flir \
  --data_root "$RGBT_ASSET_ROOT/datasets/VG/image_data" \
  --split_root "$RGBT_ASSET_ROOT/datasets/VG/ref_data_shuffled" \
  --pretrained_model_path "$RGBT_ASSET_ROOT/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16" \
  --retrain "$RGBT_ASSET_ROOT/pretrain_model/pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth" \
  --epochs 120 \
  --batch_size 6 \
  --grad_accum_steps 8 \
  --num_workers 4 \
  --lr 0.0001 \
  --infmae_alignment_lr 0.0001 \
  --lr_scheduler cosine \
  --clip_max_norm 1.0 \
  --aug_crop --aug_scale --aug_translate \
  --contrastive_loss clip \
  --image_norm dataset \
  --use_contrastive_loss \
  --use_rtcc_constrain_loss \
  --use_mask_loss \
  --infmae_alignment_mode both \
  --infmae_tir_text_weight 0.05 \
  --infmae_rgb_tir_weight 0.02 \
  --infmae_tir_text_tau 0.07 \
  --infmae_alignment_start_epoch 40 \
  --infmae_alignment_ramp_epochs 10 \
  --infmae_alignment_adapter_start_epoch 40 \
  --infmae_alignment_adapter_ramp_epochs 10 \
  --infmae_rgb_tir_start_epoch 50 \
  --infmae_rgb_tir_ramp_epochs 10 \
  --report_acc07 \
  --seed 13 \
  --output_dir "$RGBT_RUN_DIR"
```

### 训练阶段说明

| Epoch | 生效目标 |
| --- | --- |
| 0–39 | 原 grounding 目标；F2/F3 LoRA 与 A5BridgeFT 联合训练 |
| 40–49 | TIR–Text 目标区域 InfoNCE 由 0 线性升至 0.05；语义桥同步打开 |
| 50–59 | RGB→TIR 目标级余弦迁移由 0 线性升至 0.02 |
| 60–119 | 两个 InfMAE 语义目标均以完整权重参与训练 |

`lora_r_ir=48` 为统一命令接口保留参数；在 direct-InfMAE 路径中，TIR CLIP
不会被调用，真正生效的 TIR 可训练项是 F2/F3 InfMAE LoRA 与热成像适配器。

## 6. 查看训练与恢复训练

每个 epoch 都会写入结构化日志，`best_checkpoint.pth` 按验证集 Acc@0.5 的最佳值
保存，`checkpoint.pth` 为最新轮次：

```bash
tail -n 3 "$RGBT_RUN_DIR/log.txt"
tail -f "$RGBT_RUN_DIR/stdout.log"       # 仅在将 stdout 重定向到该文件时使用
ls -lh "$RGBT_RUN_DIR"/*.pth
```

若训练中断，使用**完全相同的模型和数据参数**，并将初始化参数替换为：

```bash
--resume "$RGBT_RUN_DIR/checkpoint.pth"
```

恢复时不要再将其他实验（例如 GQR、A7、A6 或旧 A5Bridge）的 checkpoint 传给
`--resume`，因为这些模型的参数结构不同。

## 7. 仅验证最佳权重

复制第 5 节命令，删除 `--retrain ...`，并替换训练相关末尾参数为：

```bash
--eval \
--resume "$RGBT_RUN_DIR/best_checkpoint.pth" \
--output_dir "$RGBT_RUN_DIR/eval_best"
```

模型构造参数（特别是 `--FusionMethod InfMAEA5BridgeFTF2`、`--open_lora True`、
`--infmae_alignment_mode both`、输入尺寸和 CLIP 路径）必须保持不变，否则
checkpoint 的状态字典无法严格加载。

## 8. 结果解释与常见问题

- RefFLIR 验证集有 608 个样本，因此 Acc@0.5 每变化一个样本约为 0.1645 个百分点。
  单个 epoch 的小波动不应直接视为结构改进。
- 本版本的已完成单种子运行峰值为 epoch 94 的 **75.16% Acc@0.5**、
  **54.11% Acc@0.7**；该结果仅用于核对配置，发布的 `v1.0` 标签不包含对应权重。
- 若报 `InfMAE checkpoint was not found`，检查 `InfMAE/InfMAE.pth` 是否存在，且必须
  从仓库根目录启动命令。
- 若报 CLIP 或数据文件找不到，优先检查 `RGBT_ASSET_ROOT` 的目录结构；不要把大文件
  复制进源码仓库。
- 若两卡 NCCL 端口冲突，将 `--master_port=29747` 改为当前机器未占用的端口。

## 9. v1.0 的范围

v1.0 固定的是源码、模型配置和训练策略；不发布数据、InfMAE/CLIP/MMVG 权重或训练
产物。后续结构实验请从新的 Git 分支开始，避免改写此标签所对应的可复现版本。
