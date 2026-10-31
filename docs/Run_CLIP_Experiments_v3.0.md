# v3.0 实验入口与运行说明

适用代码：[rgb_in_text v3.0](https://github.com/zgc2916/rgb_in_text/tree/v3.0)，源码提交 `423c026`。本文说明实际运行过的 CLIP RGB/T 指代表达定位实验，命令在项目根目录执行。数据划分为 RefFLIR：训练 7000、验证 608、测试 2104 条表达。

如果只复现最近一轮实验，准备第 2 节的资源后运行第 3 节即可；如果需要重训 C1 及细节/边界消融，使用第 4 节。两条流程的输出目录和训练预算不同。

## 1. 主要程序入口

| 要做的事 | 入口文件 | 用途 |
| --- | --- | --- |
| 最近一轮三种子后期适配与 EMA | [clip_late_adaptation_control/control.py](../script_train/clip_late_adaptation_control/control.py) | 准备源码快照，后台启动、查看状态、恢复，依次训练三个种子并自动评价 |
| 最近一轮的模型、优化器与选权重规则 | [clip_late_adaptation_control/runtime.py](../script_train/clip_late_adaptation_control/runtime.py) | 定义配方、可训练范围、差分 checkpoint 和验证门槛 |
| 最近一轮的 GPU 预检 | [clip_late_adaptation_control/preflight.py](../script_train/clip_late_adaptation_control/preflight.py) | 复现 C1 验证预测，检查梯度、真实参数更新、冻结范围及 EMA 恢复 |
| 最近一轮的独立评价 | [clip_late_adaptation_control/evaluate.py](../script_train/clip_late_adaptation_control/evaluate.py) | 加载该实验的差分权重，输出验证/测试指标和逐表达预测 |
| C1、红外细节、边界精修、联合消融 | [clip_structure/run_control.py](../script_train/clip_structure/run_control.py) | 完整队列、独立预检、单次训练、恢复和报告 |
| C1 与细节/边界模型的独立评价 | [clip_lavs_control/evaluate.py](../script_train/clip_lavs_control/evaluate.py) | 加载完整模型或受支持的精确压缩模型 |
| 所有实验共享的训练与验证循环 | [engine.py](../engine.py)、[train_val/mmvg_train.py](../train_val/mmvg_train.py) | 梯度累积、损失、验证与训练状态保存 |
| C1 主网络与最新适配封装 | [mmvg_fusion.py](../models/mmvg_fusion.py)、[late_adaptation.py](../models/late_adaptation.py) | 双模态 CLIP、LAVS、IAFv3、VL 解码和后期参数适配 |

复现实验时从控制器入口启动。`train_val/mmvg_train.py` 是底层通用训练程序，直接用默认参数启动不等于运行本次 CLIP 实验。

## 2. 环境、数据与已有基线

### 2.1 本机环境

实际实验使用 `/data/zhanggongcai/anaconda3/bin/python`，关键版本为 PyTorch 2.2.2、Transformers 4.45.2、PEFT 0.12.0。还需要 CUDA 驱动、两张可用 GPU，以及后台控制器使用的 `tmux`。依赖清单见仓库 `requirements.txt`、`environment_full.yml`；复现时应额外核对上述三个实际实验版本。

```bash
export CLIP_PROJECT=/dataset/zhanggongcai/RGBT/RGBT-GroundBench
export CLIP_PYTHON=/data/zhanggongcai/anaconda3/bin/python
export CLIP_GPUS=0,1
cd "$CLIP_PROJECT"

"$CLIP_PYTHON" -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__); print("CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())'
nvidia-smi
tmux -V
```

新机器可先获取固定源码；以下命令中的目录名可自行调整：

```bash
git clone --branch v3.0 https://github.com/zgc2916/rgb_in_text.git rgb_in_text_v3
cd rgb_in_text_v3
export CLIP_PROJECT="$PWD"
```

获取源码后，仍须准备下面的外部资源并调整路径。上面的 `CLIP_PROJECT`、`CLIP_PYTHON` 变量只便于书写命令，不会覆盖控制器代码内的固定路径。

### 2.2 必备资源

源码标签没有附带数据、预训练权重、C1 权重或原实验输出。最近一轮后期适配直接从已训练 C1 开始，所需目录包括：

```text
dataset_and_pretrain_model/
├── datasets/VG/image_data/                       # RGB/TIR 图像，保留原目录结构
├── datasets/VG/ref_data_shuffled/rgbtvg_flir/
│   ├── rgbtvg_flir_train.pth
│   ├── rgbtvg_flir_val.pth
│   └── rgbtvg_flir_test.pth
└── pretrain_model/pretrained_weights/
    ├── CLIP/clip-vit-base-patch16/               # pytorch_model.bin、配置、分词器等
    └── Bert/                                   # 原数据流水线使用的本地资源

output_clip_structure_control/2026-10-05_c1_detail_boundary/
├── formal/c1_seed13/recipe.json                 # prepare 仍会读取原配方
├── frozen/c1_seed13.pth
├── frozen/c1_seed14.pth
├── frozen/c1_seed15.pth
├── evaluation/c1_seed13/val.json                # 预检所需的逐表达验证预测
├── evaluation/c1_seed14/val.json
├── evaluation/c1_seed15/val.json
├── selected_models.json                        # 原模型登记信息
└── preregistration.json                        # 部分早期控制器还会读取此文件
```

仓库 `run_resources/c1_recipe.json`、`run_resources/c1_references.json` 提供配方与三个种子的登记指标、权重路径和 SHA256，不能替代这些权重及原验证预测。

### 2.3 迁移到其他路径

| 位置 | 需要核对的内容 |
| --- | --- |
| `script_train/clip_lavs_control/runtime.py` | `ORIGINAL`、`REFERENCE`、`PYTHON` |
| `script_train/clip_soft_control/runtime.py` | 同样有独立的 `ORIGINAL`、`REFERENCE`、`PYTHON`，可靠性及后续控制器会间接引用它 |
| `run_resources/c1_recipe.json` 和原 `formal/c1_seed13/recipe.json` | `data_root`、`split_root`、`pretrained_model_path`、`dataset_bert_model_path`、`retrain` 等资源路径 |
| `run_resources/c1_references.json`、原 `selected_models.json` | 三个 C1 checkpoint 的位置及其对应校验值和验证结果 |
| 原 checkpoint 的 `args` | 部分独立训练/评价入口继承其中的绝对资源路径，不能只修改命令里的 `--root` |

当前接口没有通用的 `--data-root`、`--epochs` 或 `--lr` 覆盖所有控制器。改变训练预算、学习率或资源配置，应在 `prepare` 前修改相应 `runtime.py`/配置，然后使用新的实验目录。完成登记后的 `source/`、配方和资源有校验，修改后不能继续当作同一次实验恢复。

若自行重训 C1 得到新的权重，需要重新登记其校验值和验证预测；它构成新的基线，不能冒用历史 C1 的校验值和指标。

## 3. 复现最近一轮：三种子后期适配与 EMA

### 3.1 本轮实际设置

| 项目 | 设置 |
| --- | --- |
| 起点 | C1 seed 13/14/15 各自对应的原始 checkpoint |
| 预算 | 三个种子依次运行，各 30 epoch，共 90 epoch |
| 两卡有效 batch | 每卡 4 × 累积 12 × 2 卡 = 96 |
| 输入与目标损失 | 224，目标保留裁剪；框损失 `2 L1 + 2 GIoU` |
| 更新范围 | 视觉 LoRA 第 8–11 层、融合与投影、VL 第 3–5 层及框头等；精确范围见 `models/late_adaptation.py` |
| 学习率 | 视觉 LoRA `4e-6`；融合/解码组 `2e-6` |
| 参数平均 | EMA，目标 decay 0.99，按有效 optimizer step 更新，带初期 decay 调整 |
| 模型选择 | 每轮分别验证 raw/EMA；同轮 Acc05 相同选 raw；全程取首次严格最高验证 Acc05，包含初始 C1 |
| 推理 | 每次只加载一个验证集选定的参数版本 |

### 3.2 准备并后台启动

下面使用新的绝对输出路径。不要先手动创建该输出目录，`prepare` 要求它不存在。

```bash
export CLIP_RUN_ROOT="$CLIP_PROJECT/output_clip_lavs_control/manual_late_adaptation_v3_run01"

"$CLIP_PYTHON" -m script_train.clip_late_adaptation_control.control prepare \
  --root "$CLIP_RUN_ROOT"

"$CLIP_PYTHON" -m script_train.clip_late_adaptation_control.control launch \
  --root "$CLIP_RUN_ROOT" --gpus "$CLIP_GPUS"
```

`prepare` 检查资源、复制源码到 `source/`，保存 `preregistration.json` 和源码/资源校验清单，不启动训练。`launch` 使用独立 tmux 会话启动控制器；可以关闭启动它的终端或断开 SSH。

也可以省略显式 `prepare`，对不存在的新目录直接执行 `launch`，控制器会先准备再启动。GPU 接口也支持 `--gpus auto`；当前启动检查要求两张不同的卡，每卡已用显存不超过 2048 MB、空闲显存至少 18000 MB。这是控制器的启动条件，不等于模型实际峰值显存。

后台依次执行：

1. 三个种子的双卡 GPU 预检；预检更新丢弃。
2. `late_adaptation_seed13`、`seed14`、`seed15` 各训练 30 轮。
3. 汇总验证结果，将三个选定 checkpoint 的路径、轮次、raw/EMA 和 SHA256 一起写入 `test_registration.json`。
4. 依次复现三个模型的验证结果并评价测试集，生成 `report.json`。

本控制器在三个模型全部选定后自动测试，即使验证提升门槛没有通过也会报告预先约定的三个结果。它不会依据测试结果修改该次选权重。

### 3.3 查看状态与日志

```bash
"$CLIP_PYTHON" -m script_train.clip_late_adaptation_control.control status \
  --root "$CLIP_RUN_ROOT"

tail -n 40 "$CLIP_RUN_ROOT/controller.log"
tail -n 5 "$CLIP_RUN_ROOT/explore/late_adaptation_seed13/log.txt"
```

种子 14/15 的日志在各自任务启动后出现。`status` 的 `live_processes` 会核对进程身份；`phase`、`current_job`、`completed_epochs` 和 `updated_at` 用于判断是否仍有进展。

### 3.4 中断后恢复

```bash
"$CLIP_PYTHON" -m script_train.clip_late_adaptation_control.control resume \
  --root "$CLIP_RUN_ROOT" --gpus "$CLIP_GPUS"
```

恢复读取 `checkpoint.pth` 中的模型差分、optimizer、scheduler、EMA、两卡 RNG 和选优记录。`best_checkpoint.pth` 用于评价，不包含完整续训状态。`resume` 默认沿用该目录记录的 GPU；若原任务仍存活，控制器会拒绝重复启动。已经 `complete` 的目录不能用来重新训练，应创建新目录。

### 3.5 结果文件怎么看

```text
实验输出目录/
├── status.json / controller.log
├── preregistration.json / source_manifest.json / resource_manifest.json
├── source/                                     # 真正运行的冻结源码
├── preflight.json / preflight/console.log
├── explore/late_adaptation_seed13/              # seed14/15 同结构
│   ├── recipe.json / initialization.json
│   ├── initial_validation.json
│   ├── log.txt / console.log / progress.json
│   ├── best.json / best_checkpoint.pth
│   ├── trained_best.json / trained_best_checkpoint.pth
│   └── checkpoint.pth
├── test_registration.json
├── evaluation/late_adaptation_seed13/            # seed14/15 同结构
│   ├── val.json
│   └── test.json
└── report.json
```

`log.txt` 每行是一轮 JSON，`raw_val_*`、`ema_val_*` 分别记录两种参数的指标，`val_*` 与该轮 `weight_view` 对应。`best.json` 是包含初始 C1 的正式选择，`trained_best.json` 只在已训练轮次中选择。轮次从 0 开始，`epoch=0` 表示第 1 轮，`epoch=-1` 表示初始 C1。

`report.json` 中重点看 `promotion`、`test`、`mean_test_acc05` 和 `mean_test_target_reached`；`phase=complete` 仅表示流程完成。评价文件使用 `summary.acc_05`，训练日志使用 `val_accu`，保存的数值为 0–1，显示百分比时乘 100。

### 3.6 只评价已选权重

训练已完成时，可以用冻结源码重新输出到一个新的评价目录。差分 checkpoint 仍依赖原 C1 权重，不能单独搬走使用。

```bash
(
  cd "$CLIP_RUN_ROOT/source"
  CUDA_VISIBLE_DEVICES="$CLIP_GPUS" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "$CLIP_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m script_train.clip_late_adaptation_control.evaluate \
    --checkpoint "$CLIP_RUN_ROOT/explore/late_adaptation_seed13/best_checkpoint.pth" \
    --output-root "$CLIP_RUN_ROOT/manual_evaluation/late_adaptation_seed13" \
    --splits val test
)
```

只复现验证集时改为 `--splits val`。种子 14/15 使用各自的 checkpoint 和输出目录。此评价器会核对原 C1 校验值，并在评价验证集时检查选定权重能否复现记录指标。

### 3.7 我们已经跑完的目录

```bash
"$CLIP_PYTHON" -m script_train.clip_late_adaptation_control.control status \
  --root "$CLIP_PROJECT/output_clip_lavs_control/2026-10-28_late_adaptation_ema_r1"
```

该目录三种子共 90 轮已经完成。验证选定位置是 seed13：epoch 0/raw；seed14：epoch 1/raw；seed15：epoch 23/EMA。测试 Acc05 分别为 72.719%、73.717%、73.241%，平均 73.226%，未达到 74%。

IDE 中打开的 `output_clip_structure_control/2026-10-05_c1_detail_boundary/status.json` 属于下面的 C1/结构消融队列；最近一轮进度应读取上面的 `late_adaptation_ema_r1/status.json`。

## 4. 重训 C1 与红外细节/边界消融

### 4.1 初始化来源和训练预算

入口是 `script_train.clip_structure.run_control`，固定参考来源为：

```bash
export CLIP_C1_SOURCE="$CLIP_PROJECT/output_clip_geometry_test/2026-10-05_seed13_safe_crop_epoch78/selected_model_only.pth"
export CLIP_STRUCTURE_ROOT="$CLIP_PROJECT/output_clip_structure_control/manual_c1_detail_boundary_v3_run01"
```

这个来源 checkpoint 提供实验配置，并为探索阶段提供 C1 参数；正式训练从其 `args.retrain` 指向的公共 MMVG 初始化开始。因而还需要该参考 checkpoint 和公共初始化权重，不能把正式训练理解为继续微调该参考模型。

| 阶段 | 方案 | 种子 | 轮数 |
| --- | --- | --- | ---: |
| 冻结 C1 的探索 | `detail`、`boundary`、`joint` | 13 | 各 20 |
| 正式训练 | `c1`、`detail`、`boundary`、`joint` | 13 | 各 120 |
| 多种子正式训练 | `c1`、`joint` | 14、15 | 各 120 |

共 11 次训练，其中 8 次正式训练。正式训练每卡 batch12、累积4、两卡有效 batch96；AdamW，lr/weight decay 为 `1e-4`。`c1` 关闭两个新增模块，`detail` 只开红外细节，`boundary` 只开边界精修，`joint` 同时开启。

### 4.2 完整队列

该早期控制器在前台运行，使用 tmux 保持断开终端后继续运行：

```bash
tmux new -s clip_structure_v3
```

在新会话中设置第 2.1 节的变量以及第 4.1 节两个路径，再执行：

```bash
"$CLIP_PYTHON" -m script_train.clip_structure.run_control suite \
  --source "$CLIP_C1_SOURCE" \
  --output-root "$CLIP_STRUCTURE_ROOT" --gpus "$CLIP_GPUS"
```

按 `Ctrl+b` 后松开，再按 `d` 可离开会话。队列自动预检、训练、冻结八个正式模型、评价 val/test 并生成 `report.json`、`report.md`；选轮不使用测试结果。中断恢复使用同一命令加 `--resume`，报告可用以下命令重新生成：

```bash
"$CLIP_PYTHON" -m script_train.clip_structure.run_control report \
  --output-root "$CLIP_STRUCTURE_ROOT"
```

### 4.3 只训练一次 C1 或某项消融

这是完整队列之外的独立运行方式；请使用新的输出目录，不要复用已经完成整队训练的目录。

```bash
CUDA_VISIBLE_DEVICES="$CLIP_GPUS" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  "$CLIP_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m script_train.clip_structure.run_control train \
  --source "$CLIP_C1_SOURCE" --output-root "$CLIP_STRUCTURE_ROOT" \
  --run-dir "$CLIP_STRUCTURE_ROOT/formal/c1_seed13" \
  --stage formal --variant c1 --seed 13
```

要训练其他方案，同时修改 `--variant` 和 `--run-dir`；种子同理。`--stage explore` 是 20 轮冻结探索，`formal` 是 120 轮正式训练。手动 `train` 不自动运行整个队列的预检或测试；可先用相同 torchrun 前缀启动 `preflight --source ... --output-root ...`。完整操作细节见 [C1 训练教学](C1_红外细节与边界定位_训练指令教学.md)。

评价已有 C1/结构权重使用下面的入口；不要用它加载后期适配的 `model_delta` 文件：

```bash
CUDA_VISIBLE_DEVICES="$CLIP_GPUS" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  "$CLIP_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m script_train.clip_lavs_control.evaluate \
  --checkpoint "$CLIP_PROJECT/output_clip_structure_control/2026-10-05_c1_detail_boundary/frozen/c1_seed13.pth" \
  --output-root "$CLIP_PROJECT/output_clip_analysis/manual_c1_seed13_v3_run01" \
  --splits val test
```

## 5. 其他已运行的优化实验

这些入口分别对应独立对照实验，不需要依次全部运行。它们复用 C1 资源，但改变的训练范围、预算、自动测试行为不同。

| 实验 | 模块入口（在 `python -m` 后） | 原输出目录（`output_clip_lavs_control/` 下） | 方案与初始预算 |
| --- | --- | --- | --- |
| 局部与文本局部交换 | `script_train.clip_lavs_control.control` | `2026-10-09_text_local_exchange` | `local`、`text_local`，初始各 120 轮 |
| 全局门控与软空间交换 | `script_train.clip_soft_control.control` | `2026-10-13_soft_spatial_exchange` | `global_gate`、`text_global_gate`、`text_soft_spatial`，初始各 120 轮 |
| 可靠性交换 | `script_train.clip_reliability_control.control` | `2026-10-18_reliable_exchange_warm` | `direction_gate`、`reliability_gate`、`reliability_spatial`，各 20 轮 |
| 增强、辅助损失及联合训练控制 | `script_train.clip_training_control.control` | `2026-10-18_clean_joint_control` | `a_clean_full`、`b_clean_box`、`c_joint_box`、`d_c1_continue`，各 20 轮 |
| 区域锚点 | `script_train.clip_anchor_control.control` | `2026-10-26_region_anchor` | `c1_continue`、`anchor_visual`、`anchor_text_shared`、`anchor_text_separate`，各 20 轮 |
| 原生分辨率再观察 | `script_train.clip_reobserve_control.control` | `2026-10-27_native_reobserve` | `c1_continue`、`fixed_local`、`text_adaptive`、`text_consistent`，各 20 轮 |
| 空间质量与局部收益估计 | `script_train.clip_spatial_quality_control.control` | `2026-10-27_spatial_quality` | `c1_continue`、`spatial_uniform`、`fixed_benefit`、`adaptive_benefit`，各 20 轮 |
| 尺度保护与相对几何 | `script_train.clip_scale_geometry_control.control` | `2026-10-28_scale_geometry` | `c1_continue`、`scale_protected`、`relative_geometry`、`scale_relative`，各 30 轮 |

前三类早期控制器及训练控制器可能在门槛通过后继续复现实验；完整预算与评价策略以相应 `preregistration.json` 为准。区域锚点、原生再观察、空间质量、尺度几何这四个后期筛选控制器只运行验证筛选，不自动测试。

例如启动尺度/几何四组对照，以下命令会自动准备快照并后台启动：

```bash
"$CLIP_PYTHON" -m script_train.clip_scale_geometry_control.control launch \
  --root "$CLIP_PROJECT/output_clip_lavs_control/manual_scale_geometry_v3_run01" \
  --gpus "$CLIP_GPUS"
```

替换表中的模块入口及新的输出目录，可运行对应实验。各入口支持 `--help`、`launch`、`status`、`resume`；后四类还支持显式 `prepare`。修改 `--variant` 不是限制 `launch` 队列只运行一种方案；整队方案由控制器登记，单次训练则使用其 `train` 子命令。

空间质量实验完成后，我们另行登记了 C1 与四个验证选定模型的固定测试计划。相应独立入口为：

```bash
"$CLIP_PYTHON" -m script_eval.clip_spatial_quality_test --help
"$CLIP_PYTHON" -m script_eval.clip_spatial_quality_test_report --help
```

该评价器有 `prepare`、`run` 两个动作，`run` 自行启动双卡子进程；检查原输出、登记和 GPU 设置后使用，不要套用后期适配的差分评价器。

## 6. 常见问题与检查方法

| 现象 | 检查/处理 |
| --- | --- |
| `prepare` 报已有目录 | 新实验换新目录；中断实验使用 `resume` |
| 找不到 recipe、val.json 或 checkpoint | 按第 2 节补齐原 C1 资源；仅有源码与 `run_resources` 不够 |
| `waiting_for_gpu` / `waiting_for_gpus` | 查看 `waiting_reason`，核对 GPU 启动条件和其他任务占用 |
| `Frozen source changed` / `Runtime resource changed` | 核对登记源码及资源；新配置使用新实验目录 |
| 选定验证结果无法复现 | 核对 torch/Transformers/PEFT、权重 SHA、数据与预处理；不同环境可能影响阈值附近样本 |
| 指标看起来与另一份状态文件不同 | 先核对实验根目录、种子、epoch 和 raw/EMA，再比较同 checkpoint 指标 |
| 已完成实验不能续训 | 部分旧目录已清理 optimizer/RNG 状态，只保留评价权重；这类文件不能精确恢复训练 |
| 差分模型缺少 base checkpoint | 保留对应 C1 原权重，或按 `checkpoint_relocations.json` 找到清理后的映射；不要直接删除依赖基座 |

源码测试不会启动训练：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  "$CLIP_PYTHON" -m pytest -q tests
```

发布前在独立目录中验证为 192 项通过、2 项跳过；跳过的是依赖本地 `InfMAE/InfMAE.pth` 的两项测试。最新 CLIP 后期适配本身不依赖这个 InfMAE 权重。本文中的 CLI 入口已逐项检查 `--help`，没有为编写说明重新启动训练。
