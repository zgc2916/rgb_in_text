# CLIP 区域锚点交互：2026-10-26 实施记录

## 当前状态

2026-10-27续记：四组各20轮、共80轮已经正常完成，验证主指标未超过原C1，状态为 `gate_not_passed`。已清理重复最佳模型副本，并将末轮可恢复状态精简为 `compact_last/*_epoch19.pth`；实际训练后的最佳权重 `trained_best_checkpoint.pth` 保留。历史启动过程见下文，最新诊断及后续方案见 [2026-10-27优化记录](CLIP_NEXT_OPTIMIZATION_2026-10-27.md)。

已实现区域锚点残差交互和四组受控续训入口。79 项相关测试通过；四组各通过 8 条验证样本的 CPU 前向检查、真实训练样本的两次优化更新，以及选择性权重、优化器和调度器恢复检查。

初次检查在受限沙箱内进行，沙箱未暴露 GPU 设备节点，导致 nvidia-smi 失败、PyTorch 检测到 0 张 CUDA 设备。这不代表宿主机显卡或驱动故障。随后在沙箱外复核，确认 8 张 RTX 4090 正常、PyTorch CUDA 运算成功，已在 GPU0、GPU1 启动独立后台控制器和双 GPU 预检。完整训练进度以 status.json 为准，尚无新精度结论。CPU 检查不替代双 GPU 预检或完整训练。

实验目录：`output_clip_lavs_control/2026-10-26_region_anchor/`。以其中 `status.json` 为当前事实。初次 `waiting_for_gpu` 已通过沙箱外启动解决；沙箱外的进程存活检查也已确认控制器和双卡预检进程运行中。

## 研究范围

参考 [RaAM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ouyang_Region-aware_Anchoring_Mechanism_for_Efficient_Referring_Visual_Grounding_ICCV_2025_paper.html)，实现文本、RGB、TIR 三路区域交互的残差适配。这是待验证的改编，不是原论文复现。

本轮保留原外部 LAVS 全局交互，因此不能称为“完整替换 LAVS”。内部文本引导、IAFv3、六层 VL Transformer 和框预测头保持原结构。先验证区域证据交互的增量收益，再决定是否值得进行完整替换实验。

- 16 个锚点对应 4×4 区域，读取完整 CLIP 文本 token。
- 通过软空间先验分别读取 RGB、TIR patch；不做硬候选截断。
- 比较平均区域记忆与两路独立区域记忆；独立模式向 RGB 写入 TIR 证据，向 TIR 写入 RGB 证据。
- 四个视觉深度共享读取器，通过层嵌入区分语义深度。
- 输出投影零初始化，初始函数与同设备 C1 一致。
- 新分支不改变 CLS，不读取 GT，并屏蔽文本和视觉 padding。
- 新增 466,688 个参数；三个新结构变体的参数数量和参数结构完全一致。

## 受控实验

| 变体 | 文本条件 | 区域记忆 | 新参数 |
|---|---|---|---:|
| `c1_continue` | 沿用 C1 | 沿用 C1 | 0 |
| `anchor_visual` | 新分支不使用文本 | RGB/TIR 平均记忆 | 466,688 |
| `anchor_text_shared` | 完整文本 token | RGB/TIR 平均记忆 | 466,688 |
| `anchor_text_separate` | 完整文本 token | RGB/TIR 独立记忆 | 466,688 |

四组均从正式 C1 seed13、epoch100 开始，使用相同目标保留增强、2×L1+2×GIoU、有效 batch96 和 20 轮预算。原外部 Q/K/V/O 投影学习率为 1e-5，新模块为 1e-4；其余参数冻结，固定主干保持 eval 行为。无合成退化，无新增辅助损失。最大四组、共80轮，不自动扩展 seed 或测试集。

这是训练后 C1 的受控续训筛选，不能代替公共初始化下的正式重训。新结构必须同时超过原 C1 和匹配续训对照，才能支持额外贡献。

## 验证结果及边界

测试命令：

```bash
python -m pytest -q tests/test_region_anchor_exchange.py tests/test_clip_training_control.py tests/test_clip_reliability_control.py tests/test_clip_lavs_control.py tests/test_clip_soft_control.py tests/test_soft_spatial_exchange.py tests/test_reliable_modal_exchange.py tests/test_local_modal_exchange.py
```

结果为 79 passed。覆盖零初始化、文本影响、共享/独立记忆差异、padding/空输入、两路及文本梯度、PEFT 切换后的冻结范围、优化器覆盖、选择性权重恢复、选模提交中断恢复、GPU 缺失状态及 GPU 占用检查。

CPU 实机检查的三个新变体相对同设备 C1 最大初始框误差均为 0。相对归档 GPU 输出，CPU C1 最大框坐标差为约 0.000557；报告单独记录这项跨设备差异，不把它冒充结构差异。双 GPU 预检仍要求完整 608 条验证结果与历史 C1 对齐，并检查真实有效 batch96 的两次更新、各 rank RNG 恢复及显存峰值。

CPU 检查只使用少量样本验证可运行性，没有计算或声明新模型的验证集、测试集精度。检查过程中产生的临时探针权重已经删除。

## 训练和恢复

在项目根目录使用当前 Python 环境运行：

```bash
python -m script_train.clip_anchor_control.control launch
```

自动选择两张使用显存不超过 2 GiB、空闲显存不少于 18000 MB 的 GPU；也可明确指定：

```bash
python -m script_train.clip_anchor_control.control launch --gpus 6,7
```

GPU 不可访问时记录 `waiting_for_gpu` 并以退出码2返回，不启动训练或等待进程。先在宿主机终端确认 nvidia-smi 正常；若执行工具的沙箱屏蔽了设备节点，应在允许访问 GPU 的执行环境运行 launch。成功启动后由独立 tmux 会话运行冻结源码，控制器不依赖 VS Code 或 SSH 会话；本次已确认沙箱外启动命令退出后，后台控制器和双卡预检进程仍在运行。

```bash
python -m script_train.clip_anchor_control.control status
python -m script_train.clip_anchor_control.control resume
```

恢复会核对源码、资源、配方 SHA256 和活跃进程，恢复选择性权重、优化器、调度器及每个 rank 的 RNG 状态。拒绝重复启动。运行中每10秒更新状态；任务失败后保留日志，由 resume 显式恢复，不无限重试。

## 权重和选模

- `best_checkpoint.pth`：首次严格达到验证 Acc@0.5 最大值的权重，包含初始 C1 保底。
- `trained_best_checkpoint.pth`：实际训练后首次达到最佳验证 Acc@0.5 的权重，独立保存，避免再次出现最佳训练轮次无权重可测的问题。
- `checkpoint.pth`：末轮及可恢复训练状态。
- 所有其他指标来自同一轮，测试集不参与筛选。
- 初筛要求相对 C1 至少净增4个验证命中，相对匹配续训至少净增2个命中；Acc@0.7、mIoU、弱光 mIoU 均不下降。通过也只代表值得复验，不能直接宣称稳定提升。

修改前源码备份：`output_code_snapshots/clip_anchor_before_2026-10-26/`。未覆盖已有实验、权重或用户暂存区。
