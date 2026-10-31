# InfMAE 前端优化路线：与 RGBT-VGNet 主干解耦

> **2026-10-04 21:31 差距解释更新：** 验证77.961%并不等于测试泛化突破。新的只读统计发现，test的小目标比例更低且中位目标更大，四个光照×尺寸组内准确率却均低于val；同样8–16像素目标，Acc@0.5从val80.612%降为test62.309%。图像ID无交集，已核对标注/IoU；尚不能归因于简单计分错误或粗分组占比。验证选峰偏差、全部12层LoRA提前开放、训练裁剪噪声等应作为待验证假设，后续优先复核seed/原配方与弱光小目标Acc@0.7保持，采用独立的train/val对照。报告：`output_clip_control_test/2026-10-04_seed13_b12_epoch111/generalization_review.json`，本轮无新训练/前向。

> **2026-10-04 17:44 最新测试结果：** 按用户明确要求，固定CLIP batch12 epoch111，在seed14复核前完成一次完整test2104评价：72.196% Acc@0.5、53.612% Acc@0.7、60.911% mIoU。相对batch6仅Acc@0.5提高0.523个百分点，高IoU准确率持平、mIoU下降0.100个百分点，仍低论文0.234/0.238个百分点。弱光/小目标的测试高IoU命中分别减少10/14个，不能把验证77.961%当作测试整体突破。冻结权重与验证口径已复核，单seed结论仍待重复；报告：`output_clip_control_test/2026-10-04_seed13_b12_epoch111/summary.json`。本轮无新训练，下方“尚无测试”为对应时点的历史记录。

> **2026-10-04 16:14 最新验证结果：** CLIP batch12/累积4的120轮已于15:30完成，保存epoch111验证77.961% Acc@0.5、56.250% Acc@0.7、65.793% mIoU，比batch6已保存epoch95提高1.809/0.822/1.986个百分点。实际checkpoint元数据与该轮日志匹配。弱光/小目标的Acc@0.7分别减少7/4个命中，整体提升尚不能替代分组和测试验证；当前只有seed13，batch12测试前向为0。后续先复核seed14，再冻结最终选择进行测试。完整记录：`output_clip_batch_control/2026-10-04_seed13_b12_a4/validation_review.json`。下方12:13的运行状态为历史记录。

> **2026-10-04 12:13 当前实验：** 用户要求解决基础CLIP本卡候选从12降到6的问题，已在GPU4/6启动120轮seed13的batch12/累积4对照，有效batch96。初始化、全部12层LoRA从epoch0开放及增强沿用已完成CLIP batch6/累积8对照；真实训练启动检查确认两卡均为12×12基础CLIP矩阵，已完成首轮验证。新入口显式记录本卡候选与有效batch，训练/选轮仅用train/val；该配置恢复不等于已证明精度提升，也未恢复论文的分阶段LoRA。当前状态见`output_clip_batch_control/2026-10-04_seed13_b12_a4/status.json`；作者权重身份与原环境复验仍待完成。

> **2026-10-04 复现优先级更新：** 六个本地RefFLIR标注与作者当前发布包逐字节一致，原IoU公式重算也一致，但本地旧MMVG发布权重缺IAFv3参数，作者当前正确候选权重尚未复验。本地CLIP训练改变了LoRA阶段和本卡对比候选数（公开脚本默认12、本地6），论文正文与公开脚本的batch/lr也有差别。下一步先核对作者权重身份并打通原环境与本地评价，再复现明确配方和seed，随后开展共同增强及InfMAE对照；不把当前缺口直接归因于测试更难或某一模块。细节见会话末节及`output_infmae_route_review/2026-10-04_paper_gap_audit/summary.json`。

> **2026-10-04 最新测试证据：** 用户要求测试本次CLIP对照，固定已完成120轮run保存的epoch95后评价完整test2104：71.673% Acc@0.5、53.612% Acc@0.7、61.012% mIoU。相对原InfMAE增加2.471/4.087/2.228个百分点，仍较论文两项准确率低0.757/0.238个百分点。验证命中数精确复现，单seed复现尚待完成。本轮仅评价该固定权重，没有新开训练或依据测试选轮；完整记录见`output_clip_control_test/2026-10-04_seed13_epoch95/summary.json`及会话文档末节。下方10月3日的“尚缺测试”为当时状态。

> **2026-10-03 路线更新：** 本地同配CLIP-TIR/IAFv3对照120轮已正常完成，实际保存best为epoch95，验证463/608、337/608、63.807%，优于InfMAE A5-F2 epoch94的457/608、329/608、63.472%；InfMAE总参数高31.262%。尚缺seed复现和CLIP最终测试。下一步复现CLIP基线，并进行原增强/目标保留裁剪/裁剪加8像素保护的三组同初始化、同预算对照。若最终推理必须使用InfMAE，再限定一次训练期TIR-CLIP教师语义迁移匹配实验。InfMAE直接替换需超过同增强下的最终CLIP基线才扩大投入。上述新实验尚未启动。下文保留早期A7设计及其历史门槛；最新方案和配对记录见`InfMAE_CHAT_SESSION_2026-09-27.md`末节。

## 目标与当前判断

目标是在 RefFLIR 验证集上将 Acc@0.5 提升至至少 76%。验证集共有
608 个样本，因此当前 A5BridgeFT 的 74.3421%（452 个正确）需要至少再
多定位正确 11 个样本，才能达到 76.1513%（463 个正确）。

论文的 AMA + LAVS + TPF 基线为 75.33%。当前直接 InfMAE 的最佳
A5BridgeFT 为 74.34%，说明尚未证明 InfMAE 前端已经完整替代原始
TIR-CLIP + 高秩 LoRA；A6SpatialFT 和 A6MassFT 都不作为主线。

## 不变的主干边界

```text
RGB CLIP + RGB LoRA ────────────────────────────┐
                                                   ├─ 原四层 LAVS RGB<->TIR
TIR InfMAE + Hierarchical Thermal Adapter ───────┘
                     │
                     └─ 原四层拼接 / 512 维投影
                              │
                           原 IAFv3
                              │
                    原 6 层 Vision-Language Transformer
                              │
                      原 regression token + MLP box head
```

LAVS、四层特征拼接、IAFv3、联合视觉语言 Transformer 和 box head 的
接口与实现保持不变。InfMAE 模块只输出原 LAVS 所需的四组
`[B, 197, 768]` TIR token，因而之后可单独研究 LAVS 或 IAFv3，而不会
重新混淆“前端表征改善”和“融合模块改善”。

## 下一代前端：A7 Hierarchical Thermal Adapter

现有适配器将同一份 F2/F3 融合 token 复制给四个独立残差 MLP。它能匹配
张量形状，但不能充分模拟原 CLIP 四层特征由浅到深的层级语义。

A7 将替换为层级、跨尺度且无文本/GT 依赖的适配器：

1. F2 先经可学习的局部下采样，保留小目标的热边缘，而不是仅双线性缩放；
2. 每个 LAVS level 都有独立的 F2/F3 门控融合；
3. 四个 level 以残差级联方式生成，低层保留结构、高层逐步补充语义；
4. 所有新增残差以保守初始化启动，确保不会扰动稳定的基础输出。

形式上，第 `s` 层为：

```text
Z0 = LN(P3(F3) + g0 * Down(P2(F2)))
Zs = LNs(Z{s-1} + gs * A_s(Z{s-1}, P2(F2), P3(F3)))
```

其中 `g_s` 由 TIR 全局统计产生，不使用文本和 GT box。文本引导仍只由
原始 LAVS 完成；GT box 仍只进入训练期的 A3/A5 对齐损失。

## 分阶段训练协议

### Phase I：前端语义适配（20--30 epoch）

- 初始化：仅加载通用 MMVG 基础 checkpoint，绝不加载 GQR、A5 或 A6 的最佳权重；
- 冻结：InfMAE 官方权重、RGB CLIP、LAVS、IAFv3、VL Transformer 和 box head；
- 训练：A7 adapter、TIR--Text projector、RGB--TIR projector；
- 损失：grounding loss 保留为接口约束，TIR--Text 先开启，RGB--TIR 在其稳定后再开启；
- 现有 `--infmae_alignment_adapter_only` 已可实现该参数范围，不需要改训练框架。

输出为“可移植的 InfMAE 前端 checkpoint”，它不是最终模型。

### Phase II：原主干集成（90--120 epoch）

- 加载：Phase I 的 model-only checkpoint；
- InfMAE：仍冻结；
- 训练：A7 adapter + 原 RGB LoRA + 原 LAVS + 原 IAFv3 + 原 VL Transformer + 原 head；
- 损失：A3 TIR--Text 和 A5 RGB--TIR 都保留，但从训练前段渐进开启，不与未校准的 adapter 同时突变；
- 判定：先要求稳定超过 75.33%，再讨论 76%。

### Phase III：InfMAE late LoRA（30--50 epoch）

- 加载：Phase II 最佳 checkpoint；
- 仅解冻：F3 最后 3 个 Transformer block 的 LoRA；InfMAE 官方权重其余保持冻结；
- 使用比 Phase II 更低的有效学习率；
- 若不带来至少约 0.3pp 的稳定收益，则回退到 Phase II，而不让 PEFT 破坏前端可复用性。

### Phase IV：后端结构研究（独立分支）

将获得的 Phase II/III 前端冻结为初始化，然后一次只修改一项：

1. LAVS 的跨模态交互；
2. IAFv3 的可靠性融合；
3. 联合 Vision-Language Transformer 的定位建模。

每个后端实验都使用同一份固定 InfMAE 前端 checkpoint；不能同时再改
InfMAE adapter 或 auxiliary loss，避免归因失效。

## 实验准入与停止标准

| 阶段 | 进入下一阶段的条件 | 回退条件 |
| --- | --- | --- |
| A7 Phase I | 对齐损失稳定，adapter 输出无数值异常 | 对齐损失与验证性能持续冲突 |
| A7 Phase II | 最佳 Acc@0.5 >= 75.33% | 未追回论文基线 |
| A7 Phase III | 多个 checkpoint 的均值优于 Phase II | 单点偶然提升或 Acc@0.7/mIoU 明显下降 |
| 后端优化 | 前端 checkpoint 固定 | 前端与后端同时变化 |

除总体 Acc@0.5 外，每个候选都需报告 Acc@0.7、mIoU、弱光/小目标子集。
当前 A6 空间注意力路线仅保留为负结果：它会增加框内注意力质量，但未稳定提升
总体定位，因此不再作为通往 76% 的主干方案。
