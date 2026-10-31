# RGBT-GroundBench 对话与模型优化记录

> 最新更新：2026-10-04（初次整理：2026-09-26）\
> 当前开发分支：`feature/infmae-v2`\
> 数据集与主要验证设置：RefFLIR (`rgbtvg_flir`)，输入 224×224，seed 13

本文按项目推进顺序整理此前讨论、已实施的模型改动、训练结果及后续约束。它是基于可见对话上下文和仓库日志整理的摘要，不是逐字聊天记录。未标注划分的Acc@0.5、Acc@0.7、mIoU指验证集指标；该验证集有608个样本，因此一个样本约对应0.1645个百分点。明确标注测试集的结果来自2104条表达。

## 先看结论

- **10月4日21:31验证/测试差距分析：** 复用冻结预测和标注，无新模型前向。train/val/test图像ID交集均0，唯一图像数3803/325/1013；这只排查ID重叠，未审计像素重复及预训练暴露。test弱光比例30.038%高于val18.750%，但small比例46.008%低于57.566%，GT短边中位数16.10像素大于12.775。四个互斥光照×尺寸组的test准确率均低于val，同样8–16像素组Acc@0.5为80.612%→62.309%，不能把差距只解释为测试小目标更多。已确认高IoU收益被弱光/小目标退步抵消；验证选峰偏差、全12层LoRA提前开放及裁剪噪声的因果贡献仍待控制实验。论文测试差约5例尚无跨seed结论，下一步应复核配方/seed及分组高IoU保持条件。报告：[泛化差距核对](output_clip_control_test/2026-10-04_seed13_b12_epoch111/generalization_review.json)。
- **10月4日CLIP batch12 epoch111完整测试已完成：** 用户明确要求即时评价，按要求在seed14复核前固定权重并完成一次全test2104前向：1519/2104（72.196%）Acc@0.5、1128/2104（53.612%）Acc@0.7、60.911%mIoU；验证474/608、342/608复现，冻结快照哈希未变。对比batch6 epoch95：Acc@0.5多11例、+0.523个百分点，Acc@0.7持平，mIoU低0.100个百分点；仍低于论文0.234/0.238个百分点。弱光632条的高IoU命中由318降至308，小目标968条由255降至241，显示验证整体收益未全面转化为测试高精度定位改善。当前无跨seed结论，不能归因于单独候选数；本轮无新训练。完整报告：[batch12完整测试](output_clip_control_test/2026-10-04_seed13_b12_epoch111/summary.json)。
- **10月4日CLIP batch12已完成120轮：** 15:30:47正常结束，实际保存best为epoch111：验证474/608（77.961%）Acc@0.5、342/608（56.250%）Acc@0.7、65.793%mIoU。比batch6 epoch95增加11/5个命中及1.809/0.822/1.986个百分点，整体三项改善；弱光Acc@0.7却由60/114降至53/114，小目标由123/350降至119/350，需要继续关注。16:14读取实际checkpoint，epoch与三项元数据均匹配该日志轮次，未新增模型前向或测试。结果属于seed13的微批配置改善，尚无跨seed及batch12测试结论，也不能把全部收益归因于候选数；下一步复核seed14再固定测试选择。记录：[batch12验证结果核对](output_clip_batch_control/2026-10-04_seed13_b12_a4/validation_review.json)。
- **10月4日12:13开始CLIP batch12对照：** 按用户要求修正基础CLIP本卡候选减少的问题，新增显式batch/累积参数，在GPU4/6从同一公共MMVG预训练权重开始120轮seed13训练。每卡12、累积4，名义有效batch96；LoRA从epoch0开放全部12层，沿用已有CLIP对照。启动检查实际完成4个微批/1次AdamW更新，两卡均确认12×12基础对比矩阵，梯度范数及各项损失有限，峰值已分配显存约9.8GiB。检查权重丢弃，正式训练重新初始化；目前尚无提升结论。该实验比较恢复微批配置后的整体效果，还会改变批内分组、增强随机数和drop_last尾部样本，不能称为纯候选数消融。状态：[CLIP batch12运行](output_clip_batch_control/2026-10-04_seed13_b12_a4/status.json)。
- **10月4日复现差距审计：** 下载并固定作者当前RefFLIR标注版本后，六个本地划分.pth均与发布包逐字节一致；原IoU公式CPU重算两种模型2104条预测均与保存值一致。本地旧`MMVG_rgbt_flir_best.pth`缺IAFv3参数，不能视为论文模型；作者当前同名权重文件已不同，尚未下载/复验。CLIP对照除了改变LoRA阶段，还把基础CLIP对比损失的本卡候选从公开脚本默认12降至6，累积不合并候选。唯一原因尚未确定，下一优先级改为核对正确作者权重及原环境评价，再复现训练配方；本轮无新训练/模型前向。审计：[原论文差距检查](output_infmae_route_review/2026-10-04_paper_gap_audit/summary.json)。
- **10月4日CLIP同配对照已完成完整测试：** 固定训练120轮后保存的epoch95，测试Acc@0.5为71.673%（1508/2104）、Acc@0.7为53.612%（1128/2104）、mIoU为61.012%；验证463/608、337/608精确复现。相对原InfMAE测试分别增加2.471/4.087/2.228个百分点，仍较论文两项准确率低0.757/0.238个百分点。弱光属性632条为68.829% Acc@0.5，小目标标签968条为51.963%；分组有重叠。它是本地CLIP配方对照，区别于原论文权重和当前默认SigLIP2脚本，仍缺跨seed复现。固定选择记录先于测试保存，本轮完整test前向1次。结果：[CLIP完整测试汇总](output_clip_control_test/2026-10-04_seed13_epoch95/summary.json)。
- 原完整训练对照中，TSAR-Ground `GQRv1` epoch57为75.329%（458/608），使用CLIP RGB/TIR与GQR，不是InfMAE直连版本。结果日志：[GQRv1 log](output_tsar_ground/TSAR_Ground_GQRv1_224/rgbtvg_flir/seed_13_modelv2/log.txt)。
- **当前保留主模型**为`InfMAEA5BridgeFTF2` epoch94：75.164%（457/608）Acc@0.5、54.112% Acc@0.7、63.472% mIoU。结果日志：[A5BridgeFTF2 log](output_infmae_a5_f2full/InfMAEA5BridgeFTF2_224/rgbtvg_flir/seed_13_full120_g2g4/log.txt)。
- 单seed低学习率续训曾出现464/608（76.316%），但高IoU下降且seed14未复现；保留的等权平均候选及10月2日匹配原配置续训各达到463/608（76.151%），均未通过高IoU保持条件。已出现76%验证候选，但尚未得到跨seed稳定、综合指标通过的最终提升模型。
- 最近测试的 `InfMAEA5BridgeFTF2GQR` 组合训练已完成 120 epoch，但最佳 Acc@0.5 只有 **73.520%（epoch 89）**，低于两个单独版本。该组合目前不应标为最佳方案。日志：[A5BridgeFTF2GQR log](output_infmae_a5_f2_gqr/InfMAEA5BridgeFTF2GQR_224/rgbtvg_flir/seed_13_scratch120_g1g2/log.txt)。
- 10月3日完成严格冻结主干的两组框头对照：三层框头最佳460/608、326/608、63.713%，最后坐标层最佳457/608、328/608、63.680%。均未通过综合提升条件，本轮最终选择原epoch94模型。
- 该固定模型完成一次全量测试：**69.202% Acc@0.5（1456/2104）、49.525% Acc@0.7（1042/2104）、58.784% mIoU**，验证准确率成功复现。测试与验证Acc@0.5相差5.963个百分点。224输入下，GT短边<16像素的测试目标Acc@0.5为48.780%，占全部测试失误的81.019%；后续优先在训练/验证集研究小目标的像素保留与泛化。结果：[本轮最终测试汇总](output_infmae_test_review/2026-10-03_final_selected/summary.json)，[误差分桶](output_infmae_test_review/2026-10-03_final_selected/error_diagnosis.json)。
- 用户要求与RGBT.pdf比较后，核对论文第12页表4及第19页表7：RGBT-VGNet的RefFLIR测试Acc@0.5为72.43%、Acc@0.7为53.85%，当前分别低3.228和4.325个百分点，未达到论文基线。复用已保存预测聚合的TestC为47.831%（论文52.53%）；TestB的632/640条匹配表达为62.500%（论文完整子集69.22%），还存在弱光泛化问题。下一阶段优先建立原CLIP-TIR/IAFv3与InfMAE的同配训练/验证对照。核对：[论文基线比较](output_infmae_test_review/2026-10-03_final_selected/paper_baseline_comparison.json)。
- 10月3日17:48在GPU3/6开始新的前端语义实验：精确加载固定epoch94，仅训练459.5万红外前端参数，RGB/定位主干冻结且eval。先做12轮RGB教师置信度加权，再做相同配置12轮等权对照，之后排队120轮公共初始化CLIP-TIR/IAFv3同配对照。真实优化步确认902个冻结张量不变，原生红外LoRA梯度有效；初始验证复现457/608和329/608，弱光76/114、小目标210/350。方案仍待验证，不替换主模型，测试集封存。进度：[实验状态](output_infmae_a5f2_semantic/2026-10-03_seed13_frontend12_lr5e6/status.json)，[启动检查](output_infmae_a5f2_semantic/2026-10-03_seed13_frontend12_lr5e6/preflight.json)。
- 新加权组初期epoch0为458/608、324/608、63.472%，epoch1为456/608、327/608、63.431%；高IoU仍低于源模型，暂无综合提升。进入第3轮后继续既定对照实验，尚未启动排队的等权和CLIP组。
- 最新已完成前4轮，epoch2/3均为453/608、327/608，mIoU分别63.428%/63.444%，弱光74/114、小目标206/350；初期仍无综合提升。实际epoch0 best再次核对902个冻结张量完全不变，父进程和torchrun存活，继续完成配对比较。
- 12轮置信度加权与等权对照已完成，最佳都为epoch0的458/608、324/608、63.472%；弱光76/114、小目标211/350。同一权重训练后期为455/608、328/608、约63.497%，整体未超过源模型的457/608、329/608、63.472%；置信度加权与等权曲线几乎相同，暂无有效增益。按原计划继续120轮CLIP-TIR/IAFv3共同初始化对照；46轮时最佳epoch43为448/608、316/608、62.697%，弱光81/114但小目标199/350。完整轮次及分组记录在会话文档和实验`comparison.json`中；测试集仍封存。
- 22:27 CLIP对照已完成92轮，最佳epoch91为459/608、340/608、64.367%，弱光82/114、小目标212/350；当前验证三项均略高于固定InfMAE主模型，尚未完成120轮/seed复现，未评测试。下一优先级来自真实训练几何审计：512条train抽样原增强产生21个空框、50个截断框（含空框），目标保留裁剪均降至0；再加8像素尺度保护，<4像素目标从143降至8。下一建议为两组目标保留增强对照，之后才检验训练过的CLIP-TIR教师蒸馏；本次只执行CPU数据审计，尚未启动这些新训练。审计：[训练增强记录](output_infmae_a5f2_target_crop/2026-10-03_audit_only/augmentation_audit.json)。
- 用户质疑InfMAE路线后再次核对：CLIP完成105轮时，epoch95达463/608、337/608、63.807%，弱光81/114、小目标216/350，已通过原综合验证门槛；InfMAE原模型为457/608、329/608、63.472%，总参数还多31.262%。冻结CLIP epoch95在val做一次配对确认，精确复现命中数：CLIP在IoU≥0.5上改对37例、改错31例，净+6；IoU≥0.7净+8。单seed/重复选轮不足以建立稳定优势；当前建议以CLIP为效果基线，将InfMAE直接替换降为有明确验收条件的研究支线。CLIP120轮训练继续、测试未评价。配对：[路线诊断](output_infmae_route_review/2026-10-03_clip_vs_infmae_val/paired_validation_comparison.json)。

## 项目目标与约束

> **2026-10-03 后续调整：** CLIP同配120轮已于23:38:01完成，实际保存best仍为epoch95（463/608、337/608、63.807%），尚无CLIP测试/seed复现。下一步先建立CLIP对照并验证原增强、目标保留裁剪、裁剪加8像素保护的三组匹配实验；若研究要求最终推理使用InfMAE，再限定一次训练期TIR-CLIP教师语义迁移对照。数据修正需共同应用，InfMAE验收改为与同增强、同seed/预算的最终CLIP比较。该调整目前是方案，尚未启动新训练；详细设置见会话文档末节。

1. 主要精度目标是 RefFLIR 验证集 **Acc@0.5 达到或超过 76%**，同时关注 Acc@0.7 和 mIoU，避免只优化低 IoU 阈值。
2. InfMAE 研究方向是增强 TIR 红外表征：RGB 使用 CLIP ViT-B/16；TIR 直接使用官方 `InfMAE/InfMAE.pth`，不走 CLIP TIR 分支。
3. 不加载已训练好的 GQR 最佳权重来初始化 InfMAE 方案。通常仍从发布的通用 MMVG mixup checkpoint 初始化检测/融合主干，并从官方 InfMAE checkpoint 初始化 TIR 编码器；这叫使用通用预训练初始化，不等于随机权重从头训练，也不等于加载 GQR 最佳模型。
4. 论文主干尽量保持：双向 LAVS cross-attention → 四层视觉特征拼接并投影到 512 维 → IAFv3 RGB/TIR 融合 → 6 层联合视觉语言 Transformer → regression token → 3 层 MLP 预测 box。InfMAE 主要替换 TIR 前端，不应无依据地重写后续主干。
5. 不改训练编排、数据加载、分布式策略和主训练循环。A3/A5 要求把对齐目标加入训练损失，因此允许改模型模块及相应 loss 接口；不把“只改模块”理解成不能接入必要的损失项。
6. 训练最多使用两张 GPU（此前上限曾讨论为四张；当前实际运行约束为优先两张）。使用固定seed和独立输出目录，保留日志、配置和诊断；用户后来授权清理不良实验权重，可清理已结束负结果权重并保存清单，保留主模型与必要候选。
7. 对照实验要明确初始化权重、每卡 batch、梯度累积、全局有效 batch、InfoNCE 候选数、学习率、epoch 和 seed。梯度累积不会增加单次前向里的 InfoNCE 候选数。

## 讨论与模型演进

### 1. TSAR-Ground / GQRv1

最初基于 TSAR-Ground 设计，在原 IAFv3 融合处增加 Grounding Quality Router（GQR）：RGB 和 TIR 各自生成文本条件 box，利用两路预测质量形成软监督，再由路由器估计模态可靠性，对 IAFv3 的 RGB/TIR 融合权重作有界修正。它的目标是按样本利用更可靠的模态，不是替代整套视觉语言主干。

训练中曾出现学习率不稳定、损失发散或进程中断的问题，随后改用稳定设置、确认进程并清理/重新开始实验。用户要求监控训练、限制 GPU 数量，并询问训练是否仍在运行。最终 GQRv1 在验证集达到本仓库已记录的最高 Acc@0.5；但它没有使用 InfMAE 直连前端。

### 2. InfMAE 直连前端 A2 与 A3/A5 语义监督

根据 `RGBT_VGNet_InfMAE_frontend_modification_plan_v2.md`，方案改为 RGB CLIP + TIR InfMAE，且不加载 GQR 最佳 checkpoint。TIR InfMAE 的官方底座冻结，之后逐步尝试轻量适配器/LoRA。

- **A2 / `InfMAEDirectV2`**：InfMAE 提供 F2/F3 特征，适配为原 LAVS 所需的多层 TIR token；后续双向 LAVS、IAFv3、联合 Transformer 和 box head 保持原接口。
- **A3**：在 GT 目标框区域池化 TIR 特征，与冻结文本语义做 TIR–Text InfoNCE 对齐。
- **A4**：在目标区域做 RGB→TIR 语义迁移，RGB 目标特征作为教师。
- **A5**：联合 A3 和 A4 两项目标级监督。
- **A5Bridge**：在 TIR token 进入 LAVS 前，用文本引导的语义残差桥接增强 TIR patch；残差零初始化，减少初期对已有前向表征的扰动。
- **A5BridgeFT**：在 A5Bridge 上对 InfMAE F3 后三层加 LoRA。
- **A5BridgeFTF2**：继续在 InfMAE F2 两个细节 CBlock 加低秩适配，同时保留 F3 后三层 LoRA，是当前 InfMAE 单模型结果最高的完整训练版本。

A5 的对齐权重采用延迟启动：TIR–Text 目标从 epoch 40 开始、10 epoch 线性升至 0.05；RGB→TIR 从 epoch 50 开始、10 epoch 升至 0.02。TIR–Text 温度为 0.07。其余训练仍持续使用原 grounding 损失。

### 3. 其他 InfMAE 探索

为增强多尺度接口，试验了 A6 spatial/mass、A7 hierarchical adapter、A7 F3 LoRA continuation、A8 local evidence、TCRC 和 F2 PEFT 等变体。结果显示，部分短续训的初始验证点看起来较高，但不能与完整 120 epoch 训练直接比较；完整实验中 A7、A6 等没有超过 A5BridgeFTF2。短续训结果只能作为探索线索，不应当作已验证的新最佳模型。

### 4. Batch 12 及 A5-F2-GQR 组合实验

为了尝试提高精度，先将每卡 batch 从 6 提高到 12、梯度累积从 8 降到 4，维持两卡全局有效 batch 96，同时把 TIR–Text 候选数从 12 提高到 24。尽管总体有效 batch 相同，单次前向中的 InfoNCE 难度变了：候选数增加，TIR–Text 对齐损失和语义桥响应增大，而定位精度下降。该次完整训练最佳为 72.862%（epoch 90），判定为负结果。

随后尝试 `InfMAEA5BridgeFTF2GQR`，把 InfMAE A5-F2 前端与 GQR 接到一起，恢复每卡 batch 6、累积 8，并用通用 MMVG 初始化而非之前 GQR 最佳权重。该组合最终最佳仅 73.520%（epoch 89），最终 epoch 119 为 71.875%。这说明两个单独有效的模块不保证直接叠加后更好；当前组合不应继续作为默认路线。

## 主要完整训练结果

以下均为验证集单 seed 结果。表中“最佳轮”是按 Acc@0.5 选出的轮次，其 Acc@0.7 和 mIoU 均取自同一轮。

| 方案 | 最佳轮 | Acc@0.5 | Acc@0.7 | mIoU | 备注 |
|---|---:|---:|---:|---:|---|
| TSAR-Ground `GQRv1` | 57/120 | **75.329%** | 53.783% | 63.721% | 全仓库修改方案最高 Acc@0.5；非 InfMAE 直连 |
| `InfMAEA5BridgeFTF2` | 94/120 | **75.164%** | 54.112% | 63.472% | InfMAE 系列最高；当前 v1.0 对应模型 |
| `InfMAEA5BridgeFT` | 97/120 | 74.342% | 53.783% | 62.742% | 不含 F2 LoRA 的 A5BridgeFT |
| `InfMAEA7HAdapterFT` | 12/40 | 74.342% | 52.138% | 62.101% | 从 phase 2 续训的短阶段，不能与完整 120 epoch 等同 |
| `InfMAEA6MassFT` | 97/120 | 73.849% | 52.796% | 62.496% | 未超过 A5-F2 |
| `InfMAEA7HAdapter` | 89/120 | 73.849% | 53.618% | 62.246% | 未超过 A5-F2 |
| `InfMAEA6SpatialFT` | 78/120 | 73.520% | 52.303% | 62.373% | 未超过 A5-F2 |
| `InfMAEDirectV2` | 109/120 | 73.520% | 53.289% | 61.820% | A2 直连控制版本 |
| `InfMAEA5BridgeFTF2GQR` | 89/120 | 73.520% | 52.632% | 62.563% | A5-F2 与 GQR 组合，负结果 |
| A5-F2 batch12 变体 | 90/120 | 72.862% | 52.796% | 61.808% | 候选数 24，低于 batch6/候选数12 |

GQRv1 最佳轮的 Acc@0.5 为 75.329%；其 epoch 119 最终验证结果为 Acc@0.5 75.000%、Acc@0.7 55.428%、mIoU 64.502%。A5BridgeFTF2 最终 epoch 119 为 Acc@0.5 74.671%、Acc@0.7 52.796%、mIoU 63.500%。因此需要区分“按 Acc@0.5 选出的最佳 checkpoint”和“训练最后一轮”，也不能把某一轮的 Acc@0.7/mIoU 与另一轮的 Acc@0.5 拼成同一个 checkpoint 成绩。

## 当前 InfMAE 最佳模型结构

当前最佳 InfMAE 方案为 `InfMAEA5BridgeFTF2`：

```text
RGB: CLIP ViT-B/16 + RGB LoRA ───────────────────────────┐
                                                         ├─ 双向 LAVS cross-attention
TIR: InfMAE F2/F3 → F2 LoRA + F3 后三层 LoRA             │
     → F2/F3 到 LAVS token 的直接适配                     │
     → A5 文本引导语义桥（进入 LAVS 前） ────────────────┘
        └─ 四层特征拼接并投影到 512
           └─ IAFv3 RGB/TIR 自适应融合
              └─ 6 层联合视觉语言 Transformer
                 └─ regression token → 3 层 MLP → box
```

训练时另有两个目标级辅助损失：GT 目标区域 TIR–Text 对齐和 RGB→TIR 目标语义迁移。官方 InfMAE 主干冻结，F2/F3 只训练插入的低秩参数；A5 适配器、对齐投影头以及原任务中可训练的融合/定位部分按对应训练配方更新。

## 训练与工程要求汇总

- 数据集：RefFLIR `rgbtvg_flir`；图像尺寸 224；常用 seed 13；主实验通常 120 epoch。
- 两卡配方：已发布 InfMAE 运行文档采用每卡 batch 6、梯度累积 8，有效全局 batch 为 `2×6×8=96`；AdamW，weight decay 1e-4，cosine 学习率，基础学习率 1e-4，梯度裁剪 1.0。
- InfoNCE 候选数由一次分布式前向可见的样本决定，不由梯度累积决定。每卡 6、两卡时记录候选数 12；每卡 12、两卡时为 24。比较两种 batch 时，即使有效全局 batch 相同，也不能称为完全相同的优化条件。
- InfMAE TIR checkpoint 是 `InfMAE/InfMAE.pth`；检测/融合公共初始化 checkpoint 是发布的 MMVG mixup checkpoint。不要把 `GQRv1` 的 best checkpoint 当成 InfMAE 的初始化权重。
- v1.0 运行教学文档：[Run_InfMAEA5BridgeFTF2_v1.0.md](docs/Run_InfMAEA5BridgeFTF2_v1.0.md)。其中仓库代码和权重/数据的边界、两卡命令及续训注意事项已有说明。
- GitHub 发布记录：此前按要求保存发布版本、创建 `v1.0` tag，并提供独立训练教学文档；当前开发仍在 `feature/infmae-v2`。研究结果、数据、预训练权重和训练 checkpoint 不应提交到公开代码仓库。

## MRP screen (2026-09-28)

| Seed | Best Acc@0.5 | Best Acc@0.7 | Best mIoU |
| --- | ---: | ---: | ---: |
| MRP 13 | 458/608 (epoch 8) | 326/608 (epoch 6) | 63.839% (epoch 2) |
| MRP 14 | 457/608 (epoch 10) | 324/608 (epoch 4) | 63.572% (epoch 10) |

A5-F2 epoch 94 baseline is 457/608 Acc@0.5, 329/608 Acc@0.7, and 63.472% mIoU. MRP did not reproduce an Acc@0.5 gain across seeds; both runs lost Acc@0.7. Paired per-sample probes show no reliable gain from global box scaling or averaging. Stop the MRP branch. Next, diagnose localization errors and test a short teacher-preserved box-regression fine-tune before another fusion module. The test set remains frozen.

## 后续判断原则

1. 先以完整 checkpoint 的验证集 Acc@0.5 判断是否超过 75.329%，再以 Acc@0.7、mIoU 检查高质量定位是否同步改善。
2. 单 seed、单轮峰值只作为候选结果；最终结论应在固定配置下复跑多个 seed，并在模型选择完成后单独评估测试集。
3. 每次实验尽量只改一个变量。保持候选数、有效 batch 和对齐损失权重可解释；不要因为语义相似度变高就断言 grounding 精度变好。
4. 继续优化时，优先围绕 InfMAE 前端与 LAVS/IAFv3 的信息接口做可控消融，同时保留论文主干和已验证的 A5BridgeFTF2 checkpoint 作对照。


## Teacher-preserved box-head screen (2026-09-28)

A5-F2 epoch 94 的 Seed13 框头微调跑 4 轮后提前停止，Acc@0.5 最好 459/608，但 Acc@0.7 最好仅 321/608，低于固定基线 329/608。逐样本探针上候选与基线 Acc@0.5 都是 458/608，Acc@0.7 为 324/608 对 329/608，mIoU 63.523% 对 63.589%；2 个样本升过 0.7，7 个跌破。判定为负结果，不运行 Seed14。保留训练日志、配置和诊断 JSON，删除本分支 checkpoint 与临时 warm-start。


## Exclusive teacher/GT gate screen (2026-09-28)

对 A5-F2 bbox head 做易/难样本互斥训练的 Seed13 首轮结果为 454/608 Acc@0.5、315/608 Acc@0.7、63.162% mIoU，低于固定基线；立即停止，不跑 Seed14。连同前一加性 teacher 一致性方案，教师框保持方向暂时判负；保留日志和配置，删除本次 checkpoint 与重复 warm-start。当前最佳验证候选仍是 A5-F2 与 Seed13 epoch 6 等权平均（463/608），测试集继续冻结。


## Recheck of the retained weight-average candidate (2026-09-28)

逐样本探针复核保留的 463/608 A5-F2+Seed13 等权平均权重：Acc@0.5 463/608、Acc@0.7 321/608、mIoU 63.829%；A5-F2 基线对应为 458/608、329/608、63.589%。高 IoU 少 8 个样本，未通过原门槛，不跑 Seed14、不评测试集。候选权重仍保留；探针结果存于 output_infmae_analysis/soup_seed13_equal_probe_2026-09-28.json。


## EIoU loss screen (2026-09-29)

An opt-in EIoU regression loss screen stopped after Seed13 epoch 0: 453/608 Acc@0.5, 317/608 Acc@0.7, 62.668% mIoU, below A5-F2 baseline 457/608, 329/608, 63.472%. No Seed14; logs/config retained and failed weights removed. The default loss remains GIoU. See [EIoU paper](https://arxiv.org/abs/2101.08158) and output_infmae_analysis/eiou_cleanup_2026-09-29.txt.


## Weight-average ratio probe (2026-09-29)

权重插值探针以 A5-F2 基线 458/608、329/608、63.589% 为参照。Seed13 续训权重比例 0.10 得到 457/608、323/608、63.593%；比例 0.25 得到 461/608、323/608、63.730%；比例 0.50 得到 463/608、321/608、63.829%。没有比例同时保持高 IoU 并达到 463/608 Acc@0.5；停止搜索、不跑 Seed14，测试集继续冻结。


## Predicted-area gate OOF probe (2026-09-29)

在 RefFLIR val 上对 A5-F2 基线与 alpha=0.5 soup 做五折交叉拟合，门控特征为基线预测框面积，q25 阈值只用每折训练侧预测面积估计。OOF 门控为 460/608 Acc@0.5、330/608 Acc@0.7、63.694% mIoU；基线探针为 458/608、329/608、63.589%。门控在 152 个小面积预测上使用 soup，其余使用基线。结果是小幅三项提升，但推理要运行两模型且 Acc@0.5 未到 463；暂不运行 Seed14或测试集评估。数据：output_infmae_analysis/soup_area_gate_oof_2026-09-29.json。

## 缩放增强消融完成（2026-10-02）

已在 GPU 3、6 启动配对续训：先关闭 `aug_scale`（缩放列表 `[224]`），完成后自动跑原配置对照（`[224,196,168,140,112,84,56]`）。两组使用同一 A5-F2 epoch94 模型权重、seed13、8轮、lr=1e-5、cosine、每卡batch6、累积8；crop/translate、224输入、损失和模型训练范围相同。新增开关和顺序执行入口为 `continue_a5f2.py --no-aug-scale` 与 `script_train/infmae_experiments/run_scale_ablation.py`。

真实数据检查通过，两组验证变换一致，启动时复现官方基线457/608。两组均完成8轮、退出码0，实验于北京时间22:43结束。关闭缩放最佳Acc@0.5轮次epoch2为461/608（75.822%）、321/608 Acc@0.7、64.010% mIoU；原配置对照最佳轮次epoch4为463/608（76.151%）、321/608、63.429%。三项指标均取同一轮。固定基线为457/608、329/608、63.472%；两组均损失8个高IoU样本，未通过综合提升条件，不替换主模型、不跑Seed14，测试集冻结。关闭缩放提升mIoU，但没有超过匹配对照的最佳Acc@0.5。输出及完整逐轮结果：`output_infmae_a5f2_scale_ablation/2026-10-02_seed13_8e/comparison.json`；`status.json`为`complete`。两组权重和日志当前保留。

## F2 局部中心修正及匹配增强对照（2026-10-02）

冻结A5-F2原模型并保持eval，仅训练272450参数的新增局部中心修正头，使用原28×28 F2与框查询。新头零初始化，预测短边16–32像素之间平滑门控，单轴中心偏移≤4像素，宽高不变。修正PEFT切换adapter重新打开LoRA梯度的问题，冻结原参数并从优化器排除原HiLoRA预留参数。此分支仅使用2:2 L1+GIoU回归。

GPU0/1、seed13、8轮、lr=1e-4、有效batch96、原数据增强；北京时间23:36启动、23:45正常完成。真实模型检查通过，正式两卡初始验证复现457/608、329/608、63.472070%。最佳Acc@0.5轮次epoch0为458/608、325/608、63.493%；epoch1保住329/608高IoU但Acc@0.5为456/608，未通过综合门槛。权重核对显示原1003个模型张量全部不变，仅11个新增头张量进入优化器。结果与检查见`output_infmae_a5f2_local_center/2026-10-02_seed13_8e_retry2/`；最初两次检查失败日志保留。

随后开始同一冻结局部头条件下的匹配增强对照，只关闭缩放，其余设置不变：`output_infmae_a5f2_local_center/2026-10-02_seed13_8e_no_scale/`。新增`--local-center`入口与`run_local_center_screen.py`负责检查/启动/结果记录，合格同轮指标另存`qualified_checkpoint.pth`。暂不替换主模型、不启动Seed14，测试集冻结。

无缩放局部头组于23:47进入训练，初始验证与原增强局部头组完全一致，配置差异仅缩放。epoch0为458/608、328/608、63.582%；epoch1为457/608、326/608、63.698%；epoch2为457/608、328/608、63.702%，继续完成8轮，尚未突破。已按此前授权清理完成的原增强局部头组epoch7负结果latest权重，释放0.991 GiB；保留best和全部日志、配置、冻结检查。清单在该组`cleanup_negative_latest_2026-10-02.json`，当前训练权重保留。

两组最终均完成8轮（无缩放组于23:55正常结束），最佳Acc@0.5都为epoch0的458/608（75.329%）：原增强组同轮Acc@0.7为325/608、mIoU63.493%；无缩放组为328/608、63.582%。固定基线为457/608、329/608、63.472%，两组均未达到463/608且保持高IoU的综合门槛。无缩放组mIoU峰值63.794%来自epoch6，该轮只有455/608与325/608，不与其他轮拼接。停止本分支长训，不跑Seed14或测试集，原主模型及此前76.151%候选保留。汇总：`output_infmae_a5f2_local_center/comparison_2026-10-02_seed13.json`；本次未进一步删除权重。
