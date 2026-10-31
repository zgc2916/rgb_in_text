# InfMAE A6SpatialFT：多尺度空间语义桥

## 目标

`InfMAEA5BridgeFT` 已证明：仅微调 InfMAE F3 最后三个 Transformer block
的 LoRA 可以提升定位准确率；但其相对冻结版的 mIoU 提升很小。A6 因而不
扩大主干或改动 LAVS/TPF，而是让文本引导的 TIR 注意力学习目标的空间范围。

## 保持不变的部分

- RGB：CLIP ViT-B/16、RGB LoRA（rank 16）与原文本引导视觉适配；
- TIR：官方 InfMAE，F2/F3 direct thermal adapter；
- PEFT：仅 F3 block 8--10 的 qkv/proj/fc1/fc2 LoRA（rank 8，alpha 16）；
- LAVS、IAFv3、VL Transformer、定位头、数据加载与训练主循环。

## A6 前端

```text
TIR -> InfMAE F2/F3 -> Direct Thermal Adapter -> 4 x TIR token stages
                                                        |
Text EOS -> frozen CLIP semantic query ----------------+-->
  stage-specific Text-Guided Thermal Spatial Bridge -> original LAVS
```

每一个原 LAVS 输入层各有一个独立桥，而不是 A5 的四层共享桥。对于第 `s`
层 patch `i`：

\[
a_i^s = \operatorname{softmax}_i(\langle \hat z_{tir,i}^s,
\hat z_{text}\rangle / \tau)
\]

\[
r_i^s = a_i^s \cdot \frac{W_t z_{text} + W_s \hat z_{tir,i}^s}{2},
\qquad
x_i^s \leftarrow x_i^s + \tanh(\gamma_s) r_i^s
\]

`gamma_s` 为每层独立的零初始化标量。因此训练第 0 步的 TIR token 与
`A5BridgeFT` 的 direct-InfMAE 前端严格一致；CLS token 不被修改。

## 训练期空间监督

GT box 被精确栅格化为 14x14 patch overlap 概率分布 `p_gt`，包括小于一个
patch 的目标。对四个层的注意力加入：

\[
L_{spatial} = 0.02 \cdot r(e) \cdot
\frac{1}{4}\sum_s \operatorname{KL}(p_{gt}\Vert a^s)
\]

其中 `r(e)` 复用 A3/A5 的对齐 warm-up：epoch 40 开始、10 个 epoch 升至
1。GT box 仅用于此训练损失；验证和推理只输入图像与 referring expression。

完整目标仍为：

\[
L = L_{ground} + 0.05 L_{TIR-Text} + 0.02 L_{RGB-TIR} + L_{spatial}.
\]

## 当前实验

- 方法：`FusionMethod=InfMAEA6SpatialFT`；
- 初始化：与 A5BridgeFT 相同的通用 MMVG mixup 预训练权重，而非续训 A5
  checkpoint；
- GPU：物理卡 4、5；每卡 batch size 6，梯度累积 8；
- 输出：`output_infmae_a6spatial_ft/InfMAEA6SpatialFT_224/rgbtvg_flir/seed_13_mmvgpretrain_120e_g4g5_spatialkl002`；
- 对照：已完成的 `InfMAEA5BridgeFT`，最佳 Acc@0.5 / Acc@0.7 / mIoU 分别为
  74.34% / 54.28% / 62.80%。

首要判断指标是 mIoU 与 Acc@0.7：若 A6 只提高 Acc@0.5 而不改善二者，说明
空间桥仅增强类别选择而未改善目标形状，应转向 F2 高分辨率细节适配；若空间
注意力目标质量上升且 Acc@0.7、mIoU 同步提高，则保留该分支并再做 LoRA 容量
消融。
