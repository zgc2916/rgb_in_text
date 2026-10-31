# InfMAE A6MassFT：局部目标内注意力优化

## 出发点

`InfMAEA6SpatialFT` 的四层空间桥可以学习 GT 目标区域：注意力落在框内的
质量从约 2% 升至 52%。但其完整框分布 KL 会要求注意力在整个目标框内近似均匀，
导致 Acc@0.5 与 mIoU 低于 `InfMAEA5BridgeFT`。

Referring expression 往往由判别性局部（如车灯、衣服、轮廓）定位目标，而非
均匀覆盖整个框。因此 A6MassFT 只约束注意力进入目标框，保留它在框内选择最有
辨识力 thermal patch 的自由。

## 与 A6SpatialFT 的唯一区别

保持四个 stage-specific text-guided thermal spatial bridge、InfMAE 最后 3 个
F3 block LoRA、A3/A5 目标级对齐以及原有下游网络不变。

训练期空间目标替换为：

\[
L_{mass}=-\log\left(\sum_{i \in B^{gt}} a_i\right)
\]

其中 `a_i` 是文本到 TIR patch 的 softmax 注意力。框仍以精确 patch overlap
栅格化，支持小目标；但只有框内支持集用于 loss，不再使用其均匀归一化分布。

| 项目 | A6SpatialFT | A6MassFT |
| --- | ---: | ---: |
| 空间目标 | KL(full-box distribution \| attention) | -log(attention mass in box) |
| 最大空间权重 | 0.02 | 0.005 |
| 空间监督开始 | e40 | e50 |
| warm-up | 10 epoch | 10 epoch |

e50 与 RGB→TIR semantic transfer 同时开始，但空间项比 A6 小四倍，以避免 A6
中 e40--e49 的剧烈干扰。推理阶段仍没有 GT box 输入。

## 实验配置

- `FusionMethod=InfMAEA6MassFT`
- 通用 MMVG 初始化；不加载 A5/A6 checkpoint
- 2 GPU（物理卡 4、5），每卡 batch size 6，梯度累积 8
- 输出目录：`output_infmae_a6mass_ft/InfMAEA6MassFT_224/rgbtvg_flir/seed_13_mmvgpretrain_120e_g4g5_mass005_e50`

判定标准：至少同时不低于 A5BridgeFT 的 Acc@0.5 与 mIoU，且 Acc@0.7 改善超过
一个验证样本（0.164pp），才认为空间桥的改动有效。
