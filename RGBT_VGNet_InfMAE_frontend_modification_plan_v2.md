# RGBT-VGNet 前端改进方案：CLIP RGB + InfMAE TIR

## 1. 改进目标

本方案只修改 **LAVS 之前的前端部分**，后续 **LAVS、TPF、Vision-Language Transformer 和定位头全部保持原实现不变**。

核心思路：

- RGB 分支继续使用 CLIP，保留已有视觉-语言语义先验；
- TIR 分支改用 InfMAE，增强红外模态表征能力；
- 使用轻量 Multi-scale Thermal Adapter 将 InfMAE 多尺度特征适配到原 LAVS 输入空间；
- 增加 **Target-aware Thermal–Text Alignment**，显式建立红外目标区域与文本表达之间的语义联系；
- 增加 **RGB→TIR Semantic Knowledge Transfer**，利用 RGB CLIP 的成熟语义表征指导 TIR 分支学习。

---

## 2. 整体结构

```text
                         Referring Expression
                                  │
                           CLIP Text Encoder
                                  │
                               F_text
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
             RGB Image                           TIR Image
                │                                   │
        CLIP Vision Encoder                    InfMAE Encoder
        + 原 RGB LoRA                              │
                │                           F2(H/8), F3(H/16)
                │                                   │
              F_rgb                    Multi-scale Thermal Adapter
                │                                   │
                │                                F_tir
                │                                   │
                │          ┌────────────────────────┼──────────────────────┐
                │          │                        │                      │
                │     RGB→TIR Semantic       Target-aware           原主干输入
                │    Knowledge Transfer      TIR-Text Alignment          │
                │          │                        │                      │
                └──────────┴────────────────────────┴──────────────────────┘
                                             │
                                      原始 LAVS（不改）
                                             │
                                      原始 TPF（不改）
                                             │
                                  原后续定位网络（不改）
```

---

## 3. 修改一：TIR 分支由 CLIP 改为 InfMAE

### 原方案

```text
RGB → CLIP Vision + RGB LoRA
TIR → CLIP Vision + 高秩 TIR LoRA
```

原方案通过更高秩的 TIR LoRA 缓解 RGB 预训练 CLIP 在红外域上的模态偏差。

### 修改后

```text
RGB → CLIP Vision + 原 RGB LoRA
TIR → InfMAE Encoder
```

TIR 分支直接加载官方 **Inf30 自监督预训练权重**，不再使用原 TIR CLIP Encoder 和对应高秩 LoRA。

这样将：

```text
RGB-centric TIR adaptation
```

改为：

```text
Infrared-specific representation learning
```

---

## 4. 修改二：Multi-scale Thermal Adapter

InfMAE 提供多尺度特征。第一版建议使用：

- `F2`：约为 `H/8 × W/8`，保留更多空间细节；
- `F3`：约为 `H/16 × W/16`，提供更强高层语义。

先分别进行维度映射：

\[
\hat F_2 = Downsample(P_2(F_2))
\]

\[
\hat F_3 = P_3(F_3)
\]

其中 `P2`、`P3` 使用 `1×1 Conv / Linear + LayerNorm`。

然后进行融合：

\[
F_{tir}^{0} = \hat F_3 + \alpha \hat F_2
\]

再使用轻量残差 Adapter：

\[
F_{tir}
=
F_{tir}^{0}
+
W_2\sigma(W_1LN(F_{tir}^{0}))
\]

最终得到与原 LAVS 输入维度和 token 形式一致的 TIR 特征：

```text
InfMAE F2 ─→ Projection ─→ Downsample ─┐
                                       ├─→ Fusion → Residual Adapter → F_tir
InfMAE F3 ─→ Projection ────────────────┘
```

---

## 5. 核心创新一：Target-aware Thermal–Text Alignment

InfMAE 主要学习红外视觉表征，其预训练过程没有文本监督，因此不能直接假设：

\[
F_{tir}
\]

已经与：

\[
F_{text}
\]

处于良好的语义空间。

因此增加 **Target-aware Thermal–Text Alignment**。

### 5.1 提取目标级 TIR 特征

利用训练集已有的 GT Bounding Box：

\[
B^{gt}
\]

从 `F_tir` 中提取目标区域 token，并进行池化：

\[
z_{tir}^{obj}
=
Pool(F_{tir},B^{gt})
\]

文本特征为：

\[
z_{text}
=
P_{text}(F_{text})
\]

### 5.2 TIR-Text 对齐损失

采用 InfoNCE：

\[
L_{TIR-Text}
=
-\log
\frac{
\exp(sim(z_{tir}^{obj},z_{text})/\tau)
}{
\sum_j
\exp(sim(z_{tir}^{obj},z_{text}^{j})/\tau)
}
\]

核心目标是：

\[
\boxed{
Target\ Thermal\ Feature
\leftrightarrow
Referring\ Expression
}
\]

而不是对整幅 TIR 图像做粗粒度的 image-text alignment。

该分支只在训练阶段提供辅助监督，推理阶段不增加额外路径。

---

## 6. 核心创新二：RGB→TIR Semantic Knowledge Transfer

RGB 分支的 CLIP 特征已经具有较成熟的视觉-语言语义先验，而 InfMAE 更擅长红外视觉结构。

因此利用同一 GT Box，在 RGB 和 TIR 两个分支中提取同一目标的区域级特征：

\[
z_{rgb}^{obj}
=
Pool(F_{rgb},B^{gt})
\]

\[
z_{tir}^{obj}
=
Pool(F_{tir},B^{gt})
\]

再经过独立投影：

\[
\tilde z_{rgb}=P_{rgb}(z_{rgb}^{obj})
\]

\[
\tilde z_{tir}=P_{tir}(z_{tir}^{obj})
\]

采用余弦距离进行目标级语义迁移：

\[
L_{RGB-TIR}
=
1-
cos(
\tilde z_{rgb},
\tilde z_{tir}
)
\]

其作用是：

```text
RGB CLIP semantic prior
          ↓
Target-level semantic transfer
          ↓
InfMAE thermal representation
```

这里不对整幅 RGB/TIR feature map 做强制一致性约束，只在 **GT 目标区域**进行软对齐，避免过度削弱 InfMAE 中的 thermal-specific 信息。

---

## 7. 三种模态的关系

最终前端形成：

```text
                   Text
                    ▲
                    │
             L_TIR-Text
                    │
                    │
                  TIR
                   ▲
                  /
                 /
          L_RGB-TIR
               /
              /
            RGB
```

其中：

- `RGB ↔ Text`：由原 CLIP 预训练先验提供；
- `TIR ↔ Text`：由 `L_TIR-Text` 建立；
- `RGB ↔ TIR`：由 `L_RGB-TIR` 建立。

这样可以让 TIR 分支同时获得：

```text
红外模态专属视觉表征
+
文本语义对齐能力
+
RGB CLIP 语义知识
```

---

## 8. 总训练目标

保留原 RGBT-VGNet 定位损失：

\[
L_{ground}
\]

加入两个辅助损失：

\[
L
=
L_{ground}
+
\lambda_1 L_{TIR-Text}
+
\lambda_2 L_{RGB-TIR}
\]

其中：

- `L_TIR-Text`：解决 InfMAE 缺少语言对齐的问题；
- `L_RGB-TIR`：利用 RGB CLIP 向 TIR 分支传递目标级语义知识。

---

## 9. 推荐训练策略

### Stage 1：前端稳定适配

- CLIP Vision：冻结 backbone，仅训练原 RGB LoRA；
- CLIP Text：冻结；
- InfMAE：冻结；
- Multi-scale Thermal Adapter：训练；
- TIR-Text Alignment projection：训练；
- RGB-TIR projection：训练；
- 原 LAVS、TPF 和后续模块：按 baseline 设置训练。

### Stage 2：红外高层适配

Stage 1 收敛后：

- 解冻 InfMAE 最后若干 Transformer Block；
- InfMAE 使用较小学习率；
- Thermal Adapter、RGB LoRA 和后续模块继续训练；
- 两个辅助对齐损失继续保留。

不建议一开始全参数微调 InfMAE。

---

## 10. 推荐消融实验

| 实验 | RGB 分支 | TIR 分支 | TIR-Text | RGB-TIR | 目的 |
|---|---|---|---|---|---|
| A0 | CLIP | 原 CLIP + AMA | × | × | 原始 RGBT-VGNet |
| A1 | CLIP | InfMAE F3 + Projection | × | × | 验证红外 backbone |
| A2 | CLIP | InfMAE F2+F3 + Adapter | × | × | 验证多尺度适配 |
| A3 | CLIP | InfMAE F2+F3 + Adapter | ✓ | × | 验证 TIR-Text 对齐 |
| A4 | CLIP | InfMAE F2+F3 + Adapter | × | ✓ | 验证 RGB→TIR 语义迁移 |
| A5 | CLIP | InfMAE F2+F3 + Adapter | ✓ | ✓ | 完整方案 |

重点观察：

- Overall Acc；
- Low-light / Very-low-light；
- Small-object；
- Long-distance。

---

## 11. 最终方案概括

```text
原方案：

RGB CLIP ───────────────┐
                        ├─→ LAVS → TPF → 后续定位网络
TIR CLIP + AMA ─────────┘


修改后：

RGB CLIP ───────────────────────────────────────────────┐
                                                       │
            ┌── RGB→TIR Semantic Knowledge Transfer ───┤
            │                                          ├─→ LAVS → TPF → 后续定位网络
TIR InfMAE → Multi-scale Thermal Adapter ───────────────┤
            │                                          │
            └── Target-aware TIR-Text Alignment ────────┘
```

### 最终三个前端改进点

1. **Infrared-specific Thermal Encoder**
   用 InfMAE 替代 TIR 分支中的 RGB-centric CLIP。

2. **Target-aware Thermal–Text Alignment**
   显式对齐目标区域红外特征与 referring expression。

3. **RGB→TIR Semantic Knowledge Transfer**
   利用 RGB CLIP 的目标级视觉语义表示指导 TIR 分支，同时保留 thermal-specific 信息。

**LAVS、TPF 及其后的网络结构全部保持不变。**
