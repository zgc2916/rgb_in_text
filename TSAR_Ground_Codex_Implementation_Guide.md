# Codex 修改指导：在 crazyxiaoxi/RGBT-GroundBench 中实现 TSAR-Ground

> **目标仓库**：https://github.com/crazyxiaoxi/RGBT-GroundBench
> **目标基线**：`MMVGFusion + IAFv3 + LAVS + AMA`
> **修改原则**：默认参数必须保持原 baseline 行为；新功能全部由显式 flag 打开。
> **优先顺序**：Baseline equivalence → GQR → TERA → Target-aware alignment → High-resolution。

---

# 0. 给 Codex 的总任务

你需要在 RGBT-GroundBench 官方仓库中实现一个**向后兼容**的改进版本，暂称 TSAR-Ground。

核心改动有两个：

1. **GQR（Grounding-Quality Reliability Router）**
   保留现有 `IAFv3` 的 patch reliability，新增一个 text-conditioned modality reliability router。训练时通过 RGB-only / TIR-only auxiliary grounding boxes 与 GT 的 IoU 构造 soft reliability teacher；推理时只使用 router 自身预测。最终在 `IAFv3` RGB 权重的 logit 上增加一个 zero-init reliability correction。

2. **TERA（Thermal Expert Residual Adapter）**
   保留现有 TIR `CLIP + LoRA(rank=48)`；额外接入 M-SpecGene ViT-B 作为 frozen expert，把 expert 特征适配到当前 512-d patch grid 后，以 zero-init gated residual 注入 TIR visual tokens。不能直接用 expert 替换 CLIP。

必须遵守：

- `--enable_gqr` 和 `--enable_thermal_expert` 均关闭时，输出应与当前 `MMVGFusion + IAFv3` 一致；
- 不删除、不重写 AMA/LAVS/IAFv3 baseline；
- 不改变官方 checkpoint 的默认加载逻辑；
- 新 checkpoint 缺失的新模块参数允许 `strict=False`，但必须打印并校验缺失 key 只属于新模块；
- validation/inference 不允许依赖 GT；
- 不自动下载外部权重；
- 不把 M-SpecGene 代码直接复制进仓库，除非许可证和依赖已经确认；优先实现适配接口并从用户指定的本地 checkpoint 读取；
- 每完成一个阶段先做 shape/smoke test，再跑训练。

---

# 1. 先读这些文件，不要直接修改

Codex 首先检查：

```text
models/mmvg_fusion.py
models/__init__.py
train_val/mmvg_train.py
engine.py
utils/loss_utils.py

script_train/RGBT_VGNet/ref_flir/train.sh
script_train/RGBT_VGNet/ref_m3fd/train.sh
script_train/RGBT_VGNet/ref_mfad/train.sh
```

必须确认以下现状：

### 1.1 官方训练入口

`script_train/RGBT_VGNet/*/train.sh` 使用：

```bash
train_val/mmvg_train.py \
  --model_name MMVGFusion \
  --FusionMethod IAFv3 \
  --modality rgbt \
  --open_lora True \
  --open_text_guided_fusion True \
  --lavs_mode lavs \
  --lora_r_rgb 16 \
  --lora_r_ir 48 \
  --epochs 120 \
  --lr 0.0001 \
  --lr_scheduler cosine \
  --imsize 224
```

并加载：

```text
merged_fixed_best_checkpoint_peft0111.pth
```

### 1.2 模型融合点

`models/mmvg_fusion.py` 中：

```python
fused_visu_src = self._fuse_visual_tokens(
    visu_src,
    visu_src_ir,
    image_tensors,
)
```

随后：

```python
vl_src = torch.cat([reg_src, fused_visu_src, text_src], dim=0)
vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)
pred_box = self.bbox_embed(vg_hs[0]).sigmoid()
```

### 1.3 IAFv3 当前实现

现有 `IAFv3` 已包含：

```text
illumination prior
token gate
channel gate
RGB/TIR difference residual
```

禁止用 GQR 完全替换它。GQR 第一版必须做 residual correction。

### 1.4 训练输出接口

当前 MMVG 路径：

```python
output, text_eos, img_cls, visu_sim, seg_mask = model(...)
```

`trans_vg_loss()` 接受上述输出。

新实现要保持 evaluation 时的 5-tuple 接口，training 在开启 GQR/TERA 时可以返回第 6 个 `aux` dict，但 `engine.py` 必须向后兼容 5/6 两种格式。

---

# 2. 建议新增文件

新增：

```text
models/tsar_modules.py
```

先只放新模块，不要把 `mmvg_fusion.py` 再复制一份。

推荐类：

```python
class TextConditionedModalityHead(nn.Module)
class GroundingQualityRouter(nn.Module)
class ThermalExpertAdapter(nn.Module)
class ThermalExpertEncoder(nn.Module)
```

如果外部 expert 的具体实现不能干净集成，`ThermalExpertEncoder` 应先做抽象 wrapper，不要污染 `MMVGFusion`。

---

# 3. Phase 1：新增 CLI，默认完全关闭

修改：

```text
train_val/mmvg_train.py
```

在 `get_args_parser()` 增加：

```python
parser.add_argument('--enable_gqr', action='store_true')
parser.add_argument('--gqr_tau', type=float, default=0.25)
parser.add_argument('--gqr_aux_weight', type=float, default=0.25)
parser.add_argument('--gqr_router_weight', type=float, default=0.20)
parser.add_argument('--gqr_start_epoch', type=int, default=5)
parser.add_argument('--gqr_ramp_epochs', type=int, default=5)
parser.add_argument('--gqr_eta_max', type=float, default=2.0)

parser.add_argument('--enable_thermal_expert', action='store_true')
parser.add_argument('--thermal_expert_type', type=str, default='none',
                    choices=['none', 'mspecgene', 'infmae'])
parser.add_argument('--thermal_expert_ckpt', type=str, default='')
parser.add_argument('--thermal_expert_dim', type=int, default=768)
parser.add_argument('--freeze_thermal_expert', action='store_true')
parser.add_argument('--expert_residual_init', type=float, default=0.0)

parser.add_argument('--enable_target_align', action='store_true')
parser.add_argument('--target_align_weight', type=float, default=0.05)

parser.add_argument('--new_module_lr', type=float, default=5e-4)
```

### 验收

不传任何新 flag 时：

```text
args.enable_gqr == False
args.enable_thermal_expert == False
args.enable_target_align == False
```

原官方脚本无需修改即可运行。

---

# 4. Phase 2：实现 GQR 基础模块

## 4.1 TextConditionedModalityHead

文件：

```text
models/tsar_modules.py
```

输入统一用 batch-first：

```text
visual_tokens: [B, N+1, D]
text_embed:    [B, D]
```

拆分：

```python
cls_token = visual_tokens[:, 0]
patch = visual_tokens[:, 1:]
```

文本作为单 query：

```python
q = self.q_proj(text_embed).unsqueeze(1)       # [B,1,D]
k = self.k_proj(patch)                         # [B,N,D]
v = self.v_proj(patch)
attn = softmax(q @ k.transpose(-1,-2) / sqrt(D), dim=-1)
target_feat = (attn @ v).squeeze(1)            # [B,D]
```

bbox head：

```python
box_feat = torch.cat(
    [target_feat, text_embed, cls_token],
    dim=-1
)

box = torch.sigmoid(self.box_mlp(box_feat))    # [B,4]
```

建议 RGB/TIR 两个 head **不共享最终 box MLP**，但 q/k/v 结构相同。

输出同时返回：

```python
{
    "box": box,
    "target_feat": target_feat,
    "attn": attn.squeeze(1),
}
```

## 4.2 GroundingQualityRouter

输入：

```text
rgb_target: [B,D]
tir_target: [B,D]
text_embed: [B,D]
```

构造：

```python
router_in = torch.cat([
    rgb_target,
    tir_target,
    torch.abs(rgb_target - tir_target),
    text_embed,
], dim=-1)
```

MLP：

```text
4D -> D -> D/2 -> 2
```

使用：

```python
router_logits = self.mlp(router_in)
router_prob = router_logits.softmax(dim=-1)
```

不要在模型中计算 GT-based teacher；GT 只在 `loss_utils.py` 中出现。

---

# 5. Phase 3：把 GQR 接入 IAFv3，但保持 baseline-preserving

## 5.1 不要直接删除 `_fusion_iafv3`

优先做小型重构。

把现有 `_fusion_iafv3()` 中计算 patch 权重和中间量的代码抽成：

```python
def _iafv3_components(
    self,
    rgb_tokens,
    ir_tokens,
    rgb_image,
):
    ...
    return {
        "cls_rgb": cls_rgb,
        "cls_ir": cls_ir,
        "rgb_patch": rgb_patch,
        "ir_patch": ir_patch,
        "channel_gate": channel_gate,
        "w_rgb_base": w_rgb,
    }
```

然后 `_fusion_iafv3()` 用这个 helper 重建**完全相同**的 fused output。

在动 GQR 前，写 baseline-equivalence test：

```python
torch.manual_seed(0)
old = old_iafv3(...)
new = refactored_iafv3(...)
assert torch.max(torch.abs(old-new)) < 1e-6
```

如果不能方便保留旧实现用于测试，就在重构前先保存固定随机输入与输出 tensor，作为 golden result。

## 5.2 新增 GQR fusion

新增：

```python
def _fusion_gqrv1(
    self,
    rgb_tokens,
    ir_tokens,
    rgb_image,
    text_embed,
):
```

取：

```python
components = self._iafv3_components(...)
w_base = components["w_rgb_base"]
```

辅助 grounding：

```python
rgb_aux = self.rgb_aux_head(rgb_tokens_bf, text_embed)
tir_aux = self.tir_aux_head(ir_tokens_bf, text_embed)
```

router：

```python
router_logits = self.gqr(
    rgb_aux["target_feat"],
    tir_aux["target_feat"],
    text_embed,
)

r_rgb = router_logits.softmax(-1)[:, 0]  # [B]
```

### 5.3 Logit residual

```python
eps = 1e-5
w = w_base.clamp(eps, 1-eps)
base_logit = torch.log(w) - torch.log1p(-w)

eta = self.gqr_eta_max * torch.tanh(self.gqr_eta_raw)
correction = eta * (2.0 * r_rgb[:, None, None] - 1.0)

w_final = torch.sigmoid(base_logit + correction)
```

其中：

```python
self.gqr_eta_raw = nn.Parameter(torch.tensor(0.0))
```

所以初始：

```text
eta = 0
w_final == w_base
```

再用现有 IAFv3 residual：

```python
mixed = w_final * rgb_patch + (1 - w_final) * ir_patch
residual = tanh(delta) * channel_gate * (rgb_patch - ir_patch)
fused_patch = mixed + residual
```

### 5.4 修改 fusion dispatch

将：

```python
def _fuse_visual_tokens(self, rgb_tokens, ir_tokens, rgb_image):
```

改成：

```python
def _fuse_visual_tokens(
    self,
    rgb_tokens,
    ir_tokens,
    rgb_image,
    text_embed=None,
):
```

原融合方法忽略 `text_embed`。

新增：

```python
if self.fusion_method == "GQRv1":
    assert text_embed is not None
    return self._fusion_gqrv1(
        rgb_tokens,
        ir_tokens,
        rgb_image,
        text_embed,
    )
```

为了让 fusion 同时返回 aux，建议内部统一：

```python
fused, fusion_aux = ...
```

但是**不要修改所有旧 fusion 的返回类型**。

更安全的办法：

```python
self._tsar_aux = {}
```

每次 `forward()` 开头清空：

```python
self._tsar_aux = {}
```

GQR path 写入：

```python
self._tsar_aux.update({
    "rgb_aux_box": rgb_aux["box"],
    "tir_aux_box": tir_aux["box"],
    "router_logits": router_logits,
    "router_prob": router_prob,
    "w_rgb_base_mean": w_base.detach().mean(),
    "w_rgb_final_mean": w_final.detach().mean(),
})
```

这样 legacy fusion 仍只返回 tensor。

如果团队更偏好纯函数接口，也可以返回 tuple，但必须把旧路径兼容性测试做完整。

---

# 6. Phase 4：让 fusion 获得 text embedding

当前 `MMVGFusion.forward()` 已有：

```python
text_eos_embed = self.clip.text_projection(
    clip_text_features.pooler_output
)
```

但后面会 normalize 用于 CLIP contrastive。

给 GQR 使用时建议单独：

```python
gqr_text_embed = F.normalize(
    self.text_proj(text_features.float())[
        torch.arange(batch_size),
        text_tensors.argmax(dim=-1)
    ],
    dim=-1,
)
```

或者使用 `text_eos_embed` 再增加一个 512→512 projection：

```python
gqr_text_embed = self.gqr_text_proj(text_eos_embed.float())
```

推荐第二种，改动较小。

然后：

```python
fused_visu_src = self._fuse_visual_tokens(
    visu_src,
    visu_src_ir,
    image_tensors,
    text_embed=gqr_text_embed,
)
```

只有 `FusionMethod=GQRv1` 使用它。

---

# 7. Phase 5：Training-only aux output

当前返回：

```python
return (
    pred_box,
    logits_per_text,
    [logits_per_image, logits_per_image_ir],
    visu_token_similarity,
    seg_mask,
)
```

改成：

```python
main_out = (
    pred_box,
    logits_per_text,
    [logits_per_image, logits_per_image_ir],
    visu_token_similarity,
    seg_mask,
)

if self.training and (
    self.args.enable_gqr
    or self.args.enable_thermal_expert
    or self.args.enable_target_align
):
    return (*main_out, self._tsar_aux)

return main_out
```

这样 evaluation 仍是 5-tuple。

---

# 8. Phase 6：修改 engine.py

当前 MMVG：

```python
output, text_eos, img_cls, visu_sim, seg_mask = model(...)
loss_dict = trans_vg_loss(...)
```

改成兼容：

```python
model_out = model(img_data, text_data)

if len(model_out) == 6:
    output, text_eos, img_cls, visu_sim, seg_mask, aux = model_out
else:
    output, text_eos, img_cls, visu_sim, seg_mask = model_out
    aux = None

loss_dict = loss_utils.trans_vg_loss(
    args,
    output,
    target,
    obj_mask,
    text_eos,
    img_cls,
    visu_sim,
    seg_mask,
    aux=aux,
    epoch=epoch,
)
```

`validate()` 不应该需要改，因为 eval mode 模型返回旧 5-tuple。

仍然要搜索整个 `engine.py` 是否存在其他 `'MMVG' in args.model_name` 路径，确保没有遗漏。

---

# 9. Phase 7：修改 loss_utils.py

函数签名改为：

```python
def trans_vg_loss(
    args,
    batch_pred,
    batch_target,
    tgt_mask,
    text_eos,
    img_cls=None,
    visu_sim=None,
    seg_mask=None,
    aux=None,
    epoch=None,
):
```

旧调用完全兼容。

## 9.1 新增 aligned IoU helper

不要依赖 pairwise matrix 后再猜 diagonal。

实现明确的 aligned IoU：

```python
def aligned_iou_xywh(box1, box2, eps=1e-6):
    # box1/box2: [B,4], normalized cxcywh
    b1 = xywh2xyxy(box1)
    b2 = xywh2xyxy(box2)

    lt = torch.max(b1[:, :2], b2[:, :2])
    rb = torch.min(b1[:, 2:], b2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    a1 = (b1[:, 2]-b1[:, 0]).clamp(min=0) * \
         (b1[:, 3]-b1[:, 1]).clamp(min=0)
    a2 = (b2[:, 2]-b2[:, 0]).clamp(min=0) * \
         (b2[:, 3]-b2[:, 1]).clamp(min=0)

    return inter / (a1 + a2 - inter + eps)
```

## 9.2 Auxiliary box loss

若：

```python
aux is not None and args.enable_gqr
```

则：

```python
rgb_box = aux["rgb_aux_box"]
tir_box = aux["tir_aux_box"]
```

复用主框思想，但权重较小：

```python
loss_aux_rgb_l1
loss_aux_rgb_giou
loss_aux_tir_l1
loss_aux_tir_giou
```

总权重由：

```text
gqr_aux_weight = 0.25
```

控制。

不要把每个子 loss 再乘主 loss 的 2.0 后又整体 0.25 而不记录，建议把系数写得明确。

## 9.3 Reliability teacher

```python
with torch.no_grad():
    q_rgb = aligned_iou_xywh(rgb_box.detach(), batch_target)
    q_tir = aligned_iou_xywh(tir_box.detach(), batch_target)

    quality = torch.stack([q_rgb, q_tir], dim=-1)
    teacher = torch.softmax(quality / args.gqr_tau, dim=-1)
```

router：

```python
log_prob = F.log_softmax(aux["router_logits"], dim=-1)
loss_router = F.kl_div(
    log_prob,
    teacher,
    reduction="batchmean",
)
```

### Warmup/ramp

```python
if epoch is None:
    gqr_scale = 1.0
elif epoch < args.gqr_start_epoch:
    gqr_scale = 0.0
else:
    gqr_scale = min(
        1.0,
        (epoch - args.gqr_start_epoch + 1)
        / max(args.gqr_ramp_epochs, 1)
    )
```

最终：

```python
losses["loss_gqr_router"] = (
    loss_router
    * args.gqr_router_weight
    * gqr_scale
)
```

注意：auxiliary box loss在 warmup 时仍训练。

---

# 10. Phase 8：实现 TERA

GQR 跑通并做完 baseline equivalence 后再做。

## 10.1 ThermalExpertEncoder wrapper

目标不是让 RGBT-GroundBench 强依赖 MMEngine/MMPretrain。

建议接口：

```python
class ThermalExpertEncoder(nn.Module):
    def __init__(
        self,
        expert_type,
        checkpoint_path,
        ...
    ):
        ...

    def forward(self, thermal_image):
        # Return either:
        # [B,N,C] or [B,C,H,W]
        return feat
```

优先支持：

```text
expert_type = mspecgene
```

M-SpecGene 官方提供 ViT-B foundation checkpoint。具体 checkpoint key/forward API 必须以用户实际下载的 M-SpecGene 版本为准，Codex 不要臆造 state_dict key。

因此实现步骤：

1. 先读取 checkpoint：
   ```python
   ckpt = torch.load(path, map_location='cpu')
   print(type(ckpt))
   print(ckpt.keys() if isinstance(ckpt, dict) else None)
   ```
2. 检查 M-SpecGene 官方 encoder config；
3. 建同结构 ViT-B；
4. 只加载 encoder；
5. 打印 missing/unexpected；
6. 缺失 key 不能静默忽略。

不要自动从网络下载 checkpoint。

## 10.2 ThermalExpertAdapter

必须支持不同 expert 输出：

```python
class ThermalExpertAdapter(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
```

增加 helper：

```python
def to_patch_tokens(feat, target_hw):
```

功能：

- `[B,C,H,W] -> [B,H*W,C]`
- `[B,L,C]` 自动判断有无 CLS
- 还原 grid
- interpolate 到 `target_hw`
- 输出 `[B,N,C]`

## 10.3 注入位置

第一版建议在：

```python
visu_src_ir = self.visu_proj(image_features_ir.float())
```

之后、`permute` 后或 fusion 前进行注入。

推荐统一转 batch-first：

```python
ir_bf = visu_src_ir.permute(1,0,2)   # [B,N+1,D]
```

专家：

```python
expert_patch = self.thermal_expert(image_tensors_ir)
expert_patch = self.thermal_adapter(
    expert_patch,
    target_hw=(patch_h, patch_w),
)  # [B,N,D]
```

CLS 可用 expert patch mean：

```python
expert_cls = expert_patch.mean(dim=1, keepdim=True)
expert_tokens = torch.cat([expert_cls, expert_patch], dim=1)
```

残差：

```python
rho = torch.tanh(self.thermal_residual_raw)
ir_bf = ir_bf + rho * expert_tokens
```

初始化：

```python
self.thermal_residual_raw = nn.Parameter(
    torch.tensor(args.expert_residual_init)
)
```

默认 `0.0`。

然后再：

```python
visu_src_ir = ir_bf.permute(1,0,2)
```

进入原有 IAFv3/GQR。

### 重要

不要第一版把 expert 注入到 CLIP 每一个 encoder layer。先在多层 CLIP 特征投影完成后注入，变量更少、更容易判断收益。

---

# 11. Phase 9：Target-aware alignment（可选）

仅当 `--enable_target_align` 开启。

模型在 training aux 中返回：

```python
aux["expert_patch_tokens"] = expert_patch
aux["tir_clip_patch_tokens"] = ir_bf[:, 1:, :]
aux["target_align_text"] = F.normalize(gqr_text_embed, dim=-1)
```

不要把 `obj_mask` 输入模型。

loss 中：

```python
mask = mdetr_interpolate(
    tgt_mask.float(),
    (patch_h, patch_w),
    mode="nearest",
)[:,0] > 0.5

mask = mask.flatten(1).float()
```

masked mean：

```python
def masked_pool(tokens, mask, eps=1e-6):
    w = mask.unsqueeze(-1)
    return (tokens * w).sum(1) / (w.sum(1) + eps)
```

得到：

```text
expert_target_proto
tir_target_proto
```

第一版只做 cosine：

```python
loss_target_align = (
    1 - F.cosine_similarity(
        F.normalize(expert_target_proto, dim=-1),
        F.normalize(text_embed, dim=-1),
        dim=-1,
    )
).mean()
```

如果稳定，再升级 batch InfoNCE。

---

# 12. Phase 10：优化器参数组

当前 `mmvg_train.py` 把 trainable 参数基本放在同一个 `args.lr` 下。

新增模块建议单独参数组：

```python
new_module_names = (
    "gqr",
    "rgb_aux_head",
    "tir_aux_head",
    "thermal_adapter",
    "thermal_residual",
    "gqr_text_proj",
)

base_params = []
new_params = []

for name, p in model_without_ddp.named_parameters():
    if not p.requires_grad:
        continue
    if any(k in name for k in new_module_names):
        new_params.append(p)
    else:
        base_params.append(p)

param_list = [
    {"params": base_params, "lr": args.lr},
    {"params": new_params, "lr": args.new_module_lr},
]
```

expert frozen 时：

```python
for p in self.thermal_expert.parameters():
    p.requires_grad_(False)
```

不要把 frozen expert 放入 optimizer。

初始推荐：

```text
base lr = 1e-4
new module lr = 5e-4
expert lr = 0
```

如果后续解冻 expert：

```text
expert lr <= 1e-5
```

---

# 13. Phase 11：Checkpoint 兼容性

官方 `--retrain` 会：

- map legacy `default` LoRA 到 `lora_rgb/lora_ir`；
- 使用 `strict=False`；
- 检查已有 base layer key。

新增 TSAR 参数后，missing keys 正常包含：

```text
gqr*
rgb_aux_head*
tir_aux_head*
thermal_expert*
thermal_adapter*
thermal_residual_raw
gqr_text_proj*
```

修改 checkpoint validation：

```python
allowed_new_prefixes = (
    "gqr",
    "rgb_aux_head",
    "tir_aux_head",
    "thermal_expert",
    "thermal_adapter",
    "thermal_residual_raw",
    "gqr_text_proj",
)
```

任何不属于这些前缀、也不属于原 legacy compatibility 的 missing key，都要报 warning 或 raise。

不要为了“加载成功”把所有异常静默过滤。

---

# 14. Phase 12：训练脚本

不要改官方：

```text
script_train/RGBT_VGNet/
```

新增：

```text
script_train/TSAR_Ground/
    ref_flir/train.sh
    ref_m3fd/train.sh
    ref_mfad/train.sh
    train_all.sh
```

以官方脚本复制为起点。

## 14.1 GQR-only 脚本

核心：

```bash
--model_name MMVGFusion \
--FusionMethod GQRv1 \
--enable_gqr \
--gqr_tau 0.25 \
--gqr_aux_weight 0.25 \
--gqr_router_weight 0.20 \
--gqr_start_epoch 5 \
--gqr_ramp_epochs 5 \
--gqr_eta_max 2.0 \
--new_module_lr 0.0005
```

其他参数和官方保持一致：

```text
LoRA 16/48
120 epochs
cosine
224
augmentation
contrastive
RTCC
mask loss
same pretrain checkpoint
```

## 14.2 TERA + GQR

额外：

```bash
--enable_thermal_expert \
--thermal_expert_type mspecgene \
--thermal_expert_ckpt "${THERMAL_EXPERT_CKPT}" \
--thermal_expert_dim 768 \
--freeze_thermal_expert \
--expert_residual_init 0.0
```

脚本中：

```bash
: "${THERMAL_EXPERT_CKPT:?Set THERMAL_EXPERT_CKPT}"
```

不要硬编码本机路径。

## 14.3 High-resolution

用环境变量：

```bash
IMGSIZE=288 BATCHSIZE=... bash ...
```

不要把 320 固定成默认。

---

# 15. Phase 13：测试

新增：

```text
tests/test_tsar_modules.py
tests/test_iafv3_equivalence.py
```

## 15.1 GQR shape

随机：

```text
B=2
N=196
D=512
```

检查：

```text
aux RGB box == [2,4]
aux TIR box == [2,4]
router logits == [2,2]
router prob.sum(-1) == 1
```

## 15.2 Zero-init equivalence

在相同 IAFv3 中间量下：

```text
eta_raw = 0
```

必须：

```python
torch.allclose(
    fused_gqr,
    fused_iafv3,
    atol=1e-6,
    rtol=1e-5,
)
```

如果失败，先修复，不训练。

## 15.3 Expert residual equivalence

```text
thermal_residual_raw = 0
```

必须：

```text
TIR enhanced tokens == original TIR tokens
```

## 15.4 Disabled-feature smoke test

命令不带新 flag，随机/最小 batch forward：

```text
model returns exactly 5 values
no KeyError
no changed tensor shape
```

## 15.5 Training output smoke

GQR 开启：

```text
model.train()
returns 6 values
aux contains required keys
all new losses finite
backward succeeds
```

## 15.6 Eval smoke

GQR 开启：

```text
model.eval()
returns 5 values
validation code unchanged
```

---

# 16. Phase 14：实验顺序，Codex 不要一次全部开启

严格按：

```text
E0 official baseline
E1 baseline refactor equivalence
E2 GQR only
E3 TERA only
E4 GQR + TERA
E5 + Target-aware alignment
E6 best + 288
E7 best + 320
```

每个实验保存独立：

```text
output_dir
args
git commit hash
best checkpoint
log.txt
```

随机种子至少：

```text
13
42
3407
```

如果算力有限，先单 seed 筛选，再对最终 E0/E2/E3/E4 做 3 seeds。

---

# 17. Go / No-Go 标准

## GQR 保留

满足至少：

```text
平均 Acc@0.5 +0.5
且任一数据集下降 <=0.3
```

最好看到：

```text
TestB / TestC 更明显上涨
```

## TERA 保留

必须看到至少：

```text
TIR-only auxiliary grounding quality 上升
或 low-light / small subset 明确提升
并且主 Test 不退化
```

如果只增加参数但主指标不涨，删除。

## Target Align 保留

必须在：

```text
TERA + GQR
```

上额外有可重复增益；否则不保留。

## HR

单独报告，不与结构贡献混为一谈。

---

# 18. 必须增加的日志

训练每个 epoch 记录：

```text
loss_aux_rgb
loss_aux_tir
loss_gqr_router
router_rgb_mean
router_tir_mean
gqr_eta
w_rgb_base_mean
w_rgb_final_mean
thermal_rho
expert_residual_ratio
```

如果能从 batch metadata 读取 weak-light/small 标签，再额外统计：

```text
router_tir_mean_lowlight
router_tir_mean_normal
acc_small
acc_non_small
```

不要为了这些诊断改动官方 metric 定义。

---

# 19. 绝对禁止的实现

Codex 不要：

1. 删除 `IAFv3`；
2. 把 TIR CLIP 直接替换为 M-SpecGene；
3. 把 GT box/mask 传进 `model.forward()` 决定 inference fusion；
4. 在 evaluation 使用 teacher reliability；
5. 改官方 `script_train/RGBT_VGNet/*`；
6. 为适配 expert 直接把整个 M-SpecGene/MMEngine 工程复制到主 repo；
7. 用 `strict=False` 后完全不检查 missing/unexpected keys；
8. 在第一版同时加入 GQR、TERA、target alignment、HR，导致无法做消融；
9. 改变 `MMVGFusion` 默认 `FusionMethod=IAFv3` 行为；
10. 未通过 zero-init equivalence test 就开始大规模训练。

---

# 20. Codex 第一轮执行任务

把下面内容直接作为第一轮 Codex prompt：

```text
Work in the root of crazyxiaoxi/RGBT-GroundBench.

Goal: implement Phase 1-7 of TSAR-Ground, GQR only. Do NOT implement the external thermal expert yet.

Hard constraints:
1. Preserve current MMVGFusion + IAFv3 behavior when all new flags are disabled.
2. Do not modify official script_train/RGBT_VGNet scripts.
3. Do not remove or replace AMA, LAVS, or IAFv3.
4. GQR must augment IAFv3 by a zero-initialized logit-space reliability correction.
5. GT may only be used in loss construction, never in inference-time fusion.
6. Validation must keep the existing 5-output model contract.
7. Training may return a sixth aux dict when GQR is enabled.
8. Add tests before considering the implementation complete.

Tasks:
A. Inspect models/mmvg_fusion.py, train_val/mmvg_train.py, engine.py, utils/loss_utils.py and verify the current interfaces.
B. Add GQR CLI flags to mmvg_train.py with defaults that leave the baseline unchanged.
C. Create models/tsar_modules.py with:
   - TextConditionedModalityHead
   - GroundingQualityRouter
D. Refactor IAFv3 only enough to expose its base RGB patch weights while preserving numerical behavior.
E. Add FusionMethod=GQRv1:
   - use RGB/TIR auxiliary language-conditioned grounding heads
   - predict sample-level RGB/TIR reliability
   - apply zero-init logit residual to IAFv3 patch weights
   - preserve the original IAFv3 difference residual term
F. During training only, expose aux:
   rgb_aux_box, tir_aux_box, router_logits,
   w_rgb_base_mean, w_rgb_final_mean.
G. Update engine.py to support both legacy 5-output and TSAR 6-output training forwards.
H. Extend trans_vg_loss with optional aux and epoch arguments:
   - auxiliary RGB/TIR box losses
   - detached IoU soft teacher
   - KL router loss
   - 5-epoch warmup/ramp
I. Add:
   tests/test_tsar_modules.py
   tests/test_iafv3_equivalence.py
J. Add script_train/TSAR_Ground/ref_flir/train.sh for GQR-only by copying the official settings and changing only GQR-related arguments/output path.

Acceptance:
- baseline flags off -> 5 outputs and unchanged shapes
- GQR train -> 6 outputs
- GQR eval -> 5 outputs
- eta=0 -> GQR fused tensor matches IAFv3 within atol 1e-6 / rtol 1e-5
- router probabilities sum to 1
- losses are finite
- a one-batch backward smoke test succeeds
- summarize modified files and exact commands used for tests
- do not start full training automatically
```

---

# 21. Codex 第二轮执行任务：接入 TERA

只有第一轮通过后，再给 Codex：

```text
Continue from the tested GQR implementation.

Now implement TERA (Thermal Expert Residual Adapter) without changing the GQR behavior.

Requirements:
1. Keep the existing TIR CLIP + LoRA(rank=48) path.
2. Add an optional external thermal/RGBT expert; preferred backend is M-SpecGene ViT-B.
3. Never download weights automatically. Require --thermal_expert_ckpt.
4. Keep the expert frozen by default.
5. Adapt expert features to the current CLIP patch grid and 512 hidden dimensions.
6. Inject expert features after the projected TIR multi-layer CLIP representation and before IAFv3/GQR fusion.
7. Use thermal_residual_raw initialized to 0:
   rho = tanh(thermal_residual_raw)
   tir_enhanced = tir_clip + rho * expert_tokens
8. When rho=0, enhanced TIR tokens must numerically match the original TIR tokens.
9. Do not inject expert features into every CLIP block in this iteration.
10. Do not implement target-aware alignment yet.

Tasks:
- add ThermalExpertEncoder and ThermalExpertAdapter in models/tsar_modules.py
- add CLI args
- inspect the actual downloaded M-SpecGene checkpoint keys before writing the loader
- report missing/unexpected keys explicitly
- integrate the residual before fusion
- return thermal_rho and expert_residual_ratio in training aux
- add shape/equivalence tests
- add TERA-only and TERA+GQR training script variants
- do not run full training automatically
```

---

# 22. Codex 第三轮：Target-aware alignment + HR

只在前两轮指标有效后执行。

```text
Implement optional target-aware thermal-text alignment as a training-only loss.
Do not feed target masks into model.forward().
Return expert patch tokens and TIR CLIP patch tokens through aux.
Use the existing obj_mask in trans_vg_loss to downsample the GT target region to the patch grid.
Start with masked target prototype cosine alignment and weight 0.05.
Keep the feature disabled by default.

Then add high-resolution script variants for IMGSIZE=288 and IMGSIZE=320.
Do not change the default 224 setting.
Confirm the existing Modified_CLIPVisionEmbeddings positional interpolation and token/mask shapes work at both resolutions.
```

---

# 23. 对 Codex 输出的人工检查清单

Codex 修改完成后人工检查：

- [ ] `models/mmvg_fusion.py` 默认路径是否仍是原始 IAFv3；
- [ ] 是否错误把 GQR teacher 放进 model forward；
- [ ] 是否 `detach()` auxiliary boxes 后才计算 teacher；
- [ ] 是否 auxiliary heads 自身仍有 bbox loss；
- [ ] `eta=0` 是否真的等价 baseline；
- [ ] `rho=0` 是否真的等价原 TIR tokens；
- [ ] 是否因第 6 个返回值破坏 validate；
- [ ] 是否修改了官方 RGBT_VGNet 脚本；
- [ ] checkpoint missing keys 是否可解释；
- [ ] external expert 是否被 optimizer 意外训练；
- [ ] expert 输入预处理是否与 M-SpecGene checkpoint 要求一致；
- [ ] 224/288/320 时 `visu_mask`、pos embed、patch token 数是否一致；
- [ ] DDP 下 `find_unused_parameters=True` 是否仍能处理关闭的模块；
- [ ] 日志是否能区分 baseline、GQR、TERA、组合实验。

---

# 24. 实现完成后建议的运行顺序

```bash
# 1. 官方 baseline
bash script_train/RGBT_VGNet/ref_flir/train.sh

# 2. GQR only
bash script_train/TSAR_Ground/ref_flir/train.sh

# 3. TERA only
TSAR_VARIANT=tera \
THERMAL_EXPERT_CKPT=/path/to/M-SpecGene_VIT-B.pth \
bash script_train/TSAR_Ground/ref_flir/train.sh

# 4. GQR + TERA
TSAR_VARIANT=tera_gqr \
THERMAL_EXPERT_CKPT=/path/to/M-SpecGene_VIT-B.pth \
bash script_train/TSAR_Ground/ref_flir/train.sh

# 5. best + 288
IMGSIZE=288 BATCHSIZE=... \
THERMAL_EXPERT_CKPT=/path/to/M-SpecGene_VIT-B.pth \
bash script_train/TSAR_Ground/ref_flir/train.sh
```

实际脚本环境变量命名以 Codex 最终实现为准，但必须保持官方脚本可直接运行。

---

# 25. 参考链接

RGBT-GroundBench:
https://github.com/crazyxiaoxi/RGBT-GroundBench

RGBT-GroundBench paper:
https://arxiv.org/abs/2512.24561

M-SpecGene:
https://github.com/CalayZhou/M-SpecGene

InfMAE:
https://github.com/liufangcen/InfMAE
