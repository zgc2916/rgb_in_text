"""C1 infrared detail compensation and independent-memory boundary localization.

Forward paths never take ground truth. Masks use True for padding throughout.
All residual output heads start at zero so a warm start preserves C1 exactly.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

NEW_MODULE_PREFIXES = ('tir_detail.', 'tir_injector.', 'tir_region_head.', 'boundary_refiner.')


def resize_padding(mask, size):
    # Conservative: any padded pixel in a feature cell invalidates that cell.
    return F.adaptive_max_pool2d(mask[:, None].float(), size)[:, 0].bool()


def conv_gn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU())


class DetailBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.GroupNorm(8, channels * 2), nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1,
                      groups=channels * 2, bias=False),
            nn.GroupNorm(8, channels * 2), nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, bias=False), nn.GroupNorm(8, channels))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class TIRDetailEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv_gn(1, 16, 2), conv_gn(16, 64, 2))
        self.stage4 = nn.Sequential(DetailBlock(64), DetailBlock(64))
        self.stage8 = nn.Sequential(conv_gn(64, 96, 2), DetailBlock(96), DetailBlock(96))
        self.stage16 = nn.Sequential(conv_gn(96, 128, 2), DetailBlock(128), DetailBlock(128))

    def forward(self, tir, padding_mask):
        padding_mask = padding_mask.bool()
        x = tir.masked_fill(padding_mask[:, None], 0)
        features, masks = [], []
        x = self.stem(x)
        for stage in (self.stage4, self.stage8, self.stage16):
            x = stage(x)
            # Keep partial cells: a tiny target at the image/padding boundary
            # must still have a feature and an area-overlap learning signal.
            mask = F.adaptive_avg_pool2d(padding_mask[:, None].float(), x.shape[-2:])[:, 0] >= 1
            x = x.masked_fill(mask[:, None], 0)
            features.append(x)
            masks.append(mask)
        return tuple(features), tuple(masks)


def patch_cells(feature, grid_size):
    """B,C,H,W -> B,patch,local_cell,C, in CLIP row-major order."""
    b, c, h, w = feature.shape
    gh, gw = grid_size
    if h % gh or w % gw:
        raise ValueError('Detail scales must align with the CLIP patch grid')
    return feature.reshape(b, c, gh, h // gh, gw, w // gw).permute(
        0, 2, 4, 3, 5, 1).reshape(b, gh * gw, -1, c)


def safe_attention(module, query, evidence, invalid):
    """Avoid softmax(all -inf); return exact zero for empty evidence rows."""
    empty = invalid.all(dim=-1)
    safe_mask = invalid.clone()
    safe_mask[empty, 0] = False
    evidence = evidence.masked_fill(invalid[..., None], 0)
    out = module(query, evidence, evidence, key_padding_mask=safe_mask,
                 need_weights=False)[0]
    return out.masked_fill(empty[:, None, None], 0), ~empty


class TextLocalTIRInjector(nn.Module):
    def __init__(self, vision_dim=768, text_dim=512, levels=4):
        super().__init__()
        self.token_norm = nn.LayerNorm(vision_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.token_query = nn.Linear(vision_dim, 128)
        self.text_query = nn.Linear(text_dim, 128)
        self.keys = nn.ModuleList(nn.Linear(c, 128) for c in (64, 96, 128))
        self.values = nn.ModuleList(nn.Linear(c, 128) for c in (64, 96, 128))
        self.scale_embed = nn.Parameter(torch.randn(3, 128) * .02)
        self.gate = nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 1))
        self.outputs = nn.ModuleList(nn.Linear(128, vision_dim) for _ in range(levels))
        for output in self.outputs:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, levels, text_eos, features, masks):
        patches = levels[0].shape[1] - 1
        side = math.isqrt(patches)
        if side * side != patches or len(levels) != len(self.outputs):
            raise ValueError('Expected four square CLIP patch sequences with CLS')
        local = [patch_cells(x, (side, side)) for x in features]
        invalid = torch.cat([patch_cells(m[:, None], (side, side))[..., 0]
                             for m in masks], dim=2)
        k = torch.cat([proj(x) + self.scale_embed[i] for i, (proj, x)
                       in enumerate(zip(self.keys, local))], dim=2)
        v = torch.cat([proj(x) for proj, x in zip(self.values, local)], dim=2)
        # B,P,21,4,32 -> B,P,4,21,32; 4 shared attention heads.
        k = k.reshape(*k.shape[:3], 4, 32).transpose(2, 3)
        v = v.reshape(*v.shape[:3], 4, 32).transpose(2, 3)
        text_q = self.text_query(self.text_norm(text_eos.float()))[:, None].expand(-1, patches, -1)
        result = []
        empty = invalid.all(-1)
        safe_mask = invalid.clone()
        safe_mask[..., 0] &= ~empty
        for tokens, output in zip(levels, self.outputs):
            token_q = self.token_query(self.token_norm(tokens[:, 1:].float()))
            q = (token_q + text_q).reshape(*token_q.shape[:2], 4, 1, 32)
            scores = (q * k).sum(-1) / math.sqrt(32)
            scores = scores.masked_fill(safe_mask[:, :, None], -torch.inf)
            weights = scores.float().softmax(-1).to(v.dtype)
            context = (weights[..., None] * v).sum(-2).reshape(*token_q.shape[:2], 128)
            context = context.masked_fill(empty[..., None], 0)
            gate = self.gate(torch.cat((token_q, text_q, context), -1)).sigmoid()
            residual = (gate * output(context)).masked_fill(empty[..., None], 0)
            result.append(torch.cat((tokens[:, :1], tokens[:, 1:] + residual.to(tokens.dtype)), 1))
        return result


class TextRegionHead(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, 64)

    def forward(self, f4, text_eos):
        query = F.normalize(self.text_projection(text_eos.float()), dim=-1)
        return (F.normalize(f4.float(), dim=1) * query[:, :, None, None]).sum(1, keepdim=True) * 8


def boundary_grid(box, image_size=224):
    """Four deterministic 3x3 bands, coordinates normalized to [0,1]."""
    cx, cy, w, h = box.unbind(-1)
    # A sampling-only floor keeps tiny boxes from collapsing all nine points.
    sw, sh = w.clamp_min(4 / image_size), h.clamp_min(4 / image_size)
    offsets = box.new_tensor([-1., 0., 1.])
    normal, tangent = torch.meshgrid(offsets, offsets, indexing='ij')
    normal, tangent = normal.flatten(), tangent.flatten()
    left = torch.stack((cx[:, None] - w[:, None] / 2 + normal * sw[:, None] * .125,
                        cy[:, None] + tangent * sh[:, None] * .5), -1)
    right = left + torch.stack((w, torch.zeros_like(w)), -1)[:, None]
    top = torch.stack((cx[:, None] + tangent * sw[:, None] * .5,
                       cy[:, None] - h[:, None] / 2 + normal * sh[:, None] * .125), -1)
    bottom = top + torch.stack((torch.zeros_like(h), h), -1)[:, None]
    return torch.stack((left, right, top, bottom), 1)


def sample_memory(feature, padding_mask, grid):
    # Float32 grid sampling is robust under AMP and differentiable wrt boxes.
    normalized = grid.float() * 2 - 1
    values = F.grid_sample(feature.float(), normalized, align_corners=False,
                           padding_mode='zeros').permute(0, 2, 3, 1)
    # Bilinear support touching a padded cell is invalid (not nearest lookup).
    valid_weight = F.grid_sample((~padding_mask)[:, None].float(), normalized,
                                 align_corners=False, padding_mode='zeros')[:, 0]
    invalid = (valid_weight < 1 - 1e-6) | ((grid < 0) | (grid > 1)).any(-1)
    return values, invalid


class BoundaryRefiner(nn.Module):
    def __init__(self, with_detail=False, image_size=224):
        super().__init__()
        self.image_size = image_size
        self.with_detail = with_detail
        self.reg_projection = nn.Linear(512, 256)
        self.eos_projection = nn.Linear(512, 256)
        self.box_projection = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 256))
        self.side_embed = nn.Parameter(torch.randn(4, 256) * .02)
        self.text_projection = nn.Linear(512, 256)
        self.memory_projections = nn.ModuleList(nn.Linear(c, 256)
                                               for c in ((512, 512, 64, 96) if with_detail else (512, 512)))
        self.source_embed = nn.Parameter(torch.randn(len(self.memory_projections), 256) * .02)
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(4))
        self.self_attention = nn.MultiheadAttention(256, 4, dropout=.1, batch_first=True)
        self.text_attention = nn.MultiheadAttention(256, 4, dropout=.1, batch_first=True)
        self.visual_attention = nn.MultiheadAttention(256, 4, dropout=.1, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 256))
        self.dropout = nn.Dropout(.1)
        self.output = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, reg, eos, text, text_mask, box, memories, masks,
                details=(), detail_masks=(), disable_independent_memory=False,
                image_padding_mask=None):
        sources = tuple(memories) + (tuple(details[:2]) if self.with_detail else ())
        source_masks = tuple(masks) + (tuple(detail_masks[:2]) if self.with_detail else ())
        if len(sources) != len(self.memory_projections):
            raise ValueError('Boundary memory sources do not match configured scales')
        text = self.text_projection(text.float())
        state = self.reg_projection(reg.float())[:, None] + self.eos_projection(eos.float())[:, None] + self.side_embed
        boxes = []
        for _ in range(2):
            q = state + self.box_projection(box.float())[:, None]
            nq = self.norms[0](q)
            q = q + self.dropout(self.self_attention(nq, nq, nq, need_weights=False)[0])
            tq, _ = safe_attention(self.text_attention, self.norms[1](q), text, text_mask.bool())
            q = q + self.dropout(tq)
            grid = boundary_grid(box, self.image_size)
            if image_padding_mask is not None:
                valid_pixels = F.grid_sample((~image_padding_mask.bool())[:, None].float(), grid.float() * 2 - 1,
                                              align_corners=False, padding_mode='zeros')[:, 0]
                pixel_invalid = valid_pixels < 1 - 1e-6
            else:
                pixel_invalid = torch.zeros(grid.shape[:-1], device=grid.device, dtype=torch.bool)
            values, invalid = [], []
            for i, (feature, mask, projection) in enumerate(zip(sources, source_masks, self.memory_projections)):
                value, bad = sample_memory(feature, mask, grid)
                bad = bad | pixel_invalid
                if disable_independent_memory and i < 2:
                    bad = torch.ones_like(bad)
                values.append(projection(value) + self.source_embed[i])
                invalid.append(bad)
            evidence = torch.cat(values, 2).flatten(0, 1)
            bad = torch.cat(invalid, 2).flatten(0, 1)
            vq, valid = safe_attention(self.visual_attention, self.norms[2](q).flatten(0, 1)[:, None], evidence, bad)
            q = q + self.dropout(vq[:, 0].reshape_as(q))
            state = q + self.dropout(self.ffn(self.norms[3](q)))
            delta = self.output(state).squeeze(-1).tanh() * .25
            delta = delta * box[:, [2, 2, 3, 3]] * valid.reshape(-1, 4)
            # Scale valid-side moves together if they would put a center or
            # size outside the normalized range. Scaling keeps an invalid
            # side's zero displacement exact, unlike clamping the final box.
            scales = []
            for axis, indices in enumerate(((0, 1), (2, 3))):
                da, db = delta[:, indices[0]], delta[:, indices[1]]
                dc, ds = (da + db) / 2, db - da
                center, extent = box[:, axis], box[:, axis + 2]
                center_limit = torch.where(dc >= 0, (1 - center) / dc.abs().clamp_min(1e-12),
                                           center / dc.abs().clamp_min(1e-12))
                size_limit = torch.where(ds >= 0, (1 - extent) / ds.abs().clamp_min(1e-12),
                                         (extent - 1e-7).clamp_min(0) / ds.abs().clamp_min(1e-12))
                scale = torch.minimum(center_limit, size_limit).clamp(0, 1)
                scales.extend((scale, scale))
            delta = delta * torch.stack(scales, -1)
            dl, dr, dt, db = delta.unbind(-1)
            update = torch.stack(((dl + dr) / 2, (dt + db) / 2, dr - dl, db - dt), -1)
            box = box + update
            boxes.append(box)
        return boxes[-1], {'boundary_b0': None, 'boundary_b1': boxes[0]}


def soft_box_occupancy(boxes, size):
    """Exact area intersection per normalized feature cell; survives tiny GT."""
    h, w = size
    cx, cy, bw, bh = boxes.float().unbind(-1)
    x0, x1 = cx - bw / 2, cx + bw / 2
    y0, y1 = cy - bh / 2, cy + bh / 2
    xs = torch.arange(w, device=boxes.device, dtype=torch.float32) / w
    ys = torch.arange(h, device=boxes.device, dtype=torch.float32) / h
    dx = (torch.minimum(x1[:, None], xs + 1 / w) - torch.maximum(x0[:, None], xs)).clamp_min(0) * w
    dy = (torch.minimum(y1[:, None], ys + 1 / h) - torch.maximum(y0[:, None], ys)).clamp_min(0) * h
    return (dy[:, :, None] * dx[:, None]).clamp(0, 1)


def balanced_region_bce(logits, boxes, padding_mask, valid_fraction=None):
    logits = logits[:, 0].float()
    target = soft_box_occupancy(boxes.detach(), logits.shape[-2:])
    valid = (~padding_mask).float()
    if valid_fraction is not None:
        valid = valid * valid_fraction.float()
    positive = target * valid
    negative = (1 - target) * valid
    pd, nd = positive.sum((1, 2)), negative.sum((1, 2))
    pl = (F.softplus(-logits) * positive).sum((1, 2)) / pd.clamp_min(1e-12)
    nl = (F.softplus(logits) * negative).sum((1, 2)) / nd.clamp_min(1e-12)
    count = (pd > 0).float() + (nd > 0).float()
    return ((pl * (pd > 0) + nl * (nd > 0)) / count.clamp_min(1)).mean()


def parameter_report(model):
    counts = {prefix[:-1]: sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))
              for prefix in NEW_MODULE_PREFIXES}
    counts['infrared_total'] = sum(counts[p] for p in ('tir_detail', 'tir_injector', 'tir_region_head'))
    counts['net_added'] = counts['infrared_total'] + counts['boundary_refiner']
    if counts['infrared_total'] > 2_000_000 or counts['net_added'] > 5_000_000:
        raise ValueError(f'Architecture exceeds the approved parameter budget: {counts}')
    return counts
