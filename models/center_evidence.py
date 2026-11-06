"""Text-conditioned stride-4 RGB/TIR center evidence for a coarse box.

The model receives only images, padding, text/REG features and a predicted
coarse box. Ground truth is accepted exclusively by the separate loss.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .tir_detail_refine import DetailBlock, conv_gn


def normalized_grid(height, width, device, dtype):
    """Return row-major normalized pixel-center coordinates in x,y order."""
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing='ij')
    return torch.stack(((x + .5) / width, (y + .5) / height), dim=-1).to(dtype)


def masked_spatial_softmax(logits, valid):
    """Probability over valid cells; fully padded examples return exact zeros."""
    if logits.ndim != 3 or valid.shape != logits.shape:
        raise ValueError('Spatial logits and validity must have shape B,H,W')
    flat = logits.flatten(1)
    mask = valid.bool().flatten(1)
    support = mask.any(-1)
    safe = mask.clone()
    safe[:, 0] |= ~support
    weights = flat.masked_fill(~safe, -1e4).softmax(-1) * mask.to(flat.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights.reshape_as(logits), support


class Stride4Encoder(nn.Module):
    """The stem/stage4 schema matches the existing trained TIR detail model."""
    def __init__(self, input_channels):
        super().__init__()
        self.stem = nn.Sequential(conv_gn(input_channels, 16, 2), conv_gn(16, 64, 2))
        self.stage4 = nn.Sequential(DetailBlock(64), DetailBlock(64))

    def forward(self, images):
        return self.stage4(self.stem(images))


class CenterEvidenceHead(nn.Module):
    """Local continuous center evidence with a zero-initialized box residual.

    Input images are the already normalized B,4,H,W RGB/TIR canvas. Both modes
    use the same text/REG-conditioned spatial scoring; ``text_gate`` also
    learns per-expression RGB/TIR weights, whereas ``fixed`` uses 0.5/0.5.
    ``aux`` supplies finite logits and a validity mask for training supervision.
    """
    def __init__(self, mode='fixed', image_size=224, hidden=64, window_scale=3.,
                 min_window_pixels=64., min_shift_pixels=12., max_shift_pixels=32.):
        super().__init__()
        if mode not in ('fixed', 'text_gate'):
            raise ValueError('Expected fixed or text_gate mode')
        if image_size < 4 or image_size % 4 or hidden < 1:
            raise ValueError('Image size must be divisible by four and hidden positive')
        if (window_scale <= 0 or min_window_pixels <= 0 or min_shift_pixels <= 0
                or max_shift_pixels <= 0 or min_shift_pixels > max_shift_pixels):
            raise ValueError('Window and displacement limits must be positive')
        self.mode, self.image_size = mode, int(image_size)
        self.window_scale = float(window_scale)
        self.min_window = min(float(min_window_pixels) / image_size, 1.)
        self.min_shift = min(float(min_shift_pixels) / image_size, 1.)
        self.max_shift = min(float(max_shift_pixels) / image_size, 1.)
        self.rgb_encoder, self.tir_encoder = Stride4Encoder(3), Stride4Encoder(1)
        self.text_norm, self.reg_norm = nn.LayerNorm(512), nn.LayerNorm(512)
        self.text_query, self.reg_query = nn.Linear(512, hidden), nn.Linear(512, hidden)
        self.rgb_projection = nn.Conv2d(64, hidden, 1)
        self.tir_projection = nn.Conv2d(64, hidden, 1)
        self.route = nn.Sequential(nn.Linear(1024, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.route[-1].weight)
        nn.init.zeros_(self.route[-1].bias)
        if mode == 'fixed':
            for parameter in self.route.parameters():
                parameter.requires_grad_(False)
        self.offset = nn.Sequential(nn.Linear(2 * hidden + 4, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, images, padding_mask, text, reg, coarse_box):
        if images.ndim != 4 or images.shape[1:] != (4, self.image_size, self.image_size):
            raise ValueError('Expected normalized B,4,image_size,image_size RGB/TIR images')
        batch = images.shape[0]
        if (padding_mask.shape != (batch, self.image_size, self.image_size)
                or text.shape != (batch, 512) or reg.shape != (batch, 512)
                or coarse_box.shape != (batch, 4)):
            raise ValueError('Padding, language, REG or coarse-box shape is invalid')
        padding = padding_mask.bool()
        clean = images.masked_fill(padding[:, None], 0.)
        rgb = self.rgb_projection(self.rgb_encoder(clean[:, :3].float()))
        tir = self.tir_projection(self.tir_encoder(clean[:, 3:4].float()))
        height, width = rgb.shape[-2:]
        if tir.shape != rgb.shape:
            raise AssertionError('Modality detail grids differ')
        valid = F.adaptive_avg_pool2d((~padding)[:, None].float(), (height, width))[:, 0] > 0
        norm_text, norm_reg = self.text_norm(text.float()), self.reg_norm(reg.float())
        query = F.normalize(self.text_query(norm_text) + self.reg_query(norm_reg), dim=-1, eps=1e-6)
        rgb_score = (F.normalize(rgb, dim=1, eps=1e-6) * query[:, :, None, None]).sum(1) * 8
        tir_score = (F.normalize(tir, dim=1, eps=1e-6) * query[:, :, None, None]).sum(1) * 8
        if self.mode == 'text_gate':
            weights = self.route(torch.cat((norm_text, norm_reg), dim=-1)).softmax(-1)
        else:
            weights = rgb.new_full((batch, 2), .5)
        score = weights[:, 0, None, None] * rgb_score + weights[:, 1, None, None] * tir_score
        grid = normalized_grid(height, width, score.device, score.dtype)
        center, size = coarse_box[:, :2].float(), coarse_box[:, 2:].float()
        window = torch.maximum(size.clamp_min(0.) * self.window_scale,
                               size.new_full(size.shape, self.min_window)).clamp_max(1.)
        displacement = grid[None] - center[:, None, None]
        in_window = (displacement.abs() <= window[:, None, None] / 2).all(-1)
        local_valid = valid & in_window
        # If a predicted coarse center is outside the valid image, retain a
        # finite global fallback instead of assigning probability to padding.
        allowed = torch.where(local_valid.flatten(1).any(-1)[:, None, None], local_valid, valid)
        sigma = (window / 4).clamp_min(1. / max(height, width))
        prior = -.5 * (displacement / sigma[:, None, None]).square().sum(-1)
        logits = score + prior
        probability, has_evidence = masked_spatial_softmax(logits, allowed)
        prior_probability, _ = masked_spatial_softmax(prior, allowed)
        candidate = (probability[..., None] * grid).sum((1, 2))
        reference = (prior_probability[..., None] * grid).sum((1, 2))
        candidate = torch.where(has_evidence[:, None], candidate, center)
        reference = torch.where(has_evidence[:, None], reference, center)
        feature = weights[:, 0, None, None, None] * rgb + weights[:, 1, None, None, None] * tir
        pooled = (feature * probability[:, None]).sum((2, 3))
        raw = self.offset(torch.cat((candidate-reference, candidate-center, pooled, query), dim=-1))
        # Bound each axis by half the coarse-box size, with a configurable
        # minimum displacement so tiny targets can still move by a few pixels.
        # The 32-pixel default cap is independent of ground truth.
        limit = torch.maximum(size.clamp_min(0.) * .5, size.new_full(size.shape, self.min_shift))
        limit = limit.clamp_max(self.max_shift)
        delta = limit * raw.tanh() * has_evidence[:, None]
        refined_center = (center + delta).clamp(0., 1.)
        refined = torch.cat((refined_center.to(coarse_box.dtype), coarse_box[:, 2:]), dim=-1)
        return refined, {'heatmap_logits': logits, 'valid_mask': valid, 'allowed_mask': allowed,
                         'probability': probability, 'candidate_center': candidate,
                         'reference_center': reference, 'center_delta': refined_center-center,
                         'modality_weights': weights, 'has_evidence': has_evidence,
                         'window_extent': window}

    def initialize_tir_from_detail_checkpoint(self, path, expected_sha256=None):
        """Strictly copy trained ``tir_detail`` stem/stage4, never alter the file."""
        path = Path(path).resolve()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        checksum = digest.hexdigest()
        if expected_sha256 is not None and checksum != expected_sha256:
            raise ValueError('Detail checkpoint checksum differs')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model', checkpoint)
        expected = self.tir_encoder.state_dict()
        prefixes = ('tir_detail.', 'base.tir_detail.', 'module.tir_detail.', 'module.base.tir_detail.')
        matching = [prefix for prefix in prefixes if all(prefix + name in state for name in expected)]
        if len(matching) != 1:
            raise ValueError('Detail checkpoint lacks a unique complete TIR stem/stage4')
        copied = {name: state[matching[0] + name] for name in expected}
        if any(copied[name].shape != value.shape or copied[name].dtype != value.dtype
               for name, value in expected.items()):
            raise ValueError('Detail TIR tensor schema differs')
        self.tir_encoder.load_state_dict(copied, strict=True)
        return {'checkpoint': str(path), 'sha256': checksum, 'source_prefix': matching[0],
                'tensors': len(copied), 'elements': sum(value.numel() for value in copied.values())}


def center_heatmap_loss(aux, target_boxes, *, return_stats=False):
    """Train center evidence using GT only in this loss, never in forward.

    Supervise a GT-sized Gaussian on the valid local window. If the GT center
    cannot be represented in the window or image, skip that sample rather
    than training a truncated location as if it were the true center.
    """
    logits, allowed = aux['heatmap_logits'], aux['allowed_mask'].bool()
    if logits.ndim != 3 or allowed.shape != logits.shape or target_boxes.shape != (logits.shape[0], 4):
        raise ValueError('Center supervision expects B,H,W logits and B,4 boxes')
    batch, height, width = logits.shape
    target = target_boxes.float()
    center, size = target[:, :2], target[:, 2:]
    finite = torch.isfinite(target).all(-1)
    inside = finite & (center >= 0).all(-1) & (center < 1).all(-1) & (size > 0).all(-1)
    safe_center = torch.where(finite[:, None], center, torch.zeros_like(center))
    ix = (safe_center[:, 0] * width).long().clamp(0, width-1)
    iy = (safe_center[:, 1] * height).long().clamp(0, height-1)
    eligible = inside & allowed[torch.arange(batch, device=logits.device), iy, ix]
    grid = normalized_grid(height, width, logits.device, logits.dtype)
    safe_size = torch.where(finite[:, None], size, torch.ones_like(size))
    sigma = torch.stack(((safe_size[:, 0] * width / 6).clamp(.5, 2.) / width,
                         (safe_size[:, 1] * height / 6).clamp(.5, 2.) / height), dim=-1)
    distance = (grid[None] - safe_center[:, None, None]) / sigma[:, None, None]
    target_map = (-.5 * distance.square().sum(-1)).exp() * allowed.to(logits.dtype)
    target_map = target_map / target_map.sum((1, 2), keepdim=True).clamp_min(1e-12)
    mask = allowed.flatten(1)
    safe = mask.clone()
    safe[:, 0] |= ~mask.any(-1)
    log_probability = logits.flatten(1).masked_fill(~safe, -1e4).log_softmax(-1)
    per_sample = -(target_map.flatten(1) * log_probability).sum(-1)
    loss = (per_sample * eligible).sum() / eligible.sum().clamp_min(1)
    if return_stats:
        return loss, {'eligible_count': eligible.sum().detach(),
                      'eligible_fraction': eligible.float().mean().detach()}
    return loss
