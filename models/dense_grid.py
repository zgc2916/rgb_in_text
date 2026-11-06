"""Dense visual grids with the original C1 checkpoint parameter layout."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.late_adaptation import LateAdaptationGrounder, parameter_group as late_group


def parameter_group(name):
    return 'reliability' if name.startswith('base.dense_reliability.') else late_group(name)


def grounding_positions(weight, grid, text_length=77, source_grid=14):
    """Preserve REG/CLS/text positions; interpolate only the used visual grid.

    C1's RGB-T table reserves a second, unused image grid after its text
    positions. Never interpret that tail as additional spatial positions.
    """
    if grid <= 0 or source_grid <= 0:
        raise ValueError('Positive square visual grid required')
    end = 2 + source_grid**2
    if weight.shape[0] < end + text_length:
        raise ValueError('C1 positional table is incomplete')
    if grid == source_grid:
        return weight[:end + text_length]
    visual = weight[2:end].T.reshape(1, weight.shape[1], source_grid, source_grid)
    visual = F.interpolate(visual.float(), size=(grid, grid), mode='bicubic',
                           align_corners=False, antialias=True).to(weight.dtype)
    return torch.cat((weight[:2], visual.flatten(2).squeeze(0).T,
                      weight[end:end + text_length]), dim=0)


class DenseTextReliability(nn.Module):
    """Bounded foreground-relative RGB/TIR correction after the existing IAFv3.

    No boxes enter this module. Padding never supplies background statistics.
    Zero initialization reproduces the plain dense-grid model exactly.
    """
    def __init__(self, hidden=32, bound=.25):
        super().__init__()
        self.bound = bound
        self.predict = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, fused, rgb, tir, text, padding):
        if text is None or padding is None:
            raise ValueError('Text and spatial padding are required')
        r, t = rgb[1:].transpose(0, 1), tir[1:].transpose(0, 1)
        side = math.isqrt(r.shape[1])
        if side**2 != r.shape[1] or padding.shape != r.shape[:2]:
            raise ValueError('Expected a square patch grid and matching padding')
        valid = (~padding.bool()).to(r.dtype)
        rn, tn = F.normalize(r.float(), dim=-1), F.normalize(t.float(), dim=-1)
        query = F.normalize(text.detach().float(), dim=-1).unsqueeze(1)
        sr, st = (rn * query).sum(-1), (tn * query).sum(-1)
        def foreground(score):
            return score - (score * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1)
        def local_contrast(score):
            numerator = F.avg_pool2d((score * valid).reshape(-1, 1, side, side), 3, 1, 1)
            denominator = F.avg_pool2d(valid.reshape(-1, 1, side, side), 3, 1, 1)
            return score - (numerator / denominator.clamp_min(1e-6)).flatten(1)
        fr, ft = foreground(sr), foreground(st)
        features = torch.stack((fr, ft, fr-ft, local_contrast(sr), local_contrast(st),
                                (rn * tn).sum(-1)), dim=-1)
        # Avoid undefined contributions even when all image cells are padding.
        features = features.masked_fill(padding.unsqueeze(-1), 0)
        correction = self.bound * self.predict(features).tanh() * valid.unsqueeze(-1)
        updated = fused[1:].transpose(0, 1) + correction.to(r.dtype) * (r-t)
        return torch.cat((fused[:1], updated.transpose(0, 1)), dim=0)


class DenseGridGrounder(LateAdaptationGrounder):
    def configure_tuning(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(parameter_group(name) is not None)
        self.base._clip_control_trainability = {
            name: p.requires_grad for name, p in self.base.clip.named_parameters()}
