"""Text-conditioned regional evidence exchange, added as a C1 residual.

Inspired by RaAM (ICCV 2025); this is a three-modality adaptation, not a
reproduction. Sixteen soft spatial anchors read full text tokens, then read
RGB and TIR independently. Patch queries read either their averaged memory
(control) or the opposite modality's memory. Zero output projections preserve
the pretrained LAVS function at initialization. LAVS remains the global path;
this experiment does not claim to replace it completely.
"""
import math

import torch
from torch import nn


MODES = ('anchor_visual', 'anchor_text_shared', 'anchor_text_separate')


def masked_read(query, key, value, padding, bias=None):
    """Attention with zero output for empty memories and no padded evidence."""
    if padding.shape != key.shape[:2] or key.shape != value.shape:
        raise ValueError('Memory and padding shapes differ')
    key = key.masked_fill(padding[..., None], 0.)
    value = value.masked_fill(padding[..., None], 0.)
    logits = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if bias is not None:
        logits = logits + bias
    logits = logits.masked_fill(padding[:, None], torch.finfo(logits.dtype).min)
    weights = logits.float().softmax(-1).to(value.dtype)
    weights = weights.masked_fill(padding[:, None], 0.)
    return weights @ value


class RegionAnchorExchange(nn.Module):
    """One shared reader across four semantic depths; no new supervision."""
    def __init__(self, grid_size=14, anchor_count=16, hidden_dim=128,
                 visual_dim=768, text_dim=512, levels=4):
        super().__init__()
        side = math.isqrt(anchor_count)
        if (anchor_count < 1 or side * side != anchor_count or grid_size < 1
                or hidden_dim < 1 or levels < 1):
            raise ValueError('Positive dimensions and a square anchor count are required')
        self.visual_dim, self.text_dim = visual_dim, text_dim
        self.grid_size, self.levels = grid_size, levels
        self.anchors = nn.Parameter(torch.empty(anchor_count, hidden_dim))
        self.level_embeddings = nn.Parameter(torch.empty(levels, hidden_dim))
        nn.init.normal_(self.anchors, std=.02)
        nn.init.normal_(self.level_embeddings, std=.02)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.read_query = nn.Linear(hidden_dim, hidden_dim)
        self.read_key = nn.Linear(hidden_dim, hidden_dim)
        self.read_value = nn.Linear(hidden_dim, hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.write_query = nn.Linear(hidden_dim, hidden_dim)
        self.write_key = nn.Linear(hidden_dim, hidden_dim)
        self.write_value = nn.Linear(hidden_dim, hidden_dim)
        self.rgb_output = nn.Linear(hidden_dim, visual_dim)
        self.tir_output = nn.Linear(hidden_dim, visual_dim)
        for output in (self.rgb_output, self.tir_output):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        patches = torch.stack((y.flatten(), x.flatten()), -1).float()
        centers = (torch.arange(side).float() + .5) * grid_size / side - .5
        ay, ax = torch.meshgrid(centers, centers, indexing='ij')
        centers = torch.stack((ay.flatten(), ax.flatten()), -1)
        distances = (centers[:, None] - patches[None]).square().sum(-1)
        # Soft support: every valid patch remains accessible to every anchor.
        self.register_buffer('spatial_bias', -distances / (2 * (grid_size / side) ** 2), persistent=False)

    def forward(self, rgb, tir, legacy_rgb, legacy_tir, text, text_padding,
                visual_padding, mode='anchor_text_separate'):
        if mode not in MODES:
            raise ValueError('Unknown anchor exchange mode')
        if not (len(rgb) == len(tir) == len(legacy_rgb) == len(legacy_tir) == self.levels):
            raise ValueError('Feature level counts differ')
        if text.ndim != 3 or text.shape[-1] != self.text_dim or text_padding.shape != text.shape[:2]:
            raise ValueError('Full text tokens and their padding mask are required')
        if visual_padding.shape != (text.shape[0], self.grid_size ** 2):
            raise ValueError('Visual padding does not match the patch grid')
        text_padding, visual_padding = text_padding.bool(), visual_padding.bool()
        text_hidden = self.text_proj(self.text_norm(text.masked_fill(text_padding[..., None], 0.)))
        outputs_rgb, outputs_tir = [], []
        anchor_padding = visual_padding.all(-1, keepdim=True).expand(-1, self.anchors.shape[0])
        for level, (r, t, base_r, base_t) in enumerate(zip(rgb, tir, legacy_rgb, legacy_tir)):
            shape = (text.shape[0], self.grid_size ** 2 + 1, self.visual_dim)
            if any(x.shape != shape for x in (r, t, base_r, base_t)):
                raise ValueError('Expected B,(1+H*W),C visual tokens')
            anchor = (self.anchors + self.level_embeddings[level])[None].expand(text.shape[0], -1, -1)
            language = masked_read(self.read_query(anchor), self.read_key(text_hidden),
                                   self.read_value(text_hidden), text_padding)
            # Keep parameter count and graph identical in the no-text control.
            anchor = anchor + language * (0. if mode == 'anchor_visual' else 1.)
            r_hidden = self.visual_proj(self.visual_norm(r[:, 1:].masked_fill(visual_padding[..., None], 0.)))
            t_hidden = self.visual_proj(self.visual_norm(t[:, 1:].masked_fill(visual_padding[..., None], 0.)))
            memories = [masked_read(self.read_query(anchor), self.read_key(x), self.read_value(x),
                                    visual_padding, self.spatial_bias.to(x.dtype))
                        for x in (r_hidden, t_hidden)]
            if mode != 'anchor_text_separate':
                shared = (memories[0] + memories[1]) * .5
                memories = [shared, shared]
            deltas = []
            for hidden, memory, output in ((r_hidden, memories[1], self.rgb_output),
                                           (t_hidden, memories[0], self.tir_output)):
                memory = self.memory_norm(anchor + memory)
                update = masked_read(self.write_query(hidden), self.write_key(memory),
                                     self.write_value(memory), anchor_padding)
                update = output(update).masked_fill(visual_padding[..., None], 0.)
                deltas.append(torch.cat((torch.zeros_like(update[:, :1]), update), dim=1))
            outputs_rgb.append(base_r + deltas[0])
            outputs_tir.append(base_t + deltas[1])
        return outputs_rgb, outputs_tir
