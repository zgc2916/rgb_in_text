"""Global RGB/TIR exchange with zero-initialized semantic spatial corrections.

All donor tokens remain accessible. Three Gaussian spatial bases bias attention
logits rather than masking distant locations. Gates and spatial coefficients are
predicted from the recipient, global cross-modal message and CLIP sentence.
Global controls construct the same routing network and disable its spatial head.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def spatial_bases(grid_size, scales=(1., 2., 3.)):
    if grid_size < 1 or len(scales) != 3 or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('A positive grid and three positive finite spatial scales are required')
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    coordinates = torch.stack((y.flatten(), x.flatten()), -1).float()
    squared = (coordinates[:, None] - coordinates[None]).square().sum(-1)
    bases = torch.zeros(3, grid_size ** 2 + 1, grid_size ** 2 + 1)
    for i, sigma in enumerate(scales):
        bases[i, 1:, 1:] = torch.exp(-squared / (2 * sigma ** 2))
    # CLS remains a global semantic channel without a coordinate prior.
    return bases


def global_read(attention, recipient, donor):
    """The original LAVS projection/attention math, with reusable logits/values."""
    if recipient.ndim != 3 or donor.shape != recipient.shape:
        raise ValueError('Expected aligned B,N,C RGB/TIR token tensors')
    batch, tokens, channels = recipient.shape
    heads, width = attention.num_heads, attention.head_dim
    if channels != attention.embed_dim or heads * width != channels:
        raise ValueError('Attention projection dimension differs from tokens')
    def split(tensor):
        return tensor.view(batch, tokens, heads, width).transpose(1, 2).contiguous().view(batch * heads, tokens, width)
    query = split(attention.q_proj(recipient) * (width ** -.5))
    key = split(attention.k_proj(donor))
    value = split(attention.v_proj(donor))
    logits = torch.bmm(query, key.transpose(1, 2))
    probabilities = F.softmax(logits, dim=-1)
    multiplier = None
    if attention.training and attention.dropout:
        # Reuse the same dropout sample for the global and corrected reads.
        multiplier = F.dropout(torch.ones_like(probabilities), attention.dropout, True)
        probabilities = probabilities * multiplier
    message = project_read(attention, probabilities, value, batch, tokens)
    return message, logits, value, multiplier


def project_read(attention, probabilities, value, batch, tokens):
    read = torch.bmm(probabilities, value)
    read = read.view(batch, attention.num_heads, tokens, attention.head_dim)
    read = read.transpose(1, 2).reshape(batch, tokens, attention.embed_dim)
    return attention.out_proj(read)


class SpatialRoutingGate(nn.Module):
    def __init__(self, visual_dim=768, text_dim=512, hidden_dim=128):
        super().__init__()
        self.recipient_norm = nn.LayerNorm(visual_dim)
        self.message_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.recipient_proj = nn.Linear(visual_dim, hidden_dim)
        self.message_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.output = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, recipient, message, text, use_text):
        own = self.recipient_proj(self.recipient_norm(recipient))
        cross = self.message_proj(self.message_norm(message))
        sentence = self.text_proj(self.text_norm(text))
        if not use_text:
            sentence = sentence * 0.
        sentence = sentence[:, None].expand_as(own)
        return self.output(torch.cat((own, cross, (own - cross).abs(), sentence), dim=-1))


class SoftSpatialModalExchange(nn.Module):
    def __init__(self, grid_size=14, visual_dim=768, text_dim=512, hidden_dim=128,
                 levels=4, scales=(1., 2., 3.), bias_bound=2.):
        super().__init__()
        if not math.isfinite(bias_bound) or bias_bound <= 0 or levels < 1:
            raise ValueError('Positive finite bias bound and level count required')
        self.bias_bound = float(bias_bound)
        self.register_buffer('bases', spatial_bases(grid_size, tuple(scales)), persistent=False)
        self.gates = nn.ModuleList(SpatialRoutingGate(visual_dim, text_dim, hidden_dim) for _ in range(levels))

    def _read(self, attention, recipient, donor, gate, text, padding, use_text, use_spatial):
        batch, tokens, _ = recipient.shape
        if padding.shape != (batch, tokens - 1) or self.bases.shape[-2:] != (tokens, tokens):
            raise ValueError('Patch grid or padding differs from token shape')
        message, logits, value, multiplier = global_read(attention, recipient, donor)
        score = gate(recipient, message, text, use_text)
        valid_patch = ~padding.bool()
        valid = torch.cat((valid_patch.any(-1, keepdim=True), valid_patch), dim=1)
        strength = (2 * score[..., :1].sigmoid()).masked_fill(~valid[..., None], 1.)
        if use_spatial:
            coefficients = self.bias_bound * score[..., 1:].tanh()
            bias = torch.einsum('bnk,knm->bnm', coefficients, self.bases.to(coefficients.dtype))
            bias = bias.masked_fill(~(valid[:, :, None] & valid[:, None, :]), 0.)
            bias = bias[:, None].expand(-1, attention.num_heads, -1, -1).reshape_as(logits)
            probabilities = F.softmax(logits + bias, dim=-1)
            if multiplier is not None:
                probabilities = probabilities * multiplier
            message = project_read(attention, probabilities, value, batch, tokens)
        # Preserve C1's original global attention, including its padding policy.
        # New routing/bias has no effect at padded queries or on spatial CLS edges.
        return strength * message

    def forward(self, rgb, tir, rgb_reads, tir_reads, text, padding,
                use_text=True, use_spatial=True):
        if not (len(rgb) == len(tir) == len(rgb_reads) == len(tir_reads) == len(self.gates)):
            raise ValueError('Feature level counts differ')
        output_rgb, output_tir = [], []
        for r, t, rg, tg, gate in zip(rgb, tir, rgb_reads, tir_reads, self.gates):
            into_rgb = self._read(rg, r, t, gate, text, padding, use_text, use_spatial)
            into_tir = self._read(tg, t, r, gate, text, padding, use_text, use_spatial)
            output_rgb.append(r + into_rgb)
            output_tir.append(t + into_tir)
        return output_rgb, output_tir
