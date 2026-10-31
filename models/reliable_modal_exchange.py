"""Direction-specific, corruption-aware RGB/TIR exchange.

The quality heads predict *synthetic corruption*, not calibrated uncertainty.
Their supervision is available only during training. Inference uses image/text
features alone. Zero output heads preserve the inherited C1 function exactly.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .soft_spatial_exchange import global_read, project_read, spatial_bases


MODES = ('direction_gate', 'reliability_gate', 'reliability_spatial')


@torch.no_grad()
def corrupt_modalities(images, padding, probability=.35, grid_size=14):
    """RefFLIR-only perturbation; leave padding and the other sensor untouched.

    RGB is darkened; TIR is blurred/noised. Half of the selected examples use a
    local rectangular corruption. Labels describe known perturbations, never
    object boxes or inferred real-world sensor quality.
    """
    if images.ndim != 4 or images.shape[1] != 4 or not 0 <= probability <= 1:
        raise ValueError('Expected normalized RefFLIR B,4,H,W and a probability')
    batch, _, height, width = images.shape
    device = images.device
    if probability == 0:
        valid = (~padding[:, None].bool()).to(images.dtype)
        fraction = F.adaptive_avg_pool2d(valid, (grid_size, grid_size))[:, 0].flatten(1)
        return images, {'damage': images.new_zeros(batch, 2, grid_size**2), 'valid_fraction': fraction}
    active = torch.rand(batch, device=device) < probability
    modality = torch.randint(2, (batch,), device=device)
    local = torch.rand(batch, device=device) < .5
    start_y = torch.randint(grid_size, (batch,), device=device)
    start_x = torch.randint(grid_size, (batch,), device=device)
    y = torch.arange(grid_size, device=device)[None, :, None]
    x = torch.arange(grid_size, device=device)[None, None, :]
    side = max(1, grid_size // 2)
    region = ((y >= start_y[:, None, None]) & (y < start_y[:, None, None] + side)
              & (x >= start_x[:, None, None]) & (x < start_x[:, None, None] + side))
    region = region | ~local[:, None, None]
    selected = active[:, None, None, None] & (modality[:, None, None, None] ==
                                            torch.arange(2, device=device)[None, :, None, None])
    mask = F.interpolate((selected & region[:, None]).float(), (height, width), mode='nearest').bool()
    mask &= ~padding[:, None].bool()
    mean = images.new_tensor([.631, .6401, .632, .5337])[None, :, None, None]
    std = images.new_tensor([.2152, .227, .2439, .2562])[None, :, None, None]
    raw = (images * std + mean).clamp(0, 1)
    dark = raw[:, :3] * (.25 + .4 * torch.rand(batch, 1, 1, 1, device=device))
    # Normalize the blur by valid support: padded pixels cannot contaminate it.
    valid = (~padding[:, None].bool()).to(images.dtype)
    support = F.avg_pool2d(valid, 9, 1, 4).clamp_min(1e-6)
    thermal = F.avg_pool2d(raw[:, 3:] * valid, 9, 1, 4) / support
    thermal = (thermal + .08 * torch.randn_like(thermal)).clamp(0, 1)
    changed = (torch.cat((dark, thermal), dim=1) - mean) / std
    channel_mask = torch.cat((mask[:, :1].expand(-1, 3, -1, -1), mask[:, 1:]), dim=1)
    result = torch.where(channel_mask, changed, images)
    valid_fraction = F.adaptive_avg_pool2d(valid, (grid_size, grid_size))[:, 0]
    target = F.adaptive_avg_pool2d(mask.float(), (grid_size, grid_size))
    target = target / valid_fraction[:, None].clamp_min(1e-6)
    return result, {'damage': target.flatten(2), 'valid_fraction': valid_fraction.flatten(1)}


class DamageHead(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens):
        return self.net(tokens).squeeze(-1)


class DirectionalRoutingGate(nn.Module):
    def __init__(self, visual_dim=768, text_dim=512, hidden_dim=128):
        super().__init__()
        self.recipient_norm = nn.LayerNorm(visual_dim)
        self.message_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.recipient_proj = nn.Linear(visual_dim, hidden_dim)
        self.message_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fuse = nn.Sequential(nn.Linear(4 * hidden_dim + 3, hidden_dim), nn.GELU())
        self.direction_heads = nn.ModuleList(nn.Linear(hidden_dim, 4) for _ in range(2))
        self.damage_heads = nn.ModuleList(DamageHead(visual_dim, hidden_dim // 2) for _ in range(2))
        self.quality_bias_logit = nn.Parameter(torch.full((2,), -2.))
        for head in self.direction_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, recipient, message, text, quality_context, direction):
        own = self.recipient_proj(self.recipient_norm(recipient))
        cross = self.message_proj(self.message_norm(message))
        sentence = self.text_proj(self.text_norm(text))[:, None].expand_as(own)
        fused = torch.cat((own, cross, (own - cross).abs(), sentence, quality_context), dim=-1)
        return self.direction_heads[direction](self.fuse(fused))


class ReliableModalExchange(nn.Module):
    def __init__(self, grid_size=14, visual_dim=768, text_dim=512, hidden_dim=128,
                 levels=4, scales=(1., 2., 3.), bias_bound=2.):
        super().__init__()
        if hidden_dim < 2 or levels < 1 or not math.isfinite(bias_bound) or bias_bound <= 0:
            raise ValueError('Invalid exchange dimensions/bias bound')
        self.bias_bound = float(bias_bound)
        self.register_buffer('bases', spatial_bases(grid_size, scales), persistent=False)
        self.gates = nn.ModuleList(DirectionalRoutingGate(visual_dim, text_dim, hidden_dim) for _ in range(levels))

    def _read(self, attention, recipient, donor, text, padding, gate, direction,
              own_damage, donor_damage, use_quality, use_spatial):
        batch, tokens, _ = recipient.shape
        patch_valid = ~padding.bool()
        valid = torch.cat((patch_valid.any(-1, keepdim=True), patch_valid), dim=1)
        message, logits, values, multiplier = global_read(attention, recipient, donor)
        probabilities = logits.softmax(-1).view(batch, attention.num_heads, tokens, tokens).mean(1)
        if use_quality:
            own_quality, donor_quality = 1 - own_damage.sigmoid(), 1 - donor_damage.sigmoid()
            read_quality = torch.bmm(probabilities, donor_quality[..., None])
            entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1, keepdim=True)
            entropy = entropy / math.log(max(tokens, 2))
            context = torch.cat((own_quality[..., None], read_quality, entropy), dim=-1)
        else:
            context = recipient.new_zeros(batch, tokens, 3)
        score = gate(recipient, message, text, context, direction)
        strength = (2 * score[..., :1].sigmoid()).masked_fill(~valid[..., None], 1.)
        bias = recipient.new_zeros(batch, tokens, tokens)
        if use_quality:
            damage = donor_damage.sigmoid()
            patch_mean = (damage[:, 1:] * patch_valid).sum(-1, keepdim=True) / patch_valid.sum(-1, keepdim=True).clamp_min(1)
            key_bias = -(damage - patch_mean) * self.bias_bound * gate.quality_bias_logit[direction].sigmoid()
            key_bias = key_bias.masked_fill(~valid, 0.)
            key_bias = torch.cat((torch.zeros_like(key_bias[:, :1]), key_bias[:, 1:]), dim=1)
            bias = bias + key_bias[:, None]
        if use_spatial:
            coefficients = self.bias_bound * score[..., 1:].tanh()
            bias = bias + torch.einsum('bnk,knm->bnm', coefficients, self.bases.to(coefficients.dtype))
        # Preserve legacy padding and global CLS connectivity. New corrections
        # are neutral at padded queries/keys; no donor token is hard-masked.
        bias = bias.masked_fill(~(valid[:, :, None] & valid[:, None, :]), 0.)
        bias = bias[:, None].expand(-1, attention.num_heads, -1, -1).reshape_as(logits)
        corrected = F.softmax(logits + bias, dim=-1)
        if multiplier is not None:
            corrected = corrected * multiplier
        message = project_read(attention, corrected, values, batch, tokens)
        regularizer = ((strength - 1).square().squeeze(-1) * valid).sum() / valid.sum().clamp_min(1)
        regularizer = regularizer + bias.square().mean()
        return strength * message, regularizer

    def forward(self, rgb, tir, rgb_reads, tir_reads, text, padding,
                mode='reliability_gate', corruption_targets=None):
        if mode not in MODES or not (len(rgb) == len(tir) == len(rgb_reads) == len(tir_reads) == len(self.gates)):
            raise ValueError('Invalid mode or feature level counts')
        if padding.shape != (rgb[0].shape[0], rgb[0].shape[1] - 1) or self.bases.shape[-1] != rgb[0].shape[1]:
            raise ValueError('Feature grid differs from padding/spatial bases')
        use_quality, use_spatial = mode != 'direction_gate', mode == 'reliability_spatial'
        output_rgb, output_tir, regularizers, damage_losses = [], [], [], []
        for r, t, rg, tg, gate in zip(rgb, tir, rgb_reads, tir_reads, self.gates):
            rd, td = (head(tokens) for head, tokens in zip(gate.damage_heads, (r, t)))
            into_rgb, reg_r = self._read(rg, r, t, text, padding, gate, 0, rd, td, use_quality, use_spatial)
            into_tir, reg_t = self._read(tg, t, r, text, padding, gate, 1, td, rd, use_quality, use_spatial)
            output_rgb.append(r + into_rgb)
            output_tir.append(t + into_tir)
            regularizers.extend((reg_r, reg_t))
            if use_quality and corruption_targets is not None:
                weights = corruption_targets['valid_fraction'] * (~padding.bool())
                targets = corruption_targets['damage']
                for direction, prediction in enumerate((rd, td)):
                    bce = F.binary_cross_entropy_with_logits(prediction[:, 1:], targets[:, direction], reduction='none',
                                                           pos_weight=prediction.new_tensor(2.))
                    damage_losses.append((bce * weights).sum() / weights.sum().clamp_min(1))
        aux = {'exchange_correction_penalty': torch.stack(regularizers).mean()}
        if damage_losses:
            aux['exchange_damage_bce'] = torch.stack(damage_losses).mean()
            aux['exchange_perturbed_fraction'] = corruption_targets['damage'].mean().detach()
        return output_rgb, output_tir, aux
