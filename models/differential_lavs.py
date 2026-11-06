"""Full-width, correctly batched language/visual differential attention.

This is an adaptation of differential attention, not the official Diff
Transformer architecture. C1 projections/FFNs can be migrated losslessly;
the historical text batch-layout error is deliberately not preserved.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .lavs_replacement import LavsReplacementGrounder, parameter_group as old_group


def differential_key(name):
    return name.startswith('differential_exchange.') or '.language_replacement.' in name


def contrast_key(name):
    return any(part in name for part in ('.reference_query.', '.reference_key.',
        '.reference_text.', '.coefficient_route.', '.head_coefficient'))


def parameter_group(name):
    if name.startswith('base.') and differential_key(name[5:]):
        return 'differential' if contrast_key(name) else 'migrated_attention'
    return old_group(name)


def selected_name(name, args):
    group = parameter_group(name)
    return group is not None and (group != 'differential' or args.differential_enabled)


def valid_mean(tokens, padding):
    valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device) if padding is None else ~padding.bool()
    weights = valid.to(tokens.dtype)
    return (tokens * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)


def masked_attention(logits, source_padding, query_padding):
    """Finite all-padding behavior, with no messages from invalid positions."""
    if source_padding is None:
        weights = logits.softmax(-1)
    else:
        valid = ~source_padding.bool()[:, None, None, :]
        weights = logits.masked_fill(~valid, -torch.finfo(logits.dtype).max).softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    if query_padding is not None:
        weights = weights * (~query_padding.bool())[:, None, :, None]
    return weights


class DifferentialRead(nn.Module):
    def __init__(self, channels=768, heads=12, context_channels=512,
                 reference_width=8, enabled=True, coefficient_bound=.25,
                 preserve_foreground_context=False):
        super().__init__()
        if channels % heads:
            raise ValueError('Channels must be divisible by heads')
        self.channels, self.heads = channels, heads
        self.width, self.reference_width = channels // heads, reference_width
        self.enabled, self.coefficient_bound = enabled, coefficient_bound
        self.preserve_foreground_context = preserve_foreground_context
        # Avoid PEFT's CLIP self_attn projection names.
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        reference_dim = heads * reference_width
        self.reference_query = nn.Linear(channels, reference_dim, bias=False)
        self.reference_key = nn.Linear(channels, reference_dim, bias=False)
        self.reference_text = nn.Linear(context_channels, reference_dim, bias=False)
        self.head_coefficient = nn.Parameter(torch.zeros(heads))
        self.coefficient_route = nn.Linear(context_channels, heads, bias=False)
        nn.init.zeros_(self.coefficient_route.weight)

    def split(self, tensor, width):
        b, n, _ = tensor.shape
        return tensor.reshape(b, n, self.heads, width).transpose(1, 2)

    def forward(self, query, source, *, context, context_padding=None,
                query_padding=None, source_padding=None, roundtrip=False):
        # All public tensors use B,T,C. In particular, never reshape S,B,C
        # into B,S,C: doing so mixes different examples' language tokens.
        if query.ndim != 3 or source.ndim != 3 or context.shape[0] != query.shape[0]:
            raise ValueError('Expected independently batched B,T,C tensors')
        q = self.split(self.query(query) * self.width**-.5, self.width)
        k = self.split(self.key(source), self.width)
        v = self.split(self.value(source), self.width)
        positive = masked_attention(q @ k.transpose(-2, -1),
            None if self.preserve_foreground_context else source_padding,
            None if self.preserve_foreground_context else query_padding)
        message = positive.transpose(-2, -1) @ (positive @ v) if roundtrip else positive @ v
        if self.enabled:
            pooled = F.layer_norm(valid_mean(context, context_padding), (context.shape[-1],))
            rq = self.reference_query(F.layer_norm(query, (self.channels,)))
            rq = rq + self.reference_text(pooled)[:, None, :]
            rk = self.reference_key(F.layer_norm(source, (self.channels,)))
            rq = self.split(rq * self.reference_width**-.5, self.reference_width)
            rk = self.split(rk, self.reference_width)
            negative = masked_attention(rq @ rk.transpose(-2, -1), source_padding, query_padding)
            reference = negative.transpose(-2, -1) @ (negative @ v) if roundtrip else negative @ v
            coefficient = self.coefficient_bound * torch.tanh(
                self.head_coefficient[None, :] + self.coefficient_route(pooled))
            message = message - coefficient[:, :, None, None] * reference
        length = source.shape[1] if roundtrip else query.shape[1]
        output = self.output(message.transpose(1, 2).reshape(query.shape[0], length, self.channels))
        destination_padding = source_padding if roundtrip else query_padding
        if destination_padding is not None and not self.preserve_foreground_context:
            output = output.masked_fill(destination_padding[..., None].bool(), 0.)
        support = torch.ones(query.shape[0], dtype=torch.bool, device=query.device)
        if source_padding is not None and not self.preserve_foreground_context:
            support &= (~source_padding.bool()).any(-1)
        if roundtrip and query_padding is not None and not self.preserve_foreground_context:
            support &= (~query_padding.bool()).any(-1)
        return output * support[:, None, None]


class LanguageBranch(nn.Module):
    def __init__(self, channels, heads, intermediate, enabled, eps):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)
        # C1 was trained with the entire fixed CLIP context, including encoded
        # canvas/text padding states. Removing those states changes its learned
        # function. Preserve that foreground context during migration; only
        # the NEW reference attention and text routing exclude padding.
        self.attention = DifferentialRead(channels, heads, channels, enabled=enabled,
                                          preserve_foreground_context=True)
        self.expand = nn.Linear(channels, intermediate)
        self.contract = nn.Linear(intermediate, channels)

    def forward(self, visual, words, word_padding):
        update = self.attention(words, self.norm(visual), context=words,
            context_padding=word_padding, query_padding=word_padding, roundtrip=True)
        update = self.expand(update)
        update = self.contract(update * torch.sigmoid(1.702 * update))
        return visual + update


class DifferentialLanguageBlock(nn.Module):
    def __init__(self, channels=768, heads=12, intermediate=3072, enabled=True, eps=1e-5):
        super().__init__()
        self.branches = nn.ModuleDict({modality: LanguageBranch(channels, heads, intermediate, enabled, eps)
                                      for modality in ('rgb', 'ir')})

    def forward(self, visual, words, padding, modality):
        return self.branches[modality](visual, words, padding)


class DifferentialVisualExchange(nn.Module):
    def __init__(self, channels=768, heads=12, levels=4, enabled=True):
        super().__init__()
        self.rgb = nn.ModuleList(DifferentialRead(channels, heads, enabled=enabled,
            preserve_foreground_context=True) for _ in range(levels))
        self.ir = nn.ModuleList(DifferentialRead(channels, heads, enabled=enabled,
            preserve_foreground_context=True) for _ in range(levels))

    def forward(self, rgb, tir, words, word_padding, padding):
        if len(rgb) != len(self.rgb) or len(tir) != len(self.ir):
            raise ValueError('Unexpected number of encoder depths')
        full_padding = torch.cat((torch.zeros_like(padding[:, :1]), padding), dim=1).bool()
        def read(blocks, own, donor):
            return [x + block(x, y, context=words, context_padding=word_padding,
                    query_padding=full_padding, source_padding=full_padding)
                    for block, x, y in zip(blocks, own, donor)]
        # Both directions read original features, before either is updated.
        return read(self.rgb, rgb, tir), read(self.ir, tir, rgb)


def migrate_lavs_state(base, original):
    """Copy all foreground projections, norms and FFNs; discard no trained tensor.

    Only the two modality-specific modules actually used by the original
    encoder are copied to their corresponding branches. C1 has no LoRA on
    cross-attention projections (LoRA targets CLIP self_attn only).
    """
    state = base.state_dict()
    migrated = {}
    def copy(destination, source):
        if destination not in state or source not in original:
            raise ValueError(f'Missing migration key: {destination} <- {source}')
        if state[destination].shape != original[source].shape:
            raise ValueError(f'Migration shape mismatch: {destination}')
        migrated[destination] = original[source]
    def projections(destination, source):
        for target, old in (('query', 'q_proj'), ('key', 'k_proj'), ('value', 'v_proj'), ('output', 'out_proj')):
            for field in ('weight', 'bias'):
                copy(f'{destination}.{target}.{field}', f'{source}.{old}.{field}')
    for key in state:
        if '.language_replacement.branches.rgb.norm.weight' not in key:
            continue
        prefix = key.split('.language_replacement.')[0]
        for modality, suffix in (('rgb', 'sv'), ('ir', 'st')):
            target = f'{prefix}.language_replacement.branches.{modality}'
            projections(target + '.attention', prefix + '.cross_attn_' + suffix)
            for field in ('weight', 'bias'):
                copy(f'{target}.norm.{field}', f'{prefix}.cross_norm_{suffix}.{field}')
                copy(f'{target}.expand.{field}', f'{prefix}.cross_mlp_{suffix}.fc1.{field}')
                copy(f'{target}.contract.{field}', f'{prefix}.cross_mlp_{suffix}.fc2.{field}')
    for modality, old in (('rgb', 'vt'), ('ir', 'tv')):
        for level in range(len(base.differential_exchange.rgb)):
            projections(f'differential_exchange.{modality}.{level}', f'cross_fusion_layers_{old}.{level}')
    expected = {key for key in state if differential_key(key) and not contrast_key(key)}
    if set(migrated) != expected:
        raise ValueError(f'Foreground migration incomplete: {expected ^ set(migrated)}')
    base.load_state_dict(migrated, strict=False)
    if any(not torch.equal(base.state_dict()[key], value) for key, value in migrated.items()):
        raise AssertionError('Migrated foreground tensor differs from original C1')
    return {'migrated_tensors': len(migrated), 'migrated_elements': sum(value.numel() for value in migrated.values()),
            'foreground_tensors_exact': True, 'historical_batch_mixing_preserved': False,
            'foreground_fixed_clip_context_preserved': True}


class DifferentialGrounder(LavsReplacementGrounder):
    def configure_tuning(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(selected_name(name, self.args))
        self.base._clip_control_trainability = {n: p.requires_grad for n, p in self.base.clip.named_parameters()}
