"""Spatially constrained RGB/TIR exchange with an optional text-conditioned gate.

The attention projections belong to the original external LAVS modules. Only
their attention graph changes, so released projection weights remain usable.
This implementation uses a dense masked attention kernel; locality does not
imply a reduction in theoretical computation here.
"""
import torch
from torch import nn
from torch.nn import functional as F


def local_attention_graph(grid_size, radius):
    """CLS reads globally; patches read a square neighborhood and donor CLS."""
    if grid_size < 1 or radius < 0:
        raise ValueError('Positive grid size and nonnegative radius required')
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    coordinates = torch.stack((y.flatten(), x.flatten()), dim=-1)
    neighbors = (coordinates[:, None] - coordinates[None]).abs().amax(-1) <= radius
    graph = torch.ones(grid_size ** 2 + 1, grid_size ** 2 + 1, dtype=torch.bool)
    graph[1:, 1:] = neighbors
    return graph


def local_cross_read(attention, recipient, donor, graph, padding):
    """Read with existing Q/K/V/out projections, respecting both stream masks."""
    if recipient.shape != donor.shape or recipient.ndim != 3:
        raise ValueError('Aligned recipient/donor tensors must have shape B,N,C')
    batch, tokens, channels = recipient.shape
    if graph.shape != (tokens, tokens) or padding.shape != (batch, tokens - 1):
        raise ValueError('Graph or patch padding shape differs from visual tokens')
    if channels != attention.embed_dim:
        raise ValueError('Projection dimension differs from visual channels')
    patch_valid = ~padding.bool()
    valid = torch.cat((patch_valid.any(-1, keepdim=True), patch_valid), dim=1)
    allowed = graph[None] & valid[:, None, :]
    support = allowed.any(-1) & valid
    # Empty attention rows get one finite sentinel; their projected updates are
    # masked AFTER out_proj, including its bias. Fully padded images stay fixed.
    allowed = allowed.clone()
    allowed[:, :, 0] |= ~allowed.any(-1)
    heads, width = attention.num_heads, attention.head_dim
    def split(tensor):
        return tensor.reshape(batch, tokens, heads, width).transpose(1, 2)
    query = split(attention.q_proj(recipient))
    key = split(attention.k_proj(donor))
    value = split(attention.v_proj(donor))
    read = F.scaled_dot_product_attention(
        query, key, value, attn_mask=allowed[:, None],
        dropout_p=attention.dropout if attention.training else 0.)
    read = read.transpose(1, 2).reshape(batch, tokens, channels)
    return attention.out_proj(read).masked_fill(~support[:, :, None], 0.)


class LocalExchangeGate(nn.Module):
    """Shared directional gate conditioned on recipient, message and sentence."""
    def __init__(self, visual_dim=768, text_dim=512, hidden_dim=128):
        super().__init__()
        self.recipient_norm = nn.LayerNorm(visual_dim)
        self.message_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.recipient_proj = nn.Linear(visual_dim, hidden_dim)
        self.message_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.output = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, recipient, message, text, use_text=True):
        own = self.recipient_proj(self.recipient_norm(recipient))
        cross = self.message_proj(self.message_norm(message))
        sentence = self.text_proj(self.text_norm(text))
        if not use_text:
            sentence = sentence * 0.
        sentence = sentence[:, None].expand_as(own)
        score = self.output(torch.cat((own, cross, (own - cross).abs(), sentence), dim=-1))
        return 2. * torch.sigmoid(score)


class LocalModalExchange(nn.Module):
    """Four-level simultaneous exchange; no boxes or supervision in forward."""
    def __init__(self, grid_size=14, radii=(1, 1, 2, 3), visual_dim=768, text_dim=512, hidden_dim=128):
        super().__init__()
        self.radii = tuple(int(r) for r in radii)
        if len(self.radii) != 4 or any(r < 0 or r >= grid_size for r in self.radii):
            raise ValueError('Four radii within the patch grid required')
        self.gates = nn.ModuleList(LocalExchangeGate(visual_dim, text_dim, hidden_dim) for _ in self.radii)
        for level, radius in enumerate(self.radii):
            self.register_buffer(f'graph_{level}', local_attention_graph(grid_size, radius), persistent=False)

    def forward(self, rgb, tir, rgb_reads, tir_reads, text, padding, use_text=True):
        if not (len(rgb) == len(tir) == len(rgb_reads) == len(tir_reads) == len(self.gates)):
            raise ValueError('Level counts differ')
        output_rgb, output_tir = [], []
        for i, (r, t, gate) in enumerate(zip(rgb, tir, self.gates)):
            graph = getattr(self, f'graph_{i}')
            into_rgb = local_cross_read(rgb_reads[i], r, t, graph, padding)
            into_tir = local_cross_read(tir_reads[i], t, r, graph, padding)
            output_rgb.append(r + gate(r, into_rgb, text, use_text) * into_rgb)
            output_tir.append(t + gate(t, into_tir, text, use_text) * into_tir)
        return output_rgb, output_tir
