import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VisionLanguageEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_visual_tokens=400,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            num_visual_tokens=num_visual_tokens,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        return self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src
        attn_list = []
        for layer in self.layers:
            output, attn = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            attn_list.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, torch.stack(attn_list, dim=0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_visual_tokens=400,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, num_visual_tokens=num_visual_tokens
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2, simi = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, simi

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, _ = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, None

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def build_vl_transformer_attbalance(args):
    divisor = 16 if args.dilation else 32
    num_visual_tokens = int((args.imsize / divisor) ** 2)
    if getattr(args, "modality", "rgb") == "rgbt":
        num_visual_tokens *= 2
    return VisionLanguageEncoder(
        d_model=args.vl_hidden_dim,
        dropout=args.vl_dropout,
        nhead=args.vl_nheads,
        dim_feedforward=args.vl_dim_feedforward,
        num_encoder_layers=args.vl_enc_layers,
        normalize_before=getattr(args, "normalize_before", False),
        num_visual_tokens=num_visual_tokens,
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, num_visual_tokens=400):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_h = embed_dim // num_heads
        self.h = num_heads
        self.num_visual_tokens = num_visual_tokens
        self.dropout = nn.Dropout(p=dropout)

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=True
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True,
        average_weights=True,
    ):
        tgt_len, bs, _ = query.shape
        src_len, _, _ = key.shape

        w_q, w_k, w_v = [
            self.in_proj_weight[i * self.embed_dim : (i + 1) * self.embed_dim]
            for i in range(3)
        ]
        b_q, b_k, b_v = [
            self.in_proj_bias[i * self.embed_dim : (i + 1) * self.embed_dim]
            for i in range(3)
        ]
        q = torch.nn.functional.linear(query, w_q, b_q)
        k = torch.nn.functional.linear(key, w_k, b_k)
        v = torch.nn.functional.linear(value, w_v, b_v)

        q = q.contiguous().view(tgt_len, bs * self.h, self.d_h).transpose(0, 1)
        k = k.contiguous().view(src_len, bs * self.h, self.d_h).transpose(0, 1)
        v = v.contiguous().view(src_len, bs * self.h, self.d_h).transpose(0, 1)

        key_padding_mask = key_padding_mask.view(bs, 1, 1, src_len).expand(-1, self.h, -1, -1)
        key_padding_mask = key_padding_mask.reshape(bs * self.h, 1, src_len)
        attn_mask_ = torch.zeros_like(key_padding_mask, dtype=q.dtype)
        attn_mask_.masked_fill_(key_padding_mask, float("-inf"))

        _, _, e = q.shape
        q_scaled = q / math.sqrt(e)
        similarity = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_mask_ + similarity, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bs, self.embed_dim)
        attn_output = torch.nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bs, self.embed_dim)

        if need_weights:
            similarity = similarity.view(bs, self.h, tgt_len, src_len)
            if average_weights:
                similarity = similarity.mean(dim=1)[:, 0, -self.num_visual_tokens :].softmax(dim=-1)
            return attn_output, similarity
        return attn_output
