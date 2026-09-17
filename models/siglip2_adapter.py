"""SigLIP2 compatibility layers for MMVGFusion.

The fixed-resolution SigLIP2 checkpoint is exposed by Transformers as a
``SiglipModel``.  MMVGFusion needs two things that the stock vision tower
does not provide: intermediate text-guided vision layers and a CLIP-like
sequence with one global token followed by the patch tokens.  This module
keeps those adaptations local to the SigLIP2 path so the original CLIP path
remains available for regression tests.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipTextGuidedAttention(nn.Module):
    """Vision-query/text-key-value attention with a text padding mask."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got {self.embed_dim} and {self.num_heads})"
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        vision_states: torch.Tensor,
        text_states: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, vision_len, _ = vision_states.shape
        text_len = text_states.shape[1]

        query = self._split_heads(self.q_proj(vision_states) * self.scale)
        key = self._split_heads(self.k_proj(text_states))
        value = self._split_heads(self.v_proj(text_states))

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if attn_weights.shape != (batch_size, self.num_heads, vision_len, text_len):
            raise RuntimeError(f"Unexpected SigLIP2 cross-attention shape: {attn_weights.shape}")

        if text_padding_mask is not None:
            mask = text_padding_mask.to(torch.bool).view(batch_size, 1, 1, text_len)
            attn_weights = attn_weights.masked_fill(mask, torch.finfo(attn_weights.dtype).min)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        output = torch.matmul(attn_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, vision_len, self.embed_dim)
        return self.out_proj(output), attn_weights


class SiglipTextGuidedMLP(nn.Module):
    """The SigLIP MLP activation used by newly initialized fusion blocks."""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, hidden_states):
        return self.fc2(self.activation(self.fc1(hidden_states)))


class SiglipEncoderLayerWithTextGuidedFusion(nn.Module):
    """A SigLIP encoder layer with the existing MMVG text-guided branch."""

    def __init__(self, args, layer_index, base_layer, config, adapt_layer):
        super().__init__()
        self.args = args
        self.layer_index = layer_index
        self.embed_dim = base_layer.embed_dim
        self.self_attn = base_layer.self_attn
        self.layer_norm1 = base_layer.layer_norm1
        self.mlp = base_layer.mlp
        self.layer_norm2 = base_layer.layer_norm2
        self.open_text_guided_fusion = bool(getattr(args, "open_text_guided_fusion", True))

        if layer_index in adapt_layer and self.open_text_guided_fusion:
            if args.modality == "rgb":
                self.cross_norm_sv = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
                self.cross_attn_sv = SiglipTextGuidedAttention(config)
                self.cross_mlp_sv = SiglipTextGuidedMLP(config)
            elif args.modality == "rgbt":
                self.cross_norm_st = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
                self.cross_norm_sv = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
                self.cross_attn_st = SiglipTextGuidedAttention(config)
                self.cross_attn_sv = SiglipTextGuidedAttention(config)
                self.cross_mlp_st = SiglipTextGuidedMLP(config)
                self.cross_mlp_sv = SiglipTextGuidedMLP(config)

    def forward(
        self,
        hidden_states,
        layer,
        adapt_layer,
        text_states,
        cur_modality,
        text_padding_mask=None,
        attention_mask=None,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        if layer in adapt_layer and self.open_text_guided_fusion:
            text_states = text_states[-1] if isinstance(text_states, (tuple, list)) else text_states
            if cur_modality == "ir" and hasattr(self, "cross_attn_st"):
                residual = hidden_states
                cross_input = self.cross_norm_st(hidden_states)
                cross_output, _ = self.cross_attn_st(cross_input, text_states, text_padding_mask)
                hidden_states = residual + self.cross_mlp_st(cross_output)
            elif hasattr(self, "cross_attn_sv"):
                residual = hidden_states
                cross_input = self.cross_norm_sv(hidden_states)
                cross_output, _ = self.cross_attn_sv(cross_input, text_states, text_padding_mask)
                hidden_states = residual + self.cross_mlp_sv(cross_output)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class SiglipEncoderWithTextGuidedFusion(nn.Module):
    def __init__(self, args, base_encoder, adapt_layer):
        super().__init__()
        self.config = base_encoder.config
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayerWithTextGuidedFusion(
                    args,
                    i,
                    base_encoder.layers[i],
                    self.config,
                    adapt_layer,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        adapt_layer,
        text_states,
        cur_modality,
        text_padding_mask=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds

        for layer_index, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                layer=layer_index,
                adapt_layer=adapt_layer,
                text_states=text_states,
                cur_modality=cur_modality,
                text_padding_mask=text_padding_mask,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in (hidden_states, encoder_states, all_attentions) if v is not None)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": encoder_states,
            "attentions": all_attentions,
        }


class SiglipVisionModelWithTextGuidedFusion(nn.Module):
    """Drop-in custom vision tower used only by MMVGFusion's SigLIP2 path."""

    def __init__(self, args, base_vision_model, adapt_layer):
        super().__init__()
        self.config = base_vision_model.config
        self.embeddings = base_vision_model.embeddings
        self.encoder = SiglipEncoderWithTextGuidedFusion(args, base_vision_model.encoder, adapt_layer)
        self.post_layernorm = base_vision_model.post_layernorm
        self.use_head = base_vision_model.use_head
        if self.use_head:
            self.head = base_vision_model.head

    def forward(
        self,
        adapt_layer,
        text_states,
        reg_src=None,
        cur_modality="rgb",
        text_padding_mask=None,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        interpolate_pos_encoding=False,
    ):
        del reg_src
        if pixel_values is None:
            raise ValueError("pixel_values must be provided to the SigLIP2 vision tower")
        output_attentions = bool(output_attentions) if output_attentions is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False
        return_dict = True if return_dict is None else return_dict

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            adapt_layer=adapt_layer,
            text_states=text_states,
            cur_modality=cur_modality,
            text_padding_mask=text_padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs["last_hidden_state"] if return_dict else encoder_outputs[0]
        pooled_input = self.post_layernorm(last_hidden_state)
        pooler_output = self.head(pooled_input) if self.use_head else None

        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]
        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
        }


def load_siglip2(path):
    """Load the fixed-resolution checkpoint without AutoModel's timm lookup."""
    from transformers import SiglipModel

    # MMVGFusion requests intermediate attentions.  Explicit eager attention
    # avoids the SDPA fallback warning in Transformers 4.50.x.
    return SiglipModel.from_pretrained(
        path,
        local_files_only=True,
        attn_implementation="eager",
    )


def load_siglip2_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def siglip2_eos_indices(input_ids, eos_token_id, pad_token_id):
    """Return EOS positions, with a last-non-pad fallback for truncated text."""
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    eos_positions = torch.where(input_ids.eq(eos_token_id), positions, -1)
    eos_indices = eos_positions.max(dim=-1).values
    fallback = input_ids.ne(pad_token_id).sum(dim=-1).clamp_min(1) - 1
    return torch.where(eos_indices >= 0, eos_indices, fallback)
