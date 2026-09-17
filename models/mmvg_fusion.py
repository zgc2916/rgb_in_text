import torch
import torch.nn as nn
import torch.nn.functional as F
from .vl_transformer import build_vl_transformer
import pdb
from .clip import *
from torchvision.transforms import Resize
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from collections import OrderedDict

# import bitsandbytes as bnb
from transformers import CLIPModel, CLIPConfig, CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel
from transformers import CLIPTokenizer, AutoTokenizer, CLIPImageProcessor
from peft import get_peft_config, PeftModel, get_peft_model, LoraConfig, TaskType,AdaLoraConfig
from torch.nn.parameter import Parameter
from typing import Any, Optional, Tuple, Union
import math
import re
from pathlib import Path

from .tsar_modules import (
    FrozenInfMAEEncoder,
    GroundingQualityRouter,
    InfMAEDirectThermalAdapter,
    InfMAEMultiScaleThermalAdapter,
    InfMAEThermalInputNormalizer,
    TargetAwareTokenPool,
    TargetFeatureProjector,
    TextGuidedThermalSemanticBridge,
    TextConditionedModalityHead,
)


class Modified_CLIPVisionEmbeddings(nn.Module):
    def __init__(self,args, clip_embed):
        super().__init__()
        self.args = args
        self.config = clip_embed.config
        self.embed_dim = clip_embed.embed_dim
        self.image_size = clip_embed.image_size
        self.patch_size = clip_embed.patch_size
        self.class_embedding = clip_embed.class_embedding  # 768
        self.patch_embedding = clip_embed.patch_embedding
        self.num_patches = clip_embed.num_patches
        self.num_positions = clip_embed.num_positions  # 197
        self.position_embedding = clip_embed.position_embedding  # 197 * 768
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:

        batch_size = pixel_values.shape[0]  # B C H W 
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid], B 768 H/16 W/16
        h, w = patch_embeds.shape[2], patch_embeds.shape[3]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # B L H
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # B * 1 * 768
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        cls_pos = self.position_embedding.weight[0:1, :]
        abs_pos = self.position_embedding.weight[1:, :]  # 196 * 768
        xy_num = abs_pos.shape[0]
        assert xy_num == self.num_patches  # 196
        size = int(math.sqrt(xy_num))  # 14
        assert size * size == xy_num

        if size != h or size != w:
            new_abs_pos = F.interpolate(  # 1 14 14 768 --> 1 768 14 14 --> 1 768 40 40
                abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            )
            new_abs_pos = new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
            position_embedding = torch.cat([cls_pos.unsqueeze(0), new_abs_pos], dim=1)  # 1 1601 768
            embeddings = embeddings + position_embedding.repeat(batch_size, 1, 1)
        else:  # 14 == 14
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class VisionEmbeddings(nn.Module):
    def __init__(self, clip_embed):
        super().__init__()
        self.config = clip_embed.config
        self.embed_dim = clip_embed.embed_dim
        self.image_size = clip_embed.image_size
        self.patch_size = clip_embed.patch_size
        self.class_embedding = clip_embed.class_embedding  # 768
        self.patch_embedding = clip_embed.patch_embedding
        self.num_patches = clip_embed.num_patches
        self.num_positions = clip_embed.num_positions  # 此时是197
        self.position_embedding = clip_embed.position_embedding  # 197 * 768
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor, position_embedding) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # B C H W
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid], B 768 H/16 W/16
        h, w = patch_embeds.shape[2], patch_embeds.shape[3]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # B L H
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # B * 1 * 768
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        embeddings = embeddings + position_embedding(self.position_ids)

        return embeddings


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

#
# ACT2CLS = {
#     "gelu": GELUActivation,
#     "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
#     "gelu_fast": FastGELUActivation,
#     "gelu_new": NewGELUActivation,
#     "gelu_python": (GELUActivation, {"use_gelu_python": True}),
#     "gelu_pytorch_tanh": PytorchGELUTanh,
#     "gelu_accurate": AccurateGELUActivation,
#     "laplace": LaplaceActivation,
#     "linear": LinearActivation,
#     "mish": MishActivation,
#     "quick_gelu": QuickGELUActivation,
#     "relu": nn.ReLU,
#     "relu2": ReLUSquaredActivation,
#     "relu6": nn.ReLU6,
#     "sigmoid": nn.Sigmoid,
#     "silu": SiLUActivation,
#     "swish": SiLUActivation,
#     "tanh": nn.Tanh,
# }
# ACT2FN = ClassInstantier(ACT2CLS)


class CLIP_Cross_Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)    
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):

        # print(torch.equal(aa.flatten(), tensor.flatten()))
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj

        query_states = self.q_proj(hidden_states) * self.scale
        query_states = self._shape(query_states, -1, bsz)  #torch.Size([32, 12, 197, 64])

        key_states = self._shape(self.k_proj(hidden_states_2), -1, bsz) #torch.Size([77, 32, 768])->torch.Size([32, 12, 77, 64])
        # todo检查self._shape
        value_states = self._shape(self.v_proj(hidden_states_2), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape) #torch.Size([32*12, 197, 64])
        key_states = key_states.view(*proj_shape)     #torch.Size([32*12, 77, 64])
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len) 
        # import pdb; pdb.set_trace()
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
   
        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) 
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
   
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)


        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped

class CLIP_Cross_Attention_VS(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)    
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):

        # print(torch.equal(aa.flatten(), tensor.flatten()))
        return tensor.contiguous().view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,  # 作为 Key/Value 的来源（原逻辑中是 Query 来源）
            text_states: torch.Tensor,  # 作为 Query 的来源（原逻辑中是 Key/Value 来源）
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        输入形状：
        - hidden_states: (bsz, src_len, embed_dim)  # 作为 Key/Value
        - text_states: (bsz, tgt_len, embed_dim)    # 作为 Query
        """

        # 获取文本序列长度（Query 长度）和批次大小
        text_len,bsz,  embed_dim = text_states.size()  # 这里改为从 text_states 取形状（因为它是 Query）
        # 获取图像/隐藏状态序列长度（Key/Value 长度）
        tgt_len = hidden_states.size(1)  # hidden_states 形状：(bsz, tgt_len, embed_dim)

        # --------------------------
        # 1. 生成 Query（来自 text_states）
        # --------------------------

        query_states = self.q_proj(text_states) * self.scale  # text 作为 Query
        query_states = self._shape(query_states, text_len, bsz)  # 形状：(bsz, num_heads, text_len, head_dim)

        # --------------------------
        # 2. 生成 Key 和 Value（来自 hidden_states）
        # --------------------------
        # Key 投影 + 形状调整：(bsz, tgt_len, embed_dim) → (bsz, num_heads, tgt_len, head_dim)
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        # Value 投影 + 形状调整：同上
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        # --------------------------
        # 3. 调整形状以进行批量矩阵乘法
        # --------------------------
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape)  # (bsz*num_heads, text_len, head_dim)
        key_states = key_states.view(*proj_shape)  # (bsz*num_heads, tgt_len, head_dim)
        value_states = value_states.view(*proj_shape)  # (bsz*num_heads,tgt_len, head_dim)

        # --------------------------
        # 4. 计算注意力权重
        # --------------------------
        # 注意力分数：(bsz*num_heads, text_len,tgt_len)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # 检查注意力权重形状是否正确
        if attn_weights.size() != (bsz * self.num_heads,text_len,tgt_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, text_len,tgt_len)}, but is"
                f" {attn_weights.size()}"
            )

        # --------------------------
        # 5. 应用掩码（若有）
        # --------------------------
        # 因果掩码（若需要，例如文本生成时的自注意力）
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1,text_len,tgt_len):
                raise ValueError(
                    f"Causal attention mask should be of size {(bsz, 1, text_len,tgt_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            # 重塑后加掩码
            attn_weights = attn_weights.view(bsz, self.num_heads, text_len, tgt_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, text_len, tgt_len)

        # 普通注意力掩码（例如padding掩码）
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1,text_len, tgt_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, text_len,tgt_len)}, but is {attention_mask.size()}"
                )
            # 重塑后加掩码
            attn_weights = attn_weights.view(bsz, self.num_heads, text_len, tgt_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, text_len, tgt_len)

        # --------------------------
        # 6. 注意力归一化与 dropout
        # --------------------------
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # --------------------------
        # 7. 计算注意力输出
        # --------------------------
        text_guided_vision= torch.bmm(attn_probs, value_states)  # (bsz*num_heads, text_len, head_dim)
        attn_weights_t = attn_weights.transpose(1, 2)
        # 聚合文本信息到视觉维度：(16*num_heads, 197, 64) → 与原始视觉长度一致
        attn_output = torch.bmm(attn_weights_t, text_guided_vision)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # --------------------------
        # 8. 重塑输出并投影
        # --------------------------
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)  # 恢复多头维度
        attn_output = attn_output.transpose(1, 2)  # (bsz, tgt_len, num_heads, head_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)  # 合并多头：(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)  # 最终线性投影

        # 输出注意力权重（若需要）
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, text_len, tgt_len)
        else:
            attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):

        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.activation_fn = ACT2FN[config.hidden_act]
        # self.activation_fn = nn.ReLU
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 768, 3072
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TOKEN_MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer_with_Crossmodal_Text_Guided_Fusion(nn.Module):

    def __init__(self, args, i, clip_encoder_layer, config: CLIPConfig, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.args = args
        self.embed_dim = clip_encoder_layer.embed_dim
        self.self_attn = clip_encoder_layer.self_attn

        """ Multi-layer Adaptive Cross-modal Text_Guided_Fusion """
        # self.enable_adaptive_weights = args.enable_adaptive_weights
        if i in adapt_layer:
            if self.args.modality == 'rgb':
                self.cross_norm_sv = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # eps=1e-05
                self.cross_attn_sv = CLIP_Cross_Attention_VS(config)
                self.cross_mlp_sv = CLIPMLP(config)
            elif self.args.modality == 'rgbt':
                self.cross_norm_st = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # eps=1e-05
                self.cross_norm_sv = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # eps=1e-05
                self.cross_attn_st = CLIP_Cross_Attention_VS(config)
                self.cross_attn_sv = CLIP_Cross_Attention_VS(config)
                self.cross_mlp_st = CLIPMLP(config)
                self.cross_mlp_sv = CLIPMLP(config)


        text_embed_dim = text_config.hidden_size  # 512 for base model, 768 for Large model
        # self.cross_gate = nn.Linear(text_embed_dim * len(extract_text_layer), self.embed_dim)  # clip vision 768
        # self.cross_adaptive_weights = nn.ModuleList([nn.Embedding(77, text_embed_dim) for i in range(len(extract_text_layer))])

        self.layer_norm1 = clip_encoder_layer.layer_norm1
        self.mlp = clip_encoder_layer.mlp
        self.layer_norm2 = clip_encoder_layer.layer_norm2

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer,
        adapt_layer,
        text_states,
        cur_modality,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        """ Multi-layer Adaptive Cross-modal Text_Guided_Fusion """
        if layer in adapt_layer:
            text_guided_fusion = True
            if text_guided_fusion == True:

                if cur_modality=='ir':
                    residual = hidden_states
                    hidden_states = self.cross_norm_st(hidden_states)
                    text_states = text_states[-1].to(hidden_states.dtype).permute(1, 0, 2)
                    
                    hidden_states, attn_weights = self.cross_attn_st(
                        hidden_states=hidden_states,
                        text_states=text_states,
                        attention_mask=attention_mask,
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=output_attentions,
                    )
                    hidden_states = self.cross_mlp_st(hidden_states)
                else:
                    residual = hidden_states
                    hidden_states = self.cross_norm_sv(hidden_states)
                    text_states = text_states[-1].to(hidden_states.dtype).permute(1, 0, 2)
                    hidden_states, attn_weights = self.cross_attn_sv(
                        hidden_states=hidden_states,
                        text_states=text_states,
                        attention_mask=attention_mask,
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=output_attentions,
                    )
                    hidden_states = self.cross_mlp_sv(hidden_states)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)


        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPEncoder_with_Crossmodal_Text_Guided_Fusion(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, args, clip_encoder, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.config = clip_encoder.config
        self.layers = nn.ModuleList([CLIPEncoderLayer_with_Crossmodal_Text_Guided_Fusion(args, i, clip_encoder.layers[i], self.config,
                                                                             adapt_layer, extract_text_layer,
                                                                             text_config)
                                     for i in range(self.config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        adapt_layer,
        text_states,
        cur_modality,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states=hidden_states,
                    layer = idx,
                    adapt_layer = adapt_layer,
                    text_states = text_states,
                    cur_modality = cur_modality,
                    attention_mask=attention_mask,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=output_attentions,
                    )

            else:
                layer_outputs = encoder_layer(

                    hidden_states=hidden_states,
                    layer = idx,
                    adapt_layer = adapt_layer,
                    text_states = text_states,
                    cur_modality=cur_modality,
                    attention_mask=attention_mask,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=output_attentions,

                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return {"last_hidden_state": hidden_states, "hidden_states": encoder_states, "attentions": all_attentions}


class CLIP_Vision_Model_with_Crossmodal_Text_Guided_Fusion(nn.Module):
    def __init__(self, args, clip_visu_model, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.config = clip_visu_model.config
        self.embeddings = Modified_CLIPVisionEmbeddings(args, clip_visu_model.embeddings)
        self.pre_layrnorm = clip_visu_model.pre_layrnorm  # 原版代码拼错了
        self.encoder = CLIPEncoder_with_Crossmodal_Text_Guided_Fusion(args, clip_visu_model.encoder, adapt_layer, extract_text_layer,
                                                          text_config)
        self.post_layernorm = clip_visu_model.post_layernorm

    # @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        adapt_layer,
        text_states,
        reg_src,
        cur_modality,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            adapt_layer=adapt_layer,
            text_states=text_states,
            cur_modality=cur_modality,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs["last_hidden_state"]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
        }


"""
   MMVG is implemented on the basis of CLIP-VG, Github: https://github.com/linhuixiao/CLIP-VG
"""


class MMVGFusion(nn.Module):
    def __init__(self, args):
        super(MMVGFusion, self).__init__()
        self.args = args
        # Keep every experimental frontend behind an explicit FusionMethod so
        # baseline IAFv3 and completed GQR runs remain numerically unchanged.
        # ``InfMAEDirectV2`` is the v2-plan A2 route: RGB uses CLIP+RGB LoRA
        # and TIR uses InfMAE directly, with no CLIP-TIR encoder or TIR LoRA.
        # A3/A4/A5 retain that exact frontend and add target-level alignment
        # supervision.  A5Bridge additionally routes the learned semantic
        # representation back into the TIR token stream before LAVS.
        self.fusion_method = getattr(args, "FusionMethod", "concat")
        self.enable_infmae_hybrid = self.fusion_method == "GQRv1InfMAE"
        self._infmae_direct_fusion_methods = {
            "InfMAEDirectV2",
            "InfMAEA3",
            "InfMAEA4",
            "InfMAEA5",
            "InfMAEA5Bridge",
        }
        self.enable_infmae_direct = self.fusion_method in self._infmae_direct_fusion_methods
        self.enable_infmae_semantic_bridge = self.fusion_method == "InfMAEA5Bridge"
        self.infmae_alignment_mode = self._resolve_infmae_alignment_mode(args)
        self.enable_infmae_tir_text_alignment = self.infmae_alignment_mode in {
            "tir_text",
            "both",
        }
        self.enable_infmae_rgb_tir_transfer = self.infmae_alignment_mode in {
            "rgb_tir",
            "both",
        }
        self.enable_infmae_alignment = (
            self.enable_infmae_tir_text_alignment
            or self.enable_infmae_rgb_tir_transfer
        )
        # The loss function sees this same args instance.  Store the resolved
        # mode rather than forcing it to duplicate FusionMethod alias logic.
        self.args.infmae_alignment_mode = self.infmae_alignment_mode
        # Retain the historical attribute for the hybrid experiment only.
        self.enable_infmae = self.enable_infmae_hybrid
        print("init MMVGFusion model...")
        self.backbone_type = getattr(args, "vl_backbone", "clip").lower()
        self.is_siglip2 = self.backbone_type == "siglip2"
        self.tokenizer = None

        if self.is_siglip2:
            from .siglip2_adapter import load_siglip2, load_siglip2_tokenizer

            pretrained_model_path = getattr(args, "pretrained_model_path", "")
            if not pretrained_model_path:
                raise ValueError("--pretrained_model_path is required when --vl_backbone=siglip2")
            print(f"init SigLIP2 from {pretrained_model_path}")
            # The fixed-resolution checkpoint is a SiglipModel.  Calling the
            # concrete class avoids AutoModel's optional timm-wrapper lookup.
            self.clip = load_siglip2(pretrained_model_path)
            self.tokenizer = load_siglip2_tokenizer(pretrained_model_path)
            self.extract_vision_layer = [1, 4, 8, 12]
            self.adapt_layer = [0, 3, 7, 11]
            self.patch_size = 16
            expected_image_size = int(self.clip.config.vision_config.image_size)
            expected_text_len = int(self.clip.config.text_config.max_position_embeddings)
            if args.imsize != expected_image_size:
                raise ValueError(
                    f"The local SigLIP2 checkpoint requires --imsize {expected_image_size}, "
                    f"got {args.imsize}."
                )
            if args.max_query_len != expected_text_len:
                raise ValueError(
                    f"The local SigLIP2 checkpoint requires --max_query_len {expected_text_len}, "
                    f"got {args.max_query_len}."
                )
        elif (args.model == "ViT-L/14-336"):
            print("init CLIP ViT-L/14-336")
            self.clip = CLIPModel.from_pretrained("/home/shared/pretrain_model/pretrained_weights/CLIP/clip-vit-large-patch14-336")
            self.extract_vision_layer = [12, 16, 20, 24]  # v4
            self.adapt_layer = [11, 15, 19, 23]
            self.patch_size = 14
        elif (args.model == "ViT-L/14"):  # main large model
            print("init CLIP ViT-L/14")
            self.clip = CLIPModel.from_pretrained("/home/shared/pretrain_model/pretrained_weights/CLIP/clip-vit-large-patch14")
            self.extract_vision_layer = [6, 12, 18, 24]  # final 版本
            self.adapt_layer = [] if args.warmup is True else [4, 10, 16, 22]  # large model is trained on two phrases
            self.patch_size = 14
        elif (args.model == "ViT-B/32"):
            print("init CLIP ViT-B/32")
            self.clip = CLIPModel.from_pretrained("/home/shared/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch32")
            self.extract_vision_layer = [1, 4, 8, 12]
            self.adapt_layer = [0, 3, 7, 11]
            self.patch_size = 32
        else:  # default base model
            print("init CLIP ViT-B/16")
            # self.clip = CLIPModel.from_pretrained("/home/shared/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16")
            self.clip = CLIPModel.from_pretrained("../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16")
            """
             Note that there is no mistake here. Note that [1, 4, 8, 12], [0, 3, 7, 11] are the same layer.
             In the internal implementation of transformers, the index at vision branch [0] is the original
             image embedding. 
            """
            self.extract_vision_layer = [1, 4, 8, 12]
            self.adapt_layer = [0, 3, 7, 11]
            self.patch_size = 16
        # set extract_text_layer
        self.mixup_pretrain = args.mixup_pretrain
        if self.mixup_pretrain:
            self.extract_text_layer = [12]
        else:
            if args.dataset == "gref_umd" or args.dataset == "gref":
                self.extract_text_layer = [i+1 for i in range(12)]
            elif args.dataset == "unc+":
                self.extract_text_layer = [6, 12]
            elif args.dataset == "unc":
                self.extract_text_layer = [12]
            elif args.dataset == "referit":
                self.extract_text_layer = [6, 12]
            else:
                self.extract_text_layer = [12]

        print("\nextract vision layer: ", self.extract_vision_layer)
        print("extract text layer: ", self.extract_text_layer)
        print("image size: ", args.imsize, " * ", args.imsize)

        print("adapt_layer: ", self.adapt_layer)

        if self.is_siglip2:
            from .siglip2_adapter import SiglipVisionModelWithTextGuidedFusion

            self.clip.vision_model = SiglipVisionModelWithTextGuidedFusion(
                args,
                self.clip.vision_model,
                self.adapt_layer,
            )
        else:
            self.clip.vision_model = CLIP_Vision_Model_with_Crossmodal_Text_Guided_Fusion(
                args,
                self.clip.vision_model,
                self.adapt_layer,
                self.extract_text_layer,
                self.clip.text_model.config,
            )
        self.cross_fusion_layers_vt = nn.ModuleList()
        self.cross_fusion_layers_tv = nn.ModuleList()
        if self.args.modality =="rgbt":
            for _ in self.extract_vision_layer:
                cross_attn_vt = CLIP_Cross_Attention(self.clip.vision_model.encoder.config)
                cross_attn_tv = CLIP_Cross_Attention(self.clip.vision_model.encoder.config)
                self.cross_fusion_layers_vt.append(cross_attn_vt)
                self.cross_fusion_layers_tv.append(cross_attn_tv)

        """
            srameter in self.clip.parameters():
                        parameter.requires_grad_(False)elf.clip.print_trainable_parameters()
            Note that the essence of the HiLoRA mechanism is a process of decomposing parameter learning, and its
            effectiveness is influenced by the learning rate and the number of epochs. Therefore, HiLoRA requires
            different learning rates and numbers of epochs at various stages for specific model configurations.
            If you do not need to enable HiLoRA, simply leave args.hi_lora_stage=0 as the default.
        """
        # open_lora = True
        self.open_lora = args.open_lora
        self.open_text_guided_fusion = args.open_text_guided_fusion

        self.set_HiLoRA(args)

        self.backbone_visual_dim = self.clip.vision_model.config.hidden_size
        self.backbone_text_dim = self.clip.text_model.config.hidden_size
        if self.is_siglip2:
            # SigLIP2 has no CLIP projection_dim.  Keep the MMVG VL head at
            # 512 while adapting the 768-wide SigLIP2 features explicitly.
            self.hidden_dim = args.vl_hidden_dim
        else:
            self.hidden_dim = self.clip.projection_dim  # base model 512，large model 768
        self.imsize = args.imsize
        self.enable_gqr = self._as_bool(getattr(args, "enable_gqr", False))
        # ``GQRv1InfMAE`` is deliberately a distinct fusion setting rather
        # than changing GQRv1 in-place.  Existing GQR checkpoints/runs retain
        # their exact architecture and the InfMAE path is opt-in.
        self._gqr_fusion_methods = {"GQRv1", "GQRv1InfMAE"}
        if self.enable_gqr and self.args.modality != "rgbt":
            raise ValueError("--enable_gqr requires --modality rgbt")
        if self.enable_gqr and self.fusion_method not in self._gqr_fusion_methods:
            raise ValueError(
                "--enable_gqr requires --FusionMethod GQRv1 or GQRv1InfMAE"
            )
        if self.fusion_method in self._gqr_fusion_methods and not self.enable_gqr:
            raise ValueError(
                f"--FusionMethod {self.fusion_method} requires --enable_gqr"
            )
        if self.enable_infmae and self.args.modality != "rgbt":
            raise ValueError("GQRv1InfMAE requires --modality rgbt")
        if self.enable_infmae and int(self.imsize) != 224:
            raise ValueError(
                "The supplied InfMAE checkpoint supports 224x224 inputs only; "
                f"got --imsize {self.imsize}."
            )
        if self.enable_infmae_direct and self.enable_gqr:
            raise ValueError(
                "Direct InfMAE A2/A3/A4/A5 modes are independent of GQR; "
                "remove --enable_gqr."
            )
        if self.enable_infmae_alignment and not self.enable_infmae_direct:
            raise ValueError(
                "Target-aware InfMAE alignment requires an InfMAE direct "
                "FusionMethod (InfMAEDirectV2/InfMAEA3/InfMAEA4/InfMAEA5/"
                "InfMAEA5Bridge)."
            )
        if self.enable_infmae_direct and self.args.modality != "rgbt":
            raise ValueError(
                "Direct InfMAE A2/A3/A4/A5/A5Bridge requires --modality rgbt"
            )
        if self.enable_infmae_direct and int(self.imsize) != 224:
            raise ValueError(
                "The supplied InfMAE checkpoint supports 224x224 inputs only; "
                f"got --imsize {self.imsize}."
            )
        if self.enable_infmae_direct and self.is_siglip2:
            raise ValueError(
                "Direct InfMAE implements the v2 CLIP-RGB + InfMAE-TIR plan; "
                "use --vl_backbone clip."
            )
        if self.enable_infmae_direct and str(args.model) != "ViT-B/16":
            raise ValueError(
                "Direct InfMAE is currently validated for --model ViT-B/16, "
                f"got {args.model!r}."
            )
        clip_visu_hidden_dim = self.backbone_visual_dim  # 768
        self.visu_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.is_siglip2:
            self.condition_text_proj = nn.Identity()
        else:
            self.condition_text_proj = nn.Linear(self.hidden_dim, clip_visu_hidden_dim)  # clip vision 768
        self.ml_text_feat_perceiver = nn.Linear(
            self.backbone_text_dim * len(self.extract_text_layer),
            clip_visu_hidden_dim,
        )
        self.text_proj = nn.Linear(self.backbone_text_dim, self.hidden_dim)
        self.reg_token = nn.Embedding(1, self.hidden_dim)
        self._init_fusion_modules()
        self._init_tsar_modules()
        self._init_infmae_modules()
        self._init_infmae_direct_modules()
        self._init_infmae_alignment_modules()

        # divisor = 16
        self.num_visu_token = int((args.imsize / self.patch_size) ** 2)
        self.num_text_token = args.max_query_len
        num_total = self.num_visu_token + 1 + self.num_text_token + 1  # v token + [cls]token + t token + [REG]token
        if args.modality=='rgbt':
            num_total = self.num_visu_token*2 + 1 + self.num_text_token + 1  # v token + [cls]token + t token + [REG]token
        self.vl_pos_embed = nn.Embedding(num_total, self.hidden_dim)

        self.vl_transformer = build_vl_transformer(args)

        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.reg_pos_embed = nn.Embedding(1, self.hidden_dim)
        self.condition_text_pos_embed = nn.Embedding(self.num_text_token, clip_visu_hidden_dim)
        self.ml_visual_projection = nn.Linear(
            len(self.extract_vision_layer) * self.backbone_visual_dim,
            self.hidden_dim,
        )
        if not self.is_siglip2:
            self.ml_visual_projection.weight = nn.Parameter(
                torch.cat(
                    [self.clip.visual_projection.weight for _ in range(len(self.extract_vision_layer))],
                    dim=1,
                )
            )
        # if args.modality=='rgbt':
        #     self.ml_visual_projection_ir = nn.Linear(len(self.extract_vision_layer) * self.clip.vision_model.config.hidden_size,
        #                                       self.hidden_dim)
        #     self.ml_visual_projection_ir.weight = nn.Parameter(self.ml_visual_projection.weight.clone())
              

        self.visu_token_norm = nn.LayerNorm(self.hidden_dim, eps=1e-05)  # 512, eps=1e-05
        self.visu_token_mlp = TOKEN_MLP(self.hidden_dim, 3072)  # 3072

        # TODO：Segmentation head for Referring Image Segmentation task. RIS works only when the seg mask is used.
        #  seg conv, 10GB, 14*14 --> 28*28 --> 56*56 --> 112*112
        hidden_dim = self.hidden_dim
        self.seg_conv1 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False
        self.seg_conv2 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False
        self.seg_conv3 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False

    def _init_tsar_modules(self):
        """Create TSAR modules only when GQR is explicitly enabled.

        Keeping these attributes absent from the baseline state dict is part of
        the checkpoint and numerical-compatibility contract.
        """
        self.gqr = None
        self.rgb_aux_head = None
        self.tir_aux_head = None
        self.gqr_text_proj = None
        self.gqr_eta_raw = None
        self.gqr_eta_max = 0.0
        # Runtime-only state: it deliberately stays out of the checkpoint so
        # previous baseline/GQR checkpoints remain compatible.
        self._gqr_correction_scale = 1.0
        self._tsar_aux = {}

        if not self.enable_gqr:
            return

        self.gqr_eta_max = float(getattr(self.args, "gqr_eta_max", 2.0))
        if self.gqr_eta_max < 0:
            raise ValueError("--gqr_eta_max must be non-negative")

        # CLIP uses a projected pooled text embedding, while SigLIP2 exposes
        # its pre-projection pooled state.  Both are adapted to MMVG's VL
        # hidden dimension before being used by the new heads.
        gqr_text_input_dim = self.backbone_text_dim if self.is_siglip2 else self.hidden_dim
        self.gqr_text_proj = nn.Linear(gqr_text_input_dim, self.hidden_dim)
        self.rgb_aux_head = TextConditionedModalityHead(self.hidden_dim)
        self.tir_aux_head = TextConditionedModalityHead(self.hidden_dim)
        self.gqr = GroundingQualityRouter(self.hidden_dim)
        self.gqr_eta_raw = nn.Parameter(torch.tensor(0.0))

    def _init_infmae_modules(self):
        """Create the optional frozen InfMAE thermal expert path.

        It is intentionally initialized only for ``GQRv1InfMAE``.  The
        original CLIP+LoRA TIR stream remains present and is enhanced later by
        a zero-init residual, so no pre-existing FusionMethod changes behavior.
        """

        self.infmae_encoder = None
        self.infmae_input_normalizer = None
        self.infmae_adapter = None
        if not self.enable_infmae:
            return

        checkpoint_path = Path(__file__).resolve().parents[1] / "InfMAE" / "InfMAE.pth"
        self.infmae_input_normalizer = InfMAEThermalInputNormalizer(
            dataset=self.args.dataset,
            image_norm=getattr(self.args, "image_norm", "dataset"),
        )
        self.infmae_encoder = FrozenInfMAEEncoder(checkpoint_path)
        self.infmae_adapter = InfMAEMultiScaleThermalAdapter(
            f2_dim=self.infmae_encoder.f2_dim,
            f3_dim=self.infmae_encoder.f3_dim,
            out_dim=self.hidden_dim,
            residual_gain_init=0.5,
        )
        print(
            "Enabled frozen InfMAE Inf30 expert: "
            f"F2={self.infmae_encoder.f2_dim}, F3={self.infmae_encoder.f3_dim}, "
            f"checkpoint={checkpoint_path}"
        )

    def _init_infmae_direct_modules(self):
        """Create the v2-plan direct InfMAE TIR frontend.

        This is intentionally separate from ``GQRv1InfMAE``: direct mode
        replaces the CLIP-TIR feature producer before LAVS, whereas the
        historical hybrid mode adds an InfMAE residual after a CLIP-TIR pass.
        Keeping the modules and state-dict prefixes distinct prevents either
        experiment from being accidentally resumed as the other.
        """

        self.infmae_direct_encoder = None
        self.infmae_direct_input_normalizer = None
        self.infmae_direct_adapter = None
        self.infmae_direct_cls_projection = None
        if not self.enable_infmae_direct:
            return

        checkpoint_path = Path(__file__).resolve().parents[1] / "InfMAE" / "InfMAE.pth"
        self.infmae_direct_input_normalizer = InfMAEThermalInputNormalizer(
            dataset=self.args.dataset,
            image_norm=getattr(self.args, "image_norm", "dataset"),
        )
        self.infmae_direct_encoder = FrozenInfMAEEncoder(checkpoint_path)
        self.infmae_direct_adapter = InfMAEDirectThermalAdapter(
            f2_dim=self.infmae_direct_encoder.f2_dim,
            f3_dim=self.infmae_direct_encoder.f3_dim,
            token_dim=self.backbone_visual_dim,
            num_feature_levels=len(self.extract_vision_layer),
            bottleneck_dim=max(self.backbone_visual_dim // 4, 1),
            f2_scale_init=0.10,
        )
        self.infmae_direct_cls_projection = nn.Linear(
            self.backbone_visual_dim,
            self.hidden_dim,
        )
        # The direct TIR global embedding is not used by the RGB CLIP loss,
        # but initializing this compatibility output from CLIP's projection
        # keeps its scale meaningful for existing diagnostics.
        if (
            not self.is_siglip2
            and tuple(self.infmae_direct_cls_projection.weight.shape)
            == tuple(self.clip.visual_projection.weight.shape)
        ):
            with torch.no_grad():
                self.infmae_direct_cls_projection.weight.copy_(
                    self.clip.visual_projection.weight
                )
                self.infmae_direct_cls_projection.bias.zero_()
        # The legacy RGBT output interface exposes an IR global-logit matrix,
        # but the unchanged CLIP contrastive objective consumes RGB logits
        # only.  Keep this projection as a fixed compatibility diagnostic
        # rather than adding an optimizer parameter with no gradient path.
        for parameter in self.infmae_direct_cls_projection.parameters():
            parameter.requires_grad_(False)
        print(
            "Enabled direct InfMAE TIR frontend (no TIR CLIP/LoRA): "
            f"F2={self.infmae_direct_encoder.f2_dim}, "
            f"F3={self.infmae_direct_encoder.f3_dim}, "
            f"checkpoint={checkpoint_path}"
        )

    def _init_infmae_alignment_modules(self):
        """Build A3/A4/A5 heads and the optional A5 semantic bridge.

        RGB CLIP text/image embeddings are used as frozen semantic anchors.
        Only the two TIR projection heads and the direct thermal adapter are
        trained by these auxiliary objectives, so RGB semantics guide thermal
        features rather than being pulled toward a random new embedding space.
        """

        self.infmae_target_pool = None
        self.infmae_tir_text_projector = None
        self.infmae_tir_rgb_projector = None
        # Runtime-only gradient gate.  It is deliberately not checkpointed:
        # the training epoch sets it before every forward pass.
        self._infmae_alignment_adapter_scale = 1.0
        # A separate gate lets an A3/A5 ablation distinguish semantic
        # adaptation from ordinary grounding fine-tuning.  It leaves all
        # forward activations unchanged and affects only the gradient that
        # the main grounding losses send back into the direct TIR adapter.
        self._infmae_grounding_adapter_gradient_scale = float(
            getattr(self.args, 'infmae_grounding_adapter_gradient_scale', 1.0)
        )
        if not 0.0 <= self._infmae_grounding_adapter_gradient_scale <= 1.0:
            raise ValueError(
                '--infmae_grounding_adapter_gradient_scale must be in [0, 1]'
            )
        if not self.enable_infmae_alignment:
            return

        # The direct A2 adapter is initialized immediately before these
        # optional heads, while the detector/VL stack is initialized after
        # them.  Preserve the CPU RNG stream so merely selecting A3/A5 or
        # A5Bridge cannot change any downstream baseline parameter at epoch
        # zero.  The model is constructed on CPU and moved to CUDA only by
        # the caller, so the CPU state is the relevant one here.
        downstream_rng_state = torch.get_rng_state()
        self.infmae_target_pool = TargetAwareTokenPool(
            pool_size=int(getattr(self.args, "infmae_target_pool_size", 4))
        )
        if self.enable_infmae_tir_text_alignment:
            self.infmae_tir_text_projector = TargetFeatureProjector(
                self.backbone_visual_dim,
                self.hidden_dim,
            )
        if self.enable_infmae_rgb_tir_transfer:
            self.infmae_tir_rgb_projector = TargetFeatureProjector(
                self.backbone_visual_dim,
                self.hidden_dim,
            )

        # Starting each TIR head from CLIP's visual projection gives the
        # frozen RGB/text semantic space a compatible scale.  The heads remain
        # trainable and learn the actual InfMAE-to-CLIP mapping from target
        # regions, rather than inheriting any TIR CLIP encoder or TIR LoRA.
        if not self.is_siglip2:
            source_projection = self.clip.visual_projection
            for projector in (
                self.infmae_tir_text_projector,
                self.infmae_tir_rgb_projector,
            ):
                if projector is None:
                    continue
                if tuple(projector.projection.weight.shape) != tuple(source_projection.weight.shape):
                    raise RuntimeError(
                        "InfMAE target projector and CLIP visual projection "
                        "must have matching shapes"
                    )
                with torch.no_grad():
                    projector.projection.weight.copy_(source_projection.weight)
                    if projector.projection.bias is not None:
                        if source_projection.bias is None:
                            projector.projection.bias.zero_()
                        else:
                            projector.projection.bias.copy_(source_projection.bias)

        if self.enable_infmae_semantic_bridge:
            if self.infmae_tir_text_projector is None:
                raise RuntimeError(
                    "InfMAEA5Bridge requires the A3 TIR-to-Text projector"
                )
            # Keep the bridge inside the direct adapter.  This both makes it
            # part of the thermal frontend checkpoint contract and lets the
            # existing adapter-only optimizer mode include it naturally.
            bridge = TextGuidedThermalSemanticBridge(
                self.backbone_visual_dim,
                self.hidden_dim,
                temperature=float(getattr(self.args, "infmae_tir_text_tau", 0.07)),
            )
            if not self.is_siglip2:
                source_projection = self.clip.visual_projection
                expected_shape = tuple(bridge.text_to_visual.weight.shape)
                source_shape = tuple(source_projection.weight.t().shape)
                if expected_shape != source_shape:
                    raise RuntimeError(
                        "InfMAE semantic bridge and CLIP visual projection "
                        "must have transposed matching shapes"
                    )
                with torch.no_grad():
                    bridge.text_to_visual.weight.copy_(source_projection.weight.t())
            self.infmae_direct_adapter.add_module("semantic_bridge", bridge)

        torch.set_rng_state(downstream_rng_state)

        print(
            "Enabled target-aware InfMAE alignment: "
            f"mode={self.infmae_alignment_mode}, "
            f"pool={self.infmae_target_pool.pool_size}x{self.infmae_target_pool.pool_size}"
        )
        if self.enable_infmae_semantic_bridge:
            print(
                "Enabled InfMAEA5Bridge: zero-init text-guided TIR semantic "
                "residual before LAVS"
            )

    _INFMAE_MISSING_PREFIXES = (
        "infmae_encoder.",
        "infmae_input_normalizer.",
        "infmae_adapter.",
    )
    _INFMAE_OMIT_FROM_CHECKPOINT_PREFIXES = (
        "infmae_encoder.",
        "infmae_input_normalizer.",
    )
    _INFMAE_DIRECT_MISSING_PREFIXES = (
        "infmae_direct_encoder.",
        "infmae_direct_input_normalizer.",
        "infmae_direct_adapter.",
        "infmae_direct_cls_projection.",
        "infmae_tir_text_projector.",
        "infmae_tir_rgb_projector.",
    )
    _INFMAE_DIRECT_OMIT_FROM_CHECKPOINT_PREFIXES = (
        "infmae_direct_encoder.",
        "infmae_direct_input_normalizer.",
    )

    def state_dict(self, *args, **kwargs):
        """Avoid serializing a duplicate frozen InfMAE pretraining checkpoint."""

        state = super().state_dict(*args, **kwargs)
        if getattr(self, "enable_infmae", False):
            for key in tuple(state):
                if key.startswith(self._INFMAE_OMIT_FROM_CHECKPOINT_PREFIXES):
                    state.pop(key)
        if getattr(self, "enable_infmae_direct", False):
            for key in tuple(state):
                if key.startswith(self._INFMAE_DIRECT_OMIT_FROM_CHECKPOINT_PREFIXES):
                    state.pop(key)
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Keep original/MMVG checkpoints compatible with GQRv1InfMAE.

        Frozen encoder weights always come from the explicitly verified local
        InfMAE checkpoint.  Adapter weights are absent when bootstrapping from
        an earlier MMVG/GQR checkpoint and are therefore expected missing keys.
        """

        try:
            incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
        except TypeError:  # Compatibility with older PyTorch releases.
            incompatible = super().load_state_dict(state_dict, strict=False)

        missing_keys = list(incompatible.missing_keys)
        if getattr(self, "enable_infmae", False):
            missing_keys = [
                key
                for key in missing_keys
                if not key.startswith(self._INFMAE_MISSING_PREFIXES)
            ]
        unexpected_keys = list(incompatible.unexpected_keys)
        if getattr(self, "enable_infmae_direct", False):
            missing_keys = [
                key
                for key in missing_keys
                if not key.startswith(self._INFMAE_DIRECT_MISSING_PREFIXES)
            ]
            # Released MMVG checkpoints contain a duplicated legacy TIR LoRA
            # adapter.  Direct InfMAE mode deliberately has no such adapter,
            # so these are expected compatibility-only keys, not an error.
            unexpected_keys = [
                key for key in unexpected_keys if ".lora_ir." not in key
            ]
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict for MMVGFusion: "
                f"missing keys={missing_keys}, unexpected keys={unexpected_keys}"
            )
        return type(incompatible)(missing_keys, unexpected_keys)

    def set_gqr_correction_scale(self, scale):
        """Gate GQR's fusion residual during auxiliary-head warmup.

        At zero, GQRv1 remains exactly equivalent to IAFv3 even after
        optimizer steps, so an untrained router cannot perturb fusion.
        """
        scale = float(scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f'GQR correction scale must be in [0, 1], got {scale}')
        self._gqr_correction_scale = scale

    def set_infmae_alignment_adapter_scale(self, scale):
        """Set the auxiliary-gradient multiplier entering the TIR adapter."""
        scale = float(scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                'InfMAE alignment adapter scale must be in [0, 1], '
                f'got {scale}'
            )
        self._infmae_alignment_adapter_scale = scale

    def set_infmae_grounding_adapter_gradient_scale(self, scale):
        """Set the main-grounding gradient multiplier for the TIR adapter.

        This is intentionally independent of
        ``set_infmae_alignment_adapter_scale``: the latter warms up the new
        auxiliary heads, while this switch can produce a clean semantic-only
        adapter ablation after those heads have been calibrated.
        """
        scale = float(scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                'InfMAE grounding adapter gradient scale must be in [0, 1], '
                f'got {scale}'
            )
        self._infmae_grounding_adapter_gradient_scale = scale

    @staticmethod
    def _scale_infmae_alignment_gradient(feature, scale):
        """Keep a feature's forward value while scaling only its gradient.

        A zero scale still lets the alignment projectors learn from the
        actual target feature, but prevents their initially uncalibrated
        losses from modifying the direct InfMAE adapter.
        """
        scale = float(scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                'InfMAE alignment gradient scale must be in [0, 1], '
                f'got {scale}'
            )
        return feature.detach() + (feature - feature.detach()) * scale

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _resolve_infmae_alignment_mode(args):
        """Resolve A2/A3/A4/A5 aliases into one explicit auxiliary mode.

        ``InfMAEDirectV2`` remains a clean A2 ablation.  It may still be used
        with an explicit CLI override for controlled studies, whereas named
        A3/A4/A5 aliases reject contradictory overrides so checkpoint names
        and enabled losses cannot silently disagree.
        """

        method_defaults = {
            "InfMAEDirectV2": "none",
            "InfMAEA3": "tir_text",
            "InfMAEA4": "rgb_tir",
            "InfMAEA5": "both",
            "InfMAEA5Bridge": "both",
        }
        requested = str(getattr(args, "infmae_alignment_mode", "auto")).lower()
        allowed = {"auto", "none", "tir_text", "rgb_tir", "both"}
        if requested not in allowed:
            raise ValueError(
                "--infmae_alignment_mode must be one of "
                "auto/none/tir_text/rgb_tir/both"
            )
        method = getattr(args, "FusionMethod", "concat")
        default = method_defaults.get(method, "none")
        if requested == "auto":
            return default
        if method in {"InfMAEA3", "InfMAEA4", "InfMAEA5", "InfMAEA5Bridge"} and requested != default:
            raise ValueError(
                f"{method} requires --infmae_alignment_mode {default!r}, "
                f"got {requested!r}"
            )
        return requested

    @staticmethod
    def _lora_layer_index(name):
        match = re.search(r"encoder\.layers\.(\d+)", name)
        return int(match.group(1)) if match else None

    def _set_lora_stage_trainable(self, stage):
        """Freeze/unfreeze existing adapters without re-wrapping the model."""
        if not self.open_lora:
            return
        stage_limits = {0: 0, 1: 5, 2: 8, 3: 12}
        if stage not in stage_limits:
            raise ValueError(f"Unsupported HiLoRA stage: {stage}")
        # The v2 frontend has one RGB LoRA adapter only.  Its Stage 1 design
        # explicitly trains that adapter from the beginning; keeping the
        # legacy 0->5->8->12 HiLoRA schedule would freeze it for 60 epochs and
        # then partially re-freeze it again.  This branch changes no baseline
        # or GQR behavior and leaves the frozen InfMAE encoder untouched.
        layer_limit = 12 if getattr(self, "enable_infmae_direct", False) else stage_limits[stage]
        for name, parameter in self.clip.named_parameters():
            is_lora_parameter = ".lora_A." in name or ".lora_B." in name
            parameter.requires_grad_(False)
            if is_lora_parameter:
                layer_index = self._lora_layer_index(name)
                parameter.requires_grad_(layer_index is not None and layer_index < layer_limit)

    def set_HiLoRA(self, args):
        self.open_lora = self._as_bool(getattr(args, "open_lora", True))
        self.open_text_guided_fusion = self._as_bool(
            getattr(args, "open_text_guided_fusion", True)
        )

        if self.open_lora and not getattr(self, "_lora_initialized", False):
            # These names match the attention projections in both CLIP and
            # SigLIP.  The old fc_in/fc_out/wte names did not match the
            # Transformers CLIP/SigLIP modules and are intentionally removed.
            target_modules = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
            ]
            peft_config_rgb = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                target_modules=target_modules,
                inference_mode=False,
                r=args.lora_r_rgb,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
            )
            self.clip = get_peft_model(
                self.clip,
                peft_config_rgb,
                adapter_name="lora_rgb",
            )
            if not getattr(self, "enable_infmae_direct", False):
                peft_config_ir = LoraConfig(
                    task_type="FEATURE_EXTRACTION",
                    target_modules=target_modules,
                    inference_mode=False,
                    r=args.lora_r_ir,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none",
                )
                self.clip.add_adapter("lora_ir", peft_config_ir)
            self._lora_initialized = True

        if self.open_lora:
            for parameter in self.clip.parameters():
                parameter.requires_grad_(False)
            self._set_lora_stage_trainable(int(getattr(args, "hi_lora_stage", 0)))

        if self.open_text_guided_fusion and hasattr(self.clip, "vision_model"):
            print("Open Multi-layer Adaptive Cross-modal Text_Guided_Fusion parameters ...")
            for name, parameter in self.clip.vision_model.encoder.layers.named_parameters():
                if any(
                    marker in name
                    for marker in (
                        "cross_attn_sv",
                        "cross_attn_st",
                        "cross_norm_sv",
                        "cross_norm_st",
                        "cross_mlp_sv",
                        "cross_mlp_st",
                    )
                ):
                    parameter.requires_grad_(True)

        if self.open_lora and hasattr(self.clip, "print_trainable_parameters"):
            self.clip.print_trainable_parameters()
    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors

        return image_tensors, texts_tensors

    def get_masks(self, images: NestedTensor, texts: NestedTensor):
        # torch_resize = Resize([14, 14])
        torch_resize = Resize([int(self.imsize / self.patch_size), int(self.imsize / self.patch_size)])  # 14 * 14 = 196， or， 16 * 16 = 256
        visu_masks = torch_resize(images.mask)
        visu_masks = visu_masks.to(torch.bool)
        visu_masks = visu_masks.flatten(1)  # visu_mask：B*L, torch.Size([B, 196])
        # text mask follow bert process
        # text_masks = texts.mask.to(torch.bool)
        # text_masks = ~text_masks
        # text_masks = text_masks.flatten(1)
        # assert text_masks is not None

        return visu_masks

    def encode_text(self, text_data, device=None):
        if self.is_siglip2:
            # The local checkpoint ships a Gemma tokenizer.  SigLIP2 was
            # trained with lower-cased, fixed-length sequences; the
            # tokenizer metadata in this checkpoint does not enforce either
            # rule, so make them explicit here.
            text_data = [str(text).lower() for text in text_data]
            tokenized = self.tokenizer(
                text_data,
                padding="max_length",
                truncation=True,
                max_length=self.num_text_token,
                return_tensors="pt",
            )
            text_tensors = tokenized.input_ids.to(device)
            text_mask = text_tensors.eq(self.tokenizer.pad_token_id).bool()
        else:
            text_tensors = clip.tokenize(text_data, context_length=77, truncate=True).to(device)  # 4 * 77
            text_mask = text_tensors.eq(0).bool()  # 4 * 77, The ones that need masking are 1.
        return text_tensors, text_mask

    @staticmethod
    def _prepend_siglip_global_token(tokens):
        """Restore MMVG's [global, 196 patch] layout after SigLIP encoding."""
        global_token = tokens.mean(dim=1, keepdim=True)
        return torch.cat([global_token, tokens], dim=1)

    def _init_fusion_modules(self):
        self.cmx = None
        self.lif_weight_net = None
        self.lif_beta = 0.4

        self.iaf_gate = None
        self.iaf_alpha = None
        self.iaf_beta = None
        self.iaf_gamma = None

        self.iafv2_token_gate = None
        self.iafv2_channel_gate = None
        self.iafv2_alpha = None
        self.iafv2_beta = None
        self.iafv2_gamma = None
        self.iafv2_res = None

        self.iafv3_token_gate = None
        self.iafv3_channel_gate = None
        self.iafv3_alpha = None
        self.iafv3_beta = None
        self.iafv3_gamma = None
        self.iafv3_delta = None
        self.iafv3_temp = None

        if self.args.modality != "rgbt":
            return

        if self.fusion_method == "CMX":
            self.cmx = CMX(
                dim=self.hidden_dim,
                reduction=1,
                num_heads=self.args.vl_nheads,
                norm_layer=nn.BatchNorm2d,
                lambda_c=0.5,
                lambda_s=0.5,
            )
        if self.fusion_method in ("LIF", "LIF_ADD"):
            self.lif_weight_net = LIFWeightNet()
        if self.fusion_method in (
            "IAF", "IAFv2.1", "IAFv3", "GQRv1", "GQRv1InfMAE",
            "InfMAEDirectV2", "InfMAEA3", "InfMAEA4", "InfMAEA5", "InfMAEA5Bridge",
        ):
            self.iaf_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
            )
            self.iaf_alpha = nn.Parameter(torch.tensor(1.2))
            self.iaf_beta = nn.Parameter(torch.tensor(0.8))
            self.iaf_gamma = nn.Parameter(torch.tensor(0.6))
        if self.fusion_method in (
            "IAFv2.1", "IAFv3", "GQRv1", "GQRv1InfMAE",
            "InfMAEDirectV2", "InfMAEA3", "InfMAEA4", "InfMAEA5", "InfMAEA5Bridge",
        ):
            self.iafv2_token_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
            )
            self.iafv2_channel_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.iafv2_alpha = nn.Parameter(torch.tensor(1.0))
            self.iafv2_beta = nn.Parameter(torch.tensor(0.9))
            self.iafv2_gamma = nn.Parameter(torch.tensor(0.7))
            self.iafv2_res = nn.Parameter(torch.tensor(0.15))
        if self.fusion_method in (
            "IAFv3", "GQRv1", "GQRv1InfMAE",
            "InfMAEDirectV2", "InfMAEA3", "InfMAEA4", "InfMAEA5", "InfMAEA5Bridge",
        ):
            self.iafv3_token_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
            )
            self.iafv3_channel_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.iafv3_alpha = nn.Parameter(torch.tensor(1.0))
            self.iafv3_beta = nn.Parameter(torch.tensor(0.9))
            self.iafv3_gamma = nn.Parameter(torch.tensor(0.6))
            self.iafv3_delta = nn.Parameter(torch.tensor(0.2))
            self.iafv3_temp = nn.Parameter(torch.tensor(1.0))

    def _tokens_to_map(self, tokens):
        feat = tokens.permute(1, 0, 2)
        cls_feat = feat[:, :1, :]
        patch_feat = feat[:, 1:, :]
        batch_size, patch_len, channels = patch_feat.shape
        patch_num = int(math.sqrt(patch_len))
        assert patch_num * patch_num == patch_len
        patch_map = patch_feat.permute(0, 2, 1).reshape(batch_size, channels, patch_num, patch_num)
        return cls_feat, patch_map

    def _map_to_tokens(self, cls_feat, patch_map):
        patch_feat = patch_map.flatten(2).permute(0, 2, 1)
        feat = torch.cat([cls_feat, patch_feat], dim=1)
        return feat.permute(1, 0, 2)

    def _illumination_prior(self, rgb_image, target_hw):
        gray = 0.299 * rgb_image[:, 0:1] + 0.587 * rgb_image[:, 1:2] + 0.114 * rgb_image[:, 2:3]
        illum = F.avg_pool2d(gray, kernel_size=8, stride=8)
        if illum.shape[-2:] != target_hw:
            illum = F.interpolate(illum, size=target_hw, mode="bilinear", align_corners=False)
        illum = (illum - illum.mean(dim=(2, 3), keepdim=True)) / (illum.std(dim=(2, 3), keepdim=True) + 1e-6)
        illum = torch.sigmoid(illum)
        return illum

    def _fusion_add(self, rgb_tokens, ir_tokens, rgb_image):
        _ = rgb_image
        return rgb_tokens + ir_tokens

    def _fusion_concat(self, rgb_tokens, ir_tokens, rgb_image):
        _ = rgb_image
        return torch.cat([rgb_tokens, ir_tokens[1:, :, :]], dim=0)

    def _fusion_cmx(self, rgb_tokens, ir_tokens, rgb_image):
        _ = rgb_image
        cls_rgb, map_rgb = self._tokens_to_map(rgb_tokens)
        cls_ir, map_ir = self._tokens_to_map(ir_tokens)
        fused_map = self.cmx([map_rgb, map_ir])
        fused_cls = 0.5 * (cls_rgb + cls_ir)
        return self._map_to_tokens(fused_cls, fused_map)

    def _fusion_lif(self, rgb_tokens, ir_tokens, rgb_image):
        cls_rgb, map_rgb = self._tokens_to_map(rgb_tokens)
        cls_ir, map_ir = self._tokens_to_map(ir_tokens)

        weight = self.lif_weight_net(rgb_image)
        if weight.shape[-2:] != map_rgb.shape[-2:]:
            weight = F.interpolate(weight, size=map_rgb.shape[-2:], mode="bilinear", align_corners=False)
        weight = torch.clamp(weight, min=0.0, max=1.0)

        fused_map = weight * map_rgb + (1.0 - weight) * map_ir
        cls_w = weight.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
        fused_cls = cls_w * cls_rgb + (1.0 - cls_w) * cls_ir
        return self._map_to_tokens(fused_cls, fused_map)

    def _fusion_lif_add(self, rgb_tokens, ir_tokens, rgb_image):
        cls_rgb, map_rgb = self._tokens_to_map(rgb_tokens)
        cls_ir, map_ir = self._tokens_to_map(ir_tokens)

        weight = self.lif_weight_net(rgb_image)
        if weight.shape[-2:] != map_rgb.shape[-2:]:
            weight = F.interpolate(weight, size=map_rgb.shape[-2:], mode="bilinear", align_corners=False)
        step1 = (weight - 0.31) / 0.63
        step2 = torch.clamp(step1, max=0.5)
        weight = self.lif_beta * step2 + 0.5
        weight = torch.clamp(weight, min=0.0, max=1.0)

        fused_map = weight * map_rgb + (1.0 - weight) * map_ir
        cls_w = weight.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
        fused_cls = cls_w * cls_rgb + (1.0 - cls_w) * cls_ir
        return self._map_to_tokens(fused_cls, fused_map)

    def _fusion_iaf(self, rgb_tokens, ir_tokens, rgb_image):
        cls_rgb, map_rgb = self._tokens_to_map(rgb_tokens)
        cls_ir, map_ir = self._tokens_to_map(ir_tokens)

        illum = self._illumination_prior(rgb_image, map_rgb.shape[-2:])
        rgb_patch = map_rgb.flatten(2).permute(0, 2, 1)
        ir_patch = map_ir.flatten(2).permute(0, 2, 1)

        rgb_norm = F.normalize(rgb_patch, dim=-1)
        ir_norm = F.normalize(ir_patch, dim=-1)
        cosine_sim = (rgb_norm * ir_norm).sum(dim=-1, keepdim=True)
        discrepancy = 1.0 - cosine_sim

        gate_input = torch.cat([rgb_patch, ir_patch, torch.abs(rgb_patch - ir_patch)], dim=-1)
        gate_token = torch.sigmoid(self.iaf_gate(gate_input))

        illum_token = illum.flatten(2).permute(0, 2, 1)
        w_rgb = torch.sigmoid(
            self.iaf_alpha * (illum_token - 0.5)
            + self.iaf_beta * (0.5 - discrepancy)
            + self.iaf_gamma * (gate_token - 0.5)
        )

        fused_patch = w_rgb * rgb_patch + (1.0 - w_rgb) * ir_patch
        fused_map = fused_patch.permute(0, 2, 1).reshape_as(map_rgb)
        cls_w = w_rgb.mean(dim=1, keepdim=True)
        fused_cls = cls_w * cls_rgb + (1.0 - cls_w) * cls_ir
        return self._map_to_tokens(fused_cls, fused_map)

    def _fusion_iafv2_1(self, rgb_tokens, ir_tokens, rgb_image):
        cls_rgb, map_rgb = self._tokens_to_map(rgb_tokens)
        cls_ir, map_ir = self._tokens_to_map(ir_tokens)

        illum = self._illumination_prior(rgb_image, map_rgb.shape[-2:])
        illum_token = illum.flatten(2).permute(0, 2, 1)

        rgb_patch = map_rgb.flatten(2).permute(0, 2, 1)
        ir_patch = map_ir.flatten(2).permute(0, 2, 1)
        diff_patch = torch.abs(rgb_patch - ir_patch)

        token_in = torch.cat([rgb_patch, ir_patch, diff_patch], dim=-1)
        token_gate = torch.sigmoid(self.iafv2_token_gate(token_in))

        rgb_global = rgb_patch.mean(dim=1)
        ir_global = ir_patch.mean(dim=1)
        ch_in = torch.cat([rgb_global, ir_global, torch.abs(rgb_global - ir_global)], dim=-1)
        channel_gate = torch.sigmoid(self.iafv2_channel_gate(ch_in)).unsqueeze(1)

        channel_bias = channel_gate.mean(dim=-1, keepdim=True)
        w_rgb = torch.sigmoid(
            self.iafv2_alpha * (illum_token - 0.5)
            + self.iafv2_beta * (token_gate - 0.5)
            + self.iafv2_gamma * (channel_bias - 0.5)
        )

        mixed = w_rgb * rgb_patch + (1.0 - w_rgb) * ir_patch
        residual = torch.tanh(self.iafv2_res) * channel_gate * (rgb_patch - ir_patch)
        fused_patch = mixed + residual

        fused_map = fused_patch.permute(0, 2, 1).reshape_as(map_rgb)
        cls_w = w_rgb.mean(dim=1, keepdim=True)
        fused_cls = cls_w * cls_rgb + (1.0 - cls_w) * cls_ir
        return self._map_to_tokens(fused_cls, fused_map)

    def _iafv3_components(self, rgb_tokens, ir_tokens, rgb_image):
        """Return IAFv3's intermediate values without changing its math.

        ``w_rgb_base`` remains the original patch-level RGB reliability.  GQR
        uses it as a prior rather than replacing any IAFv3 computation.
        """
        cls_rgb, map_rgb = self._tokens_to_map(rgb_tokens)
        cls_ir, map_ir = self._tokens_to_map(ir_tokens)

        illum = self._illumination_prior(rgb_image, map_rgb.shape[-2:])
        illum_token = illum.flatten(2).permute(0, 2, 1)
        # S1: down-weight extreme dark/bright zones where illumination prior is less reliable.
        illum_conf = 1.0 - torch.clamp(4.0 * (illum_token - 0.5) ** 2, min=0.0, max=1.0)

        rgb_patch = map_rgb.flatten(2).permute(0, 2, 1)
        ir_patch = map_ir.flatten(2).permute(0, 2, 1)
        rgb_norm = F.layer_norm(rgb_patch, (rgb_patch.shape[-1],))
        ir_norm = F.layer_norm(ir_patch, (ir_patch.shape[-1],))
        diff_patch = torch.abs(rgb_norm - ir_norm)

        token_in = torch.cat([rgb_norm, ir_norm, diff_patch], dim=-1)
        temperature = torch.clamp(self.iafv3_temp, min=0.5, max=2.0)
        token_gate = torch.sigmoid(self.iafv3_token_gate(token_in) / temperature)

        rgb_global = rgb_norm.mean(dim=1)
        ir_global = ir_norm.mean(dim=1)
        ch_in = torch.cat([rgb_global, ir_global, torch.abs(rgb_global - ir_global)], dim=-1)
        channel_gate = torch.sigmoid(self.iafv3_channel_gate(ch_in)).unsqueeze(1)
        channel_bias = channel_gate.mean(dim=-1, keepdim=True)

        w_rgb = torch.sigmoid(
            self.iafv3_alpha * (illum_token - 0.5) * illum_conf
            + self.iafv3_beta * (token_gate - 0.5)
            + self.iafv3_gamma * (channel_bias - 0.5)
        )

        return {
            "cls_rgb": cls_rgb,
            "cls_ir": cls_ir,
            "map_rgb": map_rgb,
            "rgb_patch": rgb_patch,
            "ir_patch": ir_patch,
            "channel_gate": channel_gate,
            "w_rgb_base": w_rgb,
        }

    def _compose_iafv3_tokens(self, components, w_rgb):
        """Apply IAFv3 mixing and its original RGB/TIR difference residual."""
        rgb_patch = components["rgb_patch"]
        ir_patch = components["ir_patch"]
        channel_gate = components["channel_gate"]

        fused_patch = w_rgb * rgb_patch + (1.0 - w_rgb) * ir_patch
        fused_patch = fused_patch + torch.tanh(self.iafv3_delta) * channel_gate * (rgb_patch - ir_patch)

        fused_map = fused_patch.permute(0, 2, 1).reshape_as(components["map_rgb"])
        cls_w = w_rgb.mean(dim=1, keepdim=True)
        fused_cls = cls_w * components["cls_rgb"] + (1.0 - cls_w) * components["cls_ir"]
        return self._map_to_tokens(fused_cls, fused_map)

    def _fusion_iafv3(self, rgb_tokens, ir_tokens, rgb_image):
        components = self._iafv3_components(rgb_tokens, ir_tokens, rgb_image)
        return self._compose_iafv3_tokens(components, components["w_rgb_base"])

    def _fusion_gqrv1(self, rgb_tokens, ir_tokens, rgb_image, text_embed):
        """IAFv3 plus a zero-initialized, text-conditioned logit correction."""
        if text_embed is None:
            raise ValueError("GQRv1 fusion requires a text embedding")

        components = self._iafv3_components(rgb_tokens, ir_tokens, rgb_image)
        w_rgb_base = components["w_rgb_base"]
        rgb_aux = self.rgb_aux_head(rgb_tokens.permute(1, 0, 2), text_embed)
        tir_aux = self.tir_aux_head(ir_tokens.permute(1, 0, 2), text_embed)
        router_logits = self.gqr(
            rgb_aux["target_feat"],
            tir_aux["target_feat"],
            text_embed,
            rgb_attention_confidence=rgb_aux["attention_confidence"],
            tir_attention_confidence=tir_aux["attention_confidence"],
        )
        router_prob = router_logits.softmax(dim=-1)

        eps = 1e-5
        base_logit = torch.log(w_rgb_base.clamp(eps, 1.0 - eps))
        base_logit = base_logit - torch.log1p(-w_rgb_base.clamp(eps, 1.0 - eps))
        correction_scale = self.gqr_eta_raw.new_tensor(self._gqr_correction_scale)
        eta = correction_scale * self.gqr_eta_max * torch.tanh(self.gqr_eta_raw)
        correction = eta * (2.0 * router_prob[:, 0, None, None] - 1.0)
        w_rgb_final = torch.sigmoid(base_logit + correction)

        # InfMAE may already have recorded residual diagnostics before this
        # fusion stage.  Preserve them alongside GQR's loss-facing tensors so
        # training logs can monitor whether the frozen expert is actually
        # being used.
        self._tsar_aux.update({
            "rgb_aux_box": rgb_aux["box"],
            "tir_aux_box": tir_aux["box"],
            "router_logits": router_logits,
            "router_prob": router_prob,
            "gqr_eta": eta.detach(),
            "gqr_correction_scale": correction_scale.detach(),
            "w_rgb_base_mean": w_rgb_base.detach().mean(),
            "w_rgb_final_mean": w_rgb_final.detach().mean(),
        })
        return self._compose_iafv3_tokens(components, w_rgb_final)

    def _fuse_visual_tokens(self, rgb_tokens, ir_tokens, rgb_image, text_embed=None):
        if self.fusion_method == "add":
            return self._fusion_add(rgb_tokens, ir_tokens, rgb_image)
        if self.fusion_method == "concat":
            return self._fusion_concat(rgb_tokens, ir_tokens, rgb_image)
        if self.fusion_method == "CMX":
            return self._fusion_cmx(rgb_tokens, ir_tokens, rgb_image)
        if self.fusion_method == "LIF":
            return self._fusion_lif(rgb_tokens, ir_tokens, rgb_image)
        if self.fusion_method == "LIF_ADD":
            return self._fusion_lif_add(rgb_tokens, ir_tokens, rgb_image)
        if self.fusion_method == "IAF":
            return self._fusion_iaf(rgb_tokens, ir_tokens, rgb_image)
        if self.fusion_method == "IAFv2.1":
            return self._fusion_iafv2_1(rgb_tokens, ir_tokens, rgb_image)
        if self.fusion_method in (
            "IAFv3", "InfMAEDirectV2", "InfMAEA3", "InfMAEA4", "InfMAEA5", "InfMAEA5Bridge",
        ):
            return self._fusion_iafv3(rgb_tokens, ir_tokens, rgb_image)
        if self.fusion_method in ("GQRv1", "GQRv1InfMAE"):
            return self._fusion_gqrv1(rgb_tokens, ir_tokens, rgb_image, text_embed)
        return self._fusion_concat(rgb_tokens, ir_tokens, rgb_image)

    def _apply_infmae_semantic_bridge(self, tir_features, text_eos_embed):
        """Inject A3-aligned TIR patch semantics into the LAVS input.

        This is intentionally independent of GT boxes: its text-conditioned
        patch attention works for both train and evaluation.  The separate
        A3/A5 target losses still use the raw pre-bridge adapter features, so
        the bridge cannot satisfy its own semantic target by changing only
        the downstream residual.
        """

        if not self.enable_infmae_semantic_bridge:
            return tir_features
        if self.infmae_tir_text_projector is None:
            raise RuntimeError("InfMAEA5Bridge requires a TIR-to-Text projector")
        bridge = getattr(self.infmae_direct_adapter, "semantic_bridge", None)
        if bridge is None:
            raise RuntimeError("InfMAEA5Bridge is missing its semantic bridge module")
        if not isinstance(tir_features, (list, tuple)) or not tir_features:
            raise ValueError("TIR features must be a non-empty list of token sequences")

        bridge_scale = getattr(self, "_infmae_alignment_adapter_scale", 1.0)
        text_teacher = F.normalize(text_eos_embed.float(), dim=-1).detach()
        bridged_features = []
        stage_aux = []
        for features in tir_features:
            if features.ndim != 3 or features.shape[-1] != self.backbone_visual_dim:
                raise ValueError(
                    "InfMAEA5Bridge expected [B, N+1, visual_dim] TIR features, got "
                    f"{tuple(features.shape)}"
                )
            semantic_patch_tokens = self.infmae_tir_text_projector(
                features[:, 1:, :].float()
            )
            bridged, bridge_aux = bridge(
                features,
                semantic_patch_tokens,
                text_teacher,
                residual_scale=bridge_scale,
            )
            bridged_features.append(bridged)
            stage_aux.append(bridge_aux)

        if self.training:
            for key in stage_aux[0]:
                self._tsar_aux[key] = torch.stack(
                    [aux[key].to(dtype=torch.float32) for aux in stage_aux]
                ).mean()
        return bridged_features

    def _build_infmae_alignment_aux(
        self,
        rgb_patch_tokens,
        tir_patch_tokens,
        target_boxes,
        text_eos_embed,
    ):
        """Build training-only A3/A4/A5 target embeddings.

        The RGB/text sides are detached CLIP semantic teachers.  Consequently
        the auxiliary losses refine the thermal adapter and its projectors
        without weakening the RGB CLIP representation that supplies the
        knowledge-transfer target.
        """

        if target_boxes is None:
            raise ValueError(
                "InfMAE target-aware alignment requires GT boxes during training"
            )
        if rgb_patch_tokens.shape != tir_patch_tokens.shape:
            raise ValueError(
                "RGB and TIR target-token tensors must have identical shape; got "
                f"rgb={tuple(rgb_patch_tokens.shape)}, tir={tuple(tir_patch_tokens.shape)}"
            )
        if rgb_patch_tokens.shape[-1] != self.backbone_visual_dim:
            raise ValueError(
                "Target tokens must be in the direct frontend visual dimension; got "
                f"{rgb_patch_tokens.shape[-1]} vs {self.backbone_visual_dim}"
            )

        rgb_target = self.infmae_target_pool(rgb_patch_tokens, target_boxes)
        tir_target = self.infmae_target_pool(tir_patch_tokens, target_boxes)
        tir_target_for_alignment = self._scale_infmae_alignment_gradient(
            tir_target,
            getattr(self, '_infmae_alignment_adapter_scale', 1.0),
        )
        aux = {}
        if self.enable_infmae_tir_text_alignment:
            tir_text = self.infmae_tir_text_projector(
                tir_target_for_alignment.float()
            )
            aux["tir_text_embedding"] = F.normalize(tir_text, dim=-1)
            aux["text_embedding"] = F.normalize(text_eos_embed.float(), dim=-1).detach()
        if self.enable_infmae_rgb_tir_transfer:
            tir_rgb = self.infmae_tir_rgb_projector(
                tir_target_for_alignment.float()
            )
            # ``visual_projection`` is the original RGB CLIP semantic space;
            # do not let a transfer loss update this teacher branch.
            rgb_teacher = self.clip.visual_projection(rgb_target.float()).detach()
            aux["tir_rgb_embedding"] = F.normalize(tir_rgb, dim=-1)
            aux["rgb_embedding"] = F.normalize(rgb_teacher, dim=-1)
        return aux

    def _format_model_output(self, main_out):
        """Keep validation on the legacy five-value MMVGFusion interface."""
        if self.training and (
            getattr(self, "enable_gqr", False)
            or getattr(self, "enable_infmae_alignment", False)
        ):
            return (*main_out, self._tsar_aux)
        return main_out

    def forward(self, img_data, text_data, target_boxes=None):
        # Forward-local auxiliary values are exposed only in training mode.
        # Clearing them here prevents stale tensors from a prior batch from
        # ever being returned after a feature toggle or an exception.
        self._tsar_aux = {}
        if self.open_lora:
            self.clip.set_adapter("lora_rgb")
        if self.args.modality =="rgbt":
            image_tensors_ir=img_data.tensors[:,3:,:,:].repeat(1,3,1,1)
            img_data_ir_mask=img_data.mask
            img_data_ir = NestedTensor(image_tensors_ir,img_data_ir_mask)
            img_data.tensors=img_data.tensors[:,:3,:,:]
        batch_size = img_data.tensors.shape[0]  # 得到batch_size
        image_tensors = img_data.tensors
        
        text_tensors, text_mask = self.encode_text(text_data, img_data.tensors.device)

        if self.is_siglip2:
            # SigLIP2's fixed-resolution text tower is bidirectional and its
            # official fixed-length path pools the final token.  Do not pass
            # the padding mask into the tower here; it is used separately by
            # MMVG's fusion transformer and text-guided attention.
            clip_text_features = self.clip.text_model(
                text_tensors,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            text_features = clip_text_features.last_hidden_state
            text_eos_embed = clip_text_features.pooler_output
        else:
            clip_text_features = self.clip.text_model(
                text_tensors,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )  # B * 77 * 512
            text_features = self.clip.text_projection(clip_text_features.last_hidden_state)
            text_eos_embed = self.clip.text_projection(clip_text_features.pooler_output)  # torch.Size([64, 512])

        gqr_text_embed = None
        if self.enable_gqr:
            gqr_text_embed = F.normalize(self.gqr_text_proj(text_eos_embed.float()), dim=-1)

        if self.mixup_pretrain:
            ml_text_features = [self.condition_text_proj(text_features.float())]
        else:
            ml_text_features = [clip_text_features.hidden_states[i] for i in self.extract_text_layer]

        visu_mask = self.get_masks(img_data, text_data)

        # target regression token
        reg_src = self.reg_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # B * 1 * hidden_dim

        # for i in range(13):
        #     with open(f"./bs4_output_{i}.txt", "w") as f:
        #         print(clip_image_features["hidden_states"][i][0],file=f)
        #     f.close()
        if self.open_lora:
            self.clip.set_adapter("lora_rgb")        

        vision_kwargs = {
            "adapt_layer": self.adapt_layer,
            "text_states": ml_text_features,
            "reg_src": reg_src,
            "cur_modality": "rgb",
            "pixel_values": image_tensors,
            "output_attentions": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if self.is_siglip2:
            vision_kwargs["text_padding_mask"] = text_mask
        clip_image_features = self.clip.vision_model(**vision_kwargs)
        # attention_map = clip_image_features["attentions"]  # tuple, used for draw the attention map

        ml_image_features = [clip_image_features["hidden_states"][i] for i in self.extract_vision_layer]
        if self.is_siglip2:
            ml_image_features = [self._prepend_siglip_global_token(features) for features in ml_image_features]
            img_cls_embed = clip_image_features["pooler_output"]
        else:
            img_cls_embed = self.clip.visual_projection(clip_image_features["pooler_output"])  # torch.Size([64, 512])

        capture_infmae_alignment = self.training and self.enable_infmae_alignment
        rgb_target_patch_tokens = None
        tir_target_patch_tokens = None
        tir_alignment_features = None
        if capture_infmae_alignment and target_boxes is None:
            raise ValueError(
                "FusionMethod with InfMAE A3/A4/A5 alignment requires target_boxes in training mode"
            )

        if self.args.modality =="rgbt":
            if self.enable_infmae_direct:
                # v2 frontend: TIR bypasses CLIP completely.  InfMAE emits
                # four [B, 197, 768] sequences that match the original LAVS
                # cross-fusion interface, so everything after this boundary
                # (LAVS, TPF/IAFv3, VL transformer, and head) is unchanged.
                infmae_input = self.infmae_direct_input_normalizer(image_tensors_ir)
                infmae_f2, infmae_f3 = self.infmae_direct_encoder(infmae_input)
                ml_image_features_ir, infmae_pooled = self.infmae_direct_adapter(
                    infmae_f2,
                    infmae_f3,
                )
                img_cls_embed_ir = self.infmae_direct_cls_projection(
                    infmae_pooled.float()
                )
                if capture_infmae_alignment:
                    # Preserve the unmodified adapter output for the A3/A5
                    # target-region loss, then optionally block the ordinary
                    # grounding path from updating the adapter.  Both paths
                    # see identical forward values, so evaluation/inference
                    # remains exactly unchanged.
                    tir_alignment_features = ml_image_features_ir
                    grounding_scale = self._infmae_grounding_adapter_gradient_scale
                    if grounding_scale != 1.0:
                        ml_image_features_ir = [
                            self._scale_infmae_alignment_gradient(feature, grounding_scale)
                            for feature in ml_image_features_ir
                        ]
                if self.enable_infmae_semantic_bridge:
                    ml_image_features_ir = self._apply_infmae_semantic_bridge(
                        ml_image_features_ir,
                        text_eos_embed,
                    )
            else:
                if self.open_lora:
                    self.clip.set_adapter("lora_ir")
                vision_kwargs["cur_modality"] = "ir"
                vision_kwargs["pixel_values"] = image_tensors_ir
                clip_image_features_ir = self.clip.vision_model(**vision_kwargs)
                # attention_map_ir = clip_image_features["attentions"]  # tuple, used for draw the attention map

                ml_image_features_ir = [clip_image_features_ir["hidden_states"][i] for i in self.extract_vision_layer]
                if self.is_siglip2:
                    ml_image_features_ir = [
                        self._prepend_siglip_global_token(features) for features in ml_image_features_ir
                    ]
                    img_cls_embed_ir = clip_image_features_ir["pooler_output"]
                else:
                    img_cls_embed_ir = self.clip.visual_projection(clip_image_features_ir["pooler_output"])  # torch.Size([64, 512])

            if capture_infmae_alignment:
                # Average the four pre-LAVS feature levels.  This supervises
                # every direct-InfMAE stage adapter while preserving the
                # documented boundary: alignment is formed before LAVS/TPF.
                rgb_target_patch_tokens = torch.stack(
                    [features[:, 1:, :] for features in ml_image_features],
                    dim=0,
                ).mean(dim=0)
                tir_target_patch_tokens = torch.stack(
                    [features[:, 1:, :] for features in (
                        tir_alignment_features
                        if tir_alignment_features is not None
                        else ml_image_features_ir
                    )],
                    dim=0,
                ).mean(dim=0)
            for i in range(len(self.extract_vision_layer)):

                ml_image_features_fusion_vt,_ = self.cross_fusion_layers_vt[i](ml_image_features[i], ml_image_features_ir[i],  attention_mask=None,causal_attention_mask=None,output_attentions=True)
                ml_image_features_fusion_tv,_ = self.cross_fusion_layers_tv[i](ml_image_features_ir[i], ml_image_features[i],  attention_mask=None,causal_attention_mask=None,output_attentions=True)
                ml_image_features[i] = ml_image_features[i]+ml_image_features_fusion_vt
                ml_image_features_ir[i] =  ml_image_features_ir[i] +ml_image_features_fusion_tv
            # for i,cross_fusion in enumerate(.cross_fusion_layers):
            #     #TODO
        ml_image_features = torch.cat(ml_image_features, dim=2)
        image_features = self.ml_visual_projection(ml_image_features)
        visu_src = self.visu_proj(image_features.float())  # (N*B)xC
        if self.args.modality =='rgbt':
            ml_image_features_ir = torch.cat(ml_image_features_ir, dim=2)
            image_features_ir = self.ml_visual_projection(ml_image_features_ir)

            visu_src_ir = self.visu_proj(image_features_ir.float())  # (N*B)xC
            if self.enable_infmae:
                # Keep the established CLIP+LoRA TIR stream as the reference
                # representation.  The frozen InfMAE expert contributes only
                # a zero-initialized residual on patch tokens, before the
                # existing LAVS/TPF/GQR fusion path consumes the features.
                infmae_input = self.infmae_input_normalizer(image_tensors_ir)
                infmae_f2, infmae_f3 = self.infmae_encoder(infmae_input)
                visu_src_ir, infmae_aux = self.infmae_adapter(
                    visu_src_ir,
                    infmae_f2,
                    infmae_f3,
                )
                if self.training:
                    self._tsar_aux.update(infmae_aux)
            visu_src_ir = visu_src_ir.permute(1, 0, 2)  # 197 * 4 * 512
            if self.open_lora:
                self.clip.set_adapter("lora_rgb")

        text_src = self.text_proj(text_features.float())  # B * 77 * 512

        # permute BxLenxC to LenxBxC
        visu_src = visu_src.permute(1, 0, 2)  # 197 * 4 * 512
        text_src = text_src.permute(1, 0, 2)  # 77 * 4 * 512
        reg_src = reg_src.permute(1, 0, 2)  # 1 * B * 512
        # mask
        reg_mask = torch.zeros((batch_size, 1)).to(reg_src.device).to(torch.bool)
        cls_mask = torch.zeros((batch_size, 1)).to(reg_src.device).to(torch.bool)
        text_start_idx = 2 + self.num_visu_token
        if self.args.modality == 'rgbt':
            fused_visu_src = self._fuse_visual_tokens(
                visu_src,
                visu_src_ir,
                image_tensors,
                text_embed=gqr_text_embed,
            )
            if self.fusion_method == "concat":
                vl_src = torch.cat([reg_src, fused_visu_src, text_src], dim=0)
                vl_mask = torch.cat([reg_mask, cls_mask, visu_mask, visu_mask, text_mask], dim=1)
                text_start_idx = 2 + self.num_visu_token * 2
            else:
                vl_src = torch.cat([reg_src, fused_visu_src, text_src], dim=0)
                vl_mask = torch.cat([reg_mask, cls_mask, visu_mask, text_mask], dim=1)
        else:
            vl_src = torch.cat([reg_src, visu_src, text_src], dim=0)
            vl_mask = torch.cat([reg_mask, cls_mask, visu_mask, text_mask], dim=1)

        vl_pos = self.vl_pos_embed.weight[:vl_src.size(0)].unsqueeze(1).repeat(1, batch_size, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        box_hs = vg_hs[0]
        pred_box = self.bbox_embed(box_hs).sigmoid()

        # normalized features
        img_cls_embed = img_cls_embed / img_cls_embed.norm(p=2, dim=-1, keepdim=True)
        text_eos_embed = text_eos_embed / text_eos_embed.norm(p=2, dim=-1, keepdim=True)
        if self.args.modality == 'rgbt':
            img_cls_embed_ir = img_cls_embed_ir / img_cls_embed_ir.norm(p=2, dim=-1, keepdim=True)

        if capture_infmae_alignment:
            self._tsar_aux.update(
                self._build_infmae_alignment_aux(
                    rgb_target_patch_tokens,
                    tir_target_patch_tokens,
                    target_boxes,
                    text_eos_embed,
                )
            )

        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_text = torch.matmul(text_eos_embed, img_cls_embed.t()) * logit_scale
        if self.args.modality == 'rgbt':
            logits_per_text_ir = torch.matmul(text_eos_embed, img_cls_embed_ir.t()) * logit_scale
            if self.is_siglip2:
                logits_per_text = logits_per_text + self.clip.logit_bias
                logits_per_text_ir = logits_per_text_ir + self.clip.logit_bias
            logits_per_image_ir = logits_per_text_ir.t()
        elif self.is_siglip2:
            logits_per_text = logits_per_text + self.clip.logit_bias
        logits_per_image = logits_per_text.t()

        if self.is_siglip2 and self.args.modality == "rgbt":
            contrastive_output = {
                "rgb": logits_per_text,
                "ir": logits_per_text_ir,
            }
        else:
            contrastive_output = logits_per_text
        
        # visual token align
        vg_hs_visu_features = vg_hs[2: 2 + self.num_visu_token].permute(1, 0, 2)  # B L H
        clip_last_layer_features = self.visu_token_mlp(self.visu_token_norm(vg_hs_visu_features))

        vg_hs_text = vg_hs[text_start_idx:].permute(1, 0, 2)
        if self.is_siglip2:
            from .siglip2_adapter import siglip2_eos_indices

            text_eos_indices = siglip2_eos_indices(
                text_tensors,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            )
        else:
            text_eos_indices = text_tensors.argmax(dim=-1)
        vg_hs_text_eos_embed = vg_hs_text[torch.arange(vg_hs_text.shape[0]), text_eos_indices]
        vg_hs_text_eos_embed = vg_hs_text_eos_embed / vg_hs_text_eos_embed.norm(p=2, dim=-1, keepdim=True)

        visu_token_similarity = torch.mul(vg_hs_text_eos_embed.unsqueeze(1).repeat(1, self.num_visu_token, 1),
                                          clip_last_layer_features)  # torch.Size([96, 196, 512])
        visu_token_similarity = visu_token_similarity.sum(axis=-1, keepdim=False)  # torch.Size([96, 196])

        patch_num = int(math.sqrt(vg_hs_visu_features.shape[1]))
        channel = vg_hs_visu_features.shape[2]
        assert patch_num * patch_num == vg_hs_visu_features.shape[1]
        seg_features = vg_hs_visu_features.permute(0, 2, 1).reshape(batch_size, channel, patch_num, patch_num)
        seg_features = self.seg_conv3(self.seg_conv2(self.seg_conv1(seg_features)))
        seg_features = seg_features.permute(0, 2, 3, 1)
        seg_mask = torch.mul(vg_hs_text_eos_embed.reshape(batch_size, 1, 1, vg_hs_text_eos_embed.shape[-1]).repeat(1, seg_features.shape[1], seg_features.shape[2], 1),
                             seg_features)
        seg_mask = seg_mask.sum(axis=-1, keepdim=False).unsqueeze(1)  # B 1 H W
        if self.args.modality == 'rgbt':
            main_out = (
                pred_box,
                contrastive_output,
                [logits_per_image, logits_per_image_ir],
                visu_token_similarity,
                seg_mask,
            )
        else: 
            main_out = (pred_box, contrastive_output, logits_per_image, visu_token_similarity, seg_mask)

        # Evaluation/inference intentionally keeps MMVGFusion's established
        # five-value contract. GT-derived routing supervision is constructed
        # later by the loss function, never in this forward path.
        return self._format_model_output(main_out)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LIFWeightNet(nn.Module):
    # Forward returns a low-resolution illumination weight map.
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNAct(3, 32, k=3, p=1)
        self.conv2 = ConvBNAct(32, 64, k=3, p=1)
        self.conv3 = ConvBNAct(64, 64, k=3, p=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x


###################### CMX ############################
###################### CMX ############################
###################### CMX ############################
class CMX(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.frm = FeatureRectifyModule(dim, reduction, lambda_c, lambda_s)
        self.ffm = FeatureFusionModule(dim, reduction, num_heads, norm_layer)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x1, x2 = self.frm(x1, x2)
        merged = self.ffm(x1, x2)
        return merged


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,
            out_channels=dim,
            reduction=reduction,
            norm_layer=norm_layer,
        )

    def forward(self, x1, x2):
        _, _, h, w = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)
        x1_cross, x2_cross = self.cross(x1_flat, x2_flat)
        merge = torch.cat((x1_cross, x2_cross), dim=-1)
        return self.channel_emb(merge, h, w)


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        batch_size, _, _, _ = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(batch_size, self.dim * 2)
        max_pool = self.max_pool(x).view(batch_size, self.dim * 2)
        y = torch.cat((avg, max_pool), dim=1)
        y = self.mlp(y).view(batch_size, self.dim * 2, 1)
        channel_weights = y.reshape(batch_size, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        batch_size, _, h, w = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(batch_size, 2, 1, h, w).permute(1, 0, 2, 3, 4)
        return spatial_weights


class CrossAttentionCMX(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttentionCMX, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        batch_size, num_tokens, channels = x1.shape
        q1 = x1.reshape(batch_size, -1, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(batch_size, -1, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(batch_size, -1, 2, self.num_heads, channels // self.num_heads).permute(
            2, 0, 3, 1, 4
        ).contiguous()
        k2, v2 = self.kv2(x2).reshape(batch_size, -1, 2, self.num_heads, channels // self.num_heads).permute(
            2, 0, 3, 1, 4
        ).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(batch_size, num_tokens, channels).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(batch_size, num_tokens, channels).contiguous()
        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttentionCMX(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, h, w):
        batch_size, num_tokens, channels = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size, channels, h, w).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


###################### CMX ############################
###################### CMX ############################
###################### CMX ############################


# Backward-compatible alias.
MMVG = MMVGFusion
