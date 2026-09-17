import math
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import NestedTensor
from .clip_mdetr import clip


def resize_pos_embed(clip_visual, patch_size: int, imsize_new: int):
    # for patch embedding
    cls_pos_embed = clip_visual.positional_embedding[:1, :]
    patch_pos_embed_old = clip_visual.positional_embedding[1:, :]
    patch_pos_embed_old = patch_pos_embed_old.transpose(0, 1)
    E, _Q = patch_pos_embed_old.shape
    P_H = clip_visual.input_resolution // patch_size
    P_W = clip_visual.input_resolution // patch_size
    patch_pos_embed_old = patch_pos_embed_old.view(E, P_H, P_W).unsqueeze(0)

    # for search region
    H, W = imsize_new, imsize_new
    new_P_H, new_P_W = H // patch_size, W // patch_size
    patch_pos_embed_new = nn.functional.interpolate(
        patch_pos_embed_old, size=(new_P_H, new_P_W), mode="bicubic", align_corners=False
    )
    patch_pos_embed_new = patch_pos_embed_new.flatten(2).transpose(1, 2).squeeze(0)
    patch_pos_embed_new = torch.cat([cls_pos_embed, patch_pos_embed_new], dim=0)
    patch_pos_embed_new = nn.Parameter(patch_pos_embed_new)

    clip_visual.positional_embedding = patch_pos_embed_new
    return clip_visual


def candidate_elimination(
    attn: torch.Tensor,
    tokens: torch.Tensor,
    lens_t: int,
    keep_ratio: float,
):
    """
    Candidate Elimination (CE) used by FSVG.
    tokens: [B, 1 + Ls + Lt, C] (cls + image tokens + text tokens)
    """
    lens_s = attn.shape[-1] - lens_t - 1
    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens

    # image-text attention
    attn_t = attn[:, lens_s + 1 :, 1 : lens_s + 1]  # [B, Lt, Ls]
    attn_t = attn_t.mean(dim=1)  # [B, Ls]

    _sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
    topk_idx = indices[:, :lens_keep]

    tokens_cls = tokens[:, :1]
    tokens_s = tokens[:, 1 : lens_s + 1]
    tokens_t = tokens[:, lens_s + 1 :]

    B, _L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_cls, attentive_tokens, tokens_t], dim=1)
    return tokens_new


def ceblock_forward(block, x, lens_t, keep_ratio_search=None):
    x_attn, attn = block.attention(block.ln_1(x))
    x = x + x_attn
    x = x.permute(1, 0, 2)
    x = candidate_elimination(attn, x, lens_t, keep_ratio_search)
    x = x.permute(1, 0, 2)
    x = x + block.mlp(block.ln_2(x))
    return x


class MultiLevel_Transformer(nn.Module):
    def __init__(self, clip_vit, extract_layer: List[int]):
        super().__init__()
        self.width = clip_vit.width
        self.layers = clip_vit.layers
        self.resblocks = clip_vit.resblocks
        self.extract_layer = extract_layer

    def forward(self, x: torch.Tensor, lens_t: int, ce_keep_rate: float, ce_loc: List[int]):
        for i in range(max(self.extract_layer) + 1):
            if ce_keep_rate < 1 and i in ce_loc:
                x = ceblock_forward(self.resblocks[i], x, lens_t, ce_keep_rate)
            else:
                x = self.resblocks[i](x)
        return x


class MultiLevel_ImageEncoder_modified(nn.Module):
    def __init__(self, clip_visu_model, extract_layer: List[int]):
        super().__init__()
        self.input_resolution = clip_visu_model.input_resolution
        self.output_dim = clip_visu_model.output_dim
        self.conv1 = clip_visu_model.conv1
        self.class_embedding = clip_visu_model.class_embedding
        self.positional_embedding = clip_visu_model.positional_embedding
        self.ln_pre = clip_visu_model.ln_pre
        self.transformer = MultiLevel_Transformer(clip_visu_model.transformer, extract_layer)
        self.ln_post = clip_visu_model.ln_post
        self.proj = clip_visu_model.proj
        self.positional_embedding.requires_grad_(True)

    def forward(self, x: torch.Tensor, text_tensors: torch.Tensor, ce_keep_rate: float, ce_loc: List[int]):
        txt_len = text_tensors.shape[1]
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = torch.cat([x, text_tensors], dim=1)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, txt_len, ce_keep_rate, ce_loc)
        cls_token = x[0, :, :]
        return cls_token


class TextEncoder_modified(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        return x


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size: int, output_feat_size: int, dropout: float, do_ln: bool = True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features: torch.Tensor):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        return self.dropout(x)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FSVG(nn.Module):
    """
    FSVG model integrated into xjw_codes_xjw.

    Notes:
    - Uses `models.clip_mdetr.clip` to keep `clip.load/tokenize` signatures consistent with original FSVG.
    - Accepts text as NestedTensor / torch.Tensor token ids / list[str] (will be CLIP-tokenized on the fly).
    """

    def __init__(self, args):
        super().__init__()
        print("Building FSVG model...")
        self.modality = getattr(args, "modality", "rgb")

        # CLIP Model name: ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        if args.model == "ViT-L/14":
            print("init ViT-L/14")
            self.clip, _ = clip.load("ViT-L/14", device=args.device)
            self.extract_layer = [5, 11, 17, 23]
            self.ce_layer = [6, 12, 18]
            self.patch_size = 14
            re_in_dim = 768
            re_out_dim = 1024
        elif args.model == "ViT-L/14@336px":
            # Some CLIP implementations (including `models/clip_mdetr`) don't ship the @336px variant.
            # For compatibility, fallback to ViT-L/14 and rely on positional embedding resize.
            print("init ViT-L/14@336px (fallback -> ViT-L/14)")
            self.clip, _ = clip.load("ViT-L/14", device=args.device)
            self.extract_layer = [5, 11, 17, 23]
            self.ce_layer = [6, 12, 18]
            self.patch_size = 14
            re_in_dim = 768
            re_out_dim = 1024
        elif args.model == "ViT-B/32":
            print("init ViT-B/32")
            self.clip, _ = clip.load("ViT-B/32", device=args.device)
            self.extract_layer = [2, 5, 8, 11]
            self.ce_layer = [3, 6, 9]
            self.patch_size = 32
            re_in_dim = 512
            re_out_dim = 768
        else:  # default ViT-B/16
            print("init ViT-B/16")
            self.clip, _ = clip.load("ViT-B/16", device=args.device)
            self.extract_layer = [2, 5, 8, 11]
            self.ce_layer = [3, 6, 9]
            self.patch_size = 16
            re_in_dim = 512
            re_out_dim = 768

        self.text_resize = FeatureResizer(
            input_feat_size=re_in_dim,
            output_feat_size=re_out_dim,
            dropout=0.1,
        )

        # resize pos embed
        if self.clip.visual.input_resolution != args.imsize:
            resize_pos_embed(self.clip.visual, self.patch_size, args.imsize)

        self.image_encoder = MultiLevel_ImageEncoder_modified(self.clip.visual, self.extract_layer)
        self.text_encoder = TextEncoder_modified(self.clip)

        # Keep head dim aligned with CLIP-VG pretraining head (512) for better checkpoint reuse.
        self.bbox_embed = MLP(512, 512, 4, 3)
        self.neck = nn.Linear(self.clip.visual.transformer.width, 512)
        if self.modality == "rgbt":
            self.rgbt_fusion = nn.Linear(self.clip.visual.transformer.width * 2, self.clip.visual.transformer.width)
        else:
            self.rgbt_fusion = None

    def _tensorize_images(self, images: Union[NestedTensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(images, NestedTensor):
            return images.tensors
        return images

    def _modality_adapt_images(self, image_tensors: torch.Tensor):
        """
        Convert input image tensors into CLIP-compatible RGB tensors according to args.modality.
        Returns either:
          - tensor [B,3,H,W] for rgb/ir
          - tuple(rgb_tensor, ir_tensor) for rgbt
        """
        c = image_tensors.shape[1]
        if self.modality == "rgb":
            if c >= 3:
                return image_tensors[:, :3, :, :]
            if c == 1:
                return image_tensors.repeat(1, 3, 1, 1)
            raise ValueError(f"Unsupported channel count for rgb modality: {c}")

        if self.modality == "ir":
            if c >= 4:
                ir = image_tensors[:, 3:4, :, :]
                return ir.repeat(1, 3, 1, 1)
            if c == 1:
                return image_tensors.repeat(1, 3, 1, 1)
            if c >= 3:
                ir = image_tensors[:, 0:1, :, :]
                return ir.repeat(1, 3, 1, 1)
            raise ValueError(f"Unsupported channel count for ir modality: {c}")

        if self.modality == "rgbt":
            if c < 4:
                # Fallback for malformed rgbt data.
                rgb = image_tensors[:, :3, :, :] if c >= 3 else image_tensors.repeat(1, 3, 1, 1)
                ir = rgb[:, :1, :, :].repeat(1, 3, 1, 1)
                return rgb, ir
            rgb = image_tensors[:, :3, :, :]
            ir = image_tensors[:, 3:4, :, :].repeat(1, 3, 1, 1)
            return rgb, ir

        raise ValueError(f"Unsupported modality: {self.modality}")

    def _tensorize_texts(
        self,
        texts: Union[NestedTensor, torch.Tensor, str, List[str], Tuple[str, ...]],
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(texts, NestedTensor):
            return texts.tensors.to(device)
        if torch.is_tensor(texts):
            return texts.to(device)
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, (list, tuple)):
            # list[str] from xjw_codes_xjw default dataloader
            return clip.tokenize(list(texts), truncate=True).to(device)
        raise TypeError(f"Unsupported text input type: {type(texts)}")

    def forward(self, img_data: Any, text_data: Any, ce_keep_rate: float = 1.0):
        image_tensors_raw = self._tensorize_images(img_data)
        text_tokens = self._tensorize_texts(text_data, device=image_tensors_raw.device)
        image_tensors = self._modality_adapt_images(image_tensors_raw)

        text_features = self.text_encoder(text_tokens)
        resized_text = self.text_resize(text_features)

        if self.modality == "rgbt":
            rgb_tensors, ir_tensors = image_tensors
            visu_src_rgb = self.image_encoder(
                rgb_tensors.type(self.clip.dtype),
                resized_text,
                ce_keep_rate,
                self.ce_layer,
            )
            visu_src_ir = self.image_encoder(
                ir_tensors.type(self.clip.dtype),
                resized_text,
                ce_keep_rate,
                self.ce_layer,
            )
            visu_src = self.rgbt_fusion(torch.cat([visu_src_rgb, visu_src_ir], dim=-1))
        else:
            visu_src = self.image_encoder(
                image_tensors.type(self.clip.dtype),
                resized_text,
                ce_keep_rate,
                self.ce_layer,
            )
        visu_src = self.neck(visu_src)
        pred_box = self.bbox_embed(visu_src).sigmoid()
        return pred_box
