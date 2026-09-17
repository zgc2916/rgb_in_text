"""Small, self-contained modules used by TSAR-Ground.

The modules in this file deliberately operate on the projected MMVGFusion
representation (normally 512 dimensions).  Keeping them independent from the
CLIP/SigLIP implementation makes the TSAR additions easy to test and prevents
the new routing path from changing the established visual-language encoders.
"""

from __future__ import annotations

import math
import importlib
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TextConditionedModalityHead(nn.Module):
    """Predict a light-weight, text-conditioned box for one visual modality.

    Inputs use batch-first layout: visual tokens include one global/CLS token
    followed by patch tokens, while ``text_embed`` is a single expression
    representation per image.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        hidden_dim = max(dim // 2, 1)
        self.box_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        # A text-to-patch attention map already contains a useful spatial
        # proposal.  Keep the original content-only box path at initialization
        # and let training open this residual only when it helps the auxiliary
        # grounding objective.
        self.spatial_box_gain = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _attention_spatial_box(attention: torch.Tensor) -> torch.Tensor:
        """Convert a patch-attention distribution into a normalized box prior.

        For ViT grids the attention-weighted mean provides a center and its
        second moment estimates an object extent.  The fallback still produces
        valid values for a non-square token sequence, although all supported
        TSAR image sizes use a square patch grid.
        """
        if attention.ndim != 2:
            raise ValueError(
                "attention must have shape [B, N], got "
                f"{tuple(attention.shape)}"
            )

        patch_count = attention.shape[-1]
        patch_side = math.isqrt(patch_count)
        if patch_side * patch_side == patch_count:
            grid_h = grid_w = patch_side
        else:
            grid_h, grid_w = 1, patch_count

        y = (torch.arange(grid_h, device=attention.device, dtype=attention.dtype) + 0.5) / grid_h
        x = (torch.arange(grid_w, device=attention.device, dtype=attention.dtype) + 0.5) / grid_w
        grid_y = y[:, None].expand(grid_h, grid_w).reshape(-1)
        grid_x = x[None, :].expand(grid_h, grid_w).reshape(-1)

        center_x = torch.sum(attention * grid_x, dim=-1)
        center_y = torch.sum(attention * grid_y, dim=-1)
        variance_x = torch.sum(attention * (grid_x - center_x[:, None]).square(), dim=-1)
        variance_y = torch.sum(attention * (grid_y - center_y[:, None]).square(), dim=-1)

        # For a uniform interval, width = sqrt(12) * standard deviation.
        # At least one patch remains visible for a sharply peaked attention.
        width = (math.sqrt(12.0) * torch.sqrt(variance_x.clamp_min(1e-8))).clamp(
            min=1.0 / grid_w,
            max=1.0,
        )
        height = (math.sqrt(12.0) * torch.sqrt(variance_y.clamp_min(1e-8))).clamp(
            min=1.0 / grid_h,
            max=1.0,
        )
        return torch.stack([center_x, center_y, width, height], dim=-1)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [B, N+1, D], got "
                f"{tuple(visual_tokens.shape)}"
            )
        if text_embed.ndim != 2:
            raise ValueError(
                "text_embed must have shape [B, D], got "
                f"{tuple(text_embed.shape)}"
            )
        if visual_tokens.shape[0] != text_embed.shape[0]:
            raise ValueError("visual_tokens and text_embed must share the batch dimension")
        if visual_tokens.shape[1] < 2:
            raise ValueError("visual_tokens must include one CLS token and at least one patch token")
        if visual_tokens.shape[2] != self.dim or text_embed.shape[1] != self.dim:
            raise ValueError(
                f"expected feature dimension {self.dim}, got visual={visual_tokens.shape[2]} "
                f"and text={text_embed.shape[1]}"
            )

        cls_token = visual_tokens[:, 0]
        patch_tokens = visual_tokens[:, 1:]

        query = self.q_proj(text_embed).unsqueeze(1)
        key = self.k_proj(patch_tokens)
        value = self.v_proj(patch_tokens)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim),
            dim=-1,
        )
        target_feat = torch.matmul(attention, value).squeeze(1)

        box_feat = torch.cat([target_feat, text_embed, cls_token], dim=-1)
        attention_flat = attention.squeeze(1)
        spatial_box = self._attention_spatial_box(attention_flat)
        spatial_box_logits = torch.logit(spatial_box.clamp(1e-4, 1.0 - 1e-4))
        box_logits = self.box_mlp(box_feat)
        box = torch.sigmoid(
            box_logits + torch.tanh(self.spatial_box_gain) * spatial_box_logits
        )

        attention_entropy = -torch.sum(
            attention_flat * attention_flat.clamp_min(1e-8).log(),
            dim=-1,
        )
        entropy_normalizer = attention_flat.new_tensor(math.log(max(attention_flat.shape[-1], 2)))
        attention_confidence = 1.0 - attention_entropy / entropy_normalizer
        return {
            "box": box,
            "target_feat": target_feat,
            "attn": attention_flat,
            "spatial_box": spatial_box,
            "attention_confidence": attention_confidence.clamp(0.0, 1.0),
        }


class GroundingQualityRouter(nn.Module):
    """Estimate RGB/TIR reliability with modality-exchange symmetry.

    RGB and TIR receive the same quality scorer.  Their common context is
    constructed only from symmetric cues, so swapping modalities swaps the
    two logits rather than introducing an arbitrary modality-order bias.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        hidden_dim = max(dim // 2, 1)
        self.dim = dim
        self.context_mlp = nn.Sequential(
            nn.Linear(dim * 3 + 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(dim * 3 + 1, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # An untrained router should be neutral.  The shared output layer can
        # then depart from 50/50 only when the detached quality teacher starts
        # providing reliable supervision.
        nn.init.zeros_(self.quality_mlp[-1].weight)
        nn.init.zeros_(self.quality_mlp[-1].bias)

    @staticmethod
    def _prepare_attention_confidence(
        confidence: torch.Tensor | None,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        batch_size = reference.shape[0]
        if confidence is None:
            return reference.new_zeros(batch_size, 1)
        if confidence.ndim == 1:
            confidence = confidence.unsqueeze(-1)
        if confidence.ndim != 2 or confidence.shape != (batch_size, 1):
            raise ValueError(
                f"{name} must have shape [B] or [B, 1], got {tuple(confidence.shape)}"
            )
        return confidence.to(dtype=reference.dtype)

    def forward(
        self,
        rgb_target: torch.Tensor,
        tir_target: torch.Tensor,
        text_embed: torch.Tensor,
        rgb_attention_confidence: torch.Tensor | None = None,
        tir_attention_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_shape = ("[B, D]",)
        for name, tensor in (
            ("rgb_target", rgb_target),
            ("tir_target", tir_target),
            ("text_embed", text_embed),
        ):
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape {expected_shape[0]}, got {tuple(tensor.shape)}")
            if tensor.shape[1] != self.dim:
                raise ValueError(f"{name} must have feature dimension {self.dim}, got {tensor.shape[1]}")
        if not (rgb_target.shape[0] == tir_target.shape[0] == text_embed.shape[0]):
            raise ValueError("router inputs must share the batch dimension")

        rgb_attention_confidence = self._prepare_attention_confidence(
            rgb_attention_confidence,
            rgb_target,
            "rgb_attention_confidence",
        )
        tir_attention_confidence = self._prepare_attention_confidence(
            tir_attention_confidence,
            tir_target,
            "tir_attention_confidence",
        )

        shared_context_input = torch.cat(
            [
                0.5 * (rgb_target + tir_target),
                torch.abs(rgb_target - tir_target),
                text_embed,
                0.5 * (rgb_attention_confidence + tir_attention_confidence),
                torch.abs(rgb_attention_confidence - tir_attention_confidence),
            ],
            dim=-1,
        )
        shared_context = self.context_mlp(shared_context_input)

        def modality_quality(target: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
            quality_input = torch.cat(
                [target, text_embed, shared_context, confidence],
                dim=-1,
            )
            return self.quality_mlp(quality_input)

        return torch.cat(
            [
                modality_quality(rgb_target, rgb_attention_confidence),
                modality_quality(tir_target, tir_attention_confidence),
            ],
            dim=-1,
        )


def _load_infmae_factory():
    """Load the user-supplied InfMAE implementation without copying it.

    The official release uses script-style absolute imports (for example,
    ``from vision_transformer import ...``), so its directory has to be on
    ``sys.path`` while the factory is imported.  The import happens lazily:
    normal MMVGFusion/GQR runs never import InfMAE or its dependencies.
    """

    infmae_root = Path(__file__).resolve().parents[1] / "InfMAE"
    if not infmae_root.is_dir():
        raise FileNotFoundError(
            "InfMAE source directory was not found. Expected "
            f"{infmae_root}"
        )
    root_str = str(infmae_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        module = importlib.import_module("models_infmae_skip4")
    except Exception as error:  # pragma: no cover - keeps import diagnostics intact.
        raise RuntimeError(
            "Unable to import the local InfMAE implementation from "
            f"{infmae_root}"
        ) from error
    return module.infmae_vit_base_patch16


class FrozenInfMAEEncoder(nn.Module):
    """Expose pretrained InfMAE F2/F3 encoder features for an RGBT frontend.

    InfMAE pretraining masks tokens and carries an MAE decoder.  Grounding
    needs dense, unmasked feature maps instead, so this wrapper loads the
    official checkpoint *strictly* and retains only the encoder path:

    * F2: ``[B, 384, H/8, W/8]`` after the second convolutional stage;
    * F3: ``[B, 768, H/16, W/16]`` after the transformer stage.

    The encoder is frozen and permanently held in evaluation mode.  It is
    The wrapper is shared by the residual-expert experiment and the direct
    InfMAE TIR frontend.  Which route consumes these features is decided by
    ``MMVGFusion``; this module contains no CLIP or fusion-specific behavior.
    """

    def __init__(self, checkpoint_path: str | Path):
        super().__init__()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "InfMAE checkpoint was not found. Expected "
                f"{checkpoint_path}"
            )

        factory = _load_infmae_factory()
        full_model = factory()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "InfMAE checkpoint must be a mapping containing a 'model' "
                f"state dict, got {type(checkpoint)!r}"
            )
        state_dict = checkpoint.get("model")
        if not isinstance(state_dict, dict):
            raise KeyError(
                "InfMAE checkpoint does not contain a dictionary at key 'model'"
            )
        # The shipped Inf30 checkpoint is verified strictly.  A shape/key
        # mismatch should fail at construction rather than silently degrading
        # a downstream grounding run.
        full_model.load_state_dict(state_dict, strict=True)

        # Retain only the dense encoder path.  The MAE masking stem, decoder,
        # and reconstruction-only skip projections are intentionally omitted.
        self.patch_embed1 = full_model.patch_embed1
        self.blocks1 = full_model.blocks1
        self.patch_embed2 = full_model.patch_embed2
        self.blocks2 = full_model.blocks2
        self.patch_embed3 = full_model.patch_embed3
        self.patch_embed4 = full_model.patch_embed4
        self.pos_embed = full_model.pos_embed
        self.blocks3 = full_model.blocks3
        self.norm = full_model.norm

        self.image_size = tuple(int(value) for value in full_model.patch_embed1.img_size)
        self.f2_dim = int(full_model.patch_embed2.proj.out_channels)
        self.f3_dim = int(full_model.patch_embed3.proj.out_channels)
        del full_model

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        """Keep the frozen expert deterministic when its parent enters train mode."""

        super().train(False)
        return self

    def forward(self, thermal_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if thermal_image.ndim != 4 or thermal_image.shape[1] != 3:
            raise ValueError(
                "thermal_image must have shape [B, 3, H, W], got "
                f"{tuple(thermal_image.shape)}"
            )
        spatial_size = tuple(int(value) for value in thermal_image.shape[-2:])
        if spatial_size != self.image_size:
            raise ValueError(
                "The released InfMAE checkpoint is fixed at "
                f"{self.image_size}, got input {spatial_size}."
            )

        expert_dtype = self.patch_embed1.proj.weight.dtype
        thermal_image = thermal_image.to(dtype=expert_dtype)
        with torch.no_grad():
            stage1 = self.patch_embed1(thermal_image)
            for block in self.blocks1:
                stage1 = block(stage1)

            stage2 = self.patch_embed2(stage1)
            for block in self.blocks2:
                stage2 = block(stage2)

            stage3 = self.patch_embed3(stage2)
            batch_size, channels, grid_h, grid_w = stage3.shape
            tokens = stage3.flatten(2).transpose(1, 2)
            tokens = self.patch_embed4(tokens)
            if tokens.shape[1] != self.pos_embed.shape[1]:
                raise RuntimeError(
                    "InfMAE positional embedding/token count mismatch: "
                    f"{tokens.shape[1]} vs {self.pos_embed.shape[1]}"
                )
            tokens = tokens + self.pos_embed
            for block in self.blocks3:
                tokens = block(tokens)
            tokens = self.norm(tokens)
            stage3 = tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)

        return stage2, stage3


class InfMAEThermalInputNormalizer(nn.Module):
    """Convert the existing normalized TIR channel to InfMAE's input space.

    RGBT-GroundBench hands MMVGFusion a dataset-normalized fourth channel.
    InfMAE Inf30 pretraining used repeated TIR channels normalized by
    ``mean=0.425, std=0.200``.  Reconstructing the scalar thermal image before
    applying InfMAE normalization avoids feeding the expert CLIP statistics.
    """

    _DATASET_TIR_STATS = {
        "rgbtvg_flir": (0.5337, 0.2562),
        "rgbtvg_m3fd": (0.3264, 0.1990),
        "rgbtvg_mfad": (0.3393, 0.2063),
        "rgbtvg_mixup": (0.3735, 0.2289),
    }
    _INFMAE_MEAN = 0.425
    _INFMAE_STD = 0.200

    def __init__(self, dataset: str, image_norm: str = "dataset"):
        super().__init__()
        if str(image_norm).lower() == "siglip2":
            source_mean, source_std = 0.5, 0.5
        else:
            try:
                source_mean, source_std = self._DATASET_TIR_STATS[str(dataset)]
            except KeyError as error:
                raise ValueError(
                    "No thermal normalization statistics are registered for "
                    f"dataset={dataset!r}"
                ) from error

        self.register_buffer("source_mean", torch.tensor(source_mean).view(1, 1, 1, 1))
        self.register_buffer("source_std", torch.tensor(source_std).view(1, 1, 1, 1))
        self.register_buffer("infmae_mean", torch.tensor(self._INFMAE_MEAN).view(1, 1, 1, 1))
        self.register_buffer("infmae_std", torch.tensor(self._INFMAE_STD).view(1, 1, 1, 1))

    def forward(self, thermal_image: torch.Tensor) -> torch.Tensor:
        if thermal_image.ndim != 4 or thermal_image.shape[1] not in (1, 3):
            raise ValueError(
                "thermal_image must have shape [B, 1, H, W] or [B, 3, H, W], got "
                f"{tuple(thermal_image.shape)}"
            )
        thermal = thermal_image[:, :1]
        thermal = (thermal * self.source_std + self.source_mean).clamp(0.0, 1.0)
        thermal = (thermal - self.infmae_mean) / self.infmae_std
        return thermal.expand(-1, 3, -1, -1).contiguous()


class _ThermalTokenResidualAdapter(nn.Module):
    """A small, stage-specific residual adapter on dense thermal tokens."""

    def __init__(self, dim: int, bottleneck_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        # The residual is deliberately small but nonzero.  Unlike the hybrid
        # path there is no established CLIP-TIR stream to preserve here, and
        # all stage heads need gradients from the first optimization step.
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = self.up(torch.nn.functional.gelu(self.down(self.norm(tokens))))
        return tokens + self.gain * residual


class TargetAwareTokenPool(nn.Module):
    """Pool patch tokens inside normalized ``cx, cy, w, h`` target boxes.

    The pooler is deliberately independent from the detector prediction: it
    receives the ground-truth box only during training and uses
    :func:`torch.nn.functional.grid_sample`, so gradients flow to every
    sampled visual token.  This gives the InfMAE alignment losses a reliable
    target-level representation even for objects smaller than one 14x14 ViT
    patch.
    """

    def __init__(self, pool_size: int = 4):
        super().__init__()
        if int(pool_size) <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        self.pool_size = int(pool_size)

    def forward(self, patch_tokens: torch.Tensor, boxes_xywh: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError(
                "patch_tokens must have shape [B, N, D], got "
                f"{tuple(patch_tokens.shape)}"
            )
        if boxes_xywh.ndim != 2 or boxes_xywh.shape[-1] != 4:
            raise ValueError(
                "boxes_xywh must have shape [B, 4], got "
                f"{tuple(boxes_xywh.shape)}"
            )
        if patch_tokens.shape[0] != boxes_xywh.shape[0]:
            raise ValueError("patch_tokens and boxes_xywh must share batch size")
        if not torch.isfinite(boxes_xywh).all():
            raise ValueError("boxes_xywh must contain only finite values")

        batch_size, patch_count, feature_dim = patch_tokens.shape
        grid_side = math.isqrt(patch_count)
        if grid_side * grid_side != patch_count:
            raise ValueError(
                "TargetAwareTokenPool requires a square patch grid, got "
                f"{patch_count} tokens"
            )

        boxes = boxes_xywh.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        center_x = boxes[:, 0].clamp(0.0, 1.0)
        center_y = boxes[:, 1].clamp(0.0, 1.0)
        # A positive minimum extent avoids degenerate sampling for very small
        # annotated objects after resize/pad augmentation.
        min_extent = 1.0 / float(max(grid_side * 2, 1))
        width = boxes[:, 2].abs().clamp(min=min_extent, max=1.0)
        height = boxes[:, 3].abs().clamp(min=min_extent, max=1.0)
        x1 = (center_x - 0.5 * width).clamp(0.0, 1.0 - min_extent)
        y1 = (center_y - 0.5 * height).clamp(0.0, 1.0 - min_extent)
        x2 = torch.maximum(center_x + 0.5 * width, x1 + min_extent).clamp(max=1.0)
        y2 = torch.maximum(center_y + 0.5 * height, y1 + min_extent).clamp(max=1.0)
        x1 = torch.minimum(x1, x2 - min_extent).clamp(min=0.0)
        y1 = torch.minimum(y1, y2 - min_extent).clamp(min=0.0)

        sample_positions = (
            torch.arange(
                self.pool_size,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            + 0.5
        ) / self.pool_size
        sample_x = x1[:, None] + (x2 - x1)[:, None] * sample_positions
        sample_y = y1[:, None] + (y2 - y1)[:, None] * sample_positions
        grid_x = sample_x[:, None, :].expand(-1, self.pool_size, -1)
        grid_y = sample_y[:, :, None].expand(-1, -1, self.pool_size)
        # grid_sample uses coordinates in [-1, 1].  ``align_corners=False``
        # matches the patch-grid interpolation used by the thermal adapter.
        sample_grid = torch.stack((2.0 * grid_x - 1.0, 2.0 * grid_y - 1.0), dim=-1)
        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            feature_dim,
            grid_side,
            grid_side,
        )
        pooled_map = F.grid_sample(
            feature_map,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return pooled_map.mean(dim=(-2, -1))


class TargetFeatureProjector(nn.Module):
    """A small normalized projection head for target-level alignment."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if min(int(input_dim), int(output_dim)) <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.norm = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [..., {self.input_dim}] target features, got "
                f"{tuple(features.shape)}"
            )
        return self.projection(self.norm(features))


class TextGuidedThermalSemanticBridge(nn.Module):
    """Inject aligned thermal-token semantics into the TIR grounding stream.

    A3/A5's target losses supervise a thermal projection head, but a plain
    auxiliary head is discarded before LAVS at inference.  This bridge makes
    that semantic representation useful at inference without using GT boxes:
    the referring-expression embedding attends to projected TIR patch tokens
    and produces a small text-conditioned residual on the selected patches.

    The residual gain starts exactly at zero.  Consequently, a newly enabled
    bridge is forward-identical to the original direct-InfMAE frontend and is
    opened only when the grounding objective finds it useful.  Text is a
    detached CLIP teacher, so the bridge cannot distort the RGB/text semantic
    space that supplies the A3 target.
    """

    def __init__(
        self,
        visual_dim: int,
        semantic_dim: int,
        temperature: float = 0.07,
    ):
        super().__init__()
        if min(int(visual_dim), int(semantic_dim)) <= 0:
            raise ValueError("visual_dim and semantic_dim must be positive")
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive")

        self.visual_dim = int(visual_dim)
        self.semantic_dim = int(semantic_dim)
        self.temperature = float(temperature)
        # This is initialized from the transpose of CLIP's visual projection
        # by MMVGFusion whenever that compatible teacher is available.
        self.text_to_visual = nn.Linear(self.semantic_dim, self.visual_dim, bias=False)
        nn.init.xavier_uniform_(self.text_to_visual.weight)
        self.residual_gain = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        semantic_patch_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
        residual_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                "visual_tokens must have shape [B, N+1, visual_dim], got "
                f"{tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[1] < 2:
            raise ValueError("visual_tokens must contain a CLS token and patch tokens")
        if (
            semantic_patch_tokens.ndim != 3
            or semantic_patch_tokens.shape[:2] != (
                visual_tokens.shape[0],
                visual_tokens.shape[1] - 1,
            )
            or semantic_patch_tokens.shape[-1] != self.semantic_dim
        ):
            raise ValueError(
                "semantic_patch_tokens must have shape [B, N, semantic_dim], got "
                f"{tuple(semantic_patch_tokens.shape)}"
            )
        if text_embedding.shape != (visual_tokens.shape[0], self.semantic_dim):
            raise ValueError(
                "text_embedding must have shape [B, semantic_dim], got "
                f"{tuple(text_embedding.shape)}"
            )
        residual_scale = float(residual_scale)
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError(
                f"residual_scale must be in [0, 1], got {residual_scale}"
            )

        semantic_tokens = F.normalize(semantic_patch_tokens.float(), dim=-1)
        # Text supplies an immutable semantic query rather than a trainable
        # target.  This mirrors the detached teacher used by the A3 loss.
        semantic_text = F.normalize(text_embedding.float(), dim=-1).detach()
        compatibility = torch.sum(
            semantic_tokens * semantic_text[:, None, :],
            dim=-1,
        )
        attention = torch.softmax(compatibility / self.temperature, dim=-1)

        # ``tanh(0) == 0`` preserves the original LAVS input exactly at
        # initialization.  The attention is bounded, so even after opening
        # the bridge only semantically selected patches receive a residual.
        residual = attention[:, :, None] * self.text_to_visual(semantic_text)[:, None, :]
        # Reuse the existing A3/A5 adapter warmup as a forward gate.  This
        # keeps the direct frontend untouched while the target projectors
        # first acquire a stable semantic mapping.
        gain = torch.tanh(self.residual_gain) * residual_scale
        output_patches = visual_tokens[:, 1:, :] + gain.to(visual_tokens.dtype) * residual.to(
            visual_tokens.dtype
        )
        output = torch.cat((visual_tokens[:, :1, :], output_patches), dim=1)

        entropy = -torch.sum(
            attention * attention.clamp_min(torch.finfo(attention.dtype).eps).log(),
            dim=-1,
        )
        entropy = entropy / math.log(float(attention.shape[-1]))
        return output, {
            "semantic_bridge_gain": gain.detach(),
            "semantic_bridge_attention_entropy": entropy.detach().mean(),
            "semantic_bridge_attention_peak": attention.detach().amax(dim=-1).mean(),
        }


class InfMAEDirectThermalAdapter(nn.Module):
    """Convert frozen InfMAE F2/F3 maps into the original LAVS TIR interface.

    The original RGBT-VGNet LAVS frontend consumes four CLIP feature sequences
    of shape ``[B, 197, 768]``.  This adapter makes the direct InfMAE branch
    structurally compatible without calling the TIR CLIP encoder:

    * F2 is projected then downsampled from H/8 to H/16;
    * F3 is projected at H/16 and fused with F2 through a learnable gate;
    * lightweight stage adapters emit one sequence for each original LAVS
      input level, retaining the one-global-token-plus-patch-token layout.

    It intentionally does not use text or ground-truth boxes.  Those belong
    to the optional target-aware losses, not to the inference frontend.
    """

    def __init__(
        self,
        f2_dim: int = 384,
        f3_dim: int = 768,
        token_dim: int = 768,
        num_feature_levels: int = 4,
        bottleneck_dim: int = 192,
        f2_scale_init: float = 0.10,
    ):
        super().__init__()
        if min(f2_dim, f3_dim, token_dim, num_feature_levels, bottleneck_dim) <= 0:
            raise ValueError("All direct thermal adapter dimensions must be positive")
        if not 0.0 < f2_scale_init < 1.0:
            raise ValueError("f2_scale_init must lie strictly between 0 and 1")

        self.f2_dim = int(f2_dim)
        self.f3_dim = int(f3_dim)
        self.token_dim = int(token_dim)
        self.num_feature_levels = int(num_feature_levels)

        self.f2_proj = nn.Conv2d(f2_dim, token_dim, kernel_size=1, bias=False)
        self.f3_proj = nn.Conv2d(f3_dim, token_dim, kernel_size=1, bias=False)
        self.f2_norm = nn.LayerNorm(token_dim)
        self.f3_norm = nn.LayerNorm(token_dim)
        self.fusion_norm = nn.LayerNorm(token_dim)
        self.cls_projection = nn.Linear(token_dim, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.stage_cls_bias = nn.Parameter(torch.zeros(num_feature_levels, 1, token_dim))
        self.stage_adapters = nn.ModuleList(
            _ThermalTokenResidualAdapter(token_dim, bottleneck_dim)
            for _ in range(num_feature_levels)
        )
        self.stage_norms = nn.ModuleList(
            nn.LayerNorm(token_dim) for _ in range(num_feature_levels)
        )

        # F3 already has 768 channels in the released Inf30 model.  Preserve
        # that semantic path at initialization when dimensions permit it;
        # F2 starts as a low-weight detail correction and learns its projection.
        if f3_dim == token_dim:
            with torch.no_grad():
                self.f3_proj.weight.zero_()
                diagonal = torch.arange(token_dim)
                self.f3_proj.weight[diagonal, diagonal, 0, 0] = 1.0
        nn.init.eye_(self.cls_projection.weight)
        nn.init.zeros_(self.cls_projection.bias)
        f2_logit = math.log(f2_scale_init / (1.0 - f2_scale_init))
        self.f2_scale_raw = nn.Parameter(torch.tensor(f2_logit))

    @staticmethod
    def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2)

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        if token_count != target_hw[0] * target_hw[1]:
            raise ValueError(
                "Token count does not match target grid: "
                f"{token_count} vs {target_hw}"
            )
        return tokens.transpose(1, 2).reshape(batch_size, channels, *target_hw)

    def _project_map(
        self,
        feature_map: torch.Tensor,
        projection: nn.Module,
        norm: nn.Module,
    ) -> torch.Tensor:
        projected = projection(feature_map)
        tokens = norm(self._map_to_tokens(projected))
        return self._tokens_to_map(tokens, projected.shape[-2:])

    def forward(
        self,
        f2: torch.Tensor,
        f3: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        if f2.ndim != 4 or f3.ndim != 4:
            raise ValueError("InfMAE F2/F3 inputs must be feature maps [B, C, H, W]")
        if f2.shape[0] != f3.shape[0]:
            raise ValueError("InfMAE F2 and F3 must share the batch dimension")
        if f2.shape[1] != self.f2_dim or f3.shape[1] != self.f3_dim:
            raise ValueError(
                "Unexpected InfMAE feature dimensions: "
                f"F2={f2.shape[1]} (expected {self.f2_dim}), "
                f"F3={f3.shape[1]} (expected {self.f3_dim})"
            )

        f2_projected = self._project_map(f2, self.f2_proj, self.f2_norm)
        f3_projected = self._project_map(f3, self.f3_proj, self.f3_norm)
        target_hw = f3_projected.shape[-2:]
        if f2_projected.shape[-2:] != target_hw:
            f2_projected = torch.nn.functional.interpolate(
                f2_projected,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        base_patches = self.fusion_norm(
            self._map_to_tokens(
                f3_projected + torch.sigmoid(self.f2_scale_raw) * f2_projected
            )
        )
        pooled = base_patches.mean(dim=1)
        base_cls = self.cls_projection(pooled).unsqueeze(1) + self.cls_token

        stage_tokens = []
        for stage_index, (adapter, norm) in enumerate(
            zip(self.stage_adapters, self.stage_norms)
        ):
            patches = adapter(base_patches)
            tokens = torch.cat(
                [base_cls + self.stage_cls_bias[stage_index].unsqueeze(0), patches],
                dim=1,
            )
            stage_tokens.append(norm(tokens))

        expected_tokens = target_hw[0] * target_hw[1] + 1
        if any(tokens.shape != (f2.shape[0], expected_tokens, self.token_dim) for tokens in stage_tokens):
            raise RuntimeError("Direct InfMAE adapter produced an invalid LAVS token shape")
        # The original TIR CLIP path returns a mutable list because the LAVS
        # cross-fusion loop updates every level in place.  Preserve that exact
        # interface rather than requiring any downstream framework change.
        return stage_tokens, pooled


class InfMAEMultiScaleThermalAdapter(nn.Module):
    """Adapt frozen InfMAE F2/F3 maps into a zero-init TIR residual.

    The adapter outputs a dense 512-d patch residual at the original CLIP
    grid.  Its final projection is initialized to zero, making the initial
    path an exact identity while still allowing that final projection to get
    a useful first-step gradient.  This is preferable to multiplying the
    whole expert by a zero scalar, which would initially starve every expert
    projection of gradient.  The current TIR CLIP+LoRA, LAVS, IAFv3, and GQR
    behavior is therefore preserved at initialization.
    """

    def __init__(
        self,
        f2_dim: int = 384,
        f3_dim: int = 768,
        out_dim: int = 512,
        bottleneck_dim: int = 128,
        residual_gain_init: float = 0.5,
    ):
        super().__init__()
        if min(f2_dim, f3_dim, out_dim, bottleneck_dim) <= 0:
            raise ValueError("All adapter dimensions must be positive")
        if not 0.0 < residual_gain_init < 1.0:
            raise ValueError("residual_gain_init must lie strictly between 0 and 1")

        self.out_dim = out_dim
        self.f2_proj = nn.Conv2d(f2_dim, out_dim, kernel_size=1, bias=False)
        self.f3_proj = nn.Conv2d(f3_dim, out_dim, kernel_size=1, bias=False)
        self.f2_norm = nn.LayerNorm(out_dim)
        self.f3_norm = nn.LayerNorm(out_dim)
        self.adapter_norm = nn.LayerNorm(out_dim)
        self.adapter_down = nn.Linear(out_dim, bottleneck_dim)
        self.adapter_up = nn.Linear(bottleneck_dim, out_dim)
        # The branch is exactly zero initially, but unlike a zero scalar gate
        # the output projection learns immediately from the grounding loss.
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.f2_scale_raw = nn.Parameter(torch.tensor(0.0))
        residual_gain_logit = math.log(residual_gain_init / (1.0 - residual_gain_init))
        self.residual_gain_raw = nn.Parameter(torch.tensor(residual_gain_logit))

    @staticmethod
    def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2)

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        if token_count != target_hw[0] * target_hw[1]:
            raise ValueError(
                "Token count does not match target grid: "
                f"{token_count} vs {target_hw}"
            )
        return tokens.transpose(1, 2).reshape(batch_size, channels, *target_hw)

    def _project_map(self, feature_map: torch.Tensor, projection: nn.Module, norm: nn.Module) -> torch.Tensor:
        projected = projection(feature_map)
        tokens = norm(self._map_to_tokens(projected))
        return self._tokens_to_map(tokens, projected.shape[-2:])

    def forward(
        self,
        tir_tokens: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tir_tokens.ndim != 3 or tir_tokens.shape[-1] != self.out_dim:
            raise ValueError(
                "tir_tokens must have shape [B, N+1, "
                f"{self.out_dim}], got {tuple(tir_tokens.shape)}"
            )
        if f2.ndim != 4 or f3.ndim != 4:
            raise ValueError("InfMAE F2/F3 inputs must be feature maps [B, C, H, W]")
        if not (tir_tokens.shape[0] == f2.shape[0] == f3.shape[0]):
            raise ValueError("tir_tokens, F2, and F3 must share the batch dimension")

        patch_count = tir_tokens.shape[1] - 1
        patch_side = math.isqrt(patch_count)
        if patch_side * patch_side != patch_count:
            raise ValueError(
                "The current thermal adapter requires a square CLIP patch grid, got "
                f"{patch_count} patches"
            )
        target_hw = (patch_side, patch_side)

        f2_projected = self._project_map(f2, self.f2_proj, self.f2_norm)
        f3_projected = self._project_map(f3, self.f3_proj, self.f3_norm)
        if f2_projected.shape[-2:] != target_hw:
            f2_projected = torch.nn.functional.interpolate(
                f2_projected,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if f3_projected.shape[-2:] != target_hw:
            f3_projected = torch.nn.functional.interpolate(
                f3_projected,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        fused_tokens = self._map_to_tokens(
            f3_projected + torch.sigmoid(self.f2_scale_raw) * f2_projected
        )
        adapter_delta = self.adapter_up(
            torch.nn.functional.gelu(self.adapter_down(self.adapter_norm(fused_tokens)))
        )

        residual_gain = torch.sigmoid(self.residual_gain_raw)
        residual = residual_gain * adapter_delta
        enhanced = torch.cat(
            [tir_tokens[:, :1], tir_tokens[:, 1:] + residual],
            dim=1,
        )
        base_norm = tir_tokens[:, 1:].detach().norm(dim=-1).mean().clamp_min(1e-6)
        residual_ratio = residual.detach().norm(dim=-1).mean() / base_norm
        aux = {
            "thermal_rho": residual_gain.detach(),
            "thermal_residual_ratio": residual_ratio.detach(),
            "infmae_f2_scale": torch.sigmoid(self.f2_scale_raw).detach(),
        }
        return enhanced, aux
