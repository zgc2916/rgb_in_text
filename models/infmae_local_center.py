"""Bounded center refinement from the original 28x28 InfMAE F2 map."""

import torch
from torch import nn
from torch.nn import functional as F


class InfMAELocalCenterRefiner(nn.Module):
    """Use a predicted-box crop; initialization exactly preserves all boxes."""

    def __init__(self, f2_dim, query_dim, image_size=224, max_shift_pixels=4.0):
        super().__init__()
        if image_size <= 0 or max_shift_pixels <= 0:
            raise ValueError("image_size and max_shift_pixels must be positive")
        self.image_size = int(image_size)
        self.max_shift_pixels = float(max_shift_pixels)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(f2_dim, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.query_projection = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, 64))
        self.offset_head = nn.Sequential(
            nn.Linear(25 * 64 + 64 + 4, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        coordinates = torch.linspace(-0.75, 0.75, 5)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("crop_offsets", torch.stack((xx, yy), dim=-1), persistent=False)

    def forward(self, f2, query, baseline_boxes):
        # The frozen model supplies features and boxes, never training labels.
        boxes = baseline_boxes.detach()
        feature_map = self.feature_projection(f2.detach().float())
        height, width = f2.shape[-2:]
        cell_size = boxes.new_tensor((1.0 / width, 1.0 / height))
        crop_size = torch.maximum(boxes[:, 2:], 2 * cell_size)
        grid = boxes[:, None, None, :2] + self.crop_offsets[None] * crop_size[:, None, None]
        crop = F.grid_sample(feature_map, 2 * grid - 1, mode="bilinear", align_corners=False)
        context = self.query_projection(query.detach().float())
        offset = self.offset_head(torch.cat((crop.flatten(1), context, boxes), dim=1))

        # Fully apply below a 16px predicted short side, taper to zero at 32px.
        # Limit center shifts to 25% of box size (at least one F2 cell as the
        # scale), with a separate absolute pixel cap. Width/height stay exact.
        short_side = boxes[:, 2:].amin(dim=1, keepdim=True) * self.image_size
        gate = ((32.0 - short_side) / 16.0).clamp(0, 1)
        relative_limit = 0.25 * torch.maximum(boxes[:, 2:], cell_size)
        shift_limit = relative_limit.clamp(max=self.max_shift_pixels / self.image_size)
        shift = offset.tanh() * shift_limit * gate
        center = (boxes[:, :2] + shift).clamp(0, 1)
        refined = torch.cat((center, boxes[:, 2:]), dim=1)
        diagnostics = {
            "local_center_shift_pixels": ((center - boxes[:, :2]).norm(dim=1) * self.image_size).mean().detach(),
            "local_center_active_fraction": (gate > 0).float().mean().detach(),
            "local_center_gate_mean": gate.mean().detach(),
        }
        return refined, diagnostics
