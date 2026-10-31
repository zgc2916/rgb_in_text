"""Spatial center evidence and counterfactual utility for native RGB/TIR reads.

Spatial expectations and distribution-based quality estimation have established
precedents (DSNT and GFLV2). This experiment tests their text-conditioned RGB/TIR
adaptation relative to a shared coarse prediction; novelty is not assumed.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .local_reobserve import ReobserveHead, LocalReobserveGrounder


class SpatialQualityHead(ReobserveHead):
    def __init__(self, hidden=128, mode='adaptive_benefit'):
        super().__init__(hidden=hidden)
        self.mode = mode
        for name in ('local_read', 'output_norm', 'delta', 'blend', 'accept'):
            delattr(self, name)
        self.score = nn.Linear(hidden, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.quality = nn.Sequential(nn.Linear(2 * hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    def decode(self, queries, coarse, canvas_crops, local):
        side = math.isqrt(local[0].shape[1] - 1)
        axis = (torch.arange(side, device=coarse.device, dtype=coarse.dtype) + .5) / side - .5
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        unit = torch.stack((x.flatten(), y.flatten()), -1)
        reference = coarse.detach()
        positions = canvas_crops[:, :, None, :2] + unit[None, None] * canvas_crops[:, :, None, 2:]
        sigma = torch.maximum(reference[:, None, 2:] / 2, canvas_crops[..., 2:] / 4)
        prior_logits = (-.5 * ((positions - reference[:, None, None, :2]) /
                         sigma[:, :, None].clamp(min=1e-6)).square().sum(-1)).detach()
        prior = prior_logits.softmax(-1)
        logits, contexts = [], []
        for i, tokens in enumerate(local):
            relative = (positions[:, i] - reference[:, None, :2]) / reference[:, None, 2:].clamp(min=1 / 224)
            features = self.visual_proj(self.visual_norm(tokens[:, 1:]))
            features = F.gelu(features + queries[:, i:i + 1] + self.position(relative.detach().clamp(-8, 8)))
            logits.append(prior_logits[:, i] + self.score(features)[..., 0])
            contexts.append(features)
        logits = torch.stack(logits, 1)
        posterior = logits.softmax(-1)
        prior_center = (prior[..., None] * positions).sum(-2)
        posterior_center = (posterior[..., None] * positions).sum(-2)
        center_delta = posterior_center - prior_center
        # Initial score logits are exactly zero, so this subtraction gives an
        # exact zero correction even when a discrete prior is off grid center.
        proposal_center = coarse[:, None, :2] + center_delta
        proposals = torch.cat((proposal_center, coarse[:, None, 2:].expand(-1, 2, -1)), -1)
        variance = (posterior[..., None] * (positions - posterior_center[:, :, None]).square()).sum(-2)
        entropy = -(posterior * posterior.clamp(min=1e-8).log()).sum(-1) / math.log(side ** 2)
        normalizer = reference[:, None, 2:].clamp(min=1 / 224)
        disagreement = (proposal_center[:, 0] - proposal_center[:, 1]).abs() / normalizer[:, 0]
        pooled = torch.stack([(posterior[:, i, :, None] * contexts[i]).sum(1) for i in range(2)], 1)
        geometry = torch.cat((center_delta / normalizer, (variance / normalizer.square()).log1p(),
            entropy[..., None], (center_delta / normalizer).norm(dim=-1, keepdim=True),
            disagreement[:, None].expand(-1, 2, -1)), -1).clamp(-8, 8)
        # Utility supervision cannot reshape its own proposal/uncertainty
        # features to make the utility regression artificially easier.
        quality_features = torch.cat((queries, pooled, geometry), -1).detach()
        predicted_gain = self.quality(quality_features)[..., 0]
        if self.mode == 'spatial_uniform':
            weights = torch.ones_like(predicted_gain) / 3
        else:
            positive = (predicted_gain / .05).clamp(min=0, max=10)
            weights = positive / (1 + positive.sum(-1, keepdim=True))
        center = coarse[:, :2] + (weights[..., None] * center_delta).sum(1)
        final = torch.cat((center.clamp(0, 1), coarse[:, 2:]), -1)
        return final, {'spatial_logits': logits, 'spatial_positions': positions,
            'spatial_posterior': posterior, 'spatial_entropy': entropy,
            'spatial_predicted_gain': predicted_gain, 'spatial_weights': weights,
            'reobserve_proposals': proposals, 'reobserve_crops': canvas_crops,
            'reobserve_coarse': coarse}


class SpatialQualityGrounder(LocalReobserveGrounder):
    def __init__(self, base, args):
        super().__init__(base, args)
        if args.reobserve_mode != 'c1_continue':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed)
                self.refiner = SpatialQualityHead(hidden=args.reobserve_hidden, mode=args.reobserve_mode)
            # Reuse the tested native-image encoder/crop path. Only D enables
            # geometric adaptation; B/C have the same policy network disabled.
            self.mode = 'text_adaptive' if args.reobserve_mode == 'adaptive_benefit' else 'fixed_local'
            self.configure_tuning()

    def configure_tuning(self):
        super().configure_tuning()
        if (self.args.reobserve_mode == 'spatial_uniform' and self.refiner is not None
                and hasattr(self.refiner, 'quality')):
            for parameter in self.refiner.quality.parameters():
                parameter.requires_grad_(False)
