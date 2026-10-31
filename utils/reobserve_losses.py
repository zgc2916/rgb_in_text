"""GT-based training supervision, separate from native-image crop selection."""
import torch
from torch.nn import functional as F

from utils.box_utils import xywh2xyxy, generalized_box_iou


def reobserve_losses(args, auxiliary, target):
    if auxiliary is None:
        raise ValueError('Re-observation supervision requires training auxiliary outputs')
    crops = xywh2xyxy(auxiliary['reobserve_crops'].detach())
    boxes = xywh2xyxy(target)[:, None]
    covered = ((crops[..., :2] <= boxes[..., :2]) & (crops[..., 2:] >= boxes[..., 2:])).all(-1)
    proposals = auxiliary['reobserve_proposals']
    losses = []
    for i in range(2):
        valid = covered[:, i]
        if valid.any():
            p, t = proposals[valid, i], target[valid]
            losses.append((2 * (p - t).abs().sum() +
                           2 * (1 - torch.diag(generalized_box_iou(xywh2xyxy(p), xywh2xyxy(t)))).sum()) / target.shape[0])
        else:
            losses.append(proposals[:, i].sum() * 0.)
    labels = covered.any(-1).float()
    per_item = F.binary_cross_entropy_with_logits(auxiliary['reobserve_accept_logits'], labels, reduction='none')
    positive, negative = labels.bool(), ~labels.bool()
    terms = [per_item[mask].mean() for mask in (positive, negative) if mask.any()]
    result = {'loss_reobserve_local': .1 * sum(losses) / 2,
              'loss_reobserve_coverage': .1 * sum(terms) / len(terms),
              'reobserve_coverage_fraction': labels.mean().detach()}
    if args.reobserve_mode == 'text_consistent':
        valid = covered.all(-1)
        # Both local reads correspond to potentially different scales. Compare
        # the mapped original-canvas boxes, normalizing by target extent only
        # inside the supervised training objective.
        norm = torch.cat((target[:, 2:], target[:, 2:]), -1).clamp(min=1 / 224)
        discrepancy = F.smooth_l1_loss(proposals[:, 0] / norm, proposals[:, 1] / norm, reduction='none').mean(-1)
        result['loss_reobserve_consistency'] = .05 * (discrepancy * valid).sum() / target.shape[0]
    return result
