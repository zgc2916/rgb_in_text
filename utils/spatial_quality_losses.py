"""Spatial supervision and relative-IoU utility targets; training only."""
import torch
from torch.nn import functional as F

from utils.box_utils import xywh2xyxy, bbox_iou


def spatial_quality_losses(args, aux, target):
    if aux is None:
        raise ValueError('Spatial localization supervision needs training auxiliary outputs')
    crops = aux['reobserve_crops'].detach()
    positions = aux['spatial_positions'].detach()
    difference = (positions - target[:, None, None, :2]) / torch.maximum(
        target[:, None, None, 2:] / 2, crops[:, :, None, 2:] / 14).clamp(min=1e-6)
    teacher = (-.5 * difference.square().sum(-1)).softmax(-1)
    corners = xywh2xyxy(crops)
    center_covered = ((target[:, None, :2] >= corners[..., :2]) &
                      (target[:, None, :2] <= corners[..., 2:])).all(-1)
    kl = (teacher * (teacher.clamp(min=1e-8).log() - aux['spatial_logits'].log_softmax(-1))).sum(-1)
    result = {'loss_spatial_center': .2 * (kl * center_covered).sum() / (2 * target.shape[0]),
              'spatial_center_coverage': center_covered.float().mean().detach()}
    proposals, coarse = aux['reobserve_proposals'], aux['reobserve_coarse']
    norm = target[:, None, 2:].clamp(min=1 / 224)
    center_error = F.smooth_l1_loss(proposals[..., :2] / norm, target[:, None, :2].expand(-1, 2, -1) / norm, reduction='none').mean(-1)
    result['loss_spatial_coordinate'] = .1 * (center_error * center_covered).sum() / (2 * target.shape[0])
    with torch.no_grad():
        reference = bbox_iou(xywh2xyxy(coarse.detach()).clamp(0, 1), xywh2xyxy(target))
        quality = torch.stack([bbox_iou(xywh2xyxy(proposals[:, i].detach()).clamp(0, 1),
                                        xywh2xyxy(target)) for i in range(2)], 1)
        gain = quality - reference[:, None]
    if args.reobserve_mode != 'spatial_uniform':
        result['loss_spatial_utility'] = .05 * F.smooth_l1_loss(
            aux['spatial_predicted_gain'] / .05, gain / .05)
    result['spatial_true_positive_gain_fraction'] = (gain > .01).float().mean().detach()
    result['spatial_predicted_positive_gain_fraction'] = (aux['spatial_predicted_gain'] > 0).float().mean().detach()
    return result
