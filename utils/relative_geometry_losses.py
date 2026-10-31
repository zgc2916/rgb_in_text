"""Bounded target-relative localization supervision; no inference changes."""
from torch.nn import functional as F


def relative_geometry_losses(args, prediction, target):
    floor = float(args.relative_geometry_floor_pixels) / args.imsize
    if floor <= 0:
        raise ValueError('Relative geometry requires a positive numerical size floor')
    normalizer = target[:, 2:].detach().clamp(min=floor)
    center = F.smooth_l1_loss((prediction[:, :2] - target[:, :2]) / normalizer,
                             prediction[:, :2] * 0., reduction='none').sum(-1).mean()
    log_ratio = (prediction[:, 2:].clamp(min=1e-6) / target[:, 2:].clamp(min=1e-6)).log()
    shape = F.smooth_l1_loss(log_ratio, log_ratio * 0., reduction='none').sum(-1).mean()
    return {'loss_relative_center': float(args.relative_center_weight) * center,
            'loss_relative_shape': float(args.relative_shape_weight) * shape}
