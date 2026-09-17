import mpmath
import torch
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
from torch import nn

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size, mdetr_interpolate
try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def build_target(args, gt_bbox, pred, device):
    batch_size = gt_bbox.size(0)
    num_scales = len(pred)
    coord_list, bbox_list = [], []
    for scale_ii in range(num_scales):
        this_stride = 32 // (2 ** scale_ii)
        grid = args.size // this_stride
        # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
        center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
        # Normalized by the image size
        coord = coord / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch_size):
        anch_ious = []
        for scale_ii in range(num_scales):
            this_stride = 32 // (2 ** scale_ii)
            grid = args.size // this_stride
            gw = coord_list[scale_ii][ii, 2]
            gh = coord_list[scale_ii][ii, 3]

            anchor_idxs = [x + 3 * scale_ii for x in [0, 1, 2]]
            anchors = [args.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (args.anchor_imsize / grid),
                               x[1] / (args.anchor_imsize / grid)) for x in anchors]

            gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(
                np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))

            # Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))

        # Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n // 3

        best_grid = args.size // (32 / (2 ** best_scale))
        anchor_idxs = [x + 3 * best_scale for x in [0, 1, 2]]
        anchors = [args.anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [(x[0] / (args.anchor_imsize / best_grid), \
                           x[1] / (args.anchor_imsize / best_grid)) for x in anchors]

        gi = coord_list[best_scale][ii, 0].long()
        gj = coord_list[best_scale][ii, 1].long()
        tx = coord_list[best_scale][ii, 0] - gi.float()
        ty = coord_list[best_scale][ii, 1] - gj.float()
        gw = coord_list[best_scale][ii, 2]
        gh = coord_list[best_scale][ii, 3]
        tw = torch.log(gw / scaled_anchors[best_n % 3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n % 3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n % 3, :, gj, gi] = torch.stack(
            [tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].to(device)
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1. / 5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    num_scale = len(pred_list)
    batch_size = pred_list[0].size(0)

    pred_bbox = torch.zeros(batch_size, 4).to(device)
    gt_bbox = torch.zeros(batch_size, 4).to(device)
    for ii in range(batch_size):
        pred_bbox[ii, 0:2] = torch.sigmoid(
            pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
    loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
    loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
    loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(num_scale):
        pred_conf_list.append(pred_list[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
        gt_conf_list.append(target[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf


class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):

        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(pooled_image.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0
        return loss


# The code below is copied from transformers/models/clip/modeling_clip.py
# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def siglip_loss(logits: torch.Tensor) -> torch.Tensor:
    """SigLIP's pairwise sigmoid loss for a local image/text batch."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"SigLIP loss expects square [B, B] logits, got {tuple(logits.shape)}")
    labels = 2 * torch.eye(logits.shape[0], device=logits.device, dtype=logits.dtype) - 1
    return -F.logsigmoid(labels * logits).sum(dim=-1).mean()


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    # print("[DEBUG] inputs device:", inputs.device, "targets device:", targets.device)
    # print("[DEBUG] inputs shape:", inputs.shape, "targets shape:", targets.shape)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def aligned_iou_xywh(box1, box2, eps=1e-6):
    """Return IoU for aligned normalized ``cx, cy, w, h`` box pairs.

    This intentionally avoids constructing a pairwise matrix and selecting a
    diagonal afterwards, which makes the RGB/TIR GQR teacher unambiguous.
    """
    if box1.ndim != 2 or box2.ndim != 2 or box1.shape != box2.shape or box1.shape[-1] != 4:
        raise ValueError(
            "aligned_iou_xywh expects two equally shaped [B, 4] tensors, got "
            f"{tuple(box1.shape)} and {tuple(box2.shape)}"
        )

    b1 = xywh2xyxy(box1)
    b2 = xywh2xyxy(box2)
    lt = torch.maximum(b1[:, :2], b2[:, :2])
    rb = torch.minimum(b1[:, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)


def _grounding_box_loss(batch_pred, batch_target, num_boxes):
    """Use the same explicit 2*L1 + 2*GIoU convention as the main box loss."""
    loss_l1 = F.l1_loss(batch_pred, batch_target, reduction='none').sum() / num_boxes
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target),
    ))
    return 2.0 * loss_l1 + 2.0 * (loss_giou.sum() / num_boxes)


def gqr_ramp_scale(args, epoch):
    """Return the shared GQR supervision/correction warmup scale.

    During auxiliary-head warmup, a random router must not perturb IAFv3
    through the main grounding loss.  Keeping this schedule in one helper
    ensures that the loss and fusion path use identical timing.
    """
    if epoch is None:
        return 1.0

    start_epoch = int(getattr(args, 'gqr_start_epoch', 5))
    ramp_epochs = max(int(getattr(args, 'gqr_ramp_epochs', 5)), 1)
    if epoch < start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch + 1) / ramp_epochs)


def infmae_alignment_ramp_scale(args, epoch):
    """Warm up A3/A4/A5 target-level losses without changing grounding loss."""
    if epoch is None:
        return 1.0
    start_epoch = int(getattr(args, 'infmae_alignment_start_epoch', 0))
    ramp_epochs = max(int(getattr(args, 'infmae_alignment_ramp_epochs', 5)), 1)
    if epoch < start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch + 1) / ramp_epochs)


def infmae_rgb_tir_ramp_scale(args, epoch):
    """Warm up RGB-to-TIR transfer independently of the A3 text objective.

    In a full A5 run, the TIR encoder first has to acquire a stable language
    anchor.  Enabling a second, RGB semantic teacher from the first update
    made the two target losses compete during early fine-tuning.  The two
    start/ramp controls provide a reproducible A3-to-A5 curriculum while
    retaining the original simultaneous-A5 behavior as the CLI default.
    """
    if epoch is None:
        return 1.0
    start_epoch = int(getattr(args, 'infmae_rgb_tir_start_epoch', 0))
    ramp_epochs = max(int(getattr(args, 'infmae_rgb_tir_ramp_epochs', 5)), 1)
    if epoch < start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch + 1) / ramp_epochs)


def infmae_alignment_adapter_ramp_scale(args, epoch):
    """Control when alignment gradients may refine the thermal adapter.

    The target projectors are randomly initialized (apart from their output
    projection), whereas the A2 thermal adapter is already a converged
    grounding representation.  Warming up the projectors alone prevents a
    random semantic head from immediately pulling that representation away
    from its optimum.  It does not remove the A3/A5 loss: its projector is
    trained throughout, and its gradient enters the adapter after this ramp.
    """
    if epoch is None:
        return 1.0
    start_epoch = int(getattr(args, 'infmae_alignment_adapter_start_epoch', 0))
    ramp_epochs = max(
        int(getattr(args, 'infmae_alignment_adapter_ramp_epochs', 1)),
        1,
    )
    if epoch < start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch + 1) / ramp_epochs)


def _infmae_alignment_mode(args):
    """Return the resolved A2/A3/A4/A5 auxiliary mode from training args."""
    mode = str(getattr(args, 'infmae_alignment_mode', 'none')).lower()
    if mode == 'auto':
        mode = {
            'InfMAEA3': 'tir_text',
            'InfMAEA4': 'rgb_tir',
            'InfMAEA5': 'both',
            'InfMAEA5Bridge': 'both',
        }.get(getattr(args, 'FusionMethod', ''), 'none')
    if mode not in {'none', 'tir_text', 'rgb_tir', 'both'}:
        raise ValueError(
            'infmae_alignment_mode must be none/tir_text/rgb_tir/both, '
            f'got {mode!r}'
        )
    return mode


def _gather_detached_candidates(features):
    """Gather same-sized DDP candidates while retaining a simple local API.

    A batch is intentionally the positive set for target-aware InfoNCE.  The
    train loader uses ``drop_last=True``; detecting unequal local batch sizes
    here turns a silent label mismatch into a clear configuration error.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return features.detach(), 0

    local_size = torch.tensor([features.shape[0]], device=features.device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(item.item()) for item in gathered_sizes]
    if any(size != sizes[0] for size in sizes):
        raise RuntimeError(
            'Target-aware InfoNCE requires equal per-rank batch sizes; '
            f'got {sizes}. Keep training drop_last=True.'
        )

    gathered = [torch.empty_like(features) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, features.detach())
    return torch.cat(gathered, dim=0), dist.get_rank() * features.shape[0]


def _target_aware_tir_text_infonce(tir_embedding, text_embedding, temperature):
    """Compute the document's target-TIR-to-expression InfoNCE objective."""
    if tir_embedding.ndim != 2 or text_embedding.ndim != 2:
        raise ValueError('TIR/text embeddings must both have shape [B, D]')
    if tir_embedding.shape != text_embedding.shape:
        raise ValueError(
            'TIR/text embedding shapes must match; got '
            f'{tuple(tir_embedding.shape)} and {tuple(text_embedding.shape)}'
        )
    if temperature <= 0:
        raise ValueError('--infmae_tir_text_tau must be positive')

    all_text, target_offset = _gather_detached_candidates(text_embedding)
    logits = torch.matmul(tir_embedding.float(), all_text.float().transpose(0, 1)) / temperature
    targets = torch.arange(tir_embedding.shape[0], device=tir_embedding.device) + target_offset
    loss = F.cross_entropy(logits, targets)
    positive_similarity = (tir_embedding.detach() * text_embedding.detach()).sum(dim=-1).mean()
    return loss, positive_similarity, logits.shape[-1]


def trans_vg_loss(
    args,
    batch_pred,
    batch_target,
    tgt_mask,
    text_eos,
    img_cls=None,
    visu_sim=None,
    seg_mask=None,
    aux=None,
    epoch=None,
):
    """Compute the losses related to the bounding boxes,
       including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    # torch.diag(), It means that the diagonal elements of the matrix are taken out as separate tensors
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))

    losses = {}

    coef1 = 2.0
    coef2 = 2.0
    losses['loss_bbox'] = (loss_bbox.sum() / num_boxes) * coef1
    losses['loss_giou'] = (loss_giou.sum() / num_boxes) * coef2

    if args.use_contrastive_loss:
        if isinstance(text_eos, dict):
            if getattr(args, 'contrastive_loss', 'clip') != 'siglip':
                raise ValueError('RGB/IR logits require --contrastive_loss siglip')
            loss_contrastive = torch.stack([siglip_loss(logits) for logits in text_eos.values()]).mean()
        elif getattr(args, 'contrastive_loss', 'clip') == 'siglip':
            loss_contrastive = siglip_loss(text_eos)
        else:
            loss_contrastive = clip_loss(text_eos)
        losses['loss_contrastive'] = loss_contrastive

    if args.use_rtcc_constrain_loss:
        coef_focal = 20.0
        coef_dice = 2.0
        patch_num = int(mpmath.sqrt(visu_sim.shape[-1]))
        # Downward interpolation
        obj_mask = mdetr_interpolate(tgt_mask.float(), (patch_num, patch_num), mode="nearest")[:, 0] > 0.5
        obj_mask = obj_mask.flatten(1).float()
        visu_sim = visu_sim.flatten(1)
        # print("[DEBUG] visu_sim:", visu_sim.device, "obj_mask:", obj_mask.device)
        losses['loss_algin_focal'] = sigmoid_focal_loss(visu_sim, obj_mask, num_boxes) * coef_focal
        losses['loss_align_dice'] = dice_loss(visu_sim, obj_mask, num_boxes) * coef_dice

    if args.use_mask_loss:
        coef_focal = 20.0
        coef_dice = 2.0
        src_mask = mdetr_interpolate(seg_mask, size=tgt_mask.shape[-2:], mode="bilinear", align_corners=False)
        src_mask = src_mask.flatten(1)
        tgt_mask = tgt_mask.flatten(1).float()

        losses['loss_seg_focal'] = sigmoid_focal_loss(src_mask, tgt_mask, num_boxes) * coef_focal
        losses['loss_seg_dice'] = dice_loss(src_mask, tgt_mask, num_boxes) * coef_dice

    if aux is not None and getattr(args, 'enable_gqr', False):
        required_aux = ('rgb_aux_box', 'tir_aux_box', 'router_logits')
        missing_aux = [key for key in required_aux if key not in aux]
        if missing_aux:
            raise KeyError(f"GQR auxiliary output is missing required keys: {missing_aux}")

        rgb_box = aux['rgb_aux_box']
        tir_box = aux['tir_aux_box']
        router_logits = aux['router_logits']
        if rgb_box.shape != batch_target.shape or tir_box.shape != batch_target.shape:
            raise ValueError(
                "GQR auxiliary boxes must match the target shape; got "
                f"rgb={tuple(rgb_box.shape)}, tir={tuple(tir_box.shape)}, "
                f"target={tuple(batch_target.shape)}"
            )
        if router_logits.shape != (batch_size, 2):
            raise ValueError(
                "GQR router logits must have shape [B, 2], got "
                f"{tuple(router_logits.shape)}"
            )

        aux_weight = float(getattr(args, 'gqr_aux_weight', 0.25))
        losses['loss_aux_rgb'] = aux_weight * _grounding_box_loss(rgb_box, batch_target, num_boxes)
        losses['loss_aux_tir'] = aux_weight * _grounding_box_loss(tir_box, batch_target, num_boxes)

        tau = float(getattr(args, 'gqr_tau', 0.25))
        if tau <= 0:
            raise ValueError('--gqr_tau must be positive')
        teacher_margin = float(getattr(args, 'gqr_teacher_margin', 0.0))
        if teacher_margin < 0:
            raise ValueError('--gqr_teacher_margin must be non-negative')
        # The teacher is deliberately detached: auxiliary heads learn from
        # their own box losses and cannot lower router loss by changing the
        # target distribution itself.
        with torch.no_grad():
            q_rgb = aligned_iou_xywh(rgb_box.detach(), batch_target)
            q_tir = aligned_iou_xywh(tir_box.detach(), batch_target)
            teacher = torch.softmax(torch.stack([q_rgb, q_tir], dim=-1) / tau, dim=-1)
            teacher_quality_gap = (q_rgb - q_tir).abs()
            reliable_teacher = teacher_quality_gap >= teacher_margin

        per_sample_router_loss = F.kl_div(
            F.log_softmax(router_logits, dim=-1),
            teacher,
            reduction='none',
        ).sum(dim=-1)
        # If two detached auxiliary IoUs are nearly identical, the soft
        # target is almost 50/50 and its argmax is noise.  The optional margin
        # lets a resource-constrained run avoid fitting those ambiguous
        # labels, while the default (0) preserves the original all-sample KL.
        if reliable_teacher.any():
            router_loss = per_sample_router_loss[reliable_teacher].mean()
        else:
            # Keep the router connected to the graph on a rank that has no
            # confident teacher samples, which is important for DDP.
            router_loss = router_logits.sum() * 0.0
        gqr_scale = gqr_ramp_scale(args, epoch)
        losses['loss_gqr_router'] = (
            router_loss
            * float(getattr(args, 'gqr_router_weight', 0.20))
            * gqr_scale
        )

        # Non-loss values are diagnostics only.  ``engine._sum_loss_terms``
        # excludes them from backpropagation while MetricLogger records them.
        router_prob = aux.get('router_prob', router_logits.softmax(dim=-1))
        losses['router_rgb_mean'] = router_prob[:, 0].detach().mean()
        losses['router_tir_mean'] = router_prob[:, 1].detach().mean()
        losses['teacher_rgb_mean'] = teacher[:, 0].mean()
        losses['teacher_tir_mean'] = teacher[:, 1].mean()
        if reliable_teacher.any():
            losses['router_teacher_acc'] = (
                router_prob.detach().argmax(dim=-1)[reliable_teacher]
                == teacher.argmax(dim=-1)[reliable_teacher]
            ).float().mean()
        else:
            losses['router_teacher_acc'] = batch_pred.new_tensor(0.0)
        losses['router_teacher_coverage'] = reliable_teacher.float().mean()
        losses['teacher_quality_gap_mean'] = teacher_quality_gap.mean()
        losses['gqr_ramp_scale'] = batch_pred.new_tensor(gqr_scale)
        for key in (
            'gqr_eta',
            'gqr_correction_scale',
            'w_rgb_base_mean',
            'w_rgb_final_mean',
        ):
            if key in aux:
                losses[key] = aux[key].detach()

    infmae_alignment_mode = _infmae_alignment_mode(args)
    if infmae_alignment_mode != 'none':
        if aux is None:
            raise KeyError(
                'InfMAE A3/A4/A5 requires training-time auxiliary embeddings; '
                'ensure GT target boxes are passed to MMVGFusion.'
            )
        ramp_scale = infmae_alignment_ramp_scale(args, epoch)
        if infmae_alignment_mode in {'tir_text', 'both'}:
            required_keys = ('tir_text_embedding', 'text_embedding')
            missing_keys = [key for key in required_keys if key not in aux]
            if missing_keys:
                raise KeyError(
                    'InfMAE TIR-Text alignment output is missing keys: '
                    f'{missing_keys}'
                )
            tir_text_weight = float(getattr(args, 'infmae_tir_text_weight', 0.10))
            if tir_text_weight < 0:
                raise ValueError('--infmae_tir_text_weight must be non-negative')
            tir_text_loss, positive_similarity, candidate_count = _target_aware_tir_text_infonce(
                aux['tir_text_embedding'],
                aux['text_embedding'],
                float(getattr(args, 'infmae_tir_text_tau', 0.07)),
            )
            losses['loss_infmae_tir_text'] = tir_text_loss * tir_text_weight * ramp_scale
            losses['tir_text_positive_similarity'] = positive_similarity
            losses['tir_text_candidate_count'] = batch_pred.new_tensor(candidate_count)

        if infmae_alignment_mode in {'rgb_tir', 'both'}:
            required_keys = ('tir_rgb_embedding', 'rgb_embedding')
            missing_keys = [key for key in required_keys if key not in aux]
            if missing_keys:
                raise KeyError(
                    'InfMAE RGB-to-TIR transfer output is missing keys: '
                    f'{missing_keys}'
                )
            tir_embedding = aux['tir_rgb_embedding']
            rgb_embedding = aux['rgb_embedding']
            if tir_embedding.shape != rgb_embedding.shape or tir_embedding.ndim != 2:
                raise ValueError(
                    'RGB/TIR target embeddings must share [B, D] shape; got '
                    f'tir={tuple(tir_embedding.shape)}, rgb={tuple(rgb_embedding.shape)}'
                )
            rgb_tir_weight = float(getattr(args, 'infmae_rgb_tir_weight', 0.05))
            if rgb_tir_weight < 0:
                raise ValueError('--infmae_rgb_tir_weight must be non-negative')
            # The RGB feature is a semantic teacher.  Detach again here even
            # though MMVGFusion already detaches it, so callers cannot
            # accidentally turn the transfer loss into a bidirectional drift.
            cosine_similarity = (tir_embedding * rgb_embedding.detach()).sum(dim=-1)
            rgb_tir_loss = (1.0 - cosine_similarity).mean()
            rgb_tir_ramp_scale = infmae_rgb_tir_ramp_scale(args, epoch)
            losses['loss_infmae_rgb_tir'] = (
                rgb_tir_loss * rgb_tir_weight * rgb_tir_ramp_scale
            )
            losses['rgb_tir_target_similarity'] = cosine_similarity.detach().mean()
            losses['infmae_rgb_tir_ramp_scale'] = batch_pred.new_tensor(
                rgb_tir_ramp_scale
            )

        losses['infmae_alignment_ramp_scale'] = batch_pred.new_tensor(ramp_scale)
        losses['infmae_alignment_adapter_scale'] = batch_pred.new_tensor(
            infmae_alignment_adapter_ramp_scale(args, epoch)
        )
        for key in (
            'semantic_bridge_gain',
            'semantic_bridge_attention_entropy',
            'semantic_bridge_attention_peak',
        ):
            if key in aux:
                losses[key] = aux[key].detach()

    return losses


def trans_vg_loss_from_clipvg(batch_pred, batch_target):
    """Compute the losses related to the bounding boxes,
       including the L1 regression loss and the GIoU loss
    """

    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    return losses


def trans_vg_attbalance_loss(
    batch_pred,
    batch_target,
    bce=None,
    mom=None,
    attn_ratio=None,
    ratio_box=None,
    epoch=0,
):
    """AttBalance loss: box regression + attention-to-box regularization."""
    batch_size = batch_pred.shape[0]
    num_boxes = batch_size
    losses = {}

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(xywh2xyxy(batch_pred), xywh2xyxy(batch_target)))
    iou = bbox_iou(xywh2xyxy(batch_pred), xywh2xyxy(batch_target))

    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    if bce is None or mom is None or attn_ratio is None:
        return losses

    bce = bce[2:]
    mom = mom[2:]
    if len(bce) == 0:
        losses['loss_bce'] = torch.zeros((), device=batch_pred.device)
        losses['loss_mom'] = torch.zeros((), device=batch_pred.device)
        return losses

    if ratio_box is None:
        ratio_box = torch.ones((batch_size,), device=batch_pred.device, dtype=batch_pred.dtype)
    ratio_box = ratio_box.to(batch_pred.device).reshape(-1).float()
    ratio_box = 0.5 + 1.0 / (1.0 + torch.exp(-(1.0 - ratio_box)))
    ratio = ((bce + mom).sigmoid() + 0.5)[-1] * ratio_box

    if spearmanr is not None:
        rho_list = []
        for layer in range(len(bce)):
            rho = spearmanr(
                attn_ratio[layer].detach().cpu().numpy(),
                iou.detach().cpu().numpy(),
            )[0]
            if np.isnan(rho):
                rho = 0.0
            rho_list.append(rho)
        rho = torch.tensor(rho_list, device=batch_pred.device, dtype=batch_pred.dtype).unsqueeze(-1)
    else:
        rho = torch.ones((len(bce), 1), device=batch_pred.device, dtype=batch_pred.dtype)
    rho = rho - rho.mean(dim=0, keepdim=True) + 1.0
    bce = rho * bce

    if epoch < 60:
        losses['loss_bbox'] = (ratio.unsqueeze(1) * loss_bbox).sum() / num_boxes
        losses['loss_giou'] = (ratio * loss_giou).sum() / num_boxes
        losses['loss_bce'] = bce.sum() / num_boxes
        losses['loss_mom'] = mom.sum() / num_boxes
    else:
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['loss_bce'] = (0 * bce).sum() / num_boxes
        losses['loss_mom'] = (0 * mom).sum() / num_boxes

    return losses


def one_ref_loss(args, batch_pred, batch_target, tgt_mask, contrastive_loss, visu_sim=None, seg_mask=None,
                 mim_pred=None, mim_labels=None, mim_vts_pred=None, mim_vts_labels=None,
                 mlm_loss=None, mlm_sts_pred=None, mlm_sts_labels=None):
    """Compute the losses related to the bounding boxes,
       including the L1 regression loss and the GIoU loss
    """

    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))

    losses = {}
    if args.use_regress_box:
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

    if args.use_contrastive_loss:
        # losses['loss_contrastive'] = (contrastive_loss / num_boxes) * 10.0
        losses['loss_contrastive'] = contrastive_loss / num_boxes
        """DO NOT multiply by 10, or the performance will drop sharply."""
        # losses['loss_contrastive'] = contrastive_loss * 100.0
        # losses['loss_contrastive'] = contrastive_loss * 10.0

    """ box mask constraints was proposed in HiVG """
    if args.use_box_mask_constraints or args.enable_dynamic_mim:
        coef_focal = 20.0
        coef_dice = 2.0
        if getattr(args, "modality", None) == "rgbt":
            patch_tokens = visu_sim.shape[-1] // 2
            patch_num = int(mpmath.sqrt(patch_tokens))
            obj_mask_single = mdetr_interpolate(
                tgt_mask.float(), (patch_num, patch_num), mode="nearest"
            )[:, 0] > 0.5
            obj_mask_single = obj_mask_single.flatten(1).float()
            obj_mask = obj_mask_single.repeat(1, 2)
        else:
            patch_num = int(mpmath.sqrt(visu_sim.shape[-1]))
            obj_mask = mdetr_interpolate(tgt_mask.float(), (patch_num, patch_num), mode="nearest")[:, 0] > 0.5
            obj_mask = obj_mask.flatten(1).float()
        visu_sim = visu_sim.flatten(1)
        losses['loss_mrm_focal'] = sigmoid_focal_loss(visu_sim, obj_mask, num_boxes) * coef_focal
        losses['loss_mrm_dice'] = dice_loss(visu_sim, obj_mask, num_boxes) * coef_dice

    if args.use_mask_loss:
        coef_focal = 20.0
        coef_dice = 2.0
        # Interpolation upwards, the shape of seg_mask is B C H W
        src_mask = mdetr_interpolate(seg_mask, size=tgt_mask.shape[-2:], mode="bilinear", align_corners=False)
        src_mask = src_mask.flatten(1)
        tgt_mask = tgt_mask.flatten(1).float()

        losses['loss_seg_focal'] = sigmoid_focal_loss(src_mask, tgt_mask, num_boxes) * coef_focal
        losses['loss_seg_dice'] = dice_loss(src_mask, tgt_mask, num_boxes) * coef_dice

    if args.enable_ref_mlm and mlm_loss is not None:
        # losses['loss_mlm'] = mlm_loss * 10.0
        losses['loss_mlm'] = mlm_loss
        if args.enable_mlm_sts and mlm_sts_pred.shape == mlm_sts_labels.shape:
            kl_loss = nn.KLDivLoss(reduction="batchmean")  # mlm_sts_pred is torch.Size([64, 62])
            losses['loss_mlm_sts'] = kl_loss(F.log_softmax(mlm_sts_pred, dim=-1), F.softmax(mlm_sts_labels, dim=-1))

    if args.enable_ref_mim and mim_pred is not None:
        loss_fn = nn.CrossEntropyLoss()
        if isinstance(mim_pred, list):
            loss_1 = loss_fn(input=mim_pred[0], target=mim_labels)
            loss_2 = loss_fn(input=mim_pred[1], target=mim_labels)
            losses['loss_mim'] = loss_1 + loss_2
        else:
            # mim pred shape:  torch.Size([5520, 8192]), mim_labels:  torch.Size([5520])
            # The 0-th dimension of min_pred and mim_labels varies due to the random number of mask positions
            losses['loss_mim'] = loss_fn(input=mim_pred, target=mim_labels)  # tensor(9.7763, device='cuda:0')
            if args.enable_mim_vts and mim_vts_pred is not None:
                mim_vts_loss = F.l1_loss(mim_vts_pred, mim_vts_labels, reduction='none')  # torch.Size([64, 576, 4])
                # Implementation version 2
                losses['loss_mim_vts'] = mim_vts_loss.sum(dim=-1).mean()  # tensor(1.7184, device='cuda:0')
                # Implementation version 2
                # losses['loss_mim_vts'] = mim_vts_loss.sum(dim=-1).sum(dim=-1).mean()  # tensor(429.6270, 'cuda:0')

    return losses
