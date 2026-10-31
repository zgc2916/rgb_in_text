"""Warm-start a short controlled continuation from the trained A5-F2 model.

Use --peak, --mrp, or --local-center to select one architecture change, or
--no-aug-scale to disable multi-scale shrinking. Prepare a model-only
checkpoint once, then launch this module with torchrun on two GPUs.
Use --frontend-only for an isolated thermal frontend loss comparison.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from train_val.mmvg_train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    variant_group = parser.add_mutually_exclusive_group()
    variant_group.add_argument("--peak", action="store_true")
    variant_group.add_argument("--mrp", action="store_true")
    variant_group.add_argument("--teacher-preserve", action="store_true")
    variant_group.add_argument("--teacher-gated", action="store_true")
    variant_group.add_argument("--local-center", action="store_true")
    variant_group.add_argument("--bbox-head-only", action="store_true")
    variant_group.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--rgb-transfer-weighting", choices=('uniform', 'semantic'), default='uniform')
    parser.add_argument("--report-val-groups", action="store_true")
    parser.add_argument("--target-safe-crop", action="store_true")
    parser.add_argument("--target-scale-floor-pixels", type=float, default=0.0)
    parser.add_argument("--bbox-head-last-layer-only", action="store_true")
    parser.add_argument("--local-center-max-shift-pixels", type=float, default=4.0)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--box-l1-weight", type=float, default=2.0)
    parser.add_argument("--box-giou-weight", type=float, default=2.0)
    parser.add_argument("--box-loss-type", choices=("giou", "eiou"), default="giou")
    parser.add_argument("--box-eiou-weight", type=float, default=2.0)
    parser.add_argument("--teacher-weight", type=float, default=2.0)
    parser.add_argument("--teacher-iou-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-aug-scale", action="store_true",
        help="Disable multi-scale shrinking; omission preserves the source configuration",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.prepare:
        source = torch.load(cli.source, map_location="cpu", weights_only=False)
        if source["args"].FusionMethod != "InfMAEA5BridgeFTF2":
            raise ValueError("Source checkpoint must be A5-F2 InfMAE")
        cli.warmstart.parent.mkdir(parents=True, exist_ok=True)
        state = source["model"]
        if cli.peak:
            state = dict(state)
            key = "infmae_direct_adapter.f2_peak_gate"
            if key in state:
                raise ValueError("Source checkpoint already has the F2 peak gate")
            dim = state["infmae_direct_adapter.f2_norm.weight"].numel()
            state[key] = torch.zeros(1, dim, 1, 1)
        # MRP is a newly inserted zero-init module. Strict model loading
        # accepts its missing keys through MMVGFusion.load_state_dict.
        torch.save(
            {
                "model": state,
                "source_epoch": source["epoch"],
                "source_val_accu": source["val_accu"],
            },
            cli.warmstart,
        )
        print(f"saved exact model-only warm-start: {cli.warmstart}")
        return

    if cli.output_dir is None:
        raise ValueError("--output_dir is required for training")
    if cli.epochs <= 0 or cli.lr <= 0:
        raise ValueError("--epochs and --lr must be positive")
    if cli.local_center_max_shift_pixels <= 0:
        raise ValueError("--local-center-max-shift-pixels must be positive")
    if cli.bbox_head_last_layer_only and not cli.bbox_head_only:
        raise ValueError("--bbox-head-last-layer-only requires --bbox-head-only")
    if cli.rgb_transfer_weighting != 'uniform' and not cli.frontend_only:
        raise ValueError('Semantic RGB teacher screening requires --frontend-only')
    if cli.frontend_only and cli.epochs > 60:
        raise ValueError('Frontend screens must finish before the HiLoRA stage changes')
    if cli.target_scale_floor_pixels < 0:
        raise ValueError('--target-scale-floor-pixels must be non-negative')
    if (cli.target_safe_crop or cli.target_scale_floor_pixels > 0) and not cli.frontend_only:
        raise ValueError('Target-aware augmentation screens require --frontend-only')
    if cli.target_scale_floor_pixels > 0 and not cli.target_safe_crop:
        raise ValueError('Target-aware resizing requires --target-safe-crop')
    if cli.box_l1_weight < 0 or cli.box_giou_weight < 0 or cli.box_eiou_weight < 0 or cli.teacher_weight < 0:
        raise ValueError("box and teacher loss weights must be non-negative")
    if not 0.0 <= cli.teacher_iou_threshold <= 1.0:
        raise ValueError("teacher IoU threshold must be in [0, 1]")
    source = torch.load(cli.source, map_location="cpu", weights_only=False)
    args = source["args"]
    if args.FusionMethod != "InfMAEA5BridgeFTF2" or args.dataset != "rgbtvg_flir":
        raise ValueError("Source checkpoint must be A5-F2 on RefFLIR")
    source_epoch = int(source["epoch"])
    source_aug_scale = bool(args.aug_scale)
    del source

    if cli.peak:
        args.FusionMethod = "InfMAEA5BridgeFTF2Peak"
    elif cli.mrp:
        args.FusionMethod = "InfMAEA5BridgeFTF2MRP"
    args.resume = str(cli.warmstart.resolve())
    args.retrain = ""
    args.output_dir = str(cli.output_dir.resolve())
    args.start_epoch = 0
    args.epochs = cli.epochs
    args.lr = cli.lr
    args.box_l1_weight = cli.box_l1_weight
    args.box_giou_weight = cli.box_giou_weight
    args.box_loss_type = cli.box_loss_type
    args.box_eiou_weight = cli.box_eiou_weight
    args.infmae_local_center_only = cli.local_center
    args.infmae_bbox_head_only = cli.bbox_head_only
    args.infmae_frontend_only = cli.frontend_only
    args.infmae_rgb_transfer_weighting = cli.rgb_transfer_weighting
    args.report_val_groups = cli.report_val_groups or cli.frontend_only
    args.target_safe_crop = cli.target_safe_crop
    args.target_scale_floor_pixels = cli.target_scale_floor_pixels
    args.infmae_bbox_head_last_layer_only = cli.bbox_head_last_layer_only
    args.infmae_local_center_max_shift_pixels = cli.local_center_max_shift_pixels
    args.teacher_preserve_enabled = cli.teacher_preserve or cli.teacher_gated
    args.teacher_preserve_mode = "exclusive" if cli.teacher_gated else "additive"
    args.teacher_preserve_weight = cli.teacher_weight
    args.teacher_preserve_iou_threshold = cli.teacher_iou_threshold
    if cli.seed is not None:
        args.seed = cli.seed
    if cli.no_aug_scale:
        args.aug_scale = False
    args.infmae_alignment_lr = cli.lr
    args.infmae_alignment_start_epoch = 0
    args.infmae_alignment_ramp_epochs = 1
    args.infmae_alignment_adapter_start_epoch = 0
    args.infmae_alignment_adapter_ramp_epochs = 1
    args.infmae_rgb_tir_start_epoch = 0
    args.infmae_rgb_tir_ramp_epochs = 1

    if int(os.environ.get("RANK", "0")) == 0:
        cli.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": str(cli.source.resolve()),
            "source_epoch": source_epoch,
            "fusion_method": args.FusionMethod,
            "variant": ("frontend_only" if cli.frontend_only else ("bbox_head_only" if cli.bbox_head_only else ("local_center" if cli.local_center else ("teacher_gated" if cli.teacher_gated else ("teacher_preserve" if cli.teacher_preserve else ("mrp" if cli.mrp else ("peak" if cli.peak else "base"))))))),
            "bbox_head_only": cli.bbox_head_only,
            "frontend_only": cli.frontend_only,
            "rgb_transfer_weighting": cli.rgb_transfer_weighting,
            "report_val_groups": args.report_val_groups,
            "bbox_head_last_layer_only": cli.bbox_head_last_layer_only,
            "frozen_trunk_eval": cli.local_center or cli.bbox_head_only or cli.frontend_only,
            "local_center_only": cli.local_center,
            "local_center_max_shift_pixels": cli.local_center_max_shift_pixels,
            "local_center_gate_full_below_pixels": 16,
            "local_center_gate_zero_above_pixels": 32,
            "warmstart": str(cli.warmstart.resolve()),
            "epochs": cli.epochs,
            "lr": cli.lr,
            "box_l1_weight": cli.box_l1_weight,
            "box_giou_weight": cli.box_giou_weight,
            "box_loss_type": cli.box_loss_type,
            "box_eiou_weight": cli.box_eiou_weight,
            "teacher_preserve_enabled": cli.teacher_preserve or cli.teacher_gated,
            "teacher_preserve_mode": "exclusive" if cli.teacher_gated else "additive",
            "teacher_weight": cli.teacher_weight,
            "teacher_iou_threshold": cli.teacher_iou_threshold,
            "batch_size_per_gpu": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "gpu_count": int(os.environ.get("WORLD_SIZE", "1")),
            "seed": args.seed,
            "alignment_mode": args.infmae_alignment_mode,
            "tir_text_weight": args.infmae_tir_text_weight,
            "rgb_tir_weight": args.infmae_rgb_tir_weight,
            "imsize": args.imsize,
            "lr_scheduler": args.lr_scheduler,
            "effective_global_batch": (
                args.batch_size * args.grad_accum_steps
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
            "augmentation": {
                "source_aug_scale": source_aug_scale,
                "aug_scale": bool(args.aug_scale),
                "aug_crop": bool(args.aug_crop),
                "aug_translate": bool(args.aug_translate),
                "aug_blur": bool(args.aug_blur),
                "target_safe_crop": args.target_safe_crop,
                "target_scale_floor_pixels": args.target_scale_floor_pixels,
                "resize_long_side_choices": (
                    [args.imsize - int(args.imsize / 8) * i for i in range(7)]
                    if args.aug_scale else [args.imsize]
                ),
            },
        }
        (cli.output_dir / "continuation.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Continuation augmentation: {metadata['augmentation']}", flush=True)
    train_main(args)


if __name__ == "__main__":
    main()
