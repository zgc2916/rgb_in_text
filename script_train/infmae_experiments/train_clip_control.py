"""Train a CLIP-TIR/IAFv3 control with the A5-F2 shared training recipe."""

import argparse
import copy
import json
import math
import os
from pathlib import Path

import torch

from train_val.mmvg_train import main as train_main


def control_args(source_path, output_dir, epochs=120, seed=13,
                 batch_size=None, grad_accum_steps=None,
                 target_safe_crop=None, target_scale_floor_pixels=None):
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    args = copy.deepcopy(source['args'])
    del source
    if args.dataset != 'rgbtvg_flir' or args.FusionMethod != 'InfMAEA5BridgeFTF2':
        raise ValueError('Expected the fixed RefFLIR A5-F2 source recipe')
    if not Path(args.retrain).is_file():
        raise FileNotFoundError(args.retrain)
    args.FusionMethod = 'IAFv3'
    args.infmae_alignment_mode = 'none'
    args.infmae_frontend_only = False
    args.infmae_local_center_only = False
    args.infmae_bbox_head_only = False
    args.infmae_bbox_head_last_layer_only = False
    args.teacher_preserve_enabled = False
    args.enable_gqr = False
    args.resume = ''
    args.start_epoch = 0
    args.epochs = epochs
    args.lr = 1e-4
    args.seed = seed
    if batch_size is not None:
        args.batch_size = batch_size
    if grad_accum_steps is not None:
        args.grad_accum_steps = grad_accum_steps
    if args.batch_size < 2 or args.grad_accum_steps < 1:
        raise ValueError('CLIP control requires batch size >= 2 and accumulation >= 1')
    if target_safe_crop is not None:
        args.target_safe_crop = target_safe_crop
    if target_scale_floor_pixels is not None:
        if not math.isfinite(target_scale_floor_pixels) or target_scale_floor_pixels < 0:
            raise ValueError('Target scale floor must be finite and non-negative')
        args.target_scale_floor_pixels = target_scale_floor_pixels
    args.report_acc07 = True
    args.report_val_groups = True
    args.output_dir = str(Path(output_dir).resolve())
    # A5-F2 trains all RGB LoRA layers from epoch zero. Match that rule and
    # keep both CLIP branch adapters eligible after every PEFT switch.
    args.match_infmae_lora_schedule = True
    args.hi_lora_stage = 0
    return args


def load_common_initialization(model, path):
    """Use the training loader's legacy key mapping for the GPU preflight."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    renamed = {}
    projections = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj')
    for key, value in checkpoint['model'].items():
        if 'default' in key:
            renamed[key.replace('default', 'lora_rgb')] = value
            renamed[key.replace('default', 'lora_ir')] = value.clone()
        else:
            renamed[key] = value
            if '.base_layer.' not in key and '.lora_' not in key:
                for projection in projections:
                    for field in ('weight', 'bias'):
                        suffix = f'{projection}.{field}'
                        if key.endswith(suffix):
                            renamed[key[:-len(suffix)] + f'{projection}.base_layer.{field}'] = value
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    if any('.base_layer.' in key for key in missing):
        raise RuntimeError('Released initialization did not load all PEFT base layers')
    return {'missing_keys': len(missing), 'legacy_only_keys': len(unexpected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--batch-size', type=int,
                        help='Per-GPU batch and local CLIP candidate count; defaults to the source recipe')
    parser.add_argument('--grad-accum-steps', type=int,
                        help='Microbatches per optimizer update; does not enlarge CLIP candidates')
    parser.add_argument('--target-safe-crop', action='store_true', default=None,
                        help='Use training crops that retain the complete referring target')
    parser.add_argument('--target-scale-floor-pixels', type=float,
                        help='Minimum target short side when an available training scale permits it')
    cli = parser.parse_args()
    if not 0 < cli.epochs <= 120:
        parser.error('Require 1..120 epochs')
    if cli.batch_size is not None and cli.batch_size < 2:
        parser.error('--batch-size must be at least 2')
    if cli.grad_accum_steps is not None and cli.grad_accum_steps < 1:
        parser.error('--grad-accum-steps must be positive')
    if cli.target_scale_floor_pixels is not None and (
            not math.isfinite(cli.target_scale_floor_pixels) or cli.target_scale_floor_pixels < 0):
        parser.error('--target-scale-floor-pixels must be finite and non-negative')
    args = control_args(cli.source, cli.output_dir, cli.epochs, cli.seed,
                        cli.batch_size, cli.grad_accum_steps,
                        cli.target_safe_crop, cli.target_scale_floor_pixels)
    if int(os.environ.get('RANK', '0')) == 0:
        cli.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {'reference_recipe': str(cli.source.resolve()), 'initialization': args.retrain,
                    'initialization_kind': 'released common MMVG mixup weights, not an InfMAE/GQR trained checkpoint',
                    'fusion_method': args.FusionMethod, 'epochs': args.epochs, 'seed': args.seed,
                    'lr': args.lr, 'batch_size_per_gpu': args.batch_size,
                    'grad_accum_steps': args.grad_accum_steps, 'match_infmae_lora_schedule': True,
                    'world_size': int(os.environ.get('WORLD_SIZE', '1')),
                    'nominal_effective_batch': args.batch_size * args.grad_accum_steps * int(os.environ.get('WORLD_SIZE', '1')),
                    'base_clip_candidates_per_gpu': args.batch_size,
                    'base_clip_negatives_per_anchor': args.batch_size - 1,
                    'base_clip_candidates_scope': 'local_gpu_without_gather_or_accumulation_merge',
                    'target_safe_crop': bool(getattr(args, 'target_safe_crop', False)),
                    'target_scale_floor_pixels': float(getattr(args, 'target_scale_floor_pixels', 0.0)),
                    'test_used_for_training_or_selection': False,
                    'note': 'Matched frontend control; not a claim of exact reproduction of every paper hyperparameter.'}
        (cli.output_dir / 'control_configuration.json').write_text(json.dumps(metadata, indent=2) + '\n')
    train_main(args)


if __name__ == '__main__':
    main()
