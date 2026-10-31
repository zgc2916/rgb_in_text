"""Audit real train geometry and run two target-preserving augmentations."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .run_scale_ablation import timestamp, write_json
from .run_semantic_transfer_screen import collect_results, seed_everything, thermal_preflight


def source_args(cli):
    import torch
    checkpoint = torch.load(cli.source, map_location='cpu', weights_only=False)
    args = copy.deepcopy(checkpoint['args'])
    if args.dataset != 'rgbtvg_flir' or args.FusionMethod != 'InfMAEA5BridgeFTF2':
        raise ValueError('Require the fixed RefFLIR A5-F2 source recipe')
    if not args.aug_crop or not args.aug_scale:
        raise ValueError('Expected the original crop and scale augmentation control')
    return args


def audit(cli):
    import numpy as np
    import torch
    from datasets import build_dataset

    args = source_args(cli)
    torch.set_num_threads(4)
    summaries = {}
    for name, safe_crop, floor in (('original', False, 0), ('safe_crop', True, 0),
                                  ('safe_crop_floor8', True, cli.scale_floor)):
        args.target_safe_crop, args.target_scale_floor_pixels = safe_crop, floor
        dataset = build_dataset('train', args)
        if any(min(int(row[2][2]), int(row[2][3])) <= 0 for row in dataset.images):
            raise AssertionError('Original train annotations contain an empty integer target box')
        indices = np.random.default_rng(cli.seed).choice(len(dataset), size=512, replace=False).tolist()
        geometry = dataset.transform.transforms[0]
        base_crop = geometry.transforms2.transforms[1]
        counts = dict(samples=512, crop_calls=0, accepted_crops=0, empty_after_crop=0,
                      truncated_crops=0, empty_after_geometry=0, invalid_source_boxes=0,
                      below4_after_geometry=0, below8_after_geometry=0, below16_after_geometry=0)
        examples = []

        class TracedCrop:
            def __call__(self, item):
                counts['crop_calls'] += 1
                before, image = item['box'].clone(), item['img']
                result = base_crop(item)
                if result['img'] is not image:
                    counts['accepted_crops'] += 1
                    dimensions = result['box'][2:] - result['box'][:2]
                    empty = bool((dimensions <= 0).any())
                    truncated = bool((dimensions < before[2:] - before[:2] - 1e-5).any())
                    counts['empty_after_crop'] += int(empty)
                    counts['truncated_crops'] += int(truncated)
                    if empty and len(examples) < 8:
                        examples.append({'index': index, 'image': dataset.images[index][0],
                                         'before': before.tolist(), 'after': result['box'].tolist()})
                return result

        geometry.transforms2.transforms[1] = TracedCrop()
        for position, index in enumerate(indices):
            seed_everything(cli.seed + index)
            _, image, phrase, box, _, _ = dataset.pull_item(index)
            counts['invalid_source_boxes'] += int(bool(((box[2:] - box[:2]) <= 0).any()))
            result = geometry({'img': image, 'box': box, 'text': phrase.lower()})
            shortest = float((result['box'][2:] - result['box'][:2]).min())
            counts['empty_after_geometry'] += int(shortest <= 0)
            for threshold in (4, 8, 16):
                counts[f'below{threshold}_after_geometry'] += int(shortest < threshold)
            if (position + 1) % 128 == 0:
                print(name, 'audited', position + 1, flush=True)
        if safe_crop and (counts['empty_after_geometry'] or counts['truncated_crops']):
            raise AssertionError('Target-preserving geometry lost a source target')
        summaries[name] = {'counts': counts, 'empty_examples': examples,
                           'safe_crop': safe_crop, 'scale_floor_pixels': floor}
    previous = cli.output_root / 'original_augmentation_diagnosis.json'
    if previous.is_file():
        expected = json.loads(previous.read_text())['counts']
        if any(summaries['original']['counts'][key] != value for key, value in expected.items()):
            raise AssertionError('Default augmentation no longer reproduces the original audit')
    summary = {'checked_at': timestamp(), 'split': 'train', 'seed': cli.seed,
               'samples_per_variant': 512, 'actual_rgb_tir_pairs': True,
               'sampling': 'same random subset, seed reset per sample and variant',
               'variants': summaries, 'test_forward_count': 0}
    write_json(cli.output_root / 'augmentation_audit.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def preflight(cli):
    import torch
    import torch.distributed as dist
    from datasets import build_dataset
    from utils import misc

    torch.set_num_threads(4)
    args = source_args(cli)
    misc.init_distributed_mode(args)
    original = build_dataset('val', args)
    args.target_safe_crop, args.target_scale_floor_pixels = True, cli.scale_floor
    protected = build_dataset('val', args)
    indices = range(misc.get_rank() * 4, misc.get_rank() * 4 + 4)
    first = misc.collate_fn([original[index] for index in indices])
    second = misc.collate_fn([protected[index] for index in indices])
    if first[1] != second[1] or any(not torch.equal(a, b) for a, b in (
            (first[0].tensors, second[0].tensors), (first[0].mask, second[0].mask),
            (first[2], second[2]), (first[3], second[3]))):
        raise AssertionError('Training-only augmentation changed validation samples')
    cli.rgb_transfer_weighting = 'uniform'
    scope = thermal_preflight(cli, args)
    if misc.is_main_process():
        write_json(cli.output_root / 'preflight.json',
                   {'checked_at': timestamp(), 'world_size': misc.get_world_size(),
                    'thermal': scope, 'validation_transform_exact_identity': True,
                    'validation_samples_checked': 8, 'target_safe_crop': True,
                    'scale_floor_pixels': cli.scale_floor, 'test_forward_count': 0})
        print('Training geometry, frozen scope, gradients, and validation identity passed', flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--warmstart', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--scale-floor', type=float, default=8.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-only', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    cli = parser.parse_args()
    gpu_ids = cli.gpus.split(',')
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(g.isdigit() for g in gpu_ids):
        parser.error('Require two distinct GPU IDs')
    if not 0 < cli.epochs <= 60 or cli.lr <= 0 or cli.scale_floor <= 0:
        parser.error('Invalid epochs, lr, or pixel floor')
    cli.source = cli.source.resolve(strict=True)
    cli.warmstart = cli.warmstart.resolve(strict=True)
    cli.control_dir = cli.control_dir.resolve(strict=True)
    cli.output_root = cli.output_root.resolve()
    if cli.audit_only:
        audit(cli)
        return
    if cli.preflight_only:
        preflight(cli)
        return
    root = cli.output_root
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'status.json'
    if status_path.exists() or any((root / name).exists() for name in ('safe_crop', 'safe_crop_floor8')):
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {root}')
    control_configuration = json.loads((cli.control_dir / 'continuation.json').read_text())
    if (control_configuration['epochs'], control_configuration['lr'], control_configuration['seed'],
            control_configuration['frontend_only'], control_configuration['rgb_transfer_weighting']) != (
            cli.epochs, cli.lr, cli.seed, True, 'uniform'):
        raise ValueError('Reference continuation does not match this experiment recipe')
    control = collect_results(cli.control_dir)
    if len(control['epochs']) != cli.epochs:
        raise ValueError('The reference control has not completed all epochs')
    write_json(root / 'control_reference.json',
               {'directory': str(cli.control_dir), 'configuration': control_configuration, 'results': control})
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=cli.gpus, OMP_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       MPLCONFIGDIR=str(root / 'matplotlib'))
    repository = Path(__file__).resolve().parents[2]
    status = {'pid': os.getpid(), 'started_at': timestamp(), 'gpus': gpu_ids,
              'source': str(cli.source), 'warmstart': str(cli.warmstart),
              'reference_control': str(cli.control_dir), 'epochs_per_run': cli.epochs,
              'lr': cli.lr, 'seed': cli.seed, 'scale_floor_pixels': cli.scale_floor,
              'phase': 'audit', 'runs': {}, 'test_forward_count': 0,
              'test_used_for_training_or_selection': False}
    write_json(status_path, status)
    common = ['--source', str(cli.source), '--warmstart', str(cli.warmstart),
              '--epochs', str(cli.epochs), '--lr', str(cli.lr), '--seed', str(cli.seed)]
    wrapper = ['script_train.infmae_experiments.run_target_crop_screen', *common,
               '--output-root', str(root), '--control-dir', str(cli.control_dir),
               '--gpus', cli.gpus, '--scale-floor', str(cli.scale_floor)]
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
    commands = {'audit': [sys.executable, '-m', *wrapper, '--audit-only'],
                'preflight': [*torchrun, *wrapper, '--preflight-only']}
    for name, floor in (('safe_crop', 0), ('safe_crop_floor8', cli.scale_floor)):
        commands[name] = [*torchrun, 'script_train.infmae_experiments.continue_a5f2', *common,
                          '--output_dir', str(root / name), '--frontend-only',
                          '--rgb-transfer-weighting', 'uniform', '--report-val-groups',
                          '--target-safe-crop', '--target-scale-floor-pixels', str(floor)]
    write_json(root / 'commands.json', commands)
    summary = {'reference_control': str(cli.control_dir), 'test_forward_count': 0,
               'runs': {'original': control}}
    for name, command in commands.items():
        is_training = name in ('safe_crop', 'safe_crop_floor8')
        directory = root / name if is_training else root
        directory.mkdir(exist_ok=True)
        log_path = directory / 'console.log' if is_training else root / f'{name}.log'
        with log_path.open('w') as output:
            process = subprocess.Popen(command, cwd=repository, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT)
            status.update(phase=name, active_pid=process.pid, phase_started_at=timestamp())
            status['runs'][name] = {'pid': process.pid, 'phase': 'running', 'started_at': timestamp()}
            write_json(status_path, status)
            code = process.wait()
        status['runs'][name].update(returncode=code, finished_at=timestamp(),
                                    phase='complete' if code == 0 else 'failed')
        if is_training and (directory / 'initial_validation.json').is_file():
            result = collect_results(directory)
            initial = result['initial_validation']
            identity = all(abs(initial[key] - control['initial_validation'][key]) <= 1e-7
                           for key in ('accu', 'accu_07', 'miou', 'accu_weak', 'accu_small'))
            result['initial_validation_matches_control'] = identity
            summary['runs'][name] = result
            write_json(directory / 'results.json', result)
            write_json(root / 'comparison.json', summary)
            if not identity:
                code = code or 1
                status['runs'][name]['initial_validation_mismatch'] = True
        write_json(status_path, status)
        if code:
            status.update(phase=f'failed_{name}', returncode=code, finished_at=timestamp())
            write_json(status_path, status)
            raise SystemExit(code)
    status.update(phase='complete', returncode=0, finished_at=timestamp())
    write_json(status_path, status)


if __name__ == '__main__':
    main()
