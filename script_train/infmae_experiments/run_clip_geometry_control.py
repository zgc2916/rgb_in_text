"""Compare target-preserving geometry with a completed CLIP batch12 control."""

import argparse
import copy
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .run_scale_ablation import timestamp, write_json


RUNTIME_FIELDS = {'output_dir', 'rank', 'gpu', 'world_size', 'distributed', 'dist_backend'}
GEOMETRY_DEFAULTS = {'target_safe_crop': False, 'target_scale_floor_pixels': 0.0}
QUALITY_FIELDS = ('accu', 'accu_07', 'miou', 'accu_07_weak', 'accu_07_small')


def variants(scale_floor):
    return (('original', False, 0.0), ('safe_crop', True, 0.0),
            ('safe_crop_floor8', True, scale_floor))


def recipe_differences(reference, candidate):
    reference, candidate = dict(reference), dict(candidate)
    for key, value in GEOMETRY_DEFAULTS.items():
        reference.setdefault(key, value)
        candidate.setdefault(key, value)
    return {key: {'original': reference.get(key), 'candidate': candidate.get(key)}
            for key in sorted(reference.keys() | candidate.keys())
            if key not in RUNTIME_FIELDS and reference.get(key) != candidate.get(key)}


def validation_rows(directory):
    path = directory / 'log.txt'
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('{'):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            # A live log can end in a partial write. Completed logs are
            # checked for the complete, ordered epoch range below.
            continue
        if 'validation_accu' in raw:
            stats = {key.removeprefix('validation_'): value for key, value in raw.items()
                     if key.startswith('validation_')}
            if any(key not in stats or not math.isfinite(float(stats[key]))
                   for key in QUALITY_FIELDS):
                raise ValueError('Missing or nonfinite overall/group validation metrics')
            rows.append({'epoch': raw['epoch'], **stats,
                         'correct_05': round(stats['accu'] * 608),
                         'correct_07': round(stats['accu_07'] * 608)})
    return rows


def saved_best(rows):
    # Preserve the existing training saver's first strict Acc@0.5 maximum.
    return max(rows, key=lambda row: row['accu']) if rows else None


def assess_quality(reference, candidate):
    if candidate is None:
        return None
    deltas = {key: (candidate[key] - reference[key]) * 100 for key in QUALITY_FIELDS}
    retained = all(delta >= -1e-9 for delta in deltas.values())
    improved = any(deltas[key] > 1e-9 for key in ('accu', 'accu_07', 'miou'))
    return {'delta_percentage_points': deltas,
            'overall_and_difficult_group_quality_retained': retained,
            'eligible_for_seed_replication': retained and improved}


def comparison(cli):
    baseline = validation_rows(cli.control_dir)
    reference = saved_best(baseline)
    runs = {'original': {'output_dir': str(cli.control_dir), 'reused_completed_run': True,
                         'epochs_completed': len(baseline), 'saved_best': reference}}
    for name, _, _ in variants(cli.scale_floor)[1:]:
        rows = validation_rows(cli.output_root / name)
        best = saved_best(rows)
        runs[name] = {'output_dir': str(cli.output_root / name),
                      'epochs_completed': len(rows), 'saved_best': best,
                      'quality_against_original': assess_quality(reference, best),
                      'epochs': rows}
    return {'updated_at': timestamp(), 'selection': 'first_strict_acc05_maximum',
            'quality_assessment': 'Assess the same saved best epoch; do not combine metric peaks or select using test.',
            'quality_fields': list(QUALITY_FIELDS), 'runs': runs,
            'test_used_for_training_or_selection': False, 'test_forward_count': 0}


def variant_args(cli, name, safe_crop, floor):
    from .train_clip_control import control_args
    return control_args(cli.source, cli.output_root / name, cli.epochs, cli.seed,
                        12, 4, safe_crop, floor)


def validate_reference(cli, args):
    configuration = json.loads((cli.control_dir / 'control_configuration.json').read_text())
    expected = {'epochs': cli.epochs, 'seed': cli.seed, 'lr': 1e-4,
                'batch_size_per_gpu': 12, 'grad_accum_steps': 4, 'world_size': 2,
                'nominal_effective_batch': 96, 'fusion_method': 'IAFv3',
                'match_infmae_lora_schedule': True, 'initialization': args.retrain,
                'reference_recipe': str(cli.source)}
    if any(configuration.get(key) != value for key, value in expected.items()):
        raise ValueError('The completed CLIP control has a different training recipe')
    if configuration.get('target_safe_crop', False) or configuration.get('target_scale_floor_pixels', 0.0):
        raise ValueError('The original control must use the original training geometry')
    recorded = json.loads((cli.control_dir.parent / 'recipe.json').read_text())['args']
    differences = recipe_differences(recorded, vars(args))
    if differences:
        raise ValueError(f'Original control recipe has changed: {differences}')
    rows = validation_rows(cli.control_dir)
    if [row['epoch'] for row in rows] != list(range(cli.epochs)):
        raise ValueError('Require a complete original control with the same epoch budget')
    if not (cli.control_dir / 'best_checkpoint.pth').is_file():
        raise FileNotFoundError('The original saved best checkpoint is missing')
    if (getattr(args, 'vl_backbone', 'clip'), args.FusionMethod,
            getattr(args, 'contrastive_loss', 'clip'), args.infmae_alignment_mode) != (
            'clip', 'IAFv3', 'clip', 'none'):
        raise ValueError('Require CLIP RGB/TIR and IAFv3 without InfMAE auxiliary training')
    if args.modality != 'rgbt' or not args.aug_crop or not args.aug_scale:
        raise ValueError('Require the original RGB/TIR crop and scale recipe')
    return {'directory': str(cli.control_dir), 'configuration': configuration,
            'saved_best': saved_best(rows), 'epochs_completed': len(rows),
            'recipe_differences': differences}


def audit(cli):
    import numpy as np
    from datasets import build_dataset
    from .run_semantic_transfer_screen import seed_everything

    args = variant_args(cli, 'original', False, 0.0)
    reports = {}
    for name, safe_crop, floor in variants(cli.scale_floor):
        args.target_safe_crop, args.target_scale_floor_pixels = safe_crop, floor
        dataset = build_dataset('train', args)
        if len(dataset) != 7000 or any(min(int(row[2][2]), int(row[2][3])) <= 0
                                      for row in dataset.images):
            raise AssertionError('Unexpected train split or invalid original boxes')
        indices = np.random.default_rng(cli.seed).choice(len(dataset), size=512, replace=False)
        geometry = dataset.transform.transforms[0]
        base_crop = geometry.transforms2.transforms[1]
        counts = dict(samples=512, crop_calls=0, accepted_crops=0,
                      empty_after_crop=0, truncated_crops=0, empty_after_geometry=0,
                      below4_after_geometry=0, below8_after_geometry=0,
                      below16_after_geometry=0, outside_image_after_geometry=0)

        class TracedCrop:
            def __call__(self, item):
                counts['crop_calls'] += 1
                before, image = item['box'].clone(), item['img']
                result = base_crop(item)
                if result['img'] is not image:
                    counts['accepted_crops'] += 1
                    dimensions = result['box'][2:] - result['box'][:2]
                    counts['empty_after_crop'] += int(bool((dimensions <= 0).any()))
                    counts['truncated_crops'] += int(bool(
                        (dimensions < before[2:] - before[:2] - 1e-5).any()))
                return result

        geometry.transforms2.transforms[1] = TracedCrop()
        for index in indices.tolist():
            seed_everything(cli.seed + index)
            _, image, phrase, box, _, _ = dataset.pull_item(index)
            result = geometry({'img': image, 'box': box, 'text': phrase.lower()})
            transformed = result['box']
            shortest = float((transformed[2:] - transformed[:2]).min())
            counts['empty_after_geometry'] += int(shortest <= 0)
            for threshold in (4, 8, 16):
                counts[f'below{threshold}_after_geometry'] += int(shortest < threshold)
            counts['outside_image_after_geometry'] += int(bool(
                (transformed[:2] < -1e-4).any()) or transformed[2] > result['img'].width + 1e-4
                or transformed[3] > result['img'].height + 1e-4)
        if safe_crop and any(counts[key] for key in (
                'empty_after_geometry', 'truncated_crops', 'outside_image_after_geometry')):
            raise AssertionError('Target-preserving augmentation lost a source target')
        reports[name] = {'target_safe_crop': safe_crop, 'target_scale_floor_pixels': floor,
                         'counts': counts}
        print(f'{timestamp()} geometry audit {name}: {counts}', flush=True)
    write_json(cli.output_root / 'augmentation_audit.json', {
        'checked_at': timestamp(), 'backbone': 'clip', 'split': 'train', 'seed': cli.seed,
        'sampling': 'Same 512 train expressions; reset random seed per expression and variant',
        'variants': reports, 'test_forward_count': 0})


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preflight(cli):
    import torch
    import torch.distributed as dist
    from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
    from datasets import build_dataset
    from engine import train_one_epoch
    from models import build_model
    from utils import misc
    from .run_semantic_transfer_screen import seed_everything
    from .train_clip_control import load_common_initialization

    torch.set_num_threads(4)
    base = variant_args(cli, 'original', False, 0.0)
    misc.init_distributed_mode(base)
    rank, world_size = misc.get_rank(), misc.get_world_size()
    if world_size != 2:
        raise ValueError('Require two ranks for the matched CLIP geometry experiment')
    original_val = build_dataset('val', base)
    indices = range(rank * 12, rank * 12 + 12)
    reference_batch = misc.collate_fn([original_val[index] for index in indices])
    local_reports = []
    initial_digest, initial_predictions = None, None
    for name, safe_crop, floor in variants(cli.scale_floor):
        args = copy.deepcopy(base)
        args.target_safe_crop, args.target_scale_floor_pixels = safe_crop, floor
        dataset_val = build_dataset('val', args)
        current_batch = misc.collate_fn([dataset_val[index] for index in indices])
        if current_batch[1] != reference_batch[1] or any(not torch.equal(a, b) for a, b in (
                (current_batch[0].tensors, reference_batch[0].tensors),
                (current_batch[0].mask, reference_batch[0].mask),
                (current_batch[2], reference_batch[2]), (current_batch[3], reference_batch[3]))):
            raise AssertionError('Training geometry changed validation inputs or boxes')
        seed_everything(cli.seed + rank)
        model = build_model(args).to(args.device)
        loading = load_common_initialization(model, args.retrain)
        digest = state_digest(model)
        if initial_digest is None:
            initial_digest = digest
        elif digest != initial_digest:
            raise AssertionError('Geometry variants have different initial model weights')
        model.eval()
        with torch.no_grad():
            images = misc.NestedTensor(current_batch[0].tensors.clone(), current_batch[0].mask)
            predictions = model(images.to(args.device), current_batch[1])[0].detach().cpu()
        if initial_predictions is None:
            initial_predictions = predictions
        elif not torch.equal(predictions, initial_predictions):
            raise AssertionError('Training geometry changed initial validation predictions')

        shapes, gradient_checks = [], []

        def record_candidates(module, inputs, output):
            if tuple(output[1].shape) != (12, 12):
                raise AssertionError('Base CLIP candidate matrix must be 12 by 12')
            shapes.append(list(output[1].shape))

        handle = model.register_forward_hook(record_candidates)
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        parameters = [p for key, p in model.named_parameters()
                      if p.requires_grad and 'visumodel' not in key and 'textmodel' not in key]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

        def check_gradients(optimizer, step_args, step_kwargs):
            branch_counts = {'lora_rgb': 0, 'lora_ir': 0}
            for key, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f'Nonfinite gradient: {key}')
                if key.startswith('clip.') and '.base_layer.' in key:
                    raise AssertionError('A frozen CLIP base layer received a gradient')
                for branch in branch_counts:
                    if 'vision_model' in key and f'.{branch}.' in key and parameter.grad.abs().sum() > 0:
                        branch_counts[branch] += 1
            if not all(branch_counts.values()):
                raise AssertionError('Both visual LoRA branches must receive nonzero gradients')
            gradient_checks.append(branch_counts)

        gradient_handle = optimizer.register_step_pre_hook(check_gradients)
        dataset = build_dataset('train', args)
        sampler = DistributedSampler(dataset, shuffle=True)
        sampler.set_epoch(0)
        loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler, 12, drop_last=True),
                            collate_fn=misc.collate_fn, num_workers=0)
        batches = list(itertools.islice(loader, 4))
        if len(batches) != 4:
            raise AssertionError('Need four real batch12 microbatches')
        if safe_crop and any(bool((batch[2][:, 2:] <= 0).any()) for batch in batches):
            raise AssertionError('Protected training microbatches contain empty targets')
        torch.cuda.reset_peak_memory_stats()
        stats = train_one_epoch(args, wrapped, batches, optimizer, torch.device(args.device),
                                0, args.clip_max_norm)
        if len(shapes) != 4 or len(gradient_checks) != 1 or not optimizer.state:
            raise AssertionError('A complete four-microbatch AdamW update was not observed')
        if any(not math.isfinite(float(value)) for value in stats.values()):
            raise AssertionError('Nonfinite preflight training statistics')
        if any(int(state['step'].item()) != 1 for state in optimizer.state.values()):
            raise AssertionError('Expected exactly one discarded AdamW update')
        local_reports.append({'variant': name, 'rank': rank, 'initial_model_sha256': digest,
                              'validation_inputs_exact_identity': True,
                              'initial_validation_predictions_exact_identity': True,
                              'candidate_matrix_shapes': shapes,
                              'visual_lora_gradient_counts': gradient_checks[0],
                              'optimizer_state_parameters': len(optimizer.state),
                              'trainable_parameters': sum(p.numel() for p in parameters),
                              'training_stats': stats, 'loading': loading,
                              'peak_allocated_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2})
        handle.remove()
        gradient_handle.remove()
        del wrapped, optimizer, model, parameters, batches, loader
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_reports)
    if misc.is_main_process():
        write_json(cli.output_root / 'preflight.json', {
            'checked_at': timestamp(), 'world_size': world_size, 'backbone': 'clip',
            'initialization': base.retrain, 'batch_size_per_gpu': 12, 'grad_accum_steps': 4,
            'nominal_effective_batch': 96, 'validation_samples_checked': 24,
            'discarded_optimizer_updates_per_variant': 1, 'ranks': gathered,
            'test_forward_count': 0})
        print(f'{timestamp()} all CLIP geometry preflight checks passed', flush=True)
    dist.barrier()
    dist.destroy_process_group()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_saved_checkpoint(directory, rows):
    import torch
    checkpoint = torch.load(directory / 'best_checkpoint.pth', map_location='cpu', weights_only=False)
    best = saved_best(rows)
    if checkpoint['epoch'] != best['epoch'] or any(
            abs(float(checkpoint[field]) - best[metric]) > 1e-10
            for field, metric in (('val_accu', 'accu'), ('val_accu_07', 'accu_07'), ('val_miou', 'miou'))):
        raise AssertionError('Saved checkpoint does not match its selected validation epoch')
    return {'epoch': checkpoint['epoch'], 'same_epoch_metrics_verified': True}


def run(cli):
    root = cli.output_root
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'status.json').exists() or any((root / name).exists() for name in (
            'safe_crop', 'safe_crop_floor8', 'preflight.json', 'augmentation_audit.json')):
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {root}')
    args = variant_args(cli, 'original', False, 0.0)
    reference = validate_reference(cli, args)
    reference['checkpoint_verification'] = verify_saved_checkpoint(cli.control_dir, validation_rows(cli.control_dir))
    write_json(root / 'control_reference.json', reference)
    recipes, differences = {}, {}
    for name, safe_crop, floor in variants(cli.scale_floor):
        candidate = copy.deepcopy(args)
        candidate.output_dir = str(root / name)
        candidate.target_safe_crop, candidate.target_scale_floor_pixels = safe_crop, floor
        recipes[name] = vars(candidate)
        differences[name] = recipe_differences(vars(args), vars(candidate))
        if set(differences[name]) - set(GEOMETRY_DEFAULTS):
            raise AssertionError('An objective or training setting other than geometry changed')
    write_json(root / 'recipe.json', {
        'created_at': timestamp(), 'variants': recipes, 'differences': differences,
        'initialization_kind': 'Common public MMVG mixup weights; the source supplies configuration only',
        'world_size': 2, 'nominal_effective_batch': 96,
        'lora_schedule': 'all_12_layers_from_epoch0_matching_completed_control',
        'selection': 'first_strict_acc05_maximum', 'quality_fields': list(QUALITY_FIELDS),
        'scale_floor_note': '8 pixels is a fixed screen value; use largest available scale if the floor is unreachable',
        'test_used_for_training_or_selection': False})
    repository = Path(__file__).resolve().parents[2]
    tracked = ('datasets/__init__.py', 'datasets/transforms.py', 'datasets/data_loader.py',
               'models/mmvg_fusion.py', 'engine.py', 'utils/loss_utils.py',
               'train_val/mmvg_train.py', 'script_train/infmae_experiments/train_clip_control.py',
               'script_train/infmae_experiments/run_clip_geometry_control.py')
    write_json(root / 'provenance.json', {
        'created_at': timestamp(), 'common_initialization': str(args.retrain),
        'common_initialization_sha256': sha256_file(args.retrain),
        'source_recipe_checkpoint': str(cli.source),
        'original_best_checkpoint_sha256': sha256_file(cli.control_dir / 'best_checkpoint.pth'),
        'code_sha256': {name: sha256_file(repository / name) for name in tracked},
        'baseline_code_snapshot_available': False,
        'reuse_note': 'Completed baseline recipe and selected checkpoint verified; no new original training run',
        'test_forward_count': 0})
    gpu_rows = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    gpu_memory = {row.split(',')[0].strip(): [int(value.strip()) for value in row.split(',')[1:]]
                  for row in gpu_rows.splitlines()}
    if any(gpu_memory[index][0] > 2048 or gpu_memory[index][1] < 14000 for index in cli.gpus.split(',')):
        raise RuntimeError('Selected GPUs are occupied or have insufficient free memory')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=cli.gpus, OMP_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       MPLCONFIGDIR=str(root / 'matplotlib'))
    common = ['--source', str(cli.source), '--control-dir', str(cli.control_dir),
              '--output-root', str(root), '--gpus', cli.gpus,
              '--epochs', str(cli.epochs), '--seed', str(cli.seed), '--scale-floor', str(cli.scale_floor)]
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
    commands = {
        'audit': [sys.executable, '-m', 'script_train.infmae_experiments.run_clip_geometry_control',
                  *common, '--audit-only'],
        'preflight': [*torchrun, 'script_train.infmae_experiments.run_clip_geometry_control',
                      *common, '--preflight-only'],
    }
    for name, _, floor in variants(cli.scale_floor)[1:]:
        commands[name] = [*torchrun, 'script_train.infmae_experiments.train_clip_control',
                          '--source', str(cli.source), '--output-dir', str(root / name),
                          '--epochs', str(cli.epochs), '--seed', str(cli.seed),
                          '--batch-size', '12', '--grad-accum-steps', '4',
                          '--target-safe-crop', '--target-scale-floor-pixels', str(floor)]
    write_json(root / 'commands.json', commands)
    status = {'pid': os.getpid(), 'started_at': timestamp(), 'gpus': cli.gpus.split(','),
              'source_recipe': str(cli.source), 'initialization': args.retrain,
              'reused_control_dir': str(cli.control_dir), 'epochs_per_run': cli.epochs,
              'seed': cli.seed, 'batch_size_per_gpu': 12, 'grad_accum_steps': 4,
              'nominal_effective_batch': 96, 'phase': 'prepare', 'runs': {},
              'test_forward_count': 0, 'test_used_for_training_or_selection': False}
    write_json(root / 'status.json', status)
    write_json(root / 'comparison.json', comparison(cli))
    for name, command in commands.items():
        training = name in ('safe_crop', 'safe_crop_floor8')
        directory = root / name if training else root
        directory.mkdir(exist_ok=True)
        log = directory / 'console.log' if training else root / f'{name}.log'
        with log.open('w') as output:
            process = subprocess.Popen(command, cwd=repository, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT)
            status.update(phase=name, active_pid=process.pid, phase_started_at=timestamp())
            status['runs'][name] = {'pid': process.pid, 'started_at': timestamp(), 'phase': 'running'}
            write_json(root / 'status.json', status)
            print(f'{timestamp()} starting {name}: PID {process.pid}', flush=True)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    report = comparison(cli)
                    write_json(root / 'comparison.json', report)
                    if training:
                        status['runs'][name]['epochs_completed'] = report['runs'][name]['epochs_completed']
                    status['updated_at'] = timestamp()
                    write_json(root / 'status.json', status)
        status['runs'][name].update(returncode=code, finished_at=timestamp(),
                                   phase='complete' if code == 0 else 'failed')
        if code == 0 and training:
            rows = validation_rows(directory)
            if [row['epoch'] for row in rows] != list(range(cli.epochs)):
                code = 1
                status['runs'][name]['error'] = 'Incomplete training epoch range'
            else:
                try:
                    status['runs'][name]['checkpoint_verification'] = verify_saved_checkpoint(directory, rows)
                except Exception as error:
                    code = 1
                    status['runs'][name]['error'] = str(error)
            status['runs'][name].update(returncode=code, epochs_completed=len(rows),
                                       phase='complete' if code == 0 else 'failed')
        write_json(root / 'comparison.json', comparison(cli))
        write_json(root / 'status.json', status)
        if code:
            status.update(phase=f'failed_{name}', returncode=code, finished_at=timestamp())
            write_json(root / 'status.json', status)
            raise SystemExit(code)
    status.update(phase='complete', returncode=0, finished_at=timestamp())
    write_json(root / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--scale-floor', type=float, default=8.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-only', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--detach', action='store_true')
    cli = parser.parse_args()
    gpu_ids = cli.gpus.split(',')
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(index.isdigit() for index in gpu_ids):
        parser.error('Require two distinct GPU IDs')
    if cli.epochs != 120 or cli.seed != 13 or cli.scale_floor != 8.0:
        parser.error('This matched screen requires 120 epochs, seed13 and the fixed 8-pixel floor')
    cli.source = cli.source.resolve(strict=True)
    cli.control_dir = cli.control_dir.resolve(strict=True)
    cli.output_root = cli.output_root.resolve()
    if cli.audit_only:
        audit(cli)
    elif cli.preflight_only:
        preflight(cli)
    elif cli.detach:
        cli.output_root.mkdir(parents=True, exist_ok=True)
        if (cli.output_root / 'status.json').exists():
            raise FileExistsError('The experiment has already been started')
        command = [sys.executable, '-m', 'script_train.infmae_experiments.run_clip_geometry_control',
                   *[argument for argument in sys.argv[1:] if argument != '--detach']]
        with (cli.output_root / 'runner.log').open('x') as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       cwd=Path(__file__).resolve().parents[2])
        write_json(cli.output_root / 'launcher.json', {
            'started_at': timestamp(), 'pid': process.pid, 'command': command,
            'output_root': str(cli.output_root), 'test_forward_count': 0})
        print(json.dumps({'pid': process.pid, 'output_root': str(cli.output_root)}), flush=True)
    else:
        run(cli)


if __name__ == '__main__':
    main()
