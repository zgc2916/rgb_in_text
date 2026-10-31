"""Restore twelve local CLIP candidates and compare with a completed batch-six run."""

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .run_scale_ablation import epoch_results, timestamp, write_json


def preflight(cli):
    import torch
    import torch.distributed as dist
    from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
    from datasets import build_dataset
    from engine import train_one_epoch
    from models import build_model
    from utils import misc
    from .run_semantic_transfer_screen import seed_everything
    from .train_clip_control import control_args, load_common_initialization

    torch.set_num_threads(4)
    directory = cli.output_root / 'preflight_training'
    directory.mkdir(exist_ok=True)
    args = control_args(cli.source, directory, cli.epochs, cli.seed,
                        cli.batch_size, cli.grad_accum_steps)
    misc.init_distributed_mode(args)
    seed_everything(cli.seed + misc.get_rank())
    model = build_model(args).to(args.device)
    loading = load_common_initialization(model, args.retrain)
    shapes = []

    def record_candidates(module, inputs, output):
        logits = output[1]
        if not torch.is_tensor(logits) or tuple(logits.shape) != (cli.batch_size, cli.batch_size):
            raise RuntimeError('Base CLIP logits do not contain the requested local candidates')
        shapes.append(list(logits.shape))

    handle = model.register_forward_hook(record_candidates)
    wrapped = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True)
    parameters = [p for name, p in model.named_parameters()
                  if p.requires_grad and 'visumodel' not in name and 'textmodel' not in name]
    if args.optimizer != 'adamw':
        raise ValueError('This matched experiment requires the original AdamW optimizer')
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    dataset = build_dataset('train', args)
    sampler = DistributedSampler(dataset, shuffle=True)
    sampler.set_epoch(0)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        collate_fn=misc.collate_fn, num_workers=args.num_workers)
    # Discard this optimizer update; full training initializes again in a new process.
    batches = list(itertools.islice(loader, args.grad_accum_steps))
    if len(batches) != args.grad_accum_steps:
        raise RuntimeError('Insufficient full microbatches for a real accumulated update')
    torch.cuda.reset_peak_memory_stats()
    stats = train_one_epoch(args, wrapped, batches, optimizer,
                            torch.device(args.device), 0, args.clip_max_norm)
    handle.remove()
    if len(shapes) != args.grad_accum_steps or not optimizer.state:
        raise RuntimeError('A complete accumulated optimizer update was not observed')
    if any(not math.isfinite(float(value)) for value in stats.values()):
        raise RuntimeError('Nonfinite training statistics during the batch preflight')
    local = {'rank': misc.get_rank(), 'candidate_matrix_shapes': shapes,
             'optimizer_state_parameters': len(optimizer.state),
             'trainable_parameters': sum(p.numel() for p in parameters),
             'peak_allocated_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2,
             'peak_reserved_gpu_mib': torch.cuda.max_memory_reserved() / 1024**2}
    ranks = [None] * misc.get_world_size()
    dist.all_gather_object(ranks, local)
    if misc.is_main_process():
        summary = {'checked_at': timestamp(), 'world_size': misc.get_world_size(),
                   'batch_size_per_gpu': args.batch_size,
                   'grad_accum_steps': args.grad_accum_steps,
                   'nominal_effective_batch': args.batch_size * args.grad_accum_steps * misc.get_world_size(),
                   'base_clip_candidates_per_gpu': args.batch_size,
                   'initialization': args.retrain, 'loading': loading,
                   'discarded_optimizer_updates': 1, 'training_stats': stats,
                   'ranks': ranks, 'test_forward_count': 0}
        write_json(cli.output_root / 'preflight.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def comparison(cli):
    baseline = epoch_results(cli.control_dir)
    restored = epoch_results(cli.output_root / 'batch12')
    # Match the training saver: first occurrence of the maximum Acc@0.5.
    return {'updated_at': timestamp(), 'selection': 'first_strict_acc05_maximum',
            'baseline': {'output_dir': str(cli.control_dir), 'epochs_completed': len(baseline),
                         'saved_best': max(baseline, key=lambda row: row['acc_05']) if baseline else None},
            'batch12': {'output_dir': str(cli.output_root / 'batch12'), 'epochs_completed': len(restored),
                        'saved_best': max(restored, key=lambda row: row['acc_05']) if restored else None},
            'interpretation': 'Microbatch recipe comparison, not isolation of candidate count alone; batch grouping, augmentation RNG, coupled losses and dropped samples can also change.',
            'test_forward_count': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    cli = parser.parse_args()
    gpu_ids = cli.gpus.split(',')
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(g.isdigit() for g in gpu_ids):
        parser.error('Require two distinct GPU IDs')
    if cli.batch_size != 12 or cli.grad_accum_steps != 4 or not 0 < cli.epochs <= 120:
        parser.error('This comparison requires per-GPU batch12, accumulation4 and 1..120 epochs')
    cli.source = cli.source.resolve(strict=True)
    cli.control_dir = cli.control_dir.resolve(strict=True)
    cli.output_root = cli.output_root.resolve()
    if cli.preflight_only:
        preflight(cli)
        return

    from .train_clip_control import control_args
    root = cli.output_root
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'status.json').exists() or any((root / name).exists() for name in ('batch12', 'preflight_training')):
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {root}')
    args = control_args(cli.source, root / 'batch12', cli.epochs, cli.seed,
                        cli.batch_size, cli.grad_accum_steps)
    original = control_args(cli.source, cli.control_dir, cli.epochs, cli.seed)
    reference = json.loads((cli.control_dir / 'control_configuration.json').read_text())
    expected = {'epochs': cli.epochs, 'seed': cli.seed, 'lr': args.lr,
                'batch_size_per_gpu': 6, 'grad_accum_steps': 8,
                'match_infmae_lora_schedule': True, 'fusion_method': 'IAFv3',
                'initialization': args.retrain, 'reference_recipe': str(cli.source)}
    if any(reference.get(key) != value for key, value in expected.items()):
        raise ValueError('Completed batch-six control does not match this source recipe')
    baseline_rows = epoch_results(cli.control_dir)
    if len(baseline_rows) != cli.epochs:
        raise ValueError('Require a completed control with the same epoch budget')
    if getattr(args, 'vl_backbone', 'clip') != 'clip' or getattr(args, 'contrastive_loss', 'clip') != 'clip':
        raise ValueError('Require the CLIP backbone and CLIP contrastive objective')
    differences = {key: {'baseline': value, 'batch12': vars(args).get(key)}
                   for key, value in vars(original).items()
                   if key != 'output_dir' and value != vars(args).get(key)}
    if set(differences) != {'batch_size', 'grad_accum_steps'}:
        raise ValueError(f'Unexpected recipe differences: {differences}')
    write_json(root / 'recipe.json', {'args': vars(args), 'differences': differences,
               'world_size': 2, 'nominal_effective_batch': 96,
               'base_clip_candidates_per_gpu': 12, 'base_clip_negatives_per_anchor': 11,
               'lora_schedule': 'all_12_layers_from_epoch0_matching_completed_control',
               'exact_paper_recipe_reproduction': False,
               'interpretation': 'Batch recipe effect; this also changes batch-coupled losses, augmentation grouping and drop_last behavior.',
               'test_used_for_training_or_selection': False})
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=cli.gpus, OMP_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       MPLCONFIGDIR=str(root / 'matplotlib'))
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
    batch_options = ['--batch-size', str(cli.batch_size), '--grad-accum-steps', str(cli.grad_accum_steps)]
    common = ['--source', str(cli.source), '--epochs', str(cli.epochs), '--seed', str(cli.seed), *batch_options]
    commands = {
        'preflight': [*torchrun, 'script_train.infmae_experiments.run_clip_batch_control', *common,
                      '--control-dir', str(cli.control_dir), '--output-root', str(root),
                      '--gpus', cli.gpus, '--preflight-only'],
        'batch12': [*torchrun, 'script_train.infmae_experiments.train_clip_control', *common,
                    '--output-dir', str(root / 'batch12')],
    }
    write_json(root / 'commands.json', commands)
    write_json(root / 'comparison.json', comparison(cli))
    status_path = root / 'status.json'
    status = {'pid': os.getpid(), 'started_at': timestamp(), 'gpus': gpu_ids,
              'source': str(cli.source), 'control_dir': str(cli.control_dir),
              'batch_size_per_gpu': cli.batch_size, 'grad_accum_steps': cli.grad_accum_steps,
              'nominal_effective_batch': 96, 'epochs': cli.epochs, 'seed': cli.seed,
              'base_clip_candidates_per_gpu': 12, 'phase': 'prepare', 'runs': {},
              'test_used_for_training_or_selection': False, 'test_forward_count': 0}
    write_json(status_path, status)
    for name, command in commands.items():
        log = root / 'preflight.log' if name == 'preflight' else root / 'batch12' / 'console.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('w') as output:
            process = subprocess.Popen(command, cwd=repository, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT)
            status.update(phase=name, active_pid=process.pid, phase_started_at=timestamp())
            status['runs'][name] = {'pid': process.pid, 'started_at': timestamp(), 'phase': 'running'}
            write_json(status_path, status)
            print(f'{timestamp()} starting {name}: PID {process.pid}', flush=True)
            code = process.wait()
        status['runs'][name].update(returncode=code, finished_at=timestamp(),
                                   phase='complete' if code == 0 else 'failed')
        write_json(root / 'comparison.json', comparison(cli))
        write_json(status_path, status)
        if code:
            status.update(phase=f'failed_{name}', returncode=code, finished_at=timestamp())
            write_json(status_path, status)
            raise SystemExit(code)
    status.update(phase='complete', returncode=0, finished_at=timestamp())
    write_json(status_path, status)


if __name__ == '__main__':
    main()
