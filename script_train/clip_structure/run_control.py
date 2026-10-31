"""C1 detail/refiner preflight, exploration, eight formal runs and frozen evaluation.

Run as a module. The suite uses two GPUs sequentially, stops on a failed job,
and never opens the test split until every formal checkpoint is registered.
"""
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
import time

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_C1 = REPOSITORY / 'output_clip_geometry_test/2026-10-05_seed13_safe_crop_epoch78/selected_model_only.pth'
VARIANTS = {'c1': (False, False), 'detail': (True, False),
            'boundary': (False, True), 'joint': (True, True)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def schedule():
    return [(stage, name, seed, 20 if stage == 'explore' else 120)
            for stage, names, seeds in (
                ('explore', ('detail', 'boundary', 'joint'), (13,)),
                ('formal', tuple(VARIANTS), (13,)),
                ('formal', ('c1', 'joint'), (14, 15)))
            for seed in seeds for name in names]


def make_args(source, directory, variant, stage, seed=13):
    import torch
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    args = copy.deepcopy(checkpoint['args'])
    del checkpoint
    if (args.dataset, args.FusionMethod, args.modality, args.imsize) != ('rgbtvg_flir', 'IAFv3', 'rgbt', 224):
        raise ValueError('Source must be the selected C1 CLIP checkpoint')
    if not getattr(args, 'target_safe_crop', False) or getattr(args, 'target_scale_floor_pixels', 0) != 0:
        raise ValueError('Source must use target-safe crop and no scale protection')
    args.enable_tir_detail, args.enable_boundary_refiner = VARIANTS[variant]
    args.clip_structure_new_only = stage == 'explore'
    args.clip_structure_experiment = True
    args.clip_structure_blur_detail = False
    args.clip_structure_disable_memory = False
    args.seed, args.epochs = seed, 20 if stage == 'explore' else 120
    args.batch_size, args.grad_accum_steps = 12, 4
    args.lr, args.weight_decay, args.optimizer, args.lr_scheduler = 1e-4, 1e-4, 'adamw', 'cosine'
    args.resume, args.start_epoch, args.hi_lora_stage = '', 0, 0
    args.eval = False
    args.report_acc07, args.report_val_groups = True, True
    args.match_infmae_lora_schedule = True
    args.infmae_frontend_only = args.infmae_local_center_only = args.infmae_bbox_head_only = False
    args.teacher_preserve_enabled = args.enable_gqr = False
    args.output_dir = str(Path(directory).resolve())
    if stage == 'explore':
        args.retrain = str(Path(source).resolve())
    if not Path(args.retrain).is_file():
        raise FileNotFoundError(args.retrain)
    # Reject accidental inheritance of other experimental objectives.
    if (getattr(args, 'infmae_alignment_mode', 'none') != 'none'
            or getattr(args, 'box_loss_type', 'giou') != 'giou'):
        raise ValueError('Require unmodified C1 base objectives')
    return args


def verify_runtime(source):
    # The local C1 cache records its interpreter. Different installed Torch
    # builds can move near-threshold samples even with identical weights.
    recorded = Path(source).parent / 'command.json'
    if recorded.is_file():
        expected = json.loads(recorded.read_text())['argv'][0]
        if Path(expected).resolve() != Path(sys.executable).resolve():
            raise RuntimeError(f'Use the recorded C1 interpreter: {expected}')


def train(cli):
    verify_runtime(cli.source)
    from train_val.mmvg_train import main
    args = make_args(cli.source, cli.run_dir, cli.variant, cli.stage, cli.seed)
    rank = int(os.environ.get('RANK', '0'))
    if rank == 0:
        path = cli.run_dir / 'recipe.json'
        if (cli.run_dir / 'log.txt').exists() and not cli.resume:
            raise FileExistsError('Existing training log: use explicit --resume for an interrupted run')
        write_json(path, vars(args))
    if cli.resume:
        if not (cli.run_dir / 'checkpoint.pth').is_file():
            raise FileNotFoundError('No latest checkpoint to resume')
        # Exact model structure and optimizer/scheduler state are restored.
        args.resume = str(cli.run_dir / 'checkpoint.pth')
        args.retrain = ''
        import torch
        from script_train.infmae_experiments.run_clip_geometry_control import validation_rows
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        last = checkpoint['epoch']
        del checkpoint
        rows = validation_rows(cli.run_dir)
        if [r['epoch'] for r in rows if r['epoch'] <= last] != list(range(last + 1)):
            raise ValueError('Saved checkpoint and training log do not cover the same completed epochs')
        if any(r['epoch'] > last for r in rows) and rank == 0:
            # A crash between log and checkpoint writes leaves an uncommitted
            # epoch. Preserve that log, then replay from the saved optimizer.
            log = cli.run_dir / 'log.txt'
            original = log.read_text()
            log.with_name(f'log.interrupted.{time.time_ns()}.txt').write_text(original)
            kept = []
            for line in original.splitlines():
                if line.startswith('{'):
                    try:
                        if json.loads(line).get('epoch', -1) > last:
                            continue
                    except json.JSONDecodeError:
                        continue
                kept.append(line)
            temporary = log.with_suffix('.tmp')
            temporary.write_text('\n'.join(kept) + '\n')
            temporary.replace(log)
    main(args)


def preflight(cli):
    verify_runtime(cli.source)
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
    from datasets import build_dataset
    from models import build_model
    from models.tir_detail_refine import NEW_MODULE_PREFIXES, parameter_report
    from utils import misc
    from engine import train_one_epoch, validate
    from script_train.infmae_experiments.run_semantic_transfer_screen import seed_everything
    from script_train.infmae_experiments.train_clip_control import load_common_initialization

    torch.set_num_threads(4)
    base = make_args(cli.source, cli.output_root, 'c1', 'formal')
    misc.init_distributed_mode(base)
    if misc.get_world_size() != 2:
        raise ValueError('Preflight requires two GPUs')
    device = torch.device(base.device)
    rank = misc.get_rank()
    source = torch.load(cli.source, map_location='cpu', weights_only=False)
    dataset_val = build_dataset('val', base)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=12,
                            num_workers=2, collate_fn=misc.collate_fn)
    reference = misc.collate_fn([dataset_val[i] for i in range(rank * 12, (rank + 1) * 12)])
    reports, baseline_predictions, common_predictions, common_digest = [], None, None, None
    for variant in (*VARIANTS, 'common_c1', 'common_joint'):
        args = copy.deepcopy(base)
        common = variant.startswith('common_')
        architecture = variant.removeprefix('common_')
        args.enable_tir_detail, args.enable_boundary_refiner = VARIANTS[architecture]
        args.clip_structure_new_only = variant != 'c1' and not common
        seed_everything(13 + rank)
        model = build_model(args).to(device)
        if common:
            loading = load_common_initialization(model, args.retrain)
        else:
            missing, extra = model.load_state_dict(source['model'], strict=False)
            if extra or any(not key.startswith(NEW_MODULE_PREFIXES) for key in missing):
                raise AssertionError(f'C1 checkpoint mismatch: {missing}, {extra}')
            loading = {'missing_new_keys': len(missing), 'unexpected_keys': 0}
        params = parameter_report(model)
        wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model.eval()
        def predict():
            images = misc.NestedTensor(reference[0].tensors.clone(), reference[0].mask).to(device)
            return model(images, reference[1])
        with torch.no_grad():
            out = predict()
            if len(out) != 5:
                raise AssertionError('Eval interface changed')
            predictions = out[0].detach().cpu()
            if common and common_predictions is None:
                common_predictions = predictions
            if baseline_predictions is None:
                baseline_predictions = predictions
            error = float((predictions - (common_predictions if common else baseline_predictions)).abs().max())
            if error > 1e-6:
                raise AssertionError(f'Zero initialization changed C1 by {error}')
            # Match validator batch shape/context and reproduce the full C1 split.
            stats = validate(args, wrapped, loader_val, device)
            if not common and (round(stats['accu'] * 608) != 478 or round(stats['accu_07'] * 608) != 350):
                raise AssertionError(f'C1 validation identity failed: {stats}')
            for _ in range(3):
                predict()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(10):
                predict()
            torch.cuda.synchronize()
            profile = {'inference_ms_batch12': (time.perf_counter() - start) * 100,
                       'inference_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2}
        # Frozen state includes both adapters, checked over every original byte.
        def frozen_digest():
            digest = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                if not name.startswith(NEW_MODULE_PREFIXES):
                    digest.update(name.encode())
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        frozen_before = frozen_digest() if args.clip_structure_new_only else None
        if common:
            current_digest = frozen_digest()
            if common_digest is None:
                common_digest = current_digest
            elif common_digest != current_digest:
                raise AssertionError('Formal variants differ in original random/common initialization')
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        gradient_reports = []
        def check_gradients(opt, step_args, step_kwargs):
            new_nonzero = []
            lora_branches = {'lora_rgb': 0, 'lora_ir': 0}
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise AssertionError(f'Nonfinite gradient {name}')
                if args.clip_structure_new_only and not name.startswith(NEW_MODULE_PREFIXES):
                    if p.requires_grad or p.grad is not None:
                        raise AssertionError(f'Exploration reopened original parameter {name}')
                if name.startswith(NEW_MODULE_PREFIXES):
                    if p.grad is None:
                        raise AssertionError(f'New parameter disconnected: {name}')
                    if p.grad.abs().sum() > 0:
                        new_nonzero.append(name)
                for branch in lora_branches:
                    if f'.{branch}.' in name and 'vision_model' in name and p.grad is not None and p.grad.abs().sum() > 0:
                        lora_branches[branch] += 1
            if architecture != 'c1' and not new_nonzero:
                raise AssertionError('No new-module learning signal')
            if common and not all(lora_branches.values()):
                raise AssertionError(f'Formal LoRA branches disconnected: {lora_branches}')
            gradient_reports.append(new_nonzero)
        hook = optimizer.register_step_pre_hook(check_gradients)
        dataset = build_dataset('train', args)
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler, 12, drop_last=True),
                            collate_fn=misc.collate_fn, num_workers=0)
        batches = list(itertools.islice(loader, 4))
        torch.cuda.reset_peak_memory_stats()
        training_stats = train_one_epoch(args, wrapped, batches, optimizer, device, 0, args.clip_max_norm)
        if len(gradient_reports) != 1 or any(not math.isfinite(float(x)) for x in training_stats.values()):
            raise AssertionError('Four-microbatch single update failed')
        if frozen_before is not None and frozen_before != frozen_digest():
            raise AssertionError('Original C1 state changed during exploration')
        # Check DDP actually reduced the new tensors to the same values.
        for name, p in model.named_parameters():
            if name.startswith(NEW_MODULE_PREFIXES):
                value = p.detach().clone()
                dist.broadcast(value, 0)
                if not torch.equal(value, p):
                    raise AssertionError(f'DDP parameter divergence: {name}')
        reports.append({'variant': variant, 'rank': rank, 'parameters': params,
                        'model_total_parameters': sum(p.numel() for p in model.parameters()),
                        'model_trainable_parameters': sum(p.numel() for p in parameters),
                        'zero_init_max_error': error, 'validation': stats, 'loading': loading, **profile,
                        'train_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
                        'frozen_original_unchanged': frozen_before is not None,
                        'new_nonzero_gradient_parameters': gradient_reports[0],
                        'training_stats': training_stats, 'optimizer_updates': 1})
        hook.remove()
        del wrapped, model, optimizer, parameters, batches, loader, out
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    gathered = [None] * 2
    dist.all_gather_object(gathered, reports)
    if rank == 0:
        import transformers, peft
        write_json(cli.output_root / 'preflight.json', {'passed': True, 'ranks': gathered,
                   'source_sha256': sha256(cli.source), 'test_forward_count': 0,
                   'runtime': {'python': sys.executable, 'torch': torch.__version__,
                               'transformers': transformers.__version__, 'peft': peft.__version__},
                   'world_size': 2, 'batch_size_per_gpu': 12, 'grad_accum_steps': 4})
    dist.destroy_process_group()


def collect(root):
    from script_train.infmae_experiments.run_clip_geometry_control import validation_rows, saved_best
    results = {}
    for stage, variant, seed, epochs in schedule():
        directory = root / stage / f'{variant}_seed{seed}'
        rows = validation_rows(directory)
        results[f'{stage}/{variant}_seed{seed}'] = {'epochs_completed': len(rows),
                                                   'required_epochs': epochs, 'selected': saved_best(rows)}
    return results


def orchestrate(cli):
    verify_runtime(cli.source)
    root = cli.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'status.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    if previous is not None:
        if not cli.resume:
            raise FileExistsError('Suite already exists; use --resume after inspecting the failed job')
        if previous['phase'] == 'complete':
            raise ValueError('Suite already completed')
        for pid in (previous.get('pid'), previous.get('active_pid')):
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f'Previous suite process {pid} is still alive')
    elif cli.resume:
        raise FileNotFoundError('No suite state to resume')
    if len(set(cli.gpus.split(','))) != 2:
        raise ValueError('Choose exactly two distinct GPUs')
    # Check before reserving a long sequential queue.
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    available = {r.split(',')[0].strip(): [int(x) for x in r.split(',')[1:]] for r in rows.splitlines()}
    if any(available[g][0] > 2048 or available[g][1] < 18000 for g in cli.gpus.split(',')):
        raise RuntimeError('Selected GPUs are occupied')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=cli.gpus, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', MPLCONFIGDIR=str(root / 'matplotlib'))
    module = 'script_train.clip_structure.run_control'
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
    common = ['--source', str(cli.source.resolve()), '--output-root', str(root)]
    commands = [('preflight', [*torchrun, module, 'preflight', *common], root / 'preflight.log')]
    recipes = {}
    for stage, variant, seed, _ in schedule():
        name = f'{stage}/{variant}_seed{seed}'
        directory = root / name
        args = make_args(cli.source, directory, variant, stage, seed)
        recipes[name] = vars(args)
        command = [*torchrun, module, 'train', *common, '--run-dir', str(directory),
                   '--stage', stage, '--variant', variant, '--seed', str(seed)]
        commands.append((name, command, directory / 'console.log'))
    code_files = ['models/tir_detail_refine.py', 'models/mmvg_fusion.py', 'utils/loss_utils.py',
                  'engine.py', 'train_val/mmvg_train.py', 'datasets/transforms.py',
                  'script_train/clip_structure/run_control.py', 'script_train/clip_structure/report.py',
                  'script_eval/infmae_ddp_probe.py', 'datasets/__init__.py', 'datasets/data_loader.py',
                  'models/vl_transformer.py', 'models/tsar_modules.py', 'utils/eval_utils.py']
    code_hashes = {p: sha256(REPOSITORY / p) for p in code_files}
    registration = {'schedule': schedule(), 'recipes': recipes,
        'code_sha256': code_hashes, 'source_sha256': sha256(cli.source),
        'formal_common_initialization_sha256': sha256(recipes['formal/c1_seed13']['retrain']),
        'selection': 'first strict maximum validation Acc@0.5 among trained epochs',
        'test_used_for_selection': False, 'bootstrap_unit': 'image', 'mechanism_retraining_included': False}
    if previous is not None:
        registered = json.loads((root / 'preregistration.json').read_text())
        if json.loads(json.dumps(registration)) != registered:
            raise ValueError('Code, checkpoint or recipe changed since preregistration; use a new suite directory')
    else:
        write_json(root / 'preregistration.json', registration)
        write_json(root / 'commands.json', {n: c for n, c, _ in commands})
    state = previous or {'gpus': cli.gpus, 'phase': 'prepared', 'jobs': {}, 'test_forward_count': 0}
    state['pid'] = os.getpid()
    def execute(name, command, log):
        # Avoid a mixed-source suite if active workspace files are edited later.
        changed = [p for p, digest in code_hashes.items() if sha256(REPOSITORY / p) != digest]
        if changed:
            raise RuntimeError(f'Preregistered code changed: {changed}')
        if state['jobs'].get(name, {}).get('status') == 'complete':
            return
        if cli.resume and name.startswith(('explore/', 'formal/')) and (log.parent / 'checkpoint.pth').is_file():
            command = [*command, '--resume']
        if name.startswith(('evaluate/', 'diagnostic/')):
            output_dir = Path(command[command.index('--output-root') + 1])
            split_index = command.index('--splits') + 1
            end = split_index
            while end < len(command) and not command[end].startswith('--'):
                end += 1
            missing = [split for split in command[split_index:end] if not (output_dir / f'{split}.json').is_file()]
            if not missing:
                state['jobs'][name] = {'status': 'complete', 'recovered_cached_evaluation': True}
                return
            command = command[:split_index] + missing + command[end:]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a' if cli.resume else 'w') as output:
            process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=output, stderr=subprocess.STDOUT)
            state.update(phase=name, active_pid=process.pid)
            state['jobs'][name] = {'pid': process.pid, 'status': 'running', 'started': time.time()}
            write_json(root / 'status.json', state)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    state['results'] = collect(root)
                    write_json(root / 'status.json', state)
            state['jobs'][name].update(status='complete' if process.returncode == 0 else 'failed',
                                       returncode=process.returncode, ended=time.time())
            write_json(root / 'status.json', state)
            if process.returncode:
                raise RuntimeError(f'{name} failed ({process.returncode}); see {log}')
    try:
        for name, command, log in commands:
            execute(name, command, log)
        # Freeze all eight selected checkpoints before the first test forward.
        selected = []
        import torch
        existing_selection = root / 'selected_models.json'
        if existing_selection.is_file():
            selected = json.loads(existing_selection.read_text())
            if len(selected) != 8 or any(sha256(x['checkpoint']) != x['sha256'] for x in selected):
                raise AssertionError('Frozen selection changed while resuming')
        for stage, variant, seed, epochs in schedule():
            if stage != 'formal' or existing_selection.is_file():
                continue
            directory = root / stage / f'{variant}_seed{seed}'
            from script_train.infmae_experiments.run_clip_geometry_control import validation_rows, saved_best
            rows = validation_rows(directory)
            if [row['epoch'] for row in rows] != list(range(epochs)):
                raise AssertionError('Formal training is incomplete')
            best = saved_best(rows)
            checkpoint = torch.load(directory / 'best_checkpoint.pth', map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != best['epoch']:
                raise AssertionError('Saved checkpoint does not match first strict Acc05 maximum')
            frozen = root / 'frozen' / f'{variant}_seed{seed}.pth'
            frozen.parent.mkdir(exist_ok=True)
            torch.save({key: value for key, value in checkpoint.items() if key not in ('optimizer', 'lr_scheduler')}, frozen)
            del checkpoint
            selected.append({'variant': variant, 'seed': seed, 'epoch': best['epoch'],
                             'checkpoint': str(frozen), 'sha256': sha256(frozen), 'validation': best})
        if not existing_selection.is_file():
            write_json(existing_selection, selected)
        for item in selected:
            name = f"{item['variant']}_seed{item['seed']}"
            execute(f'evaluate/{name}', [*torchrun, 'script_eval.infmae_ddp_probe', '--checkpoint', item['checkpoint'],
                    '--output-root', str(root / 'evaluation' / name), '--splits', 'val', 'test'], root / 'evaluation' / f'{name}.log')
            state['test_forward_count'] = sum((root / 'evaluation' / f"{x['variant']}_seed{x['seed']}" / 'test.json').is_file()
                                               for x in selected)
            if sha256(item['checkpoint']) != item['sha256']:
                raise AssertionError('Frozen checkpoint changed')
            if item['variant'] == 'joint':
                for flag, label in (('--blur-detail', 'blur_detail'), ('--disable-independent-memory', 'disable_memory')):
                    execute(f'diagnostic/{name}/{label}', [*torchrun, 'script_eval.infmae_ddp_probe', '--checkpoint', item['checkpoint'],
                            '--output-root', str(root / 'diagnostics' / name / label), '--splits', 'val', flag],
                            root / 'diagnostics' / name / f'{label}.log')
        from .report import build_report, write_markdown
        report = build_report(root)
        write_json(root / 'report.json', report)
        write_markdown(root, report)
        state.update(phase='complete', results=collect(root))
    except Exception as exc:
        state.update(phase='failed', error=str(exc))
        raise
    finally:
        write_json(root / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('suite', 'preflight', 'train', 'report'))
    parser.add_argument('--source', type=Path, default=DEFAULT_C1)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', default='3,4')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--stage', choices=('explore', 'formal'), default='explore')
    parser.add_argument('--variant', choices=tuple(VARIANTS), default='joint')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--resume', action='store_true')
    cli = parser.parse_args()
    if cli.action == 'train' and cli.run_dir is None:
        parser.error('train requires --run-dir')
    if cli.action == 'suite':
        orchestrate(cli)
    elif cli.action == 'preflight':
        preflight(cli)
    elif cli.action == 'train':
        train(cli)
    else:
        from .report import build_report, write_markdown
        report = build_report(cli.output_root)
        write_json(cli.output_root / 'report.json', report)
        write_markdown(cli.output_root, report)


if __name__ == '__main__':
    main()
