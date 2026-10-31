"""Check and run a validation-only thermal teacher ablation and CLIP control."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .run_scale_ablation import timestamp, write_json


THERMAL_PREFIXES = ('infmae_direct_adapter.', 'infmae_direct_cls_projection.',
                    'infmae_tir_text_projector.', 'infmae_tir_rgb_projector.')


def thermal_parameter(name):
    return name.startswith(THERMAL_PREFIXES) or (
        name.startswith('infmae_direct_encoder.') and
        ('.lora_A.' in name or '.lora_B.' in name))


def seed_everything(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def thermal_preflight(cli, args):
    import torch
    from datasets import build_dataset
    from models import build_model
    from utils import misc
    from utils.loss_utils import trans_vg_loss, _confidence_weighted_rgb_transfer

    source = torch.load(cli.source, map_location='cpu', weights_only=False)
    warm = torch.load(cli.warmstart, map_location='cpu', weights_only=False)
    if 'optimizer' in warm or set(source['model']) != set(warm['model']):
        raise AssertionError('Require an exact model-only warm start')
    if any(not torch.equal(value, warm['model'][key]) for key, value in source['model'].items()):
        raise AssertionError('Warm start differs from the fixed source')
    source_epoch = source['epoch']
    del source
    args.infmae_frontend_only = True
    args.infmae_local_center_only = False
    args.infmae_bbox_head_only = False
    args.teacher_preserve_enabled = False
    args.infmae_rgb_transfer_weighting = 'semantic'
    args.infmae_alignment_start_epoch = args.infmae_rgb_tir_start_epoch = 0
    args.infmae_alignment_ramp_epochs = args.infmae_rgb_tir_ramp_epochs = 1
    args.infmae_alignment_adapter_start_epoch = 0
    args.infmae_alignment_adapter_ramp_epochs = 1
    seed_everything(cli.seed + misc.get_rank())
    model = build_model(args).to('cuda')
    model.load_state_dict(warm['model'], strict=True)
    dataset = build_dataset('train', args)
    start = misc.get_rank() * args.batch_size
    images, texts, targets, masks = misc.collate_fn(
        [dataset[index] for index in range(start, start + args.batch_size)])
    images, targets, masks = images.to('cuda'), targets.to('cuda'), masks.to('cuda')

    def fresh_images():
        return misc.NestedTensor(images.tensors.clone(), images.mask)

    model.eval()
    with torch.no_grad():
        args.infmae_rgb_transfer_weighting = 'uniform'
        uniform_prediction = model(fresh_images(), texts)[0]
        args.infmae_rgb_transfer_weighting = 'semantic'
        semantic_prediction = model(fresh_images(), texts)[0]
    if not torch.equal(uniform_prediction, semantic_prediction):
        raise AssertionError('Training loss option changed initial inference')
    model.train()
    output = model(fresh_images(), texts, target_boxes=targets)
    if len(output) != 6:
        raise AssertionError('Training did not produce the alignment embeddings')
    pred, eos, cls, similarity, segmentation, aux = output
    if aux['rgb_embedding'].requires_grad or aux['text_embedding'].requires_grad:
        raise AssertionError('RGB/text teacher must be detached')
    active = [name for name, p in model.named_parameters() if p.requires_grad]
    if not active or any(not thermal_parameter(name) for name in active):
        raise AssertionError('Adapter switching reopened a frozen parameter')
    training_modules = [name for name, module in model.named_modules() if module.training]
    if any(not (name.startswith(tuple(p.rstrip('.') for p in THERMAL_PREFIXES))
                or (name.startswith('infmae_direct_encoder.')
                    and ('.lora_A' in name or '.lora_B' in name))) for name in training_modules):
        raise AssertionError('A frozen trunk module is in training mode')

    args.infmae_rgb_transfer_weighting = 'uniform'
    uniform = trans_vg_loss(args, pred, targets, masks, eos, cls, similarity,
                           segmentation, aux=aux, epoch=0)
    args.infmae_rgb_transfer_weighting = 'semantic'
    semantic = trans_vg_loss(args, pred, targets, masks, eos, cls, similarity,
                            segmentation, aux=aux, epoch=0)
    if any(not torch.equal(value, semantic[key]) for key, value in uniform.items()
           if key.startswith('loss_') and key != 'loss_infmae_rgb_tir'):
        raise AssertionError('An objective other than RGB transfer changed')
    candidates = int(semantic['rgb_teacher_candidate_count'].item())
    if candidates != args.batch_size * misc.get_world_size():
        raise AssertionError('Confidence candidates do not cover the DDP batch')
    if not 1.0 / candidates - 1e-7 <= semantic['rgb_teacher_weight_min'].item() <= 1.0:
        raise AssertionError('Confidence weight is outside its bounds')

    same_text = torch.zeros_like(aux['text_embedding'])
    same_text[:, 0] = 1.0
    equal_weight_loss, _, _, _ = _confidence_weighted_rgb_transfer(
        aux['tir_rgb_embedding'], aux['rgb_embedding'], same_text, args.infmae_tir_text_tau)
    ordinary = (1 - (aux['tir_rgb_embedding'] * aux['rgb_embedding']).sum(-1)).mean()
    if not torch.allclose(equal_weight_loss, ordinary, atol=1e-6, rtol=1e-6):
        raise AssertionError('Uniform confidence did not reproduce mean cosine loss')

    semantic['loss_infmae_rgb_tir'].backward(retain_graph=True)
    if any(p.grad is not None for name, p in model.named_parameters() if not thermal_parameter(name)):
        raise AssertionError('RGB transfer produced a frozen trunk gradient')
    a4_native_gradients = [name for name, p in model.named_parameters()
                           if name.startswith('infmae_direct_encoder.') and p.grad is not None
                           and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0]
    if not a4_native_gradients:
        raise AssertionError('RGB transfer did not reach native thermal LoRA')
    model.zero_grad(set_to_none=True)
    training_weighting = getattr(cli, 'rgb_transfer_weighting', 'semantic')
    if training_weighting not in ('uniform', 'semantic'):
        raise ValueError('Unknown preflight training weighting')
    training_losses = uniform if training_weighting == 'uniform' else semantic
    loss = sum(value for name, value in training_losses.items() if name.startswith('loss_'))
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError('Training loss is not finite')
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise AssertionError('Thermal gradients are missing or nonfinite')
    if any(p.grad is not None for name, p in model.named_parameters() if not thermal_parameter(name)):
        raise AssertionError('Full objective produced a frozen trunk gradient')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cli.lr)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
    optimizer.step()
    changed = []
    frozen_count = 0
    for name, value in model.state_dict().items():
        equal = torch.equal(value.detach().cpu(), warm['model'][name])
        if not thermal_parameter(name):
            frozen_count += 1
            if not equal:
                raise AssertionError(f'An optimizer step changed frozen state: {name}')
        elif not equal:
            changed.append(name)
    if not changed:
        raise AssertionError('The actual optimizer step changed no thermal weights')
    return {'source_epoch': source_epoch, 'warmstart_exact_match': True,
            'initial_predictions_exact_identity': True, 'trainable_names': active,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'training_modules': training_modules, 'frozen_state_tensors': frozen_count,
            'changed_thermal_tensors_after_discarded_step': len(changed),
            'frozen_state_exact_after_step': True, 'frozen_gradients_absent': True,
            'rgb_text_teachers_detached': True, 'native_lora_a4_gradient_tensors': len(a4_native_gradients),
            'equal_confidence_reproduces_mean_loss': True, 'only_a4_objective_changed': True,
            'global_candidates': candidates, 'training_weighting': training_weighting, 'loss': loss.item(),
            'confidence_mean': semantic['rgb_teacher_confidence'].item(),
            'weight_mean': semantic['rgb_teacher_weight_mean'].item(),
            'uniform_a4_loss': uniform['loss_infmae_rgb_tir'].item(),
            'semantic_a4_loss': semantic['loss_infmae_rgb_tir'].item(),
            'peak_gpu_memory_mib': torch.cuda.max_memory_allocated() / 1024**2}


def clip_preflight(cli):
    import torch
    from datasets import build_dataset
    from models import build_model
    from utils import misc
    from utils.box_utils import generalized_box_iou, xywh2xyxy
    from .train_clip_control import control_args, load_common_initialization

    args = control_args(cli.source, cli.output_root / 'clip_control', cli.clip_control_epochs, cli.seed)
    seed_everything(cli.seed + misc.get_rank())
    model = build_model(args).to('cuda')
    loading = load_common_initialization(model, args.retrain)
    dataset = build_dataset('train', args)
    start = misc.get_rank() * 2
    images, texts, targets, _ = misc.collate_fn([dataset[index] for index in range(start, start + 2)])
    model.train()
    targets = targets.to('cuda')
    pred = model(images.to('cuda'), texts)[0]
    loss = 2 * torch.nn.functional.l1_loss(pred, targets, reduction='sum') / len(targets)
    loss = loss + 2 * (1 - torch.diag(generalized_box_iou(xywh2xyxy(pred), xywh2xyxy(targets)))).mean()
    loss.backward()
    counts = {}
    for adapter in ('lora_rgb', 'lora_ir'):
        selected = [p for name, p in model.clip.named_parameters()
                    if 'vision_model.' in name and f'.{adapter}.' in name]
        if not selected or not all(p.requires_grad for p in selected):
            raise AssertionError(f'{adapter} trainability was lost on adapter switching')
        counts[adapter] = sum(p.grad is not None and torch.isfinite(p.grad).all().item()
                              and p.grad.abs().sum().item() > 0 for p in selected)
        if not counts[adapter]:
            raise AssertionError(f'{adapter} vision adapter has no finite nonzero regression gradient')
    if any(p.grad is not None for name, p in model.clip.named_parameters() if '.base_layer.' in name):
        raise AssertionError('Frozen CLIP base layers received gradients')
    return {'initialization': args.retrain, **loading,
            'vision_lora_gradient_tensors': counts, 'base_layer_gradients_absent': True,
            'match_infmae_lora_schedule': True, 'regression_loss': loss.item()}


def preflight(cli):
    import gc
    import torch
    import torch.distributed as dist
    from utils import misc

    args = torch.load(cli.source, map_location='cpu', weights_only=False)['args']
    torch.set_num_threads(4)
    misc.init_distributed_mode(args)
    thermal = thermal_preflight(cli, args)
    gc.collect()
    torch.cuda.empty_cache()
    control = clip_preflight(cli) if cli.clip_control_epochs else None
    if misc.is_main_process():
        summary = {'checked_at': timestamp(), 'world_size': misc.get_world_size(),
                   'thermal': thermal, 'clip_control': control, 'test_forward_count': 0}
        write_json(cli.output_root / 'preflight.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def collect_results(directory):
    initial = json.loads((directory / 'initial_validation.json').read_text())
    rows = []
    for line in (directory / 'log.txt').read_text().splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        if 'validation_accu' not in row:
            continue
        stats = {key.removeprefix('validation_'): value for key, value in row.items()
                 if key.startswith('validation_')}
        rows.append({'epoch': row['epoch'], 'stats': stats,
                     'correct_05': round(stats['accu'] * 608),
                     'correct_07': round(stats['accu_07'] * 608)})
    best = max(rows, key=lambda row: (row['stats']['accu'], row['stats']['accu_07'],
                                     row['stats']['miou'])) if rows else None
    qualified = [row for row in rows if row['correct_05'] >= 463
                 and row['stats']['accu_07'] >= 329 / 608
                 and row['stats']['miou'] >= 0.6347207019203588]
    return {'initial_validation': initial, 'epochs': rows, 'best_same_epoch': best,
            'qualified_epochs': qualified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--warmstart', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--clip-control-epochs', type=int, default=120)
    parser.add_argument('--preflight-only', action='store_true')
    cli = parser.parse_args()
    gpu_ids = cli.gpus.split(',')
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(g.isdigit() for g in gpu_ids):
        parser.error('Require two distinct GPU IDs')
    if not 0 < cli.epochs <= 60 or cli.lr <= 0 or not 0 <= cli.clip_control_epochs <= 120:
        parser.error('Invalid epoch count or learning rate')
    cli.source = cli.source.resolve(strict=True)
    cli.warmstart = cli.warmstart.resolve(strict=True)
    cli.output_root = cli.output_root.resolve()
    if cli.preflight_only:
        preflight(cli)
        return
    root = cli.output_root
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'status.json'
    if status_path.exists() or any((root / name).exists() for name in ('semantic', 'uniform', 'clip_control')):
        raise FileExistsError(f'Refusing to overwrite existing experiments: {root}')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=cli.gpus, OMP_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       MPLCONFIGDIR=str(root / 'matplotlib'))
    repository = Path(__file__).resolve().parents[2]
    status = {'pid': os.getpid(), 'started_at': timestamp(), 'gpus': gpu_ids,
              'source': str(cli.source), 'warmstart': str(cli.warmstart),
              'epochs_per_frontend_run': cli.epochs, 'frontend_lr': cli.lr, 'seed': cli.seed,
              'clip_control_epochs': cli.clip_control_epochs, 'phase': 'preflight', 'runs': {},
              'test_used_for_training_or_selection': False, 'test_forward_count': 0}
    write_json(status_path, status)
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
    common = ['--source', str(cli.source), '--warmstart', str(cli.warmstart),
              '--epochs', str(cli.epochs), '--lr', str(cli.lr), '--seed', str(cli.seed)]
    commands = {'preflight': [*torchrun, 'script_train.infmae_experiments.run_semantic_transfer_screen',
                             *common, '--output-root', str(root), '--gpus', cli.gpus,
                             '--clip-control-epochs', str(cli.clip_control_epochs), '--preflight-only']}
    for name in ('semantic', 'uniform'):
        commands[name] = [*torchrun, 'script_train.infmae_experiments.continue_a5f2', *common,
                          '--output_dir', str(root / name), '--frontend-only',
                          '--rgb-transfer-weighting', name, '--report-val-groups']
    if cli.clip_control_epochs:
        commands['clip_control'] = [*torchrun, 'script_train.infmae_experiments.train_clip_control',
                                   '--source', str(cli.source), '--output-dir', str(root / 'clip_control'),
                                   '--epochs', str(cli.clip_control_epochs), '--seed', str(cli.seed)]
    write_json(root / 'commands.json', commands)
    summary = {'source': str(cli.source), 'test_forward_count': 0, 'runs': {}}
    for name, command in commands.items():
        directory = root if name == 'preflight' else root / name
        directory.mkdir(exist_ok=True)
        log = root / 'preflight.log' if name == 'preflight' else directory / 'console.log'
        with log.open('w') as output:
            process = subprocess.Popen(command, cwd=repository, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT)
            status.update(phase=name, active_pid=process.pid, phase_started_at=timestamp())
            status['runs'][name] = {'pid': process.pid, 'started_at': timestamp(), 'phase': 'running'}
            write_json(status_path, status)
            code = process.wait()
        status['runs'][name].update(returncode=code, finished_at=timestamp(),
                                    phase='complete' if code == 0 else 'failed')
        if name != 'preflight' and (directory / 'initial_validation.json').is_file():
            summary['runs'][name] = collect_results(directory)
            write_json(directory / 'results.json', summary['runs'][name])
            write_json(root / 'comparison.json', summary)
        write_json(status_path, status)
        if code:
            status.update(phase=f'failed_{name}', returncode=code, finished_at=timestamp())
            write_json(status_path, status)
            raise SystemExit(code)
    status.update(phase='complete', returncode=0, finished_at=timestamp())
    write_json(status_path, status)


if __name__ == '__main__':
    main()
