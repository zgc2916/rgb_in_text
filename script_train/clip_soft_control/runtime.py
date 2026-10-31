import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

REPOSITORY = Path(__file__).resolve().parents[2]
ORIGINAL = Path('/dataset/zhanggongcai/RGBT/RGBT-GroundBench')
REFERENCE = ORIGINAL/'output_clip_structure_control/2026-10-05_c1_detail_boundary'
PYTHON = '/data/zhanggongcai/anaconda3/bin/python'
VARIANTS = ('global_gate', 'text_global_gate', 'text_soft_spatial')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def recipe(directory, variant='text_soft_spatial', seed=13, backend=None):
    frozen_recipe = REPOSITORY/'run_resources/c1_recipe.json'
    base = (json.loads(frozen_recipe.read_text()) if frozen_recipe.exists()
            else json.loads((REFERENCE/'formal/c1_seed13/recipe.json').read_text()))
    a = copy.deepcopy(base)
    a.update(visual_exchange='lavs' if backend == 'legacy' else variant,
             exchange_radii=[1, 1, 2, 3], exchange_gate_dim=128,
             exchange_spatial_scales=[1., 2., 3.], exchange_bias_bound=2.,
             dataset_bert_model_path=str(ORIGINAL/'dataset_and_pretrain_model/pretrain_model/pretrained_weights/Bert'),
             seed=seed, epochs=120, output_dir=str(Path(directory).resolve()), resume='',
             start_epoch=0, clip_structure_experiment=True, clip_max_norm=1., num_workers=4,
             world_size=2, distributed=False, rank=0, gpu=0, dist_url='env://', dist_backend='nccl')
    return argparse.Namespace(**a)


def mapped_initialization(path):
    import torch
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
                            renamed[key[:-len(suffix)]+f'{projection}.base_layer.{field}'] = value
    return renamed


def initialize(model, path, allowed_inherited_missing=None, allowed_inherited_extra=None):
    state = mapped_initialization(path)
    missing, extra = model.load_state_dict(state, strict=False)
    if allowed_inherited_missing is not None:
        bad_missing = [k for k in missing if not k.startswith('soft_modal_exchange.')
                       and k not in allowed_inherited_missing]
        bad_extra = [k for k in extra if k not in allowed_inherited_extra]
        if bad_missing or bad_extra:
            raise RuntimeError(f'Unexpected initialization: missing={bad_missing}, extra={bad_extra}')
    if any('.base_layer.' in k for k in missing):
        raise RuntimeError('Missing pretrained CLIP base layers')
    return {'missing_keys': missing, 'removed_or_mapped_keys': extra}


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


def atomic_checkpoint(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(value, temp)
    temp.replace(path)


def loaders(args, train=True):
    from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
    from datasets import build_dataset
    from utils import misc
    data = build_dataset('train', args) if train else None
    val = build_dataset('val', args)
    val_loader = DataLoader(val, batch_size=12, sampler=DistributedSampler(val, shuffle=False),
                            num_workers=args.num_workers, collate_fn=misc.collate_fn)
    if not train:
        return val_loader
    sampler = DistributedSampler(data, shuffle=True)
    train_loader = DataLoader(data, batch_sampler=BatchSampler(sampler, 12, drop_last=True),
                              num_workers=args.num_workers, collate_fn=misc.collate_fn)
    return train_loader, val_loader, sampler


def optimizer_for(model, args):
    import torch
    params = [p for n, p in model.named_parameters()
              if 'visumodel' not in n and 'textmodel' not in n
              and (p.requires_grad or '.lora_A.' in n or '.lora_B.' in n)]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: .5*(1+math.cos(math.pi*ep/120)))
    coverage = {id(p) for group in optimizer.param_groups for p in group['params']}
    if any(p.requires_grad and id(p) not in coverage for p in model.parameters()):
        raise AssertionError('Optimizer does not cover trainable parameters')
    return optimizer, scheduler


def trim_log(directory, epoch):
    path = directory/'log.txt'
    if not path.exists():
        if epoch >= 0:
            raise ValueError('Checkpoint lacks its epoch log')
        return
    original = path.read_text()
    entries = [json.loads(line) for line in original.splitlines() if line.strip()]
    committed = [row for row in entries if row['epoch'] <= epoch]
    if [row['epoch'] for row in committed] != list(range(epoch+1)):
        raise ValueError('Checkpoint/log epoch sequence mismatch')
    if len(entries) != len(committed):
        path.with_name(f'log.interrupted.{time.time_ns()}.txt').write_text(original)
        path.write_text(''.join(json.dumps(x)+'\n' for x in committed))


def recover_best(directory, committed_epoch):
    """Restore the best checkpoint if a crash interrupted an epoch commit."""
    best_path = directory/'best.json'
    previous = directory/'previous_best_checkpoint.pth'
    previous_json = directory/'previous_best.json'
    current_epoch = json.loads(best_path.read_text())['epoch'] if best_path.exists() else None
    ahead = current_epoch is not None and current_epoch > committed_epoch
    if previous.exists():
        if current_epoch is None or ahead or not (directory/'best_checkpoint.pth').exists():
            previous.replace(directory/'best_checkpoint.pth')
            if previous_json.exists():
                previous_json.replace(best_path)
            elif not best_path.exists():
                raise RuntimeError('Recovery journal lacks best validation metadata')
        else:
            previous.unlink()
            previous_json.unlink(missing_ok=True)
    elif ahead:
        if committed_epoch != -1:
            raise RuntimeError('Uncommitted best checkpoint lacks recovery journal')
        (directory/'best_checkpoint.pth').unlink(missing_ok=True)
        best_path.unlink(missing_ok=True)
    elif committed_epoch == -1 and current_epoch is None:
        (directory/'best_checkpoint.pth').unlink(missing_ok=True)


def train(root, directory, variant, seed, resume=False):
    import torch
    import torch.distributed as dist
    from models import build_model
    from utils import misc
    from engine import train_one_epoch, validate
    args = recipe(directory, variant, seed)
    torch.set_num_threads(4)
    misc.init_distributed_mode(args)
    if misc.get_world_size() != 2:
        raise ValueError('Formal runs require exactly two GPUs')
    rank = misc.get_rank()
    seed_all(seed+rank)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'preregistration.json').read_text())
    expected = manifest['recipes'][f'{variant}_seed{seed}']
    # Compare immutable scientific recipe; distributed rank is runtime state.
    actual = vars(recipe(directory, variant, seed))
    if actual != expected:
        raise ValueError('Recipe mismatch on recovery')
    if sha256(args.retrain) != manifest['initialization_sha256']:
        raise ValueError('Common initialization changed')
    model = build_model(args)
    latest = directory/'checkpoint.pth'
    state = None
    start, best = 0, -float('inf')
    if resume and latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['provenance'] != manifest['provenance']:
            raise ValueError('Checkpoint provenance changed')
        model.load_state_dict(state['model'], strict=True)
        start, best = state['epoch']+1, state['best_accu']
        if rank == 0:
            trim_log(directory, state['epoch'])
            recover_best(directory, state['epoch'])
    else:
        if (directory/'log.txt').exists():
            raise FileExistsError('Log exists but no valid checkpoint to recover')
        report = initialize(model, args.retrain, manifest['inherited_missing_keys'], manifest['inherited_extra_keys'])
        if rank == 0:
            write_json(directory/'initialization.json', report)
    if start >= 60:
        args.hi_lora_stage = 3 if start >= 100 else (2 if start >= 80 else 1)
        model.set_HiLoRA(args)
    model.to(args.gpu)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    optimizer, scheduler = optimizer_for(model, args)
    train_loader, val_loader, sampler = loaders(args)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['lr_scheduler'])
        restore_rng(state['rng_by_rank'][rank])
        del state
    if rank == 0:
        write_json(directory/'recipe.json', expected)
    device = torch.device('cuda', args.gpu)
    if not resume or not latest.exists():
        initial = validate(args, wrapped, val_loader, device)
        states = [None]*2
        dist.all_gather_object(states, rng_state())
        if rank == 0:
            write_json(directory/'initial_validation.json', {'excluded_from_selection': True, **initial})
            atomic_checkpoint(latest, {'model': model.state_dict(), 'epoch': -1, 'args': args,
                'best_accu': best, 'provenance': manifest['provenance'],
                'optimizer': optimizer.state_dict(), 'lr_scheduler': scheduler.state_dict(), 'rng_by_rank': states})
        dist.barrier()
    for epoch in range(start, 120):
        if rank == 0:
            write_json(directory/'progress.json', {'epoch': epoch, 'completed_epochs': epoch,
                       'epoch_in_progress': True, 'best': json.loads((directory/'best.json').read_text())
                       if (directory/'best.json').exists() else None, 'updated_at': time.time()})
        sampler.set_epoch(epoch)
        if epoch in (60, 80, 100):
            args.hi_lora_stage = {60: 1, 80: 2, 100: 3}[epoch]
            model.set_HiLoRA(args)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        training = train_one_epoch(args, wrapped, train_loader, optimizer, device, epoch, 1.)
        scheduler.step()
        validation = validate(args, wrapped, val_loader, device)
        improved = validation['accu'] > best
        best = max(best, validation['accu'])
        states = [None]*2
        dist.all_gather_object(states, rng_state())
        record = {'model': model.state_dict(), 'epoch': epoch, 'args': args,
                  'val_accu': validation['accu'], 'val_accu_07': validation['accu_07'],
                  'val_miou': validation['miou'], 'val_stats': validation,
                  'best_accu': best, 'provenance': manifest['provenance']}
        if rank == 0:
            row = {'epoch': epoch, **{f'train_{k}': v for k, v in training.items()},
                   **{f'val_{k}': v for k, v in validation.items()},
                   'seconds': time.monotonic()-tick, 'peak_memory_bytes': torch.cuda.max_memory_allocated()}
            with (directory/'log.txt').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if improved:
                if (directory/'best_checkpoint.pth').exists():
                    (directory/'best_checkpoint.pth').replace(directory/'previous_best_checkpoint.pth')
                    (directory/'best.json').replace(directory/'previous_best.json')
                atomic_checkpoint(directory/'best_checkpoint.pth', record)
                write_json(directory/'best.json', {'epoch': epoch, **validation})
            record.update(optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(), rng_by_rank=states)
            atomic_checkpoint(latest, record)
            for journal in ('previous_best_checkpoint.pth', 'previous_best.json'):
                (directory/journal).unlink(missing_ok=True)
            write_json(directory/'progress.json', {'epoch': epoch, 'completed_epochs': epoch+1,
                       'epoch_in_progress': False,
                       'best': json.loads((directory/'best.json').read_text()), 'updated_at': time.time(),
                       'peak_memory_bytes': row['peak_memory_bytes']})
        dist.barrier()
    dist.destroy_process_group()
