"""Seed-matched C1 continuation with selective, recoverable checkpoints."""
import math
import json
from pathlib import Path
import time

from script_train.clip_training_control.runtime import recipe as previous_recipe
from script_train.clip_reliability_control.runtime import reference, references
from script_train.clip_lavs_control.runtime import (
    ORIGINAL, REFERENCE, PYTHON, atomic_checkpoint, write_json, sha256,
    seed_all, loaders, trim_log, recover_best, rng_state, restore_rng,
)

REPOSITORY = Path(__file__).resolve().parents[2]
VARIANTS = ('c1_continue', 'anchor_visual', 'anchor_text_shared', 'anchor_text_separate')
EPOCHS = 20
NEW_PREFIX = 'region_anchor_exchange.'
PROJECTION_PREFIXES = ('cross_fusion_layers_vt.', 'cross_fusion_layers_tv.')
FORMAT = 'c1_region_anchor_delta_v1'


def recipe(directory, variant='anchor_text_separate', seed=13):
    if variant not in VARIANTS or seed not in (13, 14, 15):
        raise ValueError('Unregistered variant or seed')
    args = previous_recipe(directory, 'd_c1_continue', seed)
    args.experiment_case = variant
    args.visual_exchange = 'lavs' if variant == 'c1_continue' else variant
    args.exchange_train_new_only = variant != 'c1_continue'
    args.exchange_projection_tuning = True
    args.exchange_preserve_weight = 0.
    args.anchor_count, args.anchor_dim = 16, 128
    return args


def selected_name(name, args):
    return (name.startswith(PROJECTION_PREFIXES)
            or (args.exchange_train_new_only and name.startswith(NEW_PREFIX)))


def assert_scope(model, args):
    wrong = [n for n, p in model.named_parameters() if p.requires_grad != selected_name(n, args)]
    if wrong:
        raise AssertionError(f'Unexpected trainable parameters: {wrong[:12]}')


def initialize(model, args):
    import torch
    if sha256(args.exchange_base_checkpoint) != args.exchange_base_sha256:
        raise ValueError('C1 initialization checksum differs')
    state = torch.load(args.exchange_base_checkpoint, map_location='cpu', weights_only=False)
    missing, extra = model.load_state_dict(state['model'], strict=False)
    expected = {NEW_PREFIX + k for k in model.region_anchor_exchange.state_dict()} if args.exchange_train_new_only else set()
    if set(missing) != expected or extra:
        raise ValueError(f'Unexpected C1 initialization: missing={missing}, extra={extra}')
    return {'base_epoch': state['epoch'], 'base_sha256': args.exchange_base_sha256,
            'new_parameters': sum(p.numel() for n, p in model.named_parameters() if n.startswith(NEW_PREFIX))}


def optimizer_for(model, args):
    import torch
    assert_scope(model, args)
    groups = [{'name': 'existing_external_projections', 'lr': args.projection_lr,
               'params': [p for n, p in model.named_parameters() if n.startswith(PROJECTION_PREFIXES)]}]
    if args.exchange_train_new_only:
        groups.append({'name': 'regional_anchor_exchange', 'lr': args.lr,
                       'params': [p for n, p in model.named_parameters() if n.startswith(NEW_PREFIX)]})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if {id(p) for g in groups for p in g['params']} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer coverage differs from the registered scope')
    def factor(epoch):
        return min(1., (epoch + 1) / 2) * (.1 + .45 * (1 + math.cos(math.pi * min(epoch, args.epochs) / args.epochs)))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def delta_record(model, args, epoch, validation, best, provenance):
    return {'format': FORMAT, 'experiment_case': args.experiment_case, 'args': args,
            'model_delta': {n: t.detach().cpu().clone() for n, t in model.state_dict().items() if selected_name(n, args)},
            'epoch': epoch, 'val_stats': validation, 'val_accu': validation['accu'],
            'val_accu_07': validation['accu_07'], 'val_miou': validation['miou'], 'best_accu': best,
            'base_checkpoint': args.exchange_base_checkpoint, 'base_sha256': args.exchange_base_sha256,
            'provenance': provenance}


def restore_delta(model, args, checkpoint):
    if (checkpoint['format'] != FORMAT or checkpoint['experiment_case'] != args.experiment_case
            or checkpoint['base_checkpoint'] != args.exchange_base_checkpoint
            or checkpoint['base_sha256'] != args.exchange_base_sha256):
        raise ValueError('Checkpoint belongs to another initialization or variant')
    expected = {n for n in model.state_dict() if selected_name(n, args)}
    if set(checkpoint['model_delta']) != expected:
        raise ValueError('Checkpoint changes parameters outside its registered scope')
    missing, extra = model.load_state_dict(checkpoint['model_delta'], strict=False)
    if set(missing) != set(model.state_dict()) - expected or extra:
        raise ValueError('Checkpoint restoration changed the frozen/delta boundary')


def save_selection(directory, stem, record, validation):
    path, meta = directory / f'{stem}_checkpoint.pth', directory / f'{stem}.json'
    if path.exists():
        path.replace(directory / f'previous_{stem}_checkpoint.pth')
        meta.replace(directory / f'previous_{stem}.json')
    atomic_checkpoint(path, record)
    write_json(meta, {'epoch': record['epoch'], 'is_initial_c1': record['epoch'] == -1, **validation})


def recover_selection(directory, committed_epoch, stem='trained_best'):
    """Roll back a selection written before the latest epoch was committed."""
    path, meta = directory / f'{stem}_checkpoint.pth', directory / f'{stem}.json'
    previous, previous_meta = directory / f'previous_{stem}_checkpoint.pth', directory / f'previous_{stem}.json'
    current_epoch = json.loads(meta.read_text())['epoch'] if meta.exists() else None
    invalid = current_epoch is None or current_epoch > committed_epoch or not path.exists()
    if previous.exists():
        if invalid:
            previous.replace(path)
            if previous_meta.exists():
                previous_meta.replace(meta)
            elif current_epoch is None or current_epoch > committed_epoch:
                raise RuntimeError('Recovery journal lacks committed validation metadata')
        else:
            previous.unlink()
            previous_meta.unlink(missing_ok=True)
    elif current_epoch is not None and current_epoch > committed_epoch:
        if committed_epoch != -1:
            raise RuntimeError('Uncommitted trained best lacks its recovery journal')
        path.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)


def choose_promotion(baseline, runs, validation_count=608):
    control = runs['c1_continue']
    checks = {}
    for variant in VARIANTS[1:]:
        candidate = runs[variant]
        original_gain = round(candidate['accu'] * validation_count) - round(baseline['accu'] * validation_count)
        paired_gain = round(candidate['accu'] * validation_count) - round(control['accu'] * validation_count)
        quality = all(candidate[k] >= ref[k] - 1e-7 for ref in (baseline, control)
                      for k in ('accu_07', 'miou', 'miou_weak'))
        checks[variant] = {'extra_hits_vs_c1': original_gain, 'extra_hits_vs_continuation': paired_gain,
                           'quality_non_decrease': quality,
                           'passed': candidate['epoch'] >= 0 and original_gain >= 4 and paired_gain >= 2 and quality}
    eligible = [v for v, check in checks.items() if check['passed']]
    selected = max(eligible, key=lambda v: (runs[v]['accu'], runs[v]['accu_07'], runs[v]['miou'])) if eligible else None
    return {'checks': checks, 'selected_variant': selected, 'test_used': False,
            'interpretation': 'Single-seed screening only; contribution and novelty require matched replication.'}


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
        raise ValueError('Exactly two GPUs are required')
    rank = misc.get_rank()
    seed_all(seed + rank)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'preregistration.json').read_text())
    expected = manifest['recipes'][f'{variant}_seed{seed}']
    if vars(recipe(directory, variant, seed)) != expected:
        raise ValueError('Recipe changed after registration')
    model = build_model(args)
    loading = initialize(model, args)
    latest = directory / 'checkpoint.pth'
    state, start, best, trained_best = None, 0, -float('inf'), -float('inf')
    if resume and latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['provenance'] != manifest['provenance']:
            raise ValueError('Recovery provenance differs')
        restore_delta(model, args, state)
        start, best, trained_best = state['epoch'] + 1, state['best_accu'], state['trained_best_accu']
        if rank == 0:
            trim_log(directory, state['epoch'])
            recover_best(directory, state['epoch'])
            recover_selection(directory, state['epoch'])
    elif (directory / 'log.txt').exists():
        raise FileExistsError('Training log exists without a recoverable checkpoint')
    device = torch.device('cuda', args.gpu)
    model.to(device)
    assert_scope(model, args)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    optimizer, scheduler = optimizer_for(model, args)
    train_loader, val_loader, sampler = loaders(args)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['lr_scheduler'])
        restore_rng(state['rng_by_rank'][rank])
        del state
    if rank == 0:
        write_json(directory / 'recipe.json', expected)
        write_json(directory / 'initialization.json', loading)
    if start == 0 and not (resume and latest.exists()):
        initial = validate(args, wrapped, val_loader, device)
        for key in ('accu', 'accu_07', 'miou', 'miou_weak'):
            tolerance = 1e-12 if key in ('accu', 'accu_07') else 1e-5
            if abs(initial[key] - reference(seed)['validation'][key]) > tolerance:
                raise AssertionError(f'Initial C1 validation differs: {key}')
        best = initial['accu']
        states = [None] * 2
        dist.all_gather_object(states, rng_state())
        if rank == 0:
            write_json(directory / 'initial_validation.json', {'included_in_selection': True, **initial})
            record = delta_record(model, args, -1, initial, best, manifest['provenance'])
            save_selection(directory, 'best', record, initial)
            record.update(optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
                          rng_by_rank=states, trained_best_accu=trained_best)
            atomic_checkpoint(latest, record)
        dist.barrier()
    for epoch in range(start, args.epochs):
        if rank == 0:
            write_json(directory / 'progress.json', {'epoch': epoch, 'completed_epochs': epoch,
                       'epoch_in_progress': True, 'updated_at': time.time()})
        sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        training = train_one_epoch(args, wrapped, train_loader, optimizer, device, epoch, 1.)
        assert_scope(model, args)
        scheduler.step()
        validation = validate(args, wrapped, val_loader, device)
        improved, trained_improved = validation['accu'] > best, validation['accu'] > trained_best
        best, trained_best = max(best, validation['accu']), max(trained_best, validation['accu'])
        states = [None] * 2
        dist.all_gather_object(states, rng_state())
        if rank == 0:
            record = delta_record(model, args, epoch, validation, best, manifest['provenance'])
            row = {'epoch': epoch, **{f'train_{k}': v for k, v in training.items()},
                   **{f'val_{k}': v for k, v in validation.items()}, 'seconds': time.monotonic() - tick,
                   'peak_memory_bytes': torch.cuda.max_memory_allocated()}
            with (directory / 'log.txt').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            if improved:
                save_selection(directory, 'best', record, validation)
            if trained_improved:
                save_selection(directory, 'trained_best', record, validation)
            record.update(optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
                          rng_by_rank=states, trained_best_accu=trained_best)
            atomic_checkpoint(latest, record)
            for stem in ('best', 'trained_best'):
                for suffix in ('_checkpoint.pth', '.json'):
                    (directory / f'previous_{stem}{suffix}').unlink(missing_ok=True)
            write_json(directory / 'progress.json', {'epoch': epoch, 'completed_epochs': epoch + 1,
                       'epoch_in_progress': False, 'best': json.loads((directory / 'best.json').read_text()),
                       'trained_best': json.loads((directory / 'trained_best.json').read_text()),
                       'updated_at': time.time(), 'peak_memory_bytes': row['peak_memory_bytes']})
        dist.barrier()
    dist.destroy_process_group()
