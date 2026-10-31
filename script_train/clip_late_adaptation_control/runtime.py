"""Three-seed late visual/fusion adaptation with optimizer-step EMA."""
import math
import json
from pathlib import Path
import time

from script_train.clip_training_control.runtime import recipe as previous_recipe
from script_train.clip_reliability_control.runtime import reference, references
from script_train.clip_lavs_control.runtime import (
    ORIGINAL, REFERENCE, PYTHON, atomic_checkpoint, write_json, sha256,
    seed_all, trim_log, recover_best, rng_state, restore_rng,
)
from script_train.clip_reobserve_control.data import loaders
from models.late_adaptation import parameter_group
from utils.parameter_ema import ParameterEMA

REPOSITORY = Path(__file__).resolve().parents[2]
VARIANTS = ('late_adaptation',)
SEEDS = (13, 14, 15)
EPOCHS = 30
NEW_PREFIX = 'refiner.'
BASE_PREFIXES = ('base.bbox_embed.', 'base.vl_transformer.encoder.layers.5.', 'base.vl_transformer.encoder.norm.')
FORMAT = 'c1_late_adaptation_ema_delta_v1'


def recipe(directory, variant='late_adaptation', seed=13):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError('Unregistered late-adaptation run')
    a = previous_recipe(directory, 'd_c1_continue', seed)
    a.experiment_case = variant
    a.visual_exchange = 'lavs'
    a.exchange_projection_tuning = a.exchange_train_new_only = False
    a.reobserve_enabled = a.spatial_quality_enabled = a.relative_geometry_enabled = False
    a.reobserve_mode = 'c1_continue'
    a.reobserve_hidden = 128
    a.batch_size, a.grad_accum_steps, a.num_workers = 4, 12, 2
    a.epochs, a.projection_lr = EPOCHS, 2e-6
    a.late_visual_lora_lr = 4e-6
    a.clip_optimizer_groups_separately = True
    a.target_scale_floor_pixels = 0.
    a.ema_decay = .99
    return a


def build(args):
    from models import build_model
    from models.late_adaptation import LateAdaptationGrounder
    return LateAdaptationGrounder(build_model(args), args)


def selected_name(name, args):
    return parameter_group(name) is not None


def assert_scope(model,args):
    wrong=[n for n,p in model.named_parameters() if p.requires_grad != selected_name(n,args)]
    if wrong: raise AssertionError(f'Unexpected trainable parameters: {wrong[:12]}')


def initialize(model,args):
    import torch
    if sha256(args.exchange_base_checkpoint)!=args.exchange_base_sha256:
        raise ValueError('C1 initialization checksum differs')
    state=torch.load(args.exchange_base_checkpoint,map_location='cpu',weights_only=False)
    model.base.load_state_dict(state['model'],strict=True)
    model.configure_tuning()
    return {'base_epoch':state['epoch'],'base_sha256':args.exchange_base_sha256,
            'new_parameters':sum(p.numel() for n,p in model.named_parameters() if n.startswith(NEW_PREFIX)),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'trainable_base_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}


def optimizer_for(model, args):
    import torch
    assert_scope(model, args)
    groups = []
    for name in ('decoder', 'fusion', 'visual_lora'):
        parameters = [p for n, p in model.named_parameters() if parameter_group(n) == name]
        if parameters:
            groups.append({'name': name, 'params': parameters,
                'lr': args.late_visual_lora_lr if name == 'visual_lora' else args.projection_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if {id(p) for g in groups for p in g['params']} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer coverage differs from adaptation scope')
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
    write_json(meta, {'epoch': record['epoch'], 'is_initial_c1': record['epoch'] == -1, 'weight_view': record.get('weight_view', 'raw'), **validation})


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


def collect_best_runs(root):
    from script_train.clip_reliability_control.runtime import best_from_log
    return {f'late_adaptation_seed{seed}': best_from_log(
        Path(root) / 'explore' / f'late_adaptation_seed{seed}', epochs=EPOCHS) for seed in SEEDS}


def choose_promotion(runs):
    import numpy as np
    records = [runs[f'late_adaptation_seed{seed}'] for seed in SEEDS]
    baselines = [reference(seed)['validation'] for seed in SEEDS]
    gains = {key: float(np.mean([a[key] - b[key] for a, b in zip(records, baselines)]))
             for key in ('accu', 'accu_07', 'miou', 'accu_small', 'miou_weak')}
    wins = sum(a['accu'] > b['accu'] for a, b in zip(records, baselines))
    passed = (gains['accu'] >= .005 and wins >= 2 and gains['accu_small'] > 0
              and all(gains[key] >= -1e-7 for key in ('accu_07', 'miou', 'miou_weak')))
    return {'passed': passed, 'selected_variant': 'late_adaptation' if passed else None,
            'mean_validation_deltas': gains, 'improved_seeds': wins, 'test_used': False,
            'interpretation': 'Paired raw/EMA views share one trajectory; both validation results are logged.'}


def train(root, directory, variant, seed, resume=False):
    import torch
    import torch.distributed as dist
    from utils import misc
    from engine import train_one_epoch, validate
    args = recipe(directory, variant, seed)
    torch.set_num_threads(4)
    misc.init_distributed_mode(args)
    if misc.get_world_size() != 2:
        raise ValueError('Exactly two GPU ranks required')
    rank = misc.get_rank()
    seed_all(seed + rank)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'preregistration.json').read_text())
    expected = manifest['recipes'][f'{variant}_seed{seed}']
    if vars(recipe(directory, variant, seed)) != expected:
        raise ValueError('Recipe changed after registration')
    model = build(args)
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
    ema = ParameterEMA(model, [name for name, p in model.named_parameters() if p.requires_grad], args.ema_decay)
    ema_hook = optimizer.register_step_post_hook(lambda *_: ema.update(model))
    train_loader, val_loader, sampler = loaders(args)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['lr_scheduler'])
        ema.load_state_dict(state['ema_state'])
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
            record['weight_view'] = 'raw'
            save_selection(directory, 'best', record, initial)
            record.update(optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
                          rng_by_rank=states, trained_best_accu=trained_best, ema_state=ema.state_dict())
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
        raw_validation = validate(args, wrapped, val_loader, device)
        with ema.apply(model):
            ema_validation = validate(args, wrapped, val_loader, device)
        view = 'ema' if ema_validation['accu'] > raw_validation['accu'] else 'raw'
        validation = ema_validation if view == 'ema' else raw_validation
        improved, trained_improved = validation['accu'] > best, validation['accu'] > trained_best
        best, trained_best = max(best, validation['accu']), max(trained_best, validation['accu'])
        states = [None] * 2
        dist.all_gather_object(states, rng_state())
        if rank == 0:
            row = {'epoch': epoch, 'weight_view': view, 'ema_optimizer_updates': ema.updates,
                   **{f'train_{k}': v for k, v in training.items()},
                   **{f'val_{k}': v for k, v in validation.items()},
                   **{f'raw_val_{k}': v for k, v in raw_validation.items()},
                   **{f'ema_val_{k}': v for k, v in ema_validation.items()},
                   'seconds': time.monotonic() - tick, 'peak_memory_bytes': torch.cuda.max_memory_allocated()}
            with (directory / 'log.txt').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            if improved or trained_improved:
                if view == 'ema':
                    with ema.apply(model):
                        record = delta_record(model, args, epoch, validation, best, manifest['provenance'])
                else:
                    record = delta_record(model, args, epoch, validation, best, manifest['provenance'])
                record['weight_view'] = view
                if improved:
                    save_selection(directory, 'best', record, validation)
                if trained_improved:
                    save_selection(directory, 'trained_best', record, validation)
            record = delta_record(model, args, epoch, raw_validation, best, manifest['provenance'])
            record.update(weight_view='raw', optimizer=optimizer.state_dict(),
                          lr_scheduler=scheduler.state_dict(), rng_by_rank=states,
                          trained_best_accu=trained_best, ema_state=ema.state_dict())
            atomic_checkpoint(latest, record)
            for stem in ('best', 'trained_best'):
                for suffix in ('_checkpoint.pth', '.json'):
                    (directory / f'previous_{stem}{suffix}').unlink(missing_ok=True)
            write_json(directory / 'progress.json', {'epoch': epoch, 'completed_epochs': epoch + 1,
                       'epoch_in_progress': False, 'best': json.loads((directory / 'best.json').read_text()),
                       'trained_best': json.loads((directory / 'trained_best.json').read_text()),
                       'ema_optimizer_updates': ema.updates, 'updated_at': time.time(),
                       'peak_memory_bytes': row['peak_memory_bytes']})
        dist.barrier()
    ema_hook.remove()
    dist.destroy_process_group()
