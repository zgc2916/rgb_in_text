"""Complete paper-LAVS replacement: word queries and cross-level deformable exchange."""
import copy
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
from .data import loaders
from models.lavs_replacement import parameter_group, legacy_lavs_key, replacement_key
from utils.parameter_ema import ParameterEMA

REPOSITORY = Path(__file__).resolve().parents[2]
VARIANTS = ('lavs_control', 'fixed_exchange', 'deformable_exchange')
SEEDS = (13, 14, 15)
EPOCHS = 40
NEW_PREFIX = 'base.replacement_exchange.'
BASE_PREFIXES = ('base.bbox_embed.', 'base.vl_transformer.encoder.layers.5.', 'base.vl_transformer.encoder.norm.')
FORMAT = 'complete_lavs_replacement_delta_v1'


def recipe(directory, variant='deformable_exchange', seed=13):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError('Unregistered dense-grid run')
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
    a.imsize = 224
    a.dense_grid_enabled = a.dense_text_gate = False
    a.replace_lavs = variant != 'lavs_control'
    a.visual_exchange = 'deformable_lavs' if a.replace_lavs else 'lavs'
    a.replacement_fixed_offsets = variant == 'fixed_exchange'
    a.replacement_dim = 128
    a.projection_lr = 1e-5
    a.late_visual_lora_lr = 5e-6
    a.replacement_lr = 1e-4
    a.teacher_preserve_mode = 'additive'
    a.teacher_preserve_iou_threshold = .7
    a.teacher_preserve_weight = .5
    a.teacher_preserve_epochs = 10

    return a


def build(args):
    from models import build_model
    from models.lavs_replacement import LavsReplacementGrounder
    return LavsReplacementGrounder(build_model(args), args)


def teacher_for(args, device):
    teacher_args = recipe(Path(args.output_dir), 'lavs_control', args.seed)
    teacher = build(teacher_args)
    initialize(teacher, teacher_args)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.base._clip_control_trainability = {n: False for n,_ in teacher.base.clip.named_parameters()}
    return teacher.to(device).eval()


def selected_name(name, args):
    return (parameter_group(name) is not None
            and not (args.replace_lavs and args.replacement_fixed_offsets and '.offsets.' in name))


def assert_scope(model,args):
    wrong=[n for n,p in model.named_parameters() if p.requires_grad != selected_name(n,args)]
    if wrong: raise AssertionError(f'Unexpected trainable parameters: {wrong[:12]}')


def initialize(model,args):
    import torch
    if sha256(args.exchange_base_checkpoint)!=args.exchange_base_sha256:
        raise ValueError('C1 initialization checksum differs')
    state=torch.load(args.exchange_base_checkpoint,map_location='cpu',weights_only=False)
    removed = {k:v for k,v in state['model'].items() if args.replace_lavs and legacy_lavs_key(k)}
    inherited = {k:v for k,v in state['model'].items() if k not in removed}
    missing, extra = model.base.load_state_dict(inherited, strict=False)
    expected = {k for k in model.base.state_dict() if args.replace_lavs and replacement_key(k)}
    if set(missing) != expected or extra:
        raise ValueError(f'C1 load mismatch: {missing}, {extra}')
    model.configure_tuning()
    return {'base_epoch':state['epoch'],'base_sha256':args.exchange_base_sha256,
            'new_parameters':sum(p.numel() for n,p in model.base.named_parameters() if replacement_key(n)),
            'removed_lavs_elements':sum(v.numel() for v in removed.values()),
            'legacy_lavs_student_parameters':sum(p.numel() for n,p in model.base.named_parameters() if legacy_lavs_key(n)),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'student_parameters':sum(p.numel() for p in model.parameters()),
            'teacher_in_inference':False}


def optimizer_for(model, args):
    import torch
    assert_scope(model, args)
    groups = []
    for name in ('grounding', 'original_lavs', 'visual_lora', 'replacement'):
        parameters = [p for n, p in model.named_parameters() if parameter_group(n) == name and p.requires_grad]
        if parameters:
            groups.append({'name': name, 'params': parameters,
                'lr': (args.replacement_lr if name == 'replacement' else
                       args.late_visual_lora_lr if name == 'visual_lora' else args.projection_lr)})
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


def collect_best_runs(root, seeds=(13,)):
    from script_train.clip_reliability_control.runtime import best_from_log
    return {f'{variant}_seed{seed}': best_from_log(
        Path(root) / 'explore' / f'{variant}_seed{seed}', epochs=EPOCHS)
        for seed in seeds for variant in VARIANTS}


def choose_promotion(runs):
    baseline = reference(13)['validation']
    control, dense, text = [runs[v + '_seed13'] for v in VARIANTS]
    quality = ('accu_07', 'miou', 'miou_weak', 'accu_normal')
    def eligible(candidate, comparator, extra):
        return (round(candidate['accu']*608) - round(baseline['accu']*608) >= 4
                and round(candidate['accu']*608) - round(comparator['accu']*608) >= extra
                and candidate['accu_small'] > comparator['accu_small']
                and all(candidate[k] >= comparator[k]-1e-7 for k in quality)
                and all(candidate[k] >= baseline[k]-1e-7 for k in quality))
    high_pass = eligible(dense, control, 2)
    text_pass = eligible(text, control, 2) and eligible(text, dense, 2)
    selected = 'deformable_exchange' if text_pass else 'fixed_exchange' if high_pass else None
    return {'selected_variant': selected, 'passed': selected is not None,
            'fixed_replacement_passed': high_pass, 'deformable_replacement_passed': text_pass,
            'test_used': False, 'replication_seeds': [14, 15] if selected else [],
            'criterion': '>=4 hits vs original C1 and >=2 vs matched LAVS; learned sampling also >=2 vs fixed sampling; small improves, quality does not decrease'}


def replication_gate(runs, selected):
    import numpy as np
    def mean_delta(key, comparator):
        return float(np.mean([runs[f'{selected}_seed{s}'][key] - comparator(s)[key] for s in SEEDS]))
    comparisons = {'original': lambda s: reference(s)['validation'],
                   'original_lavs': lambda s: runs[f'lavs_control_seed{s}']}
    if selected == 'deformable_exchange':
        comparisons['fixed_sampling'] = lambda s: runs[f'fixed_exchange_seed{s}']
    deltas = {name: {k: mean_delta(k, comp) for k in ('accu','accu_07','miou','accu_small','miou_weak')}
              for name, comp in comparisons.items()}
    wins = {name: sum(runs[f'{selected}_seed{s}']['accu'] > comp(s)['accu'] for s in SEEDS)
            for name, comp in comparisons.items()}
    passed = (deltas['original']['accu'] >= .005 and all(v >= 2 for v in wins.values())
              and all(v['accu'] > 0 and v['accu_small'] > 0
                      and all(v[k] >= -1e-7 for k in ('accu_07','miou','miou_weak'))
                      for v in deltas.values()))
    return {'passed': passed, 'mean_validation_deltas': deltas, 'improved_seeds': wins, 'test_used': False}


def train(root, directory, variant, seed, resume=False, runtime=None):
    # Durable checkpoint/EMA/RNG handling shared by registered successors.
    # The original experiment keeps its unchanged default implementation.
    namespace = globals() if runtime is None else vars(runtime)
    recipe, build, initialize, assert_scope, optimizer_for, selected_name, delta_record, restore_delta = (
        namespace[key] for key in ('recipe', 'build', 'initialize', 'assert_scope', 'optimizer_for',
                                   'selected_name', 'delta_record', 'restore_delta'))
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
    teacher = teacher_for(args, device) if start < args.teacher_preserve_epochs else None
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
    else:
        seed_all(seed + rank)
    if rank == 0:
        write_json(directory / 'recipe.json', expected)
        write_json(directory / 'initialization.json', loading)
    if start == 0 and not (resume and latest.exists()):
        initial = validate(args, wrapped, val_loader, device)
        for key in ('accu', 'accu_07', 'miou', 'miou_weak'):
            tolerance = 1e-12 if key in ('accu', 'accu_07') else 1e-5
            if variant == 'lavs_control' and abs(initial[key] - reference(seed)['validation'][key]) > tolerance:
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
        if epoch >= args.teacher_preserve_epochs and teacher is not None:
            teacher = None
            torch.cuda.empty_cache()
        args.teacher_preserve_weight = (.5 * max(0., 1 - epoch/args.teacher_preserve_epochs)
                                        if args.teacher_preserve_epochs else 0.)
        training = train_one_epoch(args, wrapped, train_loader, optimizer, device, epoch, 1.,
                                   teacher_model=teacher if epoch < args.teacher_preserve_epochs else None)
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


