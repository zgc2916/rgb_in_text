"""Full-width C1 migration, corrected batching and text-conditioned differential reads."""
import math
import sys
from pathlib import Path

from script_train.clip_lavs_replacement_control import runtime as common
from script_train.clip_lavs_replacement_control.data import loaders
from script_train.clip_lavs_control.runtime import (
    ORIGINAL, REFERENCE, PYTHON, atomic_checkpoint, write_json, sha256,
    seed_all, trim_log, recover_best, rng_state, restore_rng,
)
from script_train.clip_reliability_control.runtime import reference, references
from models.lavs_replacement import legacy_lavs_key
from models.differential_lavs import (
    DifferentialGrounder, parameter_group, selected_name, differential_key,
    contrast_key, migrate_lavs_state,
)

REPOSITORY = Path(__file__).resolve().parents[2]
VARIANTS = ('lavs_control', 'corrected_attention', 'differential_attention')
ORDER = ('differential_attention', 'corrected_attention', 'lavs_control')
SEEDS = (13, 14, 15)
EPOCHS = 60
FORMAT = 'differential_lavs_delta_v1'
NEW_PREFIX = 'base.differential_exchange.'
BASE_PREFIXES = common.BASE_PREFIXES


def recipe(directory, variant='differential_attention', seed=13):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError('Unregistered differential-attention experiment')
    args = common.recipe(directory, 'lavs_control', seed)
    args.experiment_case = variant
    args.replace_lavs = variant != 'lavs_control'
    args.visual_exchange = 'differential_lavs' if args.replace_lavs else 'lavs'
    args.differential_enabled = variant == 'differential_attention'
    args.epochs = EPOCHS
    args.teacher_preserve_epochs = 0
    args.teacher_preserve_weight = 0.
    args.projection_lr, args.late_visual_lora_lr, args.replacement_lr = 1e-5, 5e-6, 1e-4
    return args


def build(args):
    from models import build_model
    return DifferentialGrounder(build_model(args), args)


def initialize(model, args):
    import torch
    if sha256(args.exchange_base_checkpoint) != args.exchange_base_sha256:
        raise ValueError('C1 initialization checksum differs')
    checkpoint = torch.load(args.exchange_base_checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['model']
    inherited = {name: tensor for name, tensor in state.items()
                 if not (args.replace_lavs and legacy_lavs_key(name))}
    missing, extra = model.base.load_state_dict(inherited, strict=False)
    expected = {name for name in model.base.state_dict() if differential_key(name)} if args.replace_lavs else set()
    if set(missing) != expected or extra:
        raise ValueError(f'Unexpected initialization: {missing}, {extra}')
    migration = migrate_lavs_state(model.base, state) if args.replace_lavs else None
    model.configure_tuning()
    return {'base_epoch': checkpoint['epoch'], 'base_sha256': args.exchange_base_sha256,
            'migration': migration, 'teacher_in_inference': False,
            'new_contrast_parameters': sum(p.numel() for n, p in model.named_parameters() if contrast_key(n)),
            'replacement_parameters': sum(p.numel() for n, p in model.base.named_parameters() if differential_key(n)),
            'legacy_lavs_student_parameters': sum(p.numel() for n, p in model.base.named_parameters() if legacy_lavs_key(n)),
            'student_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def assert_scope(model, args):
    wrong = [name for name, p in model.named_parameters() if p.requires_grad != selected_name(name, args)]
    if wrong:
        raise AssertionError(f'Unexpected trainable parameters: {wrong[:12]}')


def optimizer_for(model, args):
    import torch
    assert_scope(model, args)
    groups = []
    for group in ('grounding', 'original_lavs', 'visual_lora', 'migrated_attention', 'differential'):
        parameters = [p for n, p in model.named_parameters() if p.requires_grad and parameter_group(n) == group]
        if parameters:
            lr = args.replacement_lr if group == 'differential' else args.late_visual_lora_lr if group == 'visual_lora' else args.projection_lr
            groups.append({'name': group, 'params': parameters, 'lr': lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if {id(p) for g in groups for p in g['params']} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer scope mismatch')
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
        raise ValueError('Checkpoint belongs to another experiment')
    expected = {n for n in model.state_dict() if selected_name(n, args)}
    if set(checkpoint['model_delta']) != expected:
        raise ValueError('Delta scope differs from registered recipe')
    missing, extra = model.load_state_dict(checkpoint['model_delta'], strict=False)
    if set(missing) != set(model.state_dict()) - expected or extra:
        raise ValueError('Delta restoration changed the frozen parameter boundary')


def collect_best_runs(root, seeds=(13,)):
    from script_train.clip_reliability_control.runtime import best_from_log
    return {f'{variant}_seed{seed}': best_from_log(Path(root)/'explore'/f'{variant}_seed{seed}', epochs=EPOCHS)
            for seed in seeds for variant in VARIANTS}


def choose_promotion(runs):
    baseline = reference(13)['validation']
    control, corrected, differential = [runs[v + '_seed13'] for v in VARIANTS]
    quality = ('accu_07', 'miou', 'miou_weak', 'accu_normal')
    def eligible(candidate, comparator):
        return (round(candidate['accu']*608) - round(baseline['accu']*608) >= 4
                and round(candidate['accu']*608) - round(comparator['accu']*608) >= 2
                and candidate['accu_small'] > comparator['accu_small']
                and all(candidate[k] >= comparator[k]-1e-7 and candidate[k] >= baseline[k]-1e-7 for k in quality))
    corrected_pass = eligible(corrected, control)
    differential_pass = eligible(differential, control) and eligible(differential, corrected)
    selected = 'differential_attention' if differential_pass else 'corrected_attention' if corrected_pass else None
    return {'selected_variant': selected, 'passed': selected is not None,
            'corrected_attention_passed': corrected_pass, 'differential_attention_passed': differential_pass,
            'test_used': False, 'replication_seeds': [14, 15] if selected else [],
            'criterion': '>=4 hits vs original C1, >=2 vs matched control; differential also >=2 vs corrected-only; small improves and quality does not decrease'}


def replication_gate(runs, selected):
    import numpy as np
    comparisons = {'original': lambda s: reference(s)['validation'],
                   'original_lavs': lambda s: runs[f'lavs_control_seed{s}']}
    if selected == 'differential_attention':
        comparisons['corrected_only'] = lambda s: runs[f'corrected_attention_seed{s}']
    deltas = {name: {key: float(np.mean([runs[f'{selected}_seed{s}'][key]-comp(s)[key] for s in SEEDS]))
                    for key in ('accu', 'accu_07', 'miou', 'accu_small', 'miou_weak')}
              for name, comp in comparisons.items()}
    wins = {name: sum(runs[f'{selected}_seed{s}']['accu'] > comp(s)['accu'] for s in SEEDS)
            for name, comp in comparisons.items()}
    passed = (deltas['original']['accu'] >= .005 and all(value >= 2 for value in wins.values())
              and all(v['accu'] > 0 and v['accu_small'] > 0 and all(v[k] >= -1e-7 for k in ('accu_07', 'miou', 'miou_weak'))
                      for v in deltas.values()))
    return {'passed': passed, 'mean_validation_deltas': deltas, 'improved_seeds': wins, 'test_used': False}


def train(root, directory, variant, seed, resume=False):
    return common.train(root, directory, variant, seed, resume, runtime=sys.modules[__name__])
