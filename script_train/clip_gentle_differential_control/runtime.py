"""Low-learning-rate, late-scope controls for differential LAVS replacement."""
import math
import sys
from pathlib import Path

from script_train.clip_differential_control import runtime as previous
from script_train.clip_lavs_replacement_control import runtime as common
from script_train.clip_lavs_replacement_control.data import loaders
from script_train.clip_lavs_control.runtime import (
    ORIGINAL, REFERENCE, PYTHON, atomic_checkpoint, write_json, sha256,
    seed_all, trim_log, recover_best, rng_state, restore_rng,
)
from script_train.clip_reliability_control.runtime import reference, references
from models.gentle_differential import (
    VARIANTS, GentleDifferentialGrounder, parameter_group, selected_name,
    recipe_contract, assert_model_contract,
)
from models.lavs_replacement import legacy_lavs_key

REPOSITORY = Path(__file__).resolve().parents[2]
ORDER = ('gentle_differential', 'gentle_corrected')
SEEDS = (13, 14, 15)
EPOCHS = 60
FORMAT = 'gentle_differential_lavs_delta_v1'
NEW_PREFIX = 'base.differential_exchange.'
BASE_PREFIXES = common.BASE_PREFIXES


def recipe(directory, variant='gentle_differential', seed=13):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError('Unregistered gentle differential run')
    inherited = 'differential_attention' if variant == 'gentle_differential' else 'corrected_attention'
    args = previous.recipe(directory, inherited, seed)
    args.experiment_case = variant
    args.epochs = EPOCHS
    args.projection_lr = 2e-6
    args.late_visual_lora_lr = 4e-6
    args.replacement_lr = 2e-5
    args.teacher_preserve_epochs = 0
    args.teacher_preserve_weight = 0.
    recipe_contract(args)
    return args


def build(args):
    from models import build_model
    recipe_contract(args)
    return GentleDifferentialGrounder(build_model(args), args)


def assert_scope(model, args):
    assert_model_contract(model, args)
    wrong = [name for name, parameter in model.named_parameters()
             if parameter.requires_grad != selected_name(name, args)]
    if wrong:
        raise AssertionError(f'Unexpected tuning mask: {wrong[:12]}')
    legacy = [name for name, _ in model.base.named_parameters() if legacy_lavs_key(name)]
    if legacy or len(getattr(model.base, 'cross_fusion_layers_vt', ())) or len(getattr(model.base, 'cross_fusion_layers_tv', ())):
        raise AssertionError('A replacement student retained original LAVS modules')


def initialize(model, args):
    assert_model_contract(model, args)
    info = previous.initialize(model, args)
    assert_scope(model, args)
    if info['legacy_lavs_student_parameters']:
        raise AssertionError('Migrated model still contains original LAVS parameters')
    info.update(gentle_contract=recipe_contract(args), teacher_in_inference=False,
                tuning_scope='late adaptation plus optional differential reference')
    return info


def optimizer_for(model, args):
    import torch
    assert_scope(model, args)
    groups = []
    for group in ('decoder', 'fusion', 'visual_lora', 'differential'):
        parameters = [p for name, p in model.named_parameters()
                      if p.requires_grad and parameter_group(name) == group]
        if not parameters:
            continue
        lr = (args.replacement_lr if group == 'differential' else
              args.late_visual_lora_lr if group == 'visual_lora' else args.projection_lr)
        groups.append({'name': group, 'params': parameters, 'lr': lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if {id(p) for g in groups for p in g['params']} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer coverage differs from registered tuning scope')

    def factor(epoch):
        return min(1., (epoch + 1) / 2) * (.1 + .45 * (1 + math.cos(math.pi * min(epoch, args.epochs) / args.epochs)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def delta_record(model, args, epoch, validation, best, provenance):
    assert_scope(model, args)
    return {'format': FORMAT, 'experiment_case': args.experiment_case,
            'model_contract': recipe_contract(args), 'args': args,
            'model_delta': {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
                            if selected_name(name, args)},
            'epoch': epoch, 'val_stats': validation, 'val_accu': validation['accu'],
            'val_accu_07': validation['accu_07'], 'val_miou': validation['miou'], 'best_accu': best,
            'base_checkpoint': args.exchange_base_checkpoint, 'base_sha256': args.exchange_base_sha256,
            'provenance': provenance}


def restore_delta(model, args, checkpoint):
    import torch
    # A damaged trainability mask is rejected before any checkpoint tensor is
    # copied, so a failed restore cannot partially change the model.
    assert_scope(model, args)
    contract = recipe_contract(args)
    stored_args = checkpoint.get('args')
    if (checkpoint.get('format') != FORMAT or checkpoint.get('experiment_case') != args.experiment_case
            or checkpoint.get('model_contract') != contract or stored_args is None
            or recipe_contract(stored_args) != contract
            or checkpoint.get('base_checkpoint') != args.exchange_base_checkpoint
            or checkpoint.get('base_sha256') != args.exchange_base_sha256):
        raise ValueError('Checkpoint belongs to a different experiment, seed or initialization')
    state = model.state_dict()
    expected = {name for name in state if selected_name(name, args)}
    delta = checkpoint.get('model_delta')
    if not isinstance(delta, dict) or set(delta) != expected:
        raise ValueError('Delta scope differs from the registered tuning mask')
    for name, tensor in delta.items():
        if not torch.is_tensor(tensor) or tensor.shape != state[name].shape or tensor.dtype != state[name].dtype:
            raise ValueError(f'Delta tensor schema differs: {name}')
    missing, extra = model.load_state_dict(delta, strict=False)
    if set(missing) != set(state) - expected or extra:
        raise ValueError('Delta restoration changed the frozen parameter boundary')
    assert_scope(model, args)


def collect_best_runs(root, seeds=(13,)):
    from script_train.clip_reliability_control.runtime import best_from_log
    return {f'{variant}_seed{seed}': best_from_log(Path(root)/'explore'/f'{variant}_seed{seed}', epochs=EPOCHS)
            for seed in seeds for variant in VARIANTS}


def choose_promotion(runs):
    baseline = reference(13)['validation']
    corrected = runs['gentle_corrected_seed13']
    candidate = runs['gentle_differential_seed13']
    quality = ('accu_07', 'miou', 'miou_weak', 'accu_normal')
    delta_original = round(candidate['accu'] * 608) - round(baseline['accu'] * 608)
    delta_matched = round(candidate['accu'] * 608) - round(corrected['accu'] * 608)
    passed = (delta_original >= 4 and delta_matched >= 2
              and candidate['accu_small'] > corrected['accu_small']
              and all(candidate[key] >= corrected[key] - 1e-7 and candidate[key] >= baseline[key] - 1e-7
                      for key in quality))
    return {'selected_variant': 'gentle_differential' if passed else None, 'passed': passed,
            'test_used': False, 'replication_seeds': [14, 15] if passed else [],
            'additional_hits_vs_original': delta_original, 'additional_hits_vs_matched': delta_matched,
            'criterion': '>=4 validation hits vs original C1 and >=2 vs gentle corrected; small improves and quality does not decrease'}


def replication_gate(runs, selected):
    if selected != 'gentle_differential':
        raise ValueError('Only the registered differential candidate can be replicated')
    import numpy as np
    comparisons = {'original': lambda seed: reference(seed)['validation'],
                   'gentle_corrected': lambda seed: runs[f'gentle_corrected_seed{seed}']}
    metrics = ('accu', 'accu_07', 'miou', 'accu_small', 'miou_weak', 'accu_normal')
    deltas = {name: {key: float(np.mean([runs[f'{selected}_seed{seed}'][key]-compare(seed)[key]
                                       for seed in SEEDS])) for key in metrics}
              for name, compare in comparisons.items()}
    wins = {name: sum(runs[f'{selected}_seed{seed}']['accu'] > compare(seed)['accu'] for seed in SEEDS)
            for name, compare in comparisons.items()}
    passed = (deltas['original']['accu'] >= .005 and all(value >= 2 for value in wins.values())
              and all(v['accu'] > 0 and v['accu_small'] > 0
                      and all(v[key] >= -1e-7 for key in ('accu_07', 'miou', 'miou_weak', 'accu_normal'))
                      for v in deltas.values()))
    return {'passed': passed, 'mean_validation_deltas': deltas, 'improved_seeds': wins, 'test_used': False}


def train(root, directory, variant, seed, resume=False):
    return common.train(root, directory, variant, seed, resume, runtime=sys.modules[__name__])
