"""Matched center-evidence controls over the gentle differential student."""
import copy
import math
import sys
from pathlib import Path

from script_train.clip_gentle_differential_control import runtime as gentle
from script_train.clip_lavs_replacement_control import runtime as common
from script_train.clip_lavs_replacement_control.data import loaders
from script_train.clip_lavs_control.runtime import (
    ORIGINAL, REFERENCE, PYTHON, atomic_checkpoint, write_json, sha256,
    seed_all, trim_log, recover_best, rng_state, restore_rng,
)
from script_train.clip_reliability_control.runtime import reference, references

REPOSITORY = Path(__file__).resolve().parents[2]
VARIANTS = ('center_none', 'center_fixed', 'center_textgate')
ORDER = ('center_textgate', 'center_fixed', 'center_none')
SEEDS = (13, 14, 15)
EPOCHS = 40
FORMAT = 'center_evidence_differential_delta_v1'
NEW_PREFIX = 'center_head.'
BASE_PREFIXES = ('student.base.bbox_embed.', 'student.base.vl_transformer.encoder.layers.5.',
                 'student.base.vl_transformer.encoder.norm.')
DETAIL_SOURCE = REFERENCE/'frozen/detail_seed13.pth'
DETAIL_SHA256 = 'cff0ccee6265f6e30d829683df73e33aa4c9e2154cf7f7fd76c9a105d55f06a3'


def _student_args(args, deep=False):
    result = copy.deepcopy(args) if deep else copy.copy(args)
    result.experiment_case = 'gentle_differential'
    return result


def case_contract(args):
    """Validate and record the experiment identity, architecture and sources."""
    case = getattr(args, 'experiment_case', None)
    modes = {'center_none': (False, 'fixed'), 'center_fixed': (True, 'fixed'),
             'center_textgate': (True, 'text_gate')}
    if case not in modes:
        raise ValueError(f'Unregistered center-evidence variant: {case!r}')
    enabled, mode = modes[case]
    if (getattr(args, 'center_evidence_enabled', None) is not enabled
            or getattr(args, 'center_evidence_mode', None) != mode
            or getattr(args, 'center_evidence_freeze_student', None) is not False):
        raise ValueError(f'Center architecture disagrees with experiment_case {case!r}')
    if (getattr(args, 'center_evidence_loss_weight', None) != .05
            or getattr(args, 'center_evidence_lr', None) != 5e-5):
        raise ValueError('Center supervision or learning rate differs from the registered recipe')
    detail_path = Path(getattr(args, 'center_evidence_detail_checkpoint', '')).resolve()
    detail_sha = getattr(args, 'center_evidence_detail_sha256', None)
    if detail_path != DETAIL_SOURCE.resolve() or detail_sha != DETAIL_SHA256:
        raise ValueError('Detail source differs from the registered seed13 checkpoint')
    student = gentle.recipe_contract(_student_args(args))
    return {'experiment_case': case, 'seed': int(args.seed), 'student_contract': student,
            'center_evidence_enabled': enabled, 'center_evidence_mode': mode,
            'center_evidence_freeze_student': False, 'center_evidence_loss_weight': .05,
            'center_evidence_lr': 5e-5, 'imsize': int(args.imsize),
            'center_evidence_hidden': int(args.center_evidence_hidden),
            'center_evidence_window_scale': float(args.center_evidence_window_scale),
            'center_evidence_min_window_pixels': float(args.center_evidence_min_window_pixels),
            'center_evidence_min_shift_pixels': float(args.center_evidence_min_shift_pixels),
            'center_evidence_max_shift_pixels': float(args.center_evidence_max_shift_pixels),
            'base_checkpoint': str(args.exchange_base_checkpoint), 'base_sha256': args.exchange_base_sha256,
            'detail_checkpoint': str(detail_path), 'detail_sha256': detail_sha, 'detail_source_seed': 13}


def recipe(directory, variant='center_textgate', seed=13):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError('Unregistered center-evidence experiment')
    args = gentle.recipe(directory, 'gentle_differential', seed)
    args.experiment_case = variant
    args.epochs = EPOCHS
    args.center_evidence_enabled = variant != 'center_none'
    args.center_evidence_mode = 'text_gate' if variant == 'center_textgate' else 'fixed'
    args.center_evidence_freeze_student = False
    args.center_evidence_loss_weight = .05
    args.center_evidence_lr = 5e-5
    args.center_evidence_hidden = 64
    args.center_evidence_window_scale = 3.
    args.center_evidence_min_window_pixels = 64.
    args.center_evidence_min_shift_pixels = 12.
    args.center_evidence_max_shift_pixels = 32.
    args.center_evidence_detail_checkpoint = str(DETAIL_SOURCE.resolve())
    args.center_evidence_detail_sha256 = DETAIL_SHA256
    case_contract(args)
    return args


def _model_api():
    from models import center_evidence_grounder
    return center_evidence_grounder


def build(args):
    contract = case_contract(args)
    student = gentle.build(_student_args(args, deep=True))
    model = _model_api().CenterEvidenceGrounder(student)
    # Keep the outer case identity separately: student.args must remain the
    # original gentle_differential case for its immutable model contract.
    object.__setattr__(model, '_center_runtime_contract', copy.deepcopy(contract))
    assert_model_contract(model, args)
    return model


def assert_model_contract(model, args):
    contract = case_contract(args)
    if getattr(model, '_center_runtime_contract', None) != contract:
        raise ValueError('Center model was built for a different experiment, seed or source')
    if not hasattr(model, 'student') or getattr(model, 'args', None) is not model.student.args:
        raise ValueError('Center wrapper must retain its own gentle student arguments')
    student_args = model.student.args
    if getattr(student_args, 'experiment_case', None) != 'gentle_differential':
        raise ValueError('Center student must retain the gentle_differential contract')
    as_outer = copy.copy(student_args)
    as_outer.experiment_case = args.experiment_case
    if case_contract(as_outer) != contract:
        raise ValueError('Center student configuration differs from the registered outer case')
    gentle.assert_model_contract(model.student, _student_args(args))
    head = getattr(model, 'center_head', None)
    if (head is not None) != contract['center_evidence_enabled']:
        raise ValueError('Built center head disagrees with experiment_case')
    if head is not None and getattr(head, 'mode', None) != contract['center_evidence_mode']:
        raise ValueError('Built center head mode disagrees with experiment_case')
    return contract


def parameter_group(name, args):
    # The fixed route must not be grouped: its status depends on the case.
    case_contract(args)
    return _model_api().parameter_group(name, _student_args(args))


def selected_name(name, args):
    case_contract(args)
    return _model_api().selected_name(name, _student_args(args))


def assert_scope(model, args):
    contract = assert_model_contract(model, args)
    # The wrapper accepts the student's gentle arguments, never the outer case.
    _model_api().assert_scope(model, model.student.args)
    gentle.assert_scope(model.student, _student_args(args))
    wrong = [name for name, parameter in model.named_parameters()
             if parameter.requires_grad != selected_name(name, args)]
    if wrong:
        raise AssertionError(f'Unexpected center tuning mask: {wrong[:12]}')
    if contract['center_evidence_mode'] == 'fixed' and any(
            p.requires_grad for name, p in model.named_parameters() if name.startswith('center_head.route.')):
        raise AssertionError('The fixed modality gate must remain frozen')


def _assert_initialized(model, args):
    contract = assert_model_contract(model, args)
    if getattr(model, '_center_runtime_initialization', None) != contract:
        raise ValueError('Center model must be initialized from its registered sources first')


def initialize(model, args):
    contract = assert_model_contract(model, args)
    if sha256(args.center_evidence_detail_checkpoint) != args.center_evidence_detail_sha256:
        raise ValueError('Registered detail source checksum differs')
    info = gentle.initialize(model.student, _student_args(args))
    detail = None
    if contract['center_evidence_enabled']:
        detail = model.center_head.initialize_tir_from_detail_checkpoint(
            args.center_evidence_detail_checkpoint, expected_sha256=args.center_evidence_detail_sha256)
        if detail['sha256'] != args.center_evidence_detail_sha256:
            raise ValueError('Loaded detail source checksum differs')
    assert_scope(model, args)
    object.__setattr__(model, '_center_runtime_initialization', copy.deepcopy(contract))
    return {'base_epoch': info['base_epoch'], 'base_sha256': info['base_sha256'],
            'student_initialization': info, 'detail_initialization': detail,
            'legacy_lavs_student_parameters': info['legacy_lavs_student_parameters'],
            'detail_checkpoint': contract['detail_checkpoint'], 'detail_sha256': contract['detail_sha256'],
            'detail_source_seed': 13, 'detail_source_shared_across_seeds': True,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'student_parameters': sum(p.numel() for p in model.student.parameters()),
            'head_parameters': sum(p.numel() for p in model.center_head.parameters()) if model.center_head else 0,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'teacher_in_inference': False}


def optimizer_for(model, args):
    import torch
    _assert_initialized(model, args)
    assert_scope(model, args)
    groups = []
    for group in ('decoder', 'fusion', 'visual_lora', 'differential', 'center'):
        parameters = [p for name, p in model.named_parameters()
                      if p.requires_grad and parameter_group(name, args) == group]
        if not parameters:
            continue
        lr = (args.center_evidence_lr if group == 'center' else
              args.replacement_lr if group == 'differential' else
              args.late_visual_lora_lr if group == 'visual_lora' else args.projection_lr)
        groups.append({'name': group, 'params': parameters, 'lr': lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if {id(p) for g in groups for p in g['params']} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer coverage differs from center tuning scope')

    def factor(epoch):
        return min(1., (epoch + 1)/2) * (.1 + .45*(1 + math.cos(math.pi*min(epoch,args.epochs)/args.epochs)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def delta_record(model, args, epoch, validation, best, provenance):
    _assert_initialized(model, args)
    assert_scope(model, args)
    return {'format': FORMAT, 'experiment_case': args.experiment_case, 'args': args,
            'model_contract': case_contract(args),
            'model_delta': {name: tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()
                            if selected_name(name,args)},
            'epoch': epoch, 'val_stats': validation, 'val_accu': validation['accu'],
            'val_accu_07': validation['accu_07'], 'val_miou': validation['miou'], 'best_accu': best,
            'base_checkpoint': args.exchange_base_checkpoint, 'base_sha256': args.exchange_base_sha256,
            'detail_checkpoint': str(Path(args.center_evidence_detail_checkpoint).resolve()),
            'detail_sha256': args.center_evidence_detail_sha256, 'detail_source_seed': 13,
            'provenance': provenance}


def restore_delta(model, args, checkpoint):
    import torch
    _assert_initialized(model, args)
    assert_scope(model, args)
    contract = case_contract(args)
    stored_args = checkpoint.get('args')
    if (checkpoint.get('format') != FORMAT or checkpoint.get('experiment_case') != args.experiment_case
            or checkpoint.get('model_contract') != contract or stored_args is None
            or case_contract(stored_args) != contract
            or checkpoint.get('base_checkpoint') != args.exchange_base_checkpoint
            or checkpoint.get('base_sha256') != args.exchange_base_sha256
            or checkpoint.get('detail_checkpoint') != contract['detail_checkpoint']
            or checkpoint.get('detail_sha256') != contract['detail_sha256']
            or checkpoint.get('detail_source_seed') != 13):
        raise ValueError('Checkpoint belongs to a different center case, seed or source')
    if (sha256(args.exchange_base_checkpoint) != args.exchange_base_sha256
            or sha256(args.center_evidence_detail_checkpoint) != args.center_evidence_detail_sha256):
        raise ValueError('Registered initialization source checksum differs')
    state = model.state_dict()
    expected = {name for name in state if selected_name(name,args)}
    delta = checkpoint.get('model_delta')
    if not isinstance(delta, dict) or set(delta) != expected:
        raise ValueError('Center delta scope differs from the registered tuning mask')
    for name,tensor in delta.items():
        if not torch.is_tensor(tensor) or tensor.shape != state[name].shape or tensor.dtype != state[name].dtype:
            raise ValueError(f'Center delta tensor schema differs: {name}')
    missing,extra = model.load_state_dict(delta,strict=False)
    if set(missing) != set(state)-expected or extra:
        raise ValueError('Center delta restoration changed the frozen parameter boundary')
    assert_scope(model,args)


def collect_best_runs(root, seeds=(13,)):
    from script_train.clip_reliability_control.runtime import best_from_log
    return {f'{variant}_seed{seed}': best_from_log(Path(root)/'explore'/f'{variant}_seed{seed}', epochs=EPOCHS)
            for seed in seeds for variant in VARIANTS}


def choose_promotion(runs):
    baseline = reference(13)['validation']
    none, fixed, text = [runs[f'{variant}_seed13'] for variant in VARIANTS]
    quality = ('accu_07', 'miou', 'miou_weak', 'accu_normal')
    def eligible(candidate, comparator):
        return (round(candidate['accu']*608)-round(baseline['accu']*608) >= 4
                and round(candidate['accu']*608)-round(comparator['accu']*608) >= 2
                and candidate['accu_small'] > comparator['accu_small']
                and all(candidate[key] >= comparator[key]-1e-7 and candidate[key] >= baseline[key]-1e-7
                        for key in quality))
    fixed_pass = eligible(fixed,none)
    text_pass = (eligible(text,none) and round(text['accu']*608)-round(fixed['accu']*608) >= 2
                 and text['accu_small'] >= fixed['accu_small']-1e-7
                 and all(text[key] >= fixed[key]-1e-7 for key in quality))
    selected = 'center_textgate' if text_pass else 'center_fixed' if fixed_pass else None
    return {'selected_variant':selected,'passed':selected is not None,
            'center_fixed_passed':fixed_pass,'center_textgate_passed':text_pass,
            'test_used':False,'replication_seeds':[14,15] if selected else [],
            'criterion':'>=4 hits vs original C1 and >=2 vs no-center; small improves and quality does not decrease; text gate also >=2 vs fixed without quality regression'}


def replication_gate(runs, selected):
    if selected not in ('center_fixed','center_textgate'):
        raise ValueError('Only registered center candidates can be replicated')
    import numpy as np
    comparisons = {'original':lambda seed:reference(seed)['validation'],
                   'center_none':lambda seed:runs[f'center_none_seed{seed}']}
    if selected == 'center_textgate':
        comparisons['center_fixed'] = lambda seed:runs[f'center_fixed_seed{seed}']
    metrics = ('accu','accu_07','miou','accu_small','miou_weak','accu_normal')
    deltas = {name:{key:float(np.mean([runs[f'{selected}_seed{seed}'][key]-compare(seed)[key]
                                      for seed in SEEDS])) for key in metrics}
              for name,compare in comparisons.items()}
    wins = {name:sum(runs[f'{selected}_seed{seed}']['accu'] > compare(seed)['accu'] for seed in SEEDS)
            for name,compare in comparisons.items()}
    passed = (deltas['original']['accu'] >= .005 and all(value >= 2 for value in wins.values())
              and all(v['accu'] > 0 and v['accu_small'] >= (-1e-7 if name == 'center_fixed' else 0.)
                      and all(v[key] >= -1e-7 for key in ('accu_07','miou','miou_weak','accu_normal'))
                      for name,v in deltas.items()))
    return {'passed':passed,'mean_validation_deltas':deltas,'improved_seeds':wins,'test_used':False}


def train(root, directory, variant, seed, resume=False):
    return common.train(root,directory,variant,seed,resume,runtime=sys.modules[__name__])
