"""Matched query-conditioned and global target-bias LAVS controls."""
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
VARIANTS = ('query_off', 'query_loss_only', 'query_global', 'query_local')
ORDER = ('query_local', 'query_global', 'query_loss_only', 'query_off')
SEEDS = (13, 14, 15)
EPOCHS = 40
FORMAT = 'tcr_query_lavs_delta_v1'
NEW_PREFIX = 'tcr_target.'
BASE_PREFIXES = ('student.base.bbox_embed.', 'student.base.vl_transformer.encoder.layers.5.',
                 'student.base.vl_transformer.encoder.norm.')


def _student_args(args, deep=False):
    result = copy.deepcopy(args) if deep else copy.copy(args)
    result.experiment_case = 'gentle_corrected'
    return result


def case_contract(args):
    case = getattr(args, 'experiment_case', None)
    modes = {'query_off':'off', 'query_loss_only':'loss_only', 'query_global':'bias', 'query_local':'bias'}
    if case not in modes:
        raise ValueError(f'Unregistered TCR experiment: {case!r}')
    if getattr(args, 'tcr_mode', None) != modes[case]:
        raise ValueError('TCR variant and mode differ')
    if getattr(args,'tcr_query_enabled',None) is not (case == 'query_local'):
        raise ValueError('Query gate does not match its registered variant')
    if getattr(args,'tcr_initial_strength',None) != .5:
        raise ValueError('Unexpected initial target-bias strength')
    if (getattr(args, 'tcr_loss_weight', None) != .05 or getattr(args, 'tcr_lr', None) != 5e-5
            or getattr(args, 'tcr_hidden', None) != 128):
        raise ValueError('TCR loss or optimizer differs from registered recipe')
    return {'experiment_case':case, 'tcr_mode':modes[case], 'seed':int(args.seed),
            'student_contract':gentle.recipe_contract(_student_args(args)),
            'tcr_loss_weight':.05, 'tcr_lr':5e-5, 'tcr_hidden':128,
            'tcr_query_enabled':case=='query_local','tcr_initial_strength':.5,
            'base_checkpoint':str(args.exchange_base_checkpoint), 'base_sha256':args.exchange_base_sha256}


def recipe(directory, variant='query_local', seed=13):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError('Unregistered TCR run')
    args=gentle.recipe(directory, 'gentle_corrected', seed)
    args.experiment_case=variant
    args.epochs=EPOCHS
    args.tcr_mode={'query_off':'off','query_loss_only':'loss_only','query_global':'bias','query_local':'bias'}[variant]
    args.tcr_query_enabled=variant=='query_local'
    args.tcr_initial_strength=.5
    args.tcr_loss_weight=.05
    args.tcr_lr=5e-5
    args.tcr_hidden=128
    case_contract(args)
    return args


def build(args):
    from models.tcr_query_lavs import QueryTCRGrounder
    contract=case_contract(args)
    student=gentle.build(_student_args(args,deep=True))
    model=QueryTCRGrounder(student,mode=args.tcr_mode)
    object.__setattr__(model,'_tcr_runtime_contract',copy.deepcopy(contract))
    assert_model_contract(model,args)
    return model


def assert_model_contract(model,args):
    contract=case_contract(args)
    if getattr(model,'_tcr_runtime_contract',None)!=contract:
        raise ValueError('TCR model was built for a different case, seed or source')
    if not hasattr(model,'student') or getattr(model,'args',None) is not model.student.args:
        raise ValueError('TCR wrapper must retain the gentle student arguments')
    as_outer=copy.copy(model.student.args)
    as_outer.experiment_case=args.experiment_case
    if case_contract(as_outer)!=contract:
        raise ValueError('TCR student config differs from the registered outer case')
    gentle.assert_model_contract(model.student,_student_args(args))
    if getattr(model,'mode',None)!=args.tcr_mode:
        raise ValueError('Built TCR mode disagrees with registered case')
    if model.query_enabled != args.tcr_query_enabled:
        raise ValueError('Built query gate disagrees with registered case')
    head=getattr(model,'tcr_target',None)
    if (head is not None)!=(args.tcr_mode!='off'):
        raise ValueError('Built TCR target head disagrees with registered case')
    if head is not None and head.query_enabled != args.tcr_query_enabled:
        raise ValueError('Target head query gate disagrees with registered case')
    return contract


def parameter_group(name,args):
    case_contract(args)
    if name.startswith('tcr_target.'):
        if name.endswith('.bias_strength') and args.tcr_mode != 'bias':
            return None
        return 'tcr' if args.tcr_mode!='off' else None
    if name.startswith('student.'):
        return gentle.parameter_group(name[len('student.'):])
    return None


def selected_name(name,args):
    case_contract(args)
    if name.startswith('tcr_target.'):
        if name.endswith('.bias_strength') and args.tcr_mode != 'bias':
            return False
        return args.tcr_mode!='off'
    if name.startswith('student.'):
        return gentle.selected_name(name[len('student.'):],_student_args(args))
    return False


def assert_scope(model,args):
    assert_model_contract(model,args)
    gentle.assert_scope(model.student,_student_args(args))
    wrong=[name for name,p in model.named_parameters() if p.requires_grad!=selected_name(name,args)]
    if wrong:
        raise AssertionError(f'Unexpected TCR tuning mask: {wrong[:12]}')


def _assert_initialized(model,args):
    if getattr(model,'_tcr_runtime_initialization',None)!=assert_model_contract(model,args):
        raise ValueError('TCR model must first load its registered C1 sources')


def initialize(model,args):
    contract=assert_model_contract(model,args)
    info=gentle.initialize(model.student,_student_args(args))
    assert_scope(model,args)
    object.__setattr__(model,'_tcr_runtime_initialization',copy.deepcopy(contract))
    return {**info, 'student_initialization':info, 'tcr_mode':args.tcr_mode,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'new_parameters':sum(p.numel() for n,p in model.named_parameters() if n.startswith('tcr_target.')),
            'model_parameters':sum(p.numel() for p in model.parameters()), 'teacher_in_inference':False}


def optimizer_for(model,args):
    import torch
    _assert_initialized(model,args)
    assert_scope(model,args)
    groups=[]
    for group in ('decoder','fusion','visual_lora','tcr'):
        parameters=[p for n,p in model.named_parameters() if p.requires_grad and parameter_group(n,args)==group]
        if parameters:
            lr=(args.tcr_lr if group=='tcr' else args.late_visual_lora_lr if group=='visual_lora' else args.projection_lr)
            groups.append({'name':group,'params':parameters,'lr':lr})
    if {id(p) for g in groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer coverage differs from registered scope')
    optimizer=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    if args.tcr_mode == 'bias':
        def project_bias_strength(*_):
            with torch.no_grad():
                model.tcr_target.bias_strength.clamp_(0.,2.)
        # Registered before EMA hook so EMA always observes projected weights.
        optimizer.register_step_post_hook(project_bias_strength)
    def factor(epoch):
        return min(1.,(epoch+1)/2)*(.1+.45*(1+math.cos(math.pi*min(epoch,args.epochs)/args.epochs)))
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def delta_record(model,args,epoch,validation,best,provenance):
    _assert_initialized(model,args)
    assert_scope(model,args)
    return {'format':FORMAT,'experiment_case':args.experiment_case,'args':args,'model_contract':case_contract(args),
            'model_delta':{n:t.detach().cpu().clone() for n,t in model.state_dict().items() if selected_name(n,args)},
            'epoch':epoch,'val_stats':validation,'val_accu':validation['accu'],'val_accu_07':validation['accu_07'],
            'val_miou':validation['miou'],'best_accu':best,'base_checkpoint':args.exchange_base_checkpoint,
            'base_sha256':args.exchange_base_sha256,'provenance':provenance}


def restore_delta(model,args,checkpoint):
    import torch
    _assert_initialized(model,args)
    assert_scope(model,args)
    contract=case_contract(args)
    if (checkpoint.get('format')!=FORMAT or checkpoint.get('experiment_case')!=args.experiment_case
            or checkpoint.get('model_contract')!=contract or checkpoint.get('args') is None
            or case_contract(checkpoint['args'])!=contract
            or checkpoint.get('base_checkpoint')!=args.exchange_base_checkpoint
            or checkpoint.get('base_sha256')!=args.exchange_base_sha256):
        raise ValueError('Checkpoint belongs to a different TCR experiment, seed or source')
    if sha256(args.exchange_base_checkpoint)!=args.exchange_base_sha256:
        raise ValueError('Registered C1 source checksum differs')
    state=model.state_dict()
    expected={n for n in state if selected_name(n,args)}
    delta=checkpoint.get('model_delta')
    if not isinstance(delta,dict) or set(delta)!=expected:
        raise ValueError('TCR delta scope differs from registered tuning mask')
    for n,t in delta.items():
        if not torch.is_tensor(t) or t.shape!=state[n].shape or t.dtype!=state[n].dtype:
            raise ValueError(f'TCR delta tensor schema differs: {n}')
    missing,extra=model.load_state_dict(delta,strict=False)
    if set(missing)!=set(state)-expected or extra:
        raise ValueError('TCR delta restoration changed frozen parameter boundary')
    assert_scope(model,args)


def collect_best_runs(root,seeds=(13,)):
    from script_train.clip_reliability_control.runtime import best_from_log
    return {f'{v}_seed{seed}':best_from_log(Path(root)/'explore'/f'{v}_seed{seed}',epochs=EPOCHS)
            for seed in seeds for v in VARIANTS}


def choose_promotion(runs):
    baseline=reference(13)['validation']
    off,loss,global_,local=[runs[f'{v}_seed13'] for v in VARIANTS]
    quality=('accu_07','miou','miou_weak','accu_normal')
    hits=lambda x:round(x['accu']*608)
    passed=(hits(local)-hits(baseline)>=4 and all(hits(local)-hits(x)>=2 for x in (off,loss,global_))
            and local['accu_small']>off['accu_small']
            and all(local[k]>=x[k]-1e-7 for x in (baseline,off,loss,global_) for k in quality))
    return {'selected_variant':'query_local' if passed else None,'passed':passed,'test_used':False,
            'replication_seeds':[14,15] if passed else [],
            'criterion':'>=4 validation hits vs C1, >=2 vs off, loss-only and global; small improves, quality does not decrease'}


def replication_gate(runs,selected):
    if selected!='query_local':
        raise ValueError('Only the registered query-gated candidate can be replicated')
    import numpy as np
    comparisons={'original':lambda s:reference(s)['validation'],
                 'off':lambda s:runs[f'query_off_seed{s}'],
                 'loss_only':lambda s:runs[f'query_loss_only_seed{s}'],
                 'global':lambda s:runs[f'query_global_seed{s}']}
    metrics=('accu','accu_07','miou','accu_small','miou_weak','accu_normal')
    deltas={name:{k:float(np.mean([runs[f'{selected}_seed{s}'][k]-compare(s)[k] for s in SEEDS])) for k in metrics}
            for name,compare in comparisons.items()}
    wins={name:sum(runs[f'{selected}_seed{s}']['accu']>compare(s)['accu'] for s in SEEDS)
          for name,compare in comparisons.items()}
    passed=(deltas['original']['accu']>=.005 and all(v>=2 for v in wins.values())
            and all(v['accu']>0 and v['accu_small']>0
                    and all(v[k]>=-1e-7 for k in ('accu_07','miou','miou_weak','accu_normal'))
                    for v in deltas.values()))
    return {'passed':passed,'mean_validation_deltas':deltas,'improved_seeds':wins,'test_used':False}


def train(root,directory,variant,seed,resume=False):
    return common.train(root,directory,variant,seed,resume,runtime=sys.modules[__name__])
