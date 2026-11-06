"""Train the selected RefFLIR query model separately on two RGB-T datasets."""
import copy
import sys
from pathlib import Path

import torch
from script_train.clip_tcr_query_control import runtime as query
from script_train.clip_lavs_replacement_control import runtime as common
from script_train.clip_lavs_replacement_control.data import loaders
from script_train.clip_reliability_control.runtime import reference, references
from script_train.clip_lavs_control.runtime import (ORIGINAL,REFERENCE,PYTHON,atomic_checkpoint,
    write_json,sha256,seed_all,trim_log,recover_best,rng_state,restore_rng)

REPOSITORY=Path(__file__).resolve().parents[2]
VARIANTS=('cross_m3fd','cross_mfad')
DATASETS={'cross_m3fd':'rgbtvg_m3fd','cross_mfad':'rgbtvg_mfad'}
SEED=13
EPOCHS=40
FORMAT='cross_dataset_tcr_query_delta_v1'
SOURCE=ORIGINAL/'output_clip_lavs_control/2026-11-03_tcr_query/explore/query_local_seed13/best_checkpoint.pth'
SOURCE_SHA256='0910cf34df768d7cb9a196cd295f4f77c61db17ca78bd70a2b41191fefedac2f'
NEW_PREFIX=query.NEW_PREFIX
BASE_PREFIXES=query.BASE_PREFIXES


def _query_args(args,deep=False):
    a=copy.deepcopy(args) if deep else copy.copy(args)
    a.experiment_case='query_local'
    return a


def case_contract(args):
    variant=getattr(args,'experiment_case',None)
    if variant not in DATASETS or getattr(args,'dataset',None)!=DATASETS[variant]:
        raise ValueError('Cross-dataset case and dataset disagree')
    if int(args.seed)!=SEED or getattr(args,'cross_source_checkpoint',None)!=str(SOURCE.resolve()):
        raise ValueError('Cross-dataset source or seed differs')
    if getattr(args,'cross_source_sha256',None)!=SOURCE_SHA256:
        raise ValueError('Cross-dataset source SHA differs')
    return {'experiment_case':variant,'dataset':DATASETS[variant],'seed':SEED,
            'query_contract':query.case_contract(_query_args(args)),
            'initial_checkpoint':str(SOURCE.resolve()),'initial_sha256':SOURCE_SHA256,
            'initial_dataset':'rgbtvg_flir','initial_epoch_zero_based':3,'initial_weight_view':'ema',
            'normalization':'dataset-specific','epochs':EPOCHS}


def recipe(directory,variant='cross_m3fd',seed=SEED):
    if variant not in VARIANTS or seed!=SEED:
        raise ValueError('Unregistered cross-dataset run')
    a=query.recipe(directory,'query_local',seed)
    a.experiment_case=variant
    a.dataset=DATASETS[variant]
    # The shared grouped-validation implementation only supports RefFLIR.
    # Target test groups are calculated by the dataset-aware evaluator.
    a.report_val_groups=False
    a.cross_source_checkpoint=str(SOURCE.resolve())
    a.cross_source_sha256=SOURCE_SHA256
    a.epochs=EPOCHS
    case_contract(a)
    return a


def build(args):
    contract=case_contract(args)
    model=query.build(_query_args(args,deep=True))
    object.__setattr__(model,'_cross_runtime_contract',copy.deepcopy(contract))
    assert_model_contract(model,args)
    return model


def assert_model_contract(model,args):
    contract=case_contract(args)
    if getattr(model,'_cross_runtime_contract',None)!=contract:
        raise ValueError('Model was built for a different target dataset or source')
    if getattr(model.student.args,'dataset',None)!=args.dataset:
        raise ValueError('Underlying model dataset differs')
    query.assert_model_contract(model,_query_args(args))
    return contract


def parameter_group(name,args):
    case_contract(args)
    return query.parameter_group(name,_query_args(args))


def selected_name(name,args):
    case_contract(args)
    return query.selected_name(name,_query_args(args))


def assert_scope(model,args):
    assert_model_contract(model,args)
    query.assert_scope(model,_query_args(args))


def initialize(model,args):
    contract=assert_model_contract(model,args)
    if sha256(args.cross_source_checkpoint)!=args.cross_source_sha256:
        raise ValueError('RefFLIR initialization source checksum differs')
    info=query.initialize(model,_query_args(args))
    source=torch.load(args.cross_source_checkpoint,map_location='cpu',weights_only=False)
    if (source['experiment_case']!='query_local' or source['args'].dataset!='rgbtvg_flir'
            or source['epoch']!=3 or source['weight_view']!='ema'):
        raise ValueError('Selected source metadata differs')
    query.restore_delta(model,_query_args(args),source)
    assert_scope(model,args)
    object.__setattr__(model,'_cross_runtime_initialization',copy.deepcopy(contract))
    return {'source_checkpoint':args.cross_source_checkpoint,'source_sha256':args.cross_source_sha256,
            'source_epoch_zero_based':3,'source_weight_view':'ema','source_dataset':'rgbtvg_flir',
            'target_dataset':args.dataset,'source_pretraining':'supervised RefFLIR warm start',
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'query_initialization':info}


def _initialized(model,args):
    contract=assert_model_contract(model,args)
    if getattr(model,'_cross_runtime_initialization',None)!=contract:
        raise ValueError('Cross-dataset source must be initialized first')


def optimizer_for(model,args):
    _initialized(model,args)
    return query.optimizer_for(model,_query_args(args))


def delta_record(model,args,epoch,validation,best,provenance):
    _initialized(model,args)
    record=query.delta_record(model,_query_args(args),epoch,validation,best,provenance)
    record.update(format=FORMAT,experiment_case=args.experiment_case,args=copy.deepcopy(args),
                  query_model_contract=record['model_contract'],model_contract=case_contract(args),
                  source_checkpoint=args.cross_source_checkpoint,source_sha256=args.cross_source_sha256)
    return record


def restore_delta(model,args,checkpoint):
    _initialized(model,args)
    if (checkpoint.get('format')!=FORMAT or checkpoint.get('experiment_case')!=args.experiment_case
            or checkpoint.get('model_contract')!=case_contract(args)
            or checkpoint.get('args') is None or case_contract(checkpoint['args'])!=case_contract(args)
            or checkpoint.get('source_checkpoint')!=args.cross_source_checkpoint
            or checkpoint.get('source_sha256')!=args.cross_source_sha256
            or checkpoint.get('query_model_contract')!=query.case_contract(_query_args(args))):
        raise ValueError('Checkpoint dataset or initialization source differs')
    if sha256(args.cross_source_checkpoint)!=args.cross_source_sha256:
        raise ValueError('Source checkpoint changed')
    converted=dict(checkpoint)
    converted.update(format=query.FORMAT,experiment_case='query_local',args=_query_args(checkpoint['args']),
                     model_contract=checkpoint['query_model_contract'])
    query.restore_delta(model,_query_args(args),converted)
    assert_scope(model,args)


def train(root,directory,variant,seed=SEED,resume=False):
    return common.train(root,directory,variant,seed,resume,runtime=sys.modules[__name__])
