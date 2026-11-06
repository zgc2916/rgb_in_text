"""Ten-epoch matched RefFLIR warm-start screening, validation only."""
import json
from pathlib import Path
from script_train.clip_full_match import runtime as base
from script_train.clip_lavs_control.runtime import ORIGINAL,REFERENCE,PYTHON,sha256,write_json

REPOSITORY=Path(__file__).resolve().parents[2]
DATASETS={'flir':'rgbtvg_flir'}
METHODS=('joint','tcr','detail','boundary')
EPOCHS=10
SEED=13
SOURCE=ORIGINAL/'output_clip_lavs_control/2026-11-03_tcr_query/explore/query_local_seed13/best_checkpoint.pth'
SOURCE_SHA='0910cf34df768d7cb9a196cd295f4f77c61db17ca78bd70a2b41191fefedac2f'


def recipe(directory,dataset,method):
    if dataset!='flir' or method not in METHODS:raise ValueError('Unregistered screen')
    a=base.recipe(directory,'m3fd','tcr_query')
    a.dataset=DATASETS[dataset];a.full_match_dataset='flir';a.full_detail_variant=method
    a.enable_tir_detail=method in ('detail','joint')
    a.enable_boundary_refiner=method in ('boundary','joint')
    a.epochs=EPOCHS;a.batch_size=4;a.grad_accum_steps=12;a.num_workers=2
    a.lr=1e-5;a.retrain='';a.report_val_groups=False
    a.screen_source=str(SOURCE);a.screen_source_sha=SOURCE_SHA
    return a


def build(args,initialize=False):
    import torch
    from script_train.clip_tcr_query_control import runtime as q
    if args.dataset!='rgbtvg_flir' or args.full_detail_variant not in METHODS:raise ValueError('Invalid screening case')
    variant=args.full_detail_variant
    if (args.enable_tir_detail,args.enable_boundary_refiner)!=(variant in ('detail','joint'),variant in ('boundary','joint')):
        raise ValueError('Screen flags differ')
    model=base.build(args,public_migration=False)
    if not initialize:return model
    if sha256(SOURCE)!=SOURCE_SHA:raise ValueError('TCR source changed')
    checkpoint=torch.load(SOURCE,map_location='cpu',weights_only=False)
    qa=checkpoint['args']
    if checkpoint['epoch']!=3 or checkpoint['weight_view']!='ema' or qa.dataset!='rgbtvg_flir':
        raise ValueError('Unexpected TCR source')
    trained=q.build(qa);q.initialize(trained,qa);q.restore_delta(trained,qa,checkpoint)
    source=trained.state_dict();mapped={}
    for name,value in source.items():
        if name.startswith('student.base.'):target=name[len('student.base.'):]
        elif name.startswith('tcr_target.'):target=name
        else:raise ValueError(f'Unexpected trained source key {name}')
        mapped[target]=value
    target_state=model.state_dict()
    extra=set(mapped)-set(target_state)
    missing=set(target_state)-set(mapped)
    allowed=('tir_detail.','tir_injector.','tir_region_head.','boundary_refiner.')
    if extra or any(not name.startswith(allowed) for name in missing):
        raise ValueError(f'Warm-start schema mismatch missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}')
    for name,value in mapped.items():
        if target_state[name].shape!=value.shape or target_state[name].dtype!=value.dtype:
            raise ValueError(f'Warm-start shape mismatch: {name}')
    model.load_state_dict(mapped,strict=False)
    object.__setattr__(model,'_screen_loaded',True)
    return model


def train(root,dataset,method,resume=False):
    import torch
    import train_val.mmvg_train as trainer
    run=root/'runs'/f'{dataset}_{method}_seed{SEED}';run.mkdir(parents=True,exist_ok=True)
    args=recipe(run,dataset,method);manifest=json.loads((root/'preregistration.json').read_text())
    if vars(args)!=manifest['recipes'][run.name]:raise ValueError('Recipe changed')
    if resume:
        ck=run/'checkpoint.pth'
        if ck.exists():args.resume=str(ck)
        elif (run/'log.txt').exists():raise ValueError('No checkpoint for existing log')
    elif (run/'log.txt').exists():raise FileExistsError('Run already exists')
    original=trainer.build_model
    trainer.build_model=lambda a:build(a,initialize=not bool(a.resume))
    try:trainer.main(args)
    finally:trainer.build_model=original
