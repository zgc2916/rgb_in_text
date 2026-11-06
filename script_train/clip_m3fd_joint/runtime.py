"""RefM3FD 120-epoch TCR-query detail/boundary ablations."""
import json
from pathlib import Path
from script_train.clip_full_match import runtime as base
from script_train.clip_lavs_control.runtime import ORIGINAL,REFERENCE,PYTHON,sha256,write_json

REPOSITORY=Path(__file__).resolve().parents[2]
DATASETS={'m3fd':'rgbtvg_m3fd'}
METHODS=('joint',)
EPOCHS=120
SEED=13
BASELINE_ROOT=ORIGINAL/'output_clip_lavs_control/2026-11-04_full_match'
BASELINE=BASELINE_ROOT/'frozen/m3fd_tcr_query_seed13.pth'
BASELINE_SHA256='5fe061fa2ca82ba1fa9ca28620a926a630be42acd50d35c664a7a902806fbf3f'


def recipe(directory,dataset,method):
    if dataset!='m3fd' or method not in METHODS:raise ValueError('Unregistered ablation')
    a=base.recipe(directory,dataset,'tcr_query')
    a.enable_tir_detail=method in ('detail','joint')
    a.enable_boundary_refiner=method in ('boundary','joint')
    a.full_detail_variant=method
    return a


def build(args,public_migration=False):
    variant=getattr(args,'full_detail_variant',None)
    if variant not in METHODS:raise ValueError('Unregistered variant')
    if (args.enable_tir_detail,args.enable_boundary_refiner)!=(variant in ('detail','joint'),variant in ('boundary','joint')):
        raise ValueError('Ablation flags do not match')
    return base.build(args,public_migration)


def train(root,dataset,method,resume=False):
    import torch
    import train_val.mmvg_train as trainer
    run=root/'runs'/f'{dataset}_{method}_seed{SEED}';run.mkdir(parents=True,exist_ok=True)
    args=recipe(run,dataset,method);manifest=json.loads((root/'preregistration.json').read_text())
    if vars(args)!=manifest['recipes'][run.name]:raise ValueError('Recipe differs from registration')
    if resume:
        checkpoint=run/'checkpoint.pth'
        if checkpoint.exists():
            last=torch.load(checkpoint,map_location='cpu',weights_only=False)['epoch']
            args.resume=str(checkpoint);args.retrain=''
            args.hi_lora_stage=3 if last>=100 else 2 if last>=80 else 1 if last>=60 else 0
        elif (run/'log.txt').exists():raise ValueError('No checkpoint for existing log')
    elif (run/'log.txt').exists():raise FileExistsError('Run already exists')
    original=trainer.build_model
    trainer.build_model=lambda a:build(a,public_migration=not bool(a.resume))
    try:trainer.main(args)
    finally:trainer.build_model=original
