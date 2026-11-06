"""C1-matched public-initialization training and target-conditioned replacement."""
import argparse
import copy
import json
from pathlib import Path

from script_train.clip_lavs_control.runtime import ORIGINAL, REFERENCE, PYTHON, sha256, write_json

REPOSITORY=Path(__file__).resolve().parents[2]
DATASETS={'m3fd':'rgbtvg_m3fd','mfad':'rgbtvg_mfad'}
METHODS=('c1','tcr_query')
EPOCHS=120
SEED=13


def recipe(directory,dataset,method):
    if dataset not in DATASETS or method not in METHODS:raise ValueError('Unknown matched case')
    frozen=REPOSITORY/'run_resources/c1_recipe.json'
    source=frozen if frozen.exists() else REFERENCE/'formal/c1_seed13/recipe.json'
    a=argparse.Namespace(**json.loads(source.read_text()))
    a.dataset=DATASETS[dataset];a.output_dir=str(Path(directory).resolve())
    a.dataset_bert_model_path=str(ORIGINAL/'dataset_and_pretrain_model/pretrain_model/pretrained_weights/Bert')
    a.epochs=EPOCHS;a.seed=SEED;a.batch_size=12;a.grad_accum_steps=4
    a.resume='';a.start_epoch=0;a.report_val_groups=False
    a.num_workers=4;a.clip_structure_experiment=True;a.clip_structure_new_only=False
    a.hi_lora_stage=0;a.match_infmae_lora_schedule=True
    a.full_match_dataset=dataset;a.full_match_method=method
    a.replace_lavs=method=='tcr_query'
    a.visual_exchange='differential_lavs' if a.replace_lavs else 'lavs'
    a.differential_enabled=False
    a.tcr_mode='bias' if a.replace_lavs else 'off'
    a.tcr_query_enabled=a.replace_lavs;a.tcr_initial_strength=.5
    a.tcr_hidden=128;a.tcr_loss_weight=.05
    if not (a.use_contrastive_loss and a.use_rtcc_constrain_loss and a.use_mask_loss):
        raise ValueError('Original C1 objectives missing')
    return a


def install_tcr(model,args):
    from models.tcr_query_lavs import QueryTargetKeyBias
    if not args.tcr_query_enabled:raise ValueError('Expected target-conditioned replacement')
    if hasattr(model,'tcr_target'):raise ValueError('TCR already installed')
    head=QueryTargetKeyBias(channels=768,text_channels=512,hidden=args.tcr_hidden,query_enabled=True)
    head.bias_strength.data.fill_(args.tcr_initial_strength)
    model.add_module('tcr_target',head)
    captured=[]
    def exchange_hook(module,inputs,output):
        result,aux=model.tcr_target(module,inputs,output,apply_bias=True)
        captured.append(aux)
        return result
    def output_hook(module,inputs,output):
        if len(captured)!=1:raise RuntimeError('Expected one target-conditioned exchange per forward')
        aux=captured.pop()
        if not module.training:return output
        if len(output)==5:return (*output,{'tcr_target':aux})
        if len(output)==6:
            extra=dict(output[5]);extra['tcr_target']=aux
            return (*output[:5],extra)
        raise ValueError('Unexpected C1 forward output')
    model.differential_exchange.register_forward_hook(exchange_hook)
    model.register_forward_hook(output_hook)
    return model


def build(args,public_migration=False):
    from models import build_model
    if args.full_match_method=='c1':return build_model(args)
    if args.full_match_method!='tcr_query':raise ValueError('Unknown case')
    if public_migration:
        # Migrate attention from an untrained C1 loaded only with the public MMVG
        # checkpoint. No RefFLIR-trained checkpoint participates.
        from models.differential_lavs import migrate_lavs_state,differential_key
        from script_train.infmae_experiments.train_clip_control import load_common_initialization
        baseline_args=copy.deepcopy(args)
        baseline_args.replace_lavs=False;baseline_args.visual_exchange='lavs'
        baseline=build_model(baseline_args)
        info=load_common_initialization(baseline,args.retrain)
        original=baseline.state_dict()
        model=build_model(args)
        inherited={n:v for n,v in original.items() if n in model.state_dict() and not differential_key(n)}
        missing,unexpected=model.load_state_dict(inherited,strict=False)
        if unexpected or any(not differential_key(n) for n in missing):
            raise ValueError(f'Public initialization migration mismatch: {missing}, {unexpected}')
        migrated=migrate_lavs_state(model,original)
        object.__setattr__(model,'_public_migration',{'common':info,'attention':migrated})
        del original,baseline
    else:model=build_model(args)
    return install_tcr(model,args)


def train(root,dataset,method,resume=False):
    import train_val.mmvg_train as trainer
    run_dir=root/'runs'/f'{dataset}_{method}_seed{SEED}'
    run_dir.mkdir(parents=True,exist_ok=True)
    args=recipe(run_dir,dataset,method)
    manifest=json.loads((root/'preregistration.json').read_text())
    if vars(args)!=manifest['recipes'][run_dir.name]:raise ValueError('Recipe differs from registration')
    if resume:
        checkpoint=run_dir/'checkpoint.pth'
        if checkpoint.exists():
            import torch
            last=torch.load(checkpoint,map_location='cpu',weights_only=False)['epoch']
            args.resume=str(checkpoint);args.retrain=''
            args.hi_lora_stage=3 if last>=100 else 2 if last>=80 else 1 if last>=60 else 0
        elif (run_dir/'log.txt').exists():raise ValueError('Log exists without recoverable checkpoint')
    if not resume and (run_dir/'log.txt').exists():raise FileExistsError('Run already exists')
    # On recovery the complete native state is restored; public migration is
    # only needed for the initial load and must not read a fine-tuned source.
    original_build=trainer.build_model
    trainer.build_model=lambda a:build(a,public_migration=(method=='tcr_query' and not bool(a.resume)))
    try:trainer.main(args)
    finally:trainer.build_model=original_build
