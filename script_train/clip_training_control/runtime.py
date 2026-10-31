import json
import math
from pathlib import Path
import time

from script_train.clip_reliability_control.runtime import (
    recipe as previous_recipe, reference, references, initialize,
    ORIGINAL, REFERENCE, PYTHON, seed_all, rng_state, restore_rng,
    atomic_checkpoint, write_json, sha256, loaders, trim_log, recover_best, best_from_log,
)

REPOSITORY=Path(__file__).resolve().parents[2]
VARIANTS=('a_clean_full','b_clean_box','c_joint_box','d_c1_continue')
ORDER=('a_clean_full','b_clean_box','d_c1_continue','c_joint_box')
EPOCHS=20
NEW_PREFIX='reliable_modal_exchange.'
PROJECTION_PREFIXES=('cross_fusion_layers_vt.','cross_fusion_layers_tv.')
FORMAT='c1_selective_tuning_delta_v1'


def recipe(directory,variant='a_clean_full',seed=13,legacy=False):
    if variant not in VARIANTS:
        raise ValueError('Unregistered training control')
    a=previous_recipe(directory,'direction_gate',seed,legacy=legacy or variant=='d_c1_continue')
    a.experiment_case=variant
    a.exchange_projection_tuning=not legacy and variant in ('c_joint_box','d_c1_continue')
    a.exchange_corruption_probability=0.
    a.exchange_damage_weight=0.
    a.projection_lr=1e-5
    a.use_contrastive_loss=a.use_rtcc_constrain_loss=a.use_mask_loss=variant=='a_clean_full'
    return a


def selected_name(name,args):
    return ((args.exchange_train_new_only and name.startswith(NEW_PREFIX))
            or (args.exchange_projection_tuning and name.startswith(PROJECTION_PREFIXES)))


def delta_keys(model,args):
    return {name for name in model.state_dict() if selected_name(name,args)}


def assert_scope(model,args):
    wrong=[name for name,p in model.named_parameters() if p.requires_grad!=selected_name(name,args)]
    if wrong:
        raise AssertionError(f'Unexpected tuning mask: {wrong[:12]}')


def delta_record(model,args,epoch,validation,best,provenance):
    return {'format':FORMAT,'model_delta':{name:value.detach().cpu().clone()
                for name,value in model.state_dict().items() if selected_name(name,args)},
        'args':args,'epoch':epoch,'val_stats':validation,'val_accu':validation['accu'],
        'val_accu_07':validation['accu_07'],'val_miou':validation['miou'],'best_accu':best,
        'base_checkpoint':args.exchange_base_checkpoint,'base_sha256':args.exchange_base_sha256,
        'provenance':provenance,'experiment_case':args.experiment_case}


def restore_delta(model,args,checkpoint):
    if (checkpoint['format']!=FORMAT or checkpoint['base_checkpoint']!=args.exchange_base_checkpoint
            or checkpoint['base_sha256']!=args.exchange_base_sha256
            or checkpoint['experiment_case']!=args.experiment_case):
        raise ValueError('Delta checkpoint belongs to another C1/control configuration')
    expected=delta_keys(model,args)
    if set(checkpoint['model_delta'])!=expected:
        raise ValueError('Selective delta has missing or unexpected parameters')
    missing,extra=model.load_state_dict(checkpoint['model_delta'],strict=False)
    if set(missing)!=(set(model.state_dict())-expected) or extra:
        raise ValueError('Selective delta restoration changed the base/delta boundary')


def optimizer_for(model,args):
    import torch
    assert_scope(model,args)
    groups=[]
    if args.exchange_projection_tuning:
        groups.append({'params':[p for n,p in model.named_parameters() if n.startswith(PROJECTION_PREFIXES)],
                       'lr':args.projection_lr,'name':'existing_external_projections'})
    if args.exchange_train_new_only:
        groups.append({'params':[p for n,p in model.named_parameters() if n.startswith(NEW_PREFIX)],
                       'lr':args.lr,'name':'new_directional_gate'})
    optimizer=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    coverage={id(p) for group in groups for p in group['params']}
    if coverage!={id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer coverage differs from the tuning mask')
    def factor(epoch):
        return min(1.,(epoch+1)/2)*(.1+.45*(1+math.cos(math.pi*min(epoch,args.epochs)/args.epochs)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    return optimizer,scheduler


def dominates_quality(candidate,baseline):
    return all(candidate[k]>=baseline[k]-1e-7 for k in ('accu_07','miou','miou_weak'))


def choose_promotion(baseline,runs):
    joint,control=runs['c_joint_box'],runs['d_c1_continue']
    def hits(a,b):
        return round(a['accu']*608)-round(b['accu']*608)
    control_gain=hits(control,baseline)
    joint_gain=hits(joint,baseline)
    paired_gain=hits(joint,control)
    control_pass=control['epoch']>=0 and control_gain>=4 and dominates_quality(control,baseline)
    joint_pass=(joint['epoch']>=0 and joint_gain>=4 and paired_gain>=1
                and dominates_quality(joint,baseline) and dominates_quality(joint,control))
    return {'joint_passed':joint_pass,'control_passed':control_pass,
        'selected_variant':'c_joint_box' if joint_pass else ('d_c1_continue' if control_pass else None),
        'replicate_variants':['d_c1_continue','c_joint_box'] if joint_pass else (['d_c1_continue'] if control_pass else []),
        'additional_hits':{'joint_vs_c1':joint_gain,'control_vs_c1':control_gain,'joint_vs_control':paired_gain},
        'a_b_role':'diagnose augmentation and auxiliary losses; no automatic selection across post-hoc variants',
        'test_used':False}


def train(root,directory,variant,seed,resume=False):
    import torch
    import torch.distributed as dist
    from models import build_model
    from utils import misc
    from engine import train_one_epoch,validate
    args=recipe(directory,variant,seed)
    torch.set_num_threads(4)
    misc.init_distributed_mode(args)
    if misc.get_world_size()!=2:
        raise ValueError('Exactly two GPUs are required')
    rank=misc.get_rank()
    seed_all(seed+rank)
    directory.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'preregistration.json').read_text())
    expected=manifest['recipes'][f'{variant}_seed{seed}']
    if vars(recipe(directory,variant,seed))!=expected:
        raise ValueError('Recipe changed after registration')
    model=build_model(args)
    loading=initialize(model,args)
    latest=directory/'checkpoint.pth'
    state,start,best=None,0,-float('inf')
    if resume and latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['provenance']!=manifest['provenance']:
            raise ValueError('Checkpoint provenance differs')
        restore_delta(model,args,state)
        start,best=state['epoch']+1,state['best_accu']
        if rank==0:
            trim_log(directory,state['epoch'])
            recover_best(directory,state['epoch'])
    elif (directory/'log.txt').exists():
        raise FileExistsError('Training log exists without a valid recovery checkpoint')
    device=torch.device('cuda',args.gpu)
    model.to(device)
    assert_scope(model,args)
    wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
    optimizer,scheduler=optimizer_for(model,args)
    train_loader,val_loader,sampler=loaders(args)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['lr_scheduler'])
        restore_rng(state['rng_by_rank'][rank])
        del state
    if rank==0:
        write_json(directory/'recipe.json',expected)
        write_json(directory/'initialization.json',{**loading,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'optimizer_groups':[{'name':g['name'],'lr':g['initial_lr'],'parameters':sum(p.numel() for p in g['params'])}
                                for g in optimizer.param_groups]})
    if not resume or not latest.exists():
        initial=validate(args,wrapped,val_loader,device)
        base=reference(seed)['validation']
        for key in ('accu','accu_07','miou','miou_weak'):
            tolerance=1e-12 if key in ('accu','accu_07') else 1e-5
            if abs(initial[key]-base[key])>tolerance:
                raise AssertionError(f'Initial C1 validation differs: {key}')
        best=initial['accu']
        states=[None]*2
        dist.all_gather_object(states,rng_state())
        if rank==0:
            write_json(directory/'initial_validation.json',{'included_in_selection':True,**initial})
            record=delta_record(model,args,-1,initial,best,manifest['provenance'])
            atomic_checkpoint(directory/'best_checkpoint.pth',record)
            write_json(directory/'best.json',{'epoch':-1,'is_initial_c1':True,**initial})
            record.update(optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),rng_by_rank=states)
            atomic_checkpoint(latest,record)
        dist.barrier()
    for epoch in range(start,args.epochs):
        if rank==0:
            write_json(directory/'progress.json',{'epoch':epoch,'completed_epochs':epoch,'epoch_in_progress':True,
                'best':json.loads((directory/'best.json').read_text()),'updated_at':time.time()})
        sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        tick=time.monotonic()
        training=train_one_epoch(args,wrapped,train_loader,optimizer,device,epoch,1.)
        assert_scope(model,args)
        scheduler.step()
        validation=validate(args,wrapped,val_loader,device)
        improved=validation['accu']>best
        best=max(best,validation['accu'])
        states=[None]*2
        dist.all_gather_object(states,rng_state())
        if rank==0:
            record=delta_record(model,args,epoch,validation,best,manifest['provenance'])
            row={'epoch':epoch,**{f'train_{k}':v for k,v in training.items()},
                **{f'val_{k}':v for k,v in validation.items()},'seconds':time.monotonic()-tick,
                'peak_memory_bytes':torch.cuda.max_memory_allocated()}
            with (directory/'log.txt').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if improved:
                (directory/'best_checkpoint.pth').replace(directory/'previous_best_checkpoint.pth')
                (directory/'best.json').replace(directory/'previous_best.json')
                atomic_checkpoint(directory/'best_checkpoint.pth',record)
                write_json(directory/'best.json',{'epoch':epoch,'is_initial_c1':False,**validation})
            record.update(optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),rng_by_rank=states)
            atomic_checkpoint(latest,record)
            for name in ('previous_best_checkpoint.pth','previous_best.json'):
                (directory/name).unlink(missing_ok=True)
            write_json(directory/'progress.json',{'epoch':epoch,'completed_epochs':epoch+1,'epoch_in_progress':False,
                'best':json.loads((directory/'best.json').read_text()),'updated_at':time.time(),
                'peak_memory_bytes':row['peak_memory_bytes']})
        dist.barrier()
    dist.destroy_process_group()
