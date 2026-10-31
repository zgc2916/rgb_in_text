import argparse
import copy
import json
import math
from pathlib import Path
import time

from script_train.clip_soft_control.runtime import (
    ORIGINAL, REFERENCE, PYTHON, atomic_checkpoint, write_json, sha256,
    seed_all, rng_state, restore_rng, loaders, trim_log, recover_best,
)

REPOSITORY = Path(__file__).resolve().parents[2]
VARIANTS = ('direction_gate', 'reliability_gate', 'reliability_spatial')
ORDER = ('reliability_gate', 'direction_gate', 'reliability_spatial')
EPOCHS = 20
PREFIX = 'reliable_modal_exchange.'
FORMAT = 'frozen_c1_exchange_delta_v1'


def references():
    frozen = REPOSITORY/'run_resources/c1_references.json'
    source = frozen if frozen.exists() else REFERENCE/'selected_models.json'
    return [item for item in json.loads(source.read_text()) if item['variant'] == 'c1']


def reference(seed):
    return next(item for item in references() if item['seed'] == seed)


def recipe(directory, variant='reliability_gate', seed=13, legacy=False):
    frozen = REPOSITORY/'run_resources/c1_recipe.json'
    base = json.loads((frozen if frozen.exists() else REFERENCE/'formal/c1_seed13/recipe.json').read_text())
    a = copy.deepcopy(base)
    item = reference(seed)
    a.update(visual_exchange='lavs' if legacy else variant, exchange_train_new_only=not legacy,
        exchange_gate_dim=128, exchange_spatial_scales=[1., 2., 3.], exchange_bias_bound=2.,
        exchange_corruption_probability=.35, exchange_damage_weight=.05, exchange_preserve_weight=.005,
        exchange_base_checkpoint=item['checkpoint'], exchange_base_sha256=item['sha256'],
        dataset_bert_model_path=str(ORIGINAL/'dataset_and_pretrain_model/pretrain_model/pretrained_weights/Bert'),
        pretrained_model_path=str(ORIGINAL/'dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16'),
        seed=seed, epochs=EPOCHS, lr=1e-4, weight_decay=1e-4,
        output_dir=str(Path(directory).resolve()), resume='', start_epoch=0,
        clip_structure_experiment=True, clip_structure_new_only=False, num_workers=4,
        clip_max_norm=1., world_size=2, distributed=False, rank=0, gpu=0,
        dist_url='env://', dist_backend='nccl')
    if variant not in VARIANTS:
        raise ValueError('Unregistered experiment variant')
    return argparse.Namespace(**a)


def initialize(model, args):
    import torch
    if sha256(args.exchange_base_checkpoint) != args.exchange_base_sha256:
        raise ValueError('Frozen C1 initialization changed')
    state = torch.load(args.exchange_base_checkpoint, map_location='cpu', weights_only=False)
    missing, extra = model.load_state_dict(state['model'], strict=False)
    expected = {PREFIX + key for key in model.reliable_modal_exchange.state_dict()} if hasattr(model, 'reliable_modal_exchange') else set()
    if set(missing) != expected or extra:
        raise ValueError(f'C1 initialization mismatch: missing={missing}, extra={extra}')
    return {'base_epoch': state['epoch'], 'base_sha256': args.exchange_base_sha256,
            'new_parameters': sum(p.numel() for n,p in model.named_parameters() if n.startswith(PREFIX))}


def delta_record(model, args, epoch, validation, best, provenance):
    return {'format': FORMAT, 'model_delta': {key: value.detach().cpu().clone()
                for key,value in model.reliable_modal_exchange.state_dict().items()},
            'args': args, 'epoch': epoch, 'val_stats': validation,
            'val_accu': validation['accu'], 'val_accu_07': validation['accu_07'], 'val_miou': validation['miou'],
            'best_accu': best, 'base_checkpoint': args.exchange_base_checkpoint,
            'base_sha256': args.exchange_base_sha256, 'provenance': provenance}


def restore_delta(model, args, checkpoint):
    if (checkpoint['format'] != FORMAT or checkpoint['base_checkpoint'] != args.exchange_base_checkpoint
            or checkpoint['base_sha256'] != args.exchange_base_sha256):
        raise ValueError('Delta checkpoint belongs to another C1 initialization')
    model.reliable_modal_exchange.load_state_dict(checkpoint['model_delta'], strict=True)


def optimizer_for(model, args):
    import torch
    trainable = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not name.startswith(PREFIX) for name,_ in trainable):
        raise AssertionError('Only new exchange parameters may be optimized')
    optimizer = torch.optim.AdamW([p for _,p in trainable], lr=args.lr, weight_decay=args.weight_decay)
    def factor(epoch):
        return min(1., (epoch+1)/2) * (.1 + .45*(1+math.cos(math.pi*min(epoch,args.epochs)/args.epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler


def best_from_log(directory, epochs=EPOCHS):
    initial = json.loads((directory/'initial_validation.json').read_text())
    rows = [json.loads(x) for x in (directory/'log.txt').read_text().splitlines() if x.strip()]
    if [row['epoch'] for row in rows] != list(range(epochs)):
        raise ValueError(f'Incomplete exploration: {directory}')
    choices = [(-1, initial['accu'])] + [(row['epoch'],row['val_accu']) for row in rows]
    selected = max(choices, key=lambda pair: pair[1])[0]
    best = json.loads((directory/'best.json').read_text())
    if best['epoch'] != selected:
        raise ValueError('Best selection differs from first strict maximum including initial C1')
    return best


def choose_promotion(baseline, runs):
    checks, eligible = {}, []
    for variant in VARIANTS:
        run = runs[variant]
        hits = round(run['accu']*608) - round(baseline['accu']*608)
        quality = all(run[key] >= baseline[key]-1e-12 for key in ('accu_07','miou','miou_weak'))
        passed = run['epoch'] >= 0 and hits >= 4 and quality
        checks[variant] = {'additional_hits': hits, 'quality_non_decrease': quality, 'passed': passed,
                           'selected_initial_c1': run['epoch'] == -1}
        if passed:
            eligible.append(variant)
    selected = max(eligible, key=lambda v: (runs[v]['accu'],runs[v]['accu_07'],runs[v]['miou'])) if eligible else None
    return {'selected_variant': selected, 'checks': checks, 'test_used': False,
            'criterion': '>=4 extra validation Acc05 hits; Acc07, mIoU and weak-group mIoU do not decrease'}


def assert_frozen(model):
    if any(p.requires_grad for name,p in model.named_parameters() if not name.startswith(PREFIX)):
        raise AssertionError('Inherited C1 parameters became trainable')


def train(root, directory, variant, seed, resume=False):
    import torch
    import torch.distributed as dist
    from models import build_model
    from utils import misc
    from engine import train_one_epoch, validate
    args = recipe(directory, variant, seed)
    torch.set_num_threads(4)
    misc.init_distributed_mode(args)
    if misc.get_world_size() != 2:
        raise ValueError('This exploration requires exactly two GPUs')
    rank = misc.get_rank()
    seed_all(seed+rank)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'preregistration.json').read_text())
    expected = manifest['recipes'][f'{variant}_seed{seed}']
    if vars(recipe(directory,variant,seed)) != expected:
        raise ValueError('Recipe differs from preregistration')
    model = build_model(args)
    loading = initialize(model,args)
    latest = directory/'checkpoint.pth'
    state, start, best = None, 0, -float('inf')
    if resume and latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['provenance'] != manifest['provenance']:
            raise ValueError('Recovery provenance differs')
        restore_delta(model,args,state)
        start, best = state['epoch']+1,state['best_accu']
        if rank == 0:
            trim_log(directory,state['epoch'])
            recover_best(directory,state['epoch'])
    elif (directory/'log.txt').exists():
        raise FileExistsError('Training log exists without a recoverable checkpoint')
    device = torch.device('cuda',args.gpu)
    model.to(device)
    assert_frozen(model)
    wrapped = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
    optimizer,scheduler = optimizer_for(model,args)
    train_loader,val_loader,sampler = loaders(args)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['lr_scheduler'])
        restore_rng(state['rng_by_rank'][rank])
        del state
    if rank == 0:
        write_json(directory/'recipe.json',expected)
        write_json(directory/'initialization.json',loading)
    if not resume or not latest.exists():
        initial = validate(args,wrapped,val_loader,device)
        base = reference(seed)['validation']
        for key in ('accu','accu_07','miou','miou_weak'):
            tolerance=1e-12 if key in ('accu','accu_07') else 1e-5
            if abs(initial[key]-base[key]) > tolerance:
                raise AssertionError(f'Initial C1 validation differs: {key}, {initial[key]} vs {base[key]}')
        best = initial['accu']
        states = [None]*2
        dist.all_gather_object(states,rng_state())
        if rank == 0:
            write_json(directory/'initial_validation.json',{'included_in_selection':True,**initial})
            record = delta_record(model,args,-1,initial,best,manifest['provenance'])
            atomic_checkpoint(directory/'best_checkpoint.pth',record)
            write_json(directory/'best.json',{'epoch':-1,'is_initial_c1':True,**initial})
            record.update(optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),rng_by_rank=states)
            atomic_checkpoint(latest,record)
        dist.barrier()
    for epoch in range(start,args.epochs):
        if rank == 0:
            write_json(directory/'progress.json',{'epoch':epoch,'completed_epochs':epoch,'epoch_in_progress':True,
                'best':json.loads((directory/'best.json').read_text()),'updated_at':time.time()})
        sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        tick=time.monotonic()
        training=train_one_epoch(args,wrapped,train_loader,optimizer,device,epoch,1.)
        assert_frozen(model)
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
