"""Real two-GPU checks before any scheduled training is allowed to proceed."""
import gc
import hashlib
import json
import math
import time

from .runtime import (VARIANTS, PREFIX, REFERENCE, recipe, initialize, optimizer_for,
    seed_all, rng_state, restore_rng, delta_record, restore_delta, atomic_checkpoint, write_json, assert_frozen)


def inherited_digest(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not name.startswith(PREFIX):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(root):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset, DistributedSampler, BatchSampler
    from models import build_model
    from datasets import build_dataset
    from utils import misc
    from engine import train_one_epoch
    from utils.box_utils import bbox_iou, xywh2xyxy
    from script_train.clip_soft_control.evaluate import IndexedDataset, collate_indexed
    root.mkdir(parents=True,exist_ok=True)
    args=recipe(root/'preflight','reliability_spatial')
    misc.init_distributed_mode(args)
    torch.set_num_threads(4)
    if misc.get_world_size()!=2:
        raise ValueError('Exactly two ranks are required')
    rank=misc.get_rank()
    device=torch.device('cuda',args.gpu)
    seed_all(13+rank)
    data=build_dataset('val',args)
    batch=misc.collate_fn([data[index] for index in range(rank*12,(rank+1)*12)])
    def fresh():
        images,phrases,_,_=batch
        return misc.NestedTensor(images.tensors.clone().to(device),images.mask.clone().to(device)),phrases
    baseline=build_model(recipe(root/'preflight',legacy=True))
    initialize(baseline,args)
    baseline_count=sum(p.numel() for p in baseline.parameters())
    frozen_digest=inherited_digest(baseline)
    baseline.to(device).eval()
    with torch.no_grad():
        expected=baseline(*fresh())[0].detach().clone()
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    timings={}
    for variant in VARIANTS:
        seed_all(13+rank)
        model_args=recipe(root/'preflight',variant)
        model=build_model(model_args)
        loading=initialize(model,model_args)
        if inherited_digest(model)!=frozen_digest:
            raise AssertionError('Inherited C1 weights changed during construction')
        added=sum(p.numel() for p in model.parameters())-baseline_count
        if added>2_000_000:
            raise AssertionError('Added-parameter budget exceeded')
        assert_frozen(model)
        model.to(device).eval()
        exchange_error=[]
        def check_exchange(module,inputs,output):
            rgb,tir,rg,tg=inputs[:4]
            for own,donor,read,result in [*zip(rgb,tir,rg,output[0]),*zip(tir,rgb,tg,output[1])]:
                original=own+read(own,donor,output_attentions=True)[0]
                exchange_error.append((original-result).abs().max().item())
        hook=model.reliable_modal_exchange.register_forward_hook(check_exchange)
        with torch.no_grad():
            output=model(*fresh())
            delta=(output[0]-expected).abs().max().item()
            # Freezing C1 itself changes GPU linear-kernel rounding. The new
            # exchange must be exactly neutral; the complete frozen model has
            # a separately disclosed numerical tolerance (< .003 input pixel).
            if delta>1e-5 or max(exchange_error)>1e-6 or len(output)!=5 or not torch.isfinite(output[0]).all():
                raise AssertionError(f'{variant} initial C1 equivalence failed: {delta}')
            hook.remove()
            torch.cuda.synchronize()
            tick=time.monotonic()
            model(*fresh())
            torch.cuda.synchronize()
        timings[variant]={'max_initial_box_error':delta,'max_exchange_initial_error':max(exchange_error),
                         'net_added_parameters':added,
                         'batch12_seconds':time.monotonic()-tick,'initialization':loading}
        del model,output
        gc.collect()
        torch.cuda.empty_cache()
    seed_all(13+rank)
    model=build_model(args)
    initialize(model,args)
    model.to(device).eval()
    indexed=IndexedDataset(data)
    loader=DataLoader(indexed,batch_size=12,sampler=DistributedSampler(indexed,shuffle=False),
                      num_workers=2,collate_fn=collate_indexed)
    old_report=json.loads((REFERENCE/'evaluation/c1_seed13/val.json').read_text())
    old=old_report['samples']
    error=0.
    local_ious=[]
    with torch.no_grad():
        for indices,(images,phrases,targets,_) in loader:
            boxes=model(images.to(device),phrases)[0].cpu()
            ious=bbox_iou(xywh2xyxy(boxes).clamp(0,1),xywh2xyxy(targets))
            for index,phrase,target,box in zip(indices,phrases,targets,boxes):
                if phrase!=old[index]['phrase'] or target.tolist()!=old[index]['target_xywh']:
                    raise AssertionError('Validation identity changed')
                error=max(error,(box-torch.tensor(old[index]['pred_xywh'])).abs().max().item())
            local_ious.extend((index,value.item()) for index,value in zip(indices,ious))
    errors=[None]*2
    dist.all_gather_object(errors,error)
    if max(errors)>1e-5:
        raise AssertionError(f'Full validation fails C1 equivalence: {errors}')
    gathered_ious=[None]*2
    dist.all_gather_object(gathered_ious,local_ious)
    by_index=dict(pair for rows in gathered_ious for pair in rows)
    validation_ious=torch.tensor([by_index[index] for index in range(len(data))],dtype=torch.float64)
    counts={'correct_05':int((validation_ious>=.5).sum()),'correct_07':int((validation_ious>=.7).sum())}
    if any(counts[key]!=old_report['summary'][key] for key in counts):
        raise AssertionError(f'Freezing changed validation hit counts: {counts}')
    miou_drift=float(validation_ious.mean())-old_report['summary']['miou']
    if abs(miou_drift)>1e-5:
        raise AssertionError(f'Frozen C1 mIoU drift exceeds numerical tolerance: {miou_drift}')
    training=Subset(build_dataset('train',args),list(range(192)))
    sampler=DistributedSampler(training,shuffle=True)
    train_loader=DataLoader(training,batch_sampler=BatchSampler(sampler,12,drop_last=True),
                            num_workers=2,collate_fn=misc.collate_fn)
    wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
    optimizer,scheduler=optimizer_for(model,args)
    watch={name:p.detach().clone() for name,p in model.named_parameters() if name.startswith(PREFIX)
           and ('.direction_heads.' in name and name.endswith('weight')
                or '.damage_heads.' in name and name.endswith('net.3.weight')
                or name.endswith('text_proj.weight'))}
    gradients,hooks={},[]
    for name,parameter in model.named_parameters():
        if name in watch:
            def record(gradient,name=name):
                gradients[name]=gradients.get(name,0.)+gradient.detach().abs().sum().item()
            hooks.append(parameter.register_hook(record))
    torch.cuda.reset_peak_memory_stats()
    stats=train_one_epoch(args,wrapped,train_loader,optimizer,device,0,1.)
    scheduler.step()
    for hook in hooks:
        hook.remove()
    sums={branch:sum(value for name,value in gradients.items() if branch in name)
          for branch in ('.direction_heads.0.','.direction_heads.1.','.damage_heads.0.','.damage_heads.1.','text_proj')}
    if not all(value>0 and math.isfinite(value) for value in sums.values()):
        raise AssertionError(f'Both directions, both damage heads and text need gradients: {sums}')
    steps={int(state['step'].item()) for state in optimizer.state.values() if 'step' in state}
    if steps!={2}:
        raise AssertionError(f'Expected two effective-batch96 updates, got {steps}')
    updated={branch:[name for name,before in watch.items() if branch in name
             and not torch.equal(before,dict(model.named_parameters())[name])] for branch in sums}
    if not all(updated.values()) or any(not torch.isfinite(p).all() for p in model.reliable_modal_exchange.parameters()):
        raise AssertionError('Required weights did not update or are nonfinite')
    assert_frozen(model)
    if model.clip.training or model.vl_transformer.training or not model.reliable_modal_exchange.training:
        raise AssertionError('Frozen trunk/dropout mode changed')
    if inherited_digest(model)!=frozen_digest:
        raise AssertionError('Training changed inherited C1 weights')
    manifest=json.loads((root/'preregistration.json').read_text()) if (root/'preregistration.json').exists() else {'provenance':{'manual_preflight':True}}
    states=[None]*2
    dist.all_gather_object(states,rng_state())
    probe=root/'preflight/recovery_probe.pth'
    if rank==0:
        validation={'accu':.5,'accu_07':.3,'miou':.4}
        checkpoint=delta_record(model,args,0,validation,.5,manifest['provenance'])
        checkpoint.update(optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),rng_by_rank=states)
        atomic_checkpoint(probe,checkpoint)
    dist.barrier()
    checkpoint=torch.load(probe,map_location='cpu',weights_only=False)
    before={name:value.detach().cpu().clone() for name,value in model.reliable_modal_exchange.state_dict().items()}
    # Exercise a real restore, including a deliberately modified parameter.
    with torch.no_grad():
        next(model.reliable_modal_exchange.parameters()).add_(1.)
    restore_delta(model,args,checkpoint)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    restore_rng(checkpoint['rng_by_rank'][rank])
    draw=torch.rand(16,device=device)
    restore_rng(states[rank])
    if not torch.equal(draw,torch.rand(16,device=device)):
        raise AssertionError('Per-rank CUDA RNG restore failed')
    if any(not torch.equal(value,model.reliable_modal_exchange.state_dict()[name].cpu()) for name,value in before.items()):
        raise AssertionError('Compact checkpoint restore failed')
    ranks=[None]*2
    dist.all_gather_object(ranks,{'rank':rank,'gradient_l1':sums,'updated_parameters':updated,
        'optimizer_updates':2,'frozen_weights_unchanged':True,'trunk_eval':True,
        'compact_checkpoint_rng_restored':True,'peak_memory_bytes':torch.cuda.max_memory_allocated()})
    if rank==0:
        write_json(root/'preflight.json',{'passed':True,'baseline_parameters':baseline_count,
            'variants':timings,'full_c1_validation_max_box_error':max(errors),'validation_examples':608,
            'full_model_numerical_tolerance':1e-5,'exact_validation_hit_counts':counts,'initial_miou_drift':miou_drift,
            'two_gpu_update':ranks,'training_metrics':stats})
        probe.unlink()
    dist.barrier()
    dist.destroy_process_group()
