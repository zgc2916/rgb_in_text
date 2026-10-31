"""Real two-GPU acceptance for C1 and all regional-anchor controls."""
import gc
import hashlib
import json
import math

from .runtime import (VARIANTS, NEW_PREFIX, PROJECTION_PREFIXES, REFERENCE, recipe, initialize,
    assert_scope, optimizer_for, seed_all, rng_state, restore_rng, delta_record, restore_delta,
    selected_name, atomic_checkpoint, write_json)


def frozen_digest(model,args):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not selected_name(name,args):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(root):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,DistributedSampler,Subset,BatchSampler
    from models import build_model
    from datasets import build_dataset
    from utils import misc
    from utils.box_utils import bbox_iou,xywh2xyxy
    from engine import train_one_epoch
    from script_train.clip_soft_control.evaluate import IndexedDataset,collate_indexed
    root.mkdir(parents=True,exist_ok=True)
    args=recipe(root/'preflight')
    misc.init_distributed_mode(args)
    torch.set_num_threads(4)
    if misc.get_world_size()!=2:
        raise ValueError('Exactly two GPU ranks required')
    rank=misc.get_rank()
    device=torch.device('cuda',args.gpu)
    old=json.loads((REFERENCE/'evaluation/c1_seed13/val.json').read_text())
    all_results={}
    for variant in VARIANTS:
        seed_all(13+rank)
        a=recipe(root/'preflight',variant)
        a.gpu=args.gpu
        a.rank=rank
        a.distributed=True
        model=build_model(a)
        loading=initialize(model,a)
        assert_scope(model,a)
        before_digest=frozen_digest(model,a)
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.to(device).eval()
        data=build_dataset('val',a)
        indexed=IndexedDataset(data)
        loader=DataLoader(indexed,batch_size=12,sampler=DistributedSampler(indexed,shuffle=False),
                          num_workers=2,collate_fn=collate_indexed)
        error,local=0.,[]
        with torch.no_grad():
            for indices,(images,phrases,targets,_) in loader:
                result=model(images.to(device),phrases)
                if len(result)!=5:
                    raise AssertionError('Evaluation changed the five-value API')
                boxes=result[0].cpu()
                values=bbox_iou(xywh2xyxy(boxes).clamp(0,1),xywh2xyxy(targets))
                for index,phrase,target,box,value in zip(indices,phrases,targets,boxes,values):
                    if phrase!=old['samples'][index]['phrase'] or target.tolist()!=old['samples'][index]['target_xywh']:
                        raise AssertionError('Validation annotation identity changed')
                    error=max(error,(box-torch.tensor(old['samples'][index]['pred_xywh'])).abs().max().item())
                    local.append((index,value.item()))
        gathered=[None]*2
        dist.all_gather_object(gathered,{'error':error,'ious':local})
        maximum=max(item['error'] for item in gathered)
        by_index=dict(pair for item in gathered for pair in item['ious'])
        values=torch.tensor([by_index[index] for index in range(len(data))],dtype=torch.float64)
        counts={'correct_05':int((values>=.5).sum()),'correct_07':int((values>=.7).sum())}
        drift=values.mean().item()-old['summary']['miou']
        if maximum>1e-5 or abs(drift)>1e-5 or any(counts[k]!=old['summary'][k] for k in counts):
            raise AssertionError(f'{variant} initial validation differs: {maximum}, {counts}, {drift}')
        # Two genuine effective-batch96 updates under the real data/loss path.
        training=Subset(build_dataset('train',a),list(range(192)))
        sampler=DistributedSampler(training,shuffle=True)
        train_loader=DataLoader(training,batch_sampler=BatchSampler(sampler,12,drop_last=True),
                                num_workers=2,collate_fn=misc.collate_fn)
        wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
        optimizer,scheduler=optimizer_for(model,a)
        watched={name:p.detach().clone() for name,p in model.named_parameters()
            if selected_name(name,a) and ((name.startswith(NEW_PREFIX) and name.endswith('weight'))
                or name.endswith(('q_proj.weight','k_proj.weight','v_proj.weight','out_proj.weight')))}
        gradients,hooks={},[]
        for name,p in model.named_parameters():
            if name in watched:
                def record(grad,name=name):
                    gradients[name]=gradients.get(name,0.)+grad.detach().abs().sum().item()
                hooks.append(p.register_hook(record))
        torch.cuda.reset_peak_memory_stats()
        stats=train_one_epoch(a,wrapped,train_loader,optimizer,device,0,1.)
        scheduler.step()
        for hook in hooks:
            hook.remove()
        assert_scope(model,a)
        if model.clip.training or model.vl_transformer.training or model.cross_fusion_layers_vt[0].training:
            raise AssertionError('Original modules must keep deterministic eval behavior')
        if frozen_digest(model,a)!=before_digest:
            raise AssertionError('Training changed a parameter outside the allowed scope')
        branches=[]
        if a.exchange_train_new_only:
            branches.extend((NEW_PREFIX+'rgb_output.', NEW_PREFIX+'tir_output.', NEW_PREFIX+'text_proj.' if variant != 'anchor_visual' else NEW_PREFIX+'visual_proj.'))
        if a.exchange_projection_tuning:
            branches.extend(PROJECTION_PREFIXES)
        gradient_sums={branch:sum(value for name,value in gradients.items() if branch in name) for branch in branches}
        if not all(value>0 and math.isfinite(value) for value in gradient_sums.values()):
            raise AssertionError(f'Missing required gradients: {gradient_sums}')
        updates={branch:[name for name,before in watched.items() if branch in name
                  and not torch.equal(before,dict(model.named_parameters())[name])] for branch in branches}
        if not all(updates.values()):
            raise AssertionError('Required weights did not update')
        steps={int(s['step'].item()) for s in optimizer.state.values() if 'step' in s}
        if steps!={2}:
            raise AssertionError(f'Expected exactly two optimizer updates: {steps}')
        if any(key in stats for key in ['loss_seg_focal','loss_seg_dice','loss_align_dice','loss_algin_focal','loss_contrastive']):
            raise AssertionError('Box-only control accidentally retained original auxiliary losses')
        if 'loss_exchange_damage' in stats:
            raise AssertionError('Synthetic damage supervision must be disabled')
        states=[None]*2
        dist.all_gather_object(states,rng_state())
        probe=root/'preflight'/f'{variant}_recovery_probe.pth'
        if rank==0:
            record=delta_record(model,a,0,{'accu':.5,'accu_07':.3,'miou':.4},.5,{'preflight':True})
            record.update(optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),rng_by_rank=states)
            atomic_checkpoint(probe,record)
        dist.barrier()
        checkpoint=torch.load(probe,map_location='cpu',weights_only=False)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if selected_name(name,a):
                    p.add_(1.)
        restore_delta(model,a,checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        restore_rng(checkpoint['rng_by_rank'][rank])
        draw=torch.rand(16,device=device)
        restore_rng(states[rank])
        if not torch.equal(draw,torch.rand(16,device=device)):
            raise AssertionError('Per-rank CUDA RNG recovery changed')
        if any(not torch.equal(value,model.state_dict()[name].cpu()) for name,value in checkpoint['model_delta'].items()):
            raise AssertionError('Selective delta restoration failed')
        ranked=[None]*2
        dist.all_gather_object(ranked,{'rank':rank,'gradient_l1':gradient_sums,'updated_parameters':updates,
            'optimizer_updates':2,'frozen_parameters_unchanged':True,'checkpoint_rng_restored':True,
            'peak_memory_bytes':torch.cuda.max_memory_allocated()})
        all_results[variant]={'initialization':loading,'max_initial_box_error':maximum,
            'initial_hit_counts':counts,'miou_drift':drift,'trainable_parameters':trainable,
            'optimizer_groups':[{'name':g['name'],'lr':g['initial_lr']} for g in optimizer.param_groups],
            'training_metrics':stats,'two_gpu_update':ranked}
        if rank==0:
            write_json(root/'preflight/progress.json',{'completed_variants':list(all_results)})
            probe.unlink()
            print(f'Preflight passed: {variant}, trainable={trainable}, initial max error={maximum}',flush=True)
        dist.barrier()
        del checkpoint,model,wrapped,optimizer,scheduler,loader,train_loader,watched,training,data,indexed,result
        gc.collect()
        torch.cuda.empty_cache()
    if rank==0:
        write_json(root/'preflight.json',{'passed':True,'variants':all_results,'validation_examples_per_case':608,
            'corruption_probability':0.,'backbone_eval':True,'no_test_evaluation':True})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
