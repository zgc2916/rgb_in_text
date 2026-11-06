"""Two-GPU acceptance of complete LAVS removal, teacher isolation and learned sampling."""
import gc
import hashlib
import json
import math
import time

from .runtime import (VARIANTS, SEEDS, NEW_PREFIX, BASE_PREFIXES, REFERENCE, recipe, initialize,
    assert_scope, optimizer_for, seed_all, rng_state, restore_rng, delta_record, restore_delta,
    selected_name, atomic_checkpoint, write_json, build, teacher_for)


def frozen_digest(model,args):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not selected_name(name,args):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_l1(gradient):
    return 0. if gradient is None else gradient.detach().abs().sum().item()


def run(root):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,DistributedSampler,Subset,BatchSampler
    from .data import dataset as NativeDataset
    from utils.misc import collate_fn as collate_native
    from models.lavs_replacement import legacy_lavs_key
    from utils.parameter_ema import ParameterEMA
    from utils import misc
    from utils.box_utils import bbox_iou,xywh2xyxy
    from engine import train_one_epoch
    from script_train.clip_soft_control.evaluate import IndexedDataset
    def collate_indexed(rows):
        indices,values=zip(*rows)
        return indices,collate_native(values)
    root.mkdir(parents=True,exist_ok=True)
    args=recipe(root/'preflight')
    misc.init_distributed_mode(args)
    torch.set_num_threads(4)
    if misc.get_world_size()!=2:
        raise ValueError('Exactly two GPU ranks required')
    rank=misc.get_rank()
    device=torch.device('cuda',args.gpu)
    all_results={}
    fixed_predictions = None
    for variant in VARIANTS:
        seed = 13
        key = f'{variant}_seed{seed}'
        old=json.loads((REFERENCE/f'evaluation/c1_seed{seed}/val.json').read_text())
        seed_all(seed+rank)
        a=recipe(root/'preflight',variant,seed)
        a.gpu=args.gpu
        a.rank=rank
        a.distributed=True
        model=build(a)
        loading=initialize(model,a)
        assert_scope(model,a)
        before_digest=frozen_digest(model,a)
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if a.replace_lavs:
            if any(legacy_lavs_key(n) for n,_ in model.base.named_parameters()):
                raise AssertionError('Student retains original LAVS parameters')
            if len(model.base.cross_fusion_layers_vt) or len(model.base.cross_fusion_layers_tv):
                raise AssertionError('Student retains original external attention')
        model.to(device).eval()
        data=NativeDataset('val',a)
        indexed=IndexedDataset(data)
        loader=DataLoader(indexed,batch_size=12,sampler=DistributedSampler(indexed,shuffle=False),
                          num_workers=2,collate_fn=collate_indexed)
        error,local=0.,[]
        local_predictions=[]
        with torch.no_grad():
            for indices,(images,phrases,targets,_) in loader:
                result=model(images.to(device),phrases)
                if len(result)!=5:
                    raise AssertionError('Evaluation changed the five-value API')
                boxes=result[0].cpu()
                values=bbox_iou(xywh2xyxy(boxes).clamp(0,1),xywh2xyxy(targets))
                for index,phrase,target,box,value in zip(indices,phrases,targets,boxes,values):
                    if phrase!=old['samples'][index]['phrase'] or (variant=='lavs_control' and target.tolist()!=old['samples'][index]['target_xywh']):
                        raise AssertionError('Validation annotation identity changed')
                    error=max(error,(box-torch.tensor(old['samples'][index]['pred_xywh'])).abs().max().item())
                    local.append((index,value.item()))
                    local_predictions.append((index,box.tolist()))
        gathered=[None]*2
        dist.all_gather_object(gathered,{'error':error,'ious':local,'boxes':local_predictions})
        maximum=max(item['error'] for item in gathered)
        by_index=dict(pair for item in gathered for pair in item['ious'])
        values=torch.tensor([by_index[index] for index in range(len(data))],dtype=torch.float64)
        counts={'correct_05':int((values>=.5).sum()),'correct_07':int((values>=.7).sum())}
        drift=values.mean().item()-old['summary']['miou']
        predictions=dict(pair for item in gathered for pair in item['boxes'])
        if variant=='fixed_exchange':
            fixed_predictions=predictions
        if variant=='deformable_exchange':
            zero_gate_error=max((torch.tensor(box)-torch.tensor(fixed_predictions[i])).abs().max().item() for i,box in predictions.items())
            if zero_gate_error>1e-6:
                raise AssertionError(f'Fixed/learned exchange zero initialization differs: {zero_gate_error}')
        if variant=='lavs_control' and (maximum>1e-5 or abs(drift)>1e-5 or any(counts[k]!=old['summary'][k] for k in counts)):
            raise AssertionError(f'{variant} initial validation differs: {maximum}, {counts}, {drift}')
        benchmark_images=images.to(device)
        with torch.no_grad():
            for _ in range(3):
                model(benchmark_images,phrases)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick=time.perf_counter()
            for _ in range(10):
                model(benchmark_images,phrases)
            torch.cuda.synchronize()
            inference_ms=(time.perf_counter()-tick)*1000/10
            inference_peak=torch.cuda.max_memory_allocated()
        # Two genuine effective-batch96 updates under the real data/loss path.
        training=Subset(NativeDataset('train',a),list(range(192)))
        sampler=DistributedSampler(training,shuffle=True)
        train_loader=DataLoader(training,batch_sampler=BatchSampler(sampler,a.batch_size,drop_last=True),
                                num_workers=2,collate_fn=collate_native)
        wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
        optimizer,scheduler=optimizer_for(model,a)
        ema=ParameterEMA(model,[n for n,p in model.named_parameters() if p.requires_grad],a.ema_decay)
        ema_hook=optimizer.register_step_post_hook(lambda *_: ema.update(model))
        watched={name:p.detach().clone() for name,p in model.named_parameters()
                 if selected_name(name,a)}
        gradients,hooks={},[]
        for name,p in model.named_parameters():
            if name in watched:
                def record(grad,name=name):
                    gradients[name]=gradients.get(name,0.)+gradient_l1(grad)
                hooks.append(p.register_hook(record))
        torch.cuda.reset_peak_memory_stats()
        teacher=teacher_for(a,device)
        teacher_before={n:t.detach().cpu().clone() for n,t in teacher.state_dict().items()}
        stats=train_one_epoch(a,wrapped,train_loader,optimizer,device,0,1.,teacher_model=teacher)
        if 'loss_teacher_preserve' not in stats or not 0 < stats['teacher_preserve_fraction'] <= 1:
            raise AssertionError('Common training-only preservation loss did not run')
        if any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
            raise AssertionError('Training-only teacher received gradients')
        if any(not torch.equal(t,teacher.state_dict()[n].cpu()) for n,t in teacher_before.items()):
            raise AssertionError('Teacher weights changed')
        scheduler.step()
        if ema.updates != 2:
            raise AssertionError('EMA must update once per effective optimizer step')
        raw = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        with ema.apply(model):
            if not any(not torch.equal(raw[n],p) for n,p in model.named_parameters() if n in raw):
                raise AssertionError('EMA weights did not diverge from the trained raw weights')
        if any(not torch.equal(raw[n],p) for n,p in model.named_parameters() if n in raw):
            raise AssertionError('EMA validation failed to restore raw training parameters')
        del raw
        for hook in hooks:
            hook.remove()
        assert_scope(model,a)
        if model.base.clip.training or model.base.vl_transformer.training:
            raise AssertionError('Original modules must keep deterministic eval behavior')
        if frozen_digest(model,a)!=before_digest:
            raise AssertionError('Training changed a parameter outside the allowed scope')
        branches=['base.iafv3_token_gate.',
                  'base.clip.base_model.model.vision_model.encoder.layers.8.self_attn.q_proj.lora_A.lora_rgb.',
                  'base.clip.base_model.model.vision_model.encoder.layers.8.self_attn.q_proj.lora_A.lora_ir.']
        branches.extend(BASE_PREFIXES[:2])
        if a.replace_lavs:
            branches.extend(['.word_conditioner.output.', 'base.replacement_exchange.outputs.'])
            if not a.replacement_fixed_offsets:
                branches.append('base.replacement_exchange.offsets.')
        else:
            branches.append('base.cross_fusion_layers_vt.')
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
        probe=root/'preflight'/f'{key}_recovery_probe.pth'
        if rank==0:
            record=delta_record(model,a,0,{'accu':.5,'accu_07':.3,'miou':.4},.5,{'preflight':True})
            record.update(optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),rng_by_rank=states,ema_state=ema.state_dict())
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
        saved_ema=ema.state_dict()
        ema.shadow[next(iter(ema.shadow))].add_(1.)
        ema.load_state_dict(checkpoint['ema_state'])
        if ema.updates!=2 or any(not torch.equal(saved_ema['shadow'][n],v.cpu()) for n,v in ema.shadow.items()):
            raise AssertionError('EMA checkpoint restoration failed')
        restore_rng(checkpoint['rng_by_rank'][rank])
        draw=torch.rand(16,device=device)
        restore_rng(states[rank])
        if not torch.equal(draw,torch.rand(16,device=device)):
            raise AssertionError('Per-rank CUDA RNG recovery changed')
        if any(not torch.equal(value,model.state_dict()[name].cpu()) for name,value in checkpoint['model_delta'].items()):
            raise AssertionError('Selective delta restoration failed')
        ranked=[None]*2
        dist.all_gather_object(ranked,{'rank':rank,'gradient_l1':gradient_sums,'updated_parameters':updates,
            'optimizer_updates':2,'inference_ms':inference_ms,
            'inference_batch_size':benchmark_images.tensors.shape[0],'inference_peak_bytes':inference_peak,
            'frozen_parameters_unchanged':True,'checkpoint_rng_restored':True,
            'peak_memory_bytes':torch.cuda.max_memory_allocated()})
        all_results[key]={'replacement':a.replace_lavs,'teacher_frozen':True,'initialization':loading,
            'max_initial_box_error_vs_224':maximum,
            'zero_gate_max_error':zero_gate_error if variant=='deformable_exchange' else None,
            'initial_hit_counts':counts,'miou_drift':drift,'trainable_parameters':trainable,
            'optimizer_groups':[{'name':g['name'],'lr':g['initial_lr']} for g in optimizer.param_groups],
            'training_metrics':stats,'two_gpu_update':ranked}
        if rank==0:
            write_json(root/'preflight/progress.json',{'completed_variants':list(all_results)})
            probe.unlink()
            print(f'Preflight passed: {key}, trainable={trainable}, initial max error={maximum}',flush=True)
        dist.barrier()
        ema_hook.remove()
        del checkpoint,model,wrapped,optimizer,scheduler,loader,train_loader,watched,training,data,indexed,result,ema,saved_ema,teacher,teacher_before
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
