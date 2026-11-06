"""Two-GPU acceptance of target-conditioned LAVS ablations."""
import gc
import hashlib
import json
import math
import time

from .runtime import (VARIANTS, SEEDS, NEW_PREFIX, BASE_PREFIXES, REFERENCE, recipe, initialize,
    assert_scope, optimizer_for, seed_all, rng_state, restore_rng, delta_record, restore_delta,
    selected_name, atomic_checkpoint, write_json, build)


def frozen_digest(model,args):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not selected_name(name,args):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_l1(gradient):
    return 0. if gradient is None else gradient.detach().abs().sum().item()


def measure_tcr_attention_effect(model,images,phrases):
    """Compare each actual external attention output immediately before/after TCR.

    The same forward and weights provide both values, so student optimization
    or a changed data batch cannot masquerade as an attention-bias effect.
    """
    import torch
    from models.tcr_lavs import neutral_key_prior
    from utils.misc import NestedTensor
    captured=[]
    def observe(module,inputs,result):
        before_rgb,before_tir=inputs[2]
        (after_rgb,after_tir),aux=result
        if not (len(before_rgb)==len(after_rgb)==len(before_tir)==len(after_tir)):
            raise AssertionError('TCR changed the number of visual levels')
        capture={
            'rgb_output_max_abs':max((x-y).abs().max().item() for x,y in zip(after_rgb,before_rgb)),
            'tir_output_max_abs':max((x-y).abs().max().item() for x,y in zip(after_tir,before_tir)),
            'rgb_prior_max_abs':max(neutral_key_prior(x,aux['patch_padding']).abs().max().item()
                                    for x in aux['rgb_logits']),
            'tir_prior_max_abs':max(neutral_key_prior(x,aux['patch_padding']).abs().max().item()
                                    for x in aux['tir_logits']),
        }
        captured.append(capture)
    was_training=model.training
    hook=model.tcr_target.register_forward_hook(observe)
    try:
        model.eval()
        fresh=NestedTensor(images.tensors.clone(),images.mask.clone())
        with torch.no_grad():
            model(fresh,phrases)
    finally:
        hook.remove()
        model.train(was_training)
    if len(captured)!=1:
        raise AssertionError('TCR attention effect audit did not capture exactly one forward')
    effect=captured[0]
    if any(not math.isfinite(value) or value<=0 for value in effect.values()):
        raise AssertionError(f'TCR target priors or attention outputs have no finite effect: {effect}')
    return effect


def measure_query_gate_effect(model,images,phrases):
    """Compare query gate on/off using identical trained weights and inputs."""
    import torch
    from utils.misc import NestedTensor
    if not model.query_enabled or not model.tcr_target.query_enabled:
        raise ValueError('Query-gate intervention requires the registered query variant')
    captured=[]
    def observe(module,inputs,result):
        outputs=result[0]
        captured.append([[x.detach().clone() for x in direction] for direction in outputs])
    was_training=model.training
    hook=model.tcr_target.register_forward_hook(observe)
    try:
        model.eval()
        with torch.no_grad():
            for enabled in (False,True):
                model.tcr_target.query_enabled=enabled
                fresh=NestedTensor(images.tensors.clone(),images.mask.clone())
                model(fresh,phrases)
    finally:
        model.tcr_target.query_enabled=True
        hook.remove()
        model.train(was_training)
    if len(captured)!=2:
        raise AssertionError('Query-gate intervention did not capture both forwards')
    effect={name:max((a-b).abs().max().item() for a,b in zip(captured[1][direction],captured[0][direction]))
            for direction,name in enumerate(('rgb_query_gate_max_abs','tir_query_gate_max_abs'))}
    if any(not math.isfinite(x) or x<=0 for x in effect.values()):
        raise AssertionError(f'Query gate has no finite effect on both visual streams: {effect}')
    return effect


def batch_independence(model,images,phrases):
    import torch
    from utils.misc import NestedTensor
    was_training=model.training
    model.eval()
    count=min(4,len(phrases))
    if count<2:
        raise AssertionError('Batch audit requires multiple examples')
    def predict(indices,texts=None):
        indices=torch.tensor(indices,device=images.tensors.device)
        fresh=NestedTensor(images.tensors[indices].clone(),images.mask[indices])
        return model(fresh,texts or [phrases[i] for i in indices.tolist()])[0]
    with torch.no_grad():
        original=predict(list(range(count)))
        separate=torch.cat([predict([i]) for i in range(count)])
        reverse=predict(list(reversed(range(count)))).flip(0)
        changed=list(phrases[:count]);changed[1]='a white building on the right'
        modified=predict(list(range(count)),changed)
    model.train(was_training)
    return {'independent_max_abs':(original-separate).abs().max().item(),
            'permuted_max_abs':(original-reverse).abs().max().item(),
            'other_text_sample0_max_abs':(original[0]-modified[0]).abs().max().item()}


def run(root):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,DistributedSampler,Subset,BatchSampler
    from script_train.clip_lavs_replacement_control.data import dataset as NativeDataset
    from utils.misc import collate_fn as collate_native
    from models.lavs_replacement import legacy_lavs_key
    from models.differential_lavs import contrast_key
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
    baseline_predictions = None
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
        inner_foreground=[(n,p) for n,p in model.named_parameters()
                          if '.language_replacement.' in n and not contrast_key(n)]
        if not inner_foreground or any(p.requires_grad for _,p in inner_foreground):
            raise AssertionError('Migrated foreground inside CLIP must be frozen')
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if a.replace_lavs:
            if any(legacy_lavs_key(n) for n,_ in model.student.base.named_parameters()):
                raise AssertionError('Student retains original LAVS parameters')
            if len(model.student.base.cross_fusion_layers_vt) or len(model.student.base.cross_fusion_layers_tv):
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
                    if phrase!=old['samples'][index]['phrase'] or target.tolist()!=old['samples'][index]['target_xywh']:
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
        if variant=='query_off':
            baseline_predictions=predictions
        else:
            zero_gate_error=max((torch.tensor(box)-torch.tensor(baseline_predictions[i])).abs().max().item() for i,box in predictions.items())
            if zero_gate_error>1e-6:
                raise AssertionError(f'Target head does not preserve initial box: {zero_gate_error}')
        if a.replace_lavs and any(counts[k]<old['summary'][k]-2 for k in counts):
            raise AssertionError(f'Migrated initialization loses more than two validation hits: {counts}')
        independence_before=batch_independence(model,images.to(device),phrases)
        if a.replace_lavs and max(independence_before.values())>1e-5:
            raise AssertionError(f'New model mixes examples: {independence_before}')
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
        # Query gating depends on two learned signals and needs enough actual
        # updates to observe an effect at float32 precision.
        expected_updates=8 if a.tcr_query_enabled else 2
        training=Subset(NativeDataset('train',a),list(range(96*expected_updates)))
        sampler=DistributedSampler(training,shuffle=True)
        train_loader=DataLoader(training,batch_sampler=BatchSampler(sampler,a.batch_size,drop_last=True),
                                num_workers=2,collate_fn=collate_native)
        wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
        # DDP first synchronizes constructor-initialized inactive parameters.
        # Freeze checks start at this common, synchronized training state.
        before_digest=frozen_digest(model,a)
        optimizer,scheduler=optimizer_for(model,a)
        ema=ParameterEMA(model,[n for n,p in model.named_parameters() if p.requires_grad],a.ema_decay)
        ema_hook=optimizer.register_step_post_hook(lambda *_: ema.update(model))
        watched={name:p.detach().clone() for name,p in model.named_parameters()
                 if selected_name(name,a)}
        if a.tcr_mode == 'bias':
            initial_strength=watched['tcr_target.bias_strength']
            if not torch.allclose(initial_strength,torch.full_like(initial_strength,.5),rtol=0.,atol=1e-8):
                raise AssertionError('TCR bias strength does not have its registered initial value')
        gradients,hooks={},[]
        for name,p in model.named_parameters():
            if name in watched:
                def record(grad,name=name):
                    gradients[name]=gradients.get(name,0.)+gradient_l1(grad)
                hooks.append(p.register_hook(record))
        torch.cuda.reset_peak_memory_stats()
        stats=train_one_epoch(a,wrapped,train_loader,optimizer,device,0,1.)
        if 'loss_teacher_preserve' in stats:
            raise AssertionError('Unregistered teacher loss ran')
        if a.tcr_mode != 'off':
            if 'loss_tcr_target' not in stats:
                raise AssertionError('Target loss missing from TCR training')
        elif 'loss_tcr_target' in stats:
            raise AssertionError('Off baseline received TCR target loss')
        scheduler.step()
        if ema.updates != expected_updates:
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
        if model.student.base.clip.training or model.student.base.vl_transformer.training:
            raise AssertionError('Original modules must keep deterministic eval behavior')
        if frozen_digest(model,a)!=before_digest:
            raise AssertionError('Training changed a parameter outside the allowed scope')
        branches=['student.base.iafv3_token_gate.',
                  'student.base.clip.base_model.model.vision_model.encoder.layers.8.self_attn.q_proj.lora_A.lora_rgb.',
                  'student.base.clip.base_model.model.vision_model.encoder.layers.8.self_attn.q_proj.lora_A.lora_ir.']
        branches.extend(BASE_PREFIXES[:2])
        branches.extend(['student.base.differential_exchange.rgb.0.output.',
                         'student.base.differential_exchange.ir.0.output.'])
        if a.differential_enabled:
            branches.extend(['.head_coefficient', '.coefficient_route.',
                '.language_replacement.branches.rgb.attention.reference_query.',
                '.language_replacement.branches.ir.attention.reference_query.',
                'student.base.differential_exchange.rgb.0.reference_query.',
                'student.base.differential_exchange.ir.0.reference_query.'])
        if a.tcr_mode != 'off':
            branches.extend(['tcr_target.rgb.', 'tcr_target.tir.'])
        gradient_sums={branch:sum(value for name,value in gradients.items() if branch in name) for branch in branches}
        if not all(value>0 and math.isfinite(value) for value in gradient_sums.values()):
            raise AssertionError(f'Missing required gradients: {gradient_sums}')
        attention_effect=None
        query_effect=None
        strength_before=None
        strength_after=None
        if a.tcr_mode == 'bias':
            strength_grad=sum(value for name,value in gradients.items() if name.endswith('tcr_target.bias_strength'))
            strength=model.tcr_target.bias_strength.detach()
            if not math.isfinite(strength_grad) or strength_grad <= 0:
                raise AssertionError('TCR attention bias strength did not receive a finite gradient')
            if not ((strength >= 0) & (strength <= 2)).all():
                raise AssertionError('TCR attention bias strength escaped the projection interval')
            if torch.equal(strength,initial_strength):
                raise AssertionError('TCR bias strength did not change after the two real updates')
            if not (strength > 0).all():
                raise AssertionError('TCR bias strength is not positive after real updates')
            gradient_sums['tcr_target.bias_strength']=strength_grad
            strength_before=initial_strength.item()
            strength_after=strength.item()
            attention_effect=measure_tcr_attention_effect(model,images.to(device),phrases)
            if a.tcr_query_enabled:
                query_effect=measure_query_gate_effect(model,images.to(device),phrases)
        updates={branch:[name for name,before in watched.items() if branch in name
                  and not torch.equal(before,dict(model.named_parameters())[name])] for branch in branches}
        if not all(updates.values()):
            raise AssertionError('Required weights did not update')
        steps={int(s['step'].item()) for s in optimizer.state.values() if 'step' in s}
        if steps!={expected_updates}:
            raise AssertionError(f'Unexpected optimizer update count: {steps}')
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
        if ema.updates!=expected_updates or any(not torch.equal(saved_ema['shadow'][n],v.cpu()) for n,v in ema.shadow.items()):
            raise AssertionError('EMA checkpoint restoration failed')
        restore_rng(checkpoint['rng_by_rank'][rank])
        draw=torch.rand(16,device=device)
        restore_rng(states[rank])
        if not torch.equal(draw,torch.rand(16,device=device)):
            raise AssertionError('Per-rank CUDA RNG recovery changed')
        if any(not torch.equal(value,model.state_dict()[name].cpu()) for name,value in checkpoint['model_delta'].items()):
            raise AssertionError('Selective delta restoration failed')
        independence_after=batch_independence(model,images.to(device),phrases)
        if a.replace_lavs and max(independence_after.values())>1e-5:
            raise AssertionError(f'Trained model mixes examples: {independence_after}')
        ranked=[None]*2
        dist.all_gather_object(ranked,{'rank':rank,'gradient_l1':gradient_sums,'updated_parameters':updates,
            'optimizer_updates':expected_updates,'batch_independence_before':independence_before,
            'batch_independence_after':independence_after,'inference_ms':inference_ms,
            'inference_batch_size':benchmark_images.tensors.shape[0],'inference_peak_bytes':inference_peak,
            'tcr_bias_strength_before':strength_before,'tcr_bias_strength_after':strength_after,
            'tcr_attention_effect':attention_effect,
            'query_gate_effect':query_effect,
            'frozen_parameters_unchanged':True,'checkpoint_rng_restored':True,
            'peak_memory_bytes':torch.cuda.max_memory_allocated()})
        all_results[key]={'replacement':a.replace_lavs,'teacher_used':False,'initialization':loading,
            'max_initial_box_error_vs_224':maximum,
            'zero_gate_max_error':zero_gate_error if variant!='query_off' else None,
            'initial_hit_counts':counts,'miou_drift':drift,'trainable_parameters':trainable,
            'optimizer_groups':[{'name':g['name'],'lr':g['initial_lr']} for g in optimizer.param_groups],
            'training_metrics':stats,'two_gpu_update':ranked}
        if rank==0:
            write_json(root/'preflight/progress.json',{'completed_variants':list(all_results)})
            probe.unlink()
            print(f'Preflight passed: {key}, trainable={trainable}, initial max error={maximum}',flush=True)
        dist.barrier()
        ema_hook.remove()
        del checkpoint,model,wrapped,optimizer,scheduler,loader,train_loader,watched,training,data,indexed,result,ema,saved_ema
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
