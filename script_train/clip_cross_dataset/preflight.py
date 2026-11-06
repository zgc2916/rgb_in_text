"""Real two-GPU training and recovery acceptance for each target dataset."""
import gc
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler,Subset,BatchSampler

from . import runtime as r
from script_train.clip_tcr_query_control.preflight import (batch_independence,
    measure_tcr_attention_effect,measure_query_gate_effect)
from script_train.clip_lavs_replacement_control.data import dataset
from script_train.clip_soft_control.evaluate import IndexedDataset
from utils import misc
from utils.misc import collate_fn
from utils.box_utils import bbox_iou,xywh2xyxy
from utils.parameter_ema import ParameterEMA
from engine import train_one_epoch,validate


def collate_indexed(rows):
    indices,values=zip(*rows)
    return indices,collate_fn(values)


def frozen_digest(model,args):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not r.selected_name(name,args):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(root):
    args=r.recipe(root/'preflight',r.VARIANTS[0],r.SEED)
    misc.init_distributed_mode(args)
    torch.set_num_threads(4)
    if misc.get_world_size()!=2:raise ValueError('Exactly two GPU ranks required')
    rank=misc.get_rank()
    device=torch.device('cuda',args.gpu)
    all_results={}
    for variant in r.VARIANTS:
        r.seed_all(r.SEED+rank)
        a=r.recipe(root/'preflight'/variant,variant,r.SEED)
        a.gpu=args.gpu;a.rank=rank;a.distributed=True
        model=r.build(a)
        initialization=r.initialize(model,a)
        model.to(device).eval()
        data=dataset('val',a)
        indexed=IndexedDataset(data)
        loader=DataLoader(indexed,batch_size=12,sampler=DistributedSampler(indexed,shuffle=False),
                          num_workers=2,collate_fn=collate_indexed)
        local=[]
        with torch.no_grad():
            for indices,(images,phrases,targets,_) in loader:
                boxes=model(images.to(device),phrases)[0]
                if not torch.isfinite(boxes).all():raise AssertionError('Nonfinite target validation box')
                values=bbox_iou(xywh2xyxy(boxes).clamp(0,1),xywh2xyxy(targets.to(device)))
                local.extend((i,float(v)) for i,v in zip(indices,values.cpu()))
        gathered=[None]*2
        dist.all_gather_object(gathered,local)
        by_index=dict(x for rank_rows in gathered for x in rank_rows)
        if set(by_index)!=set(range(len(data))):raise AssertionError('Incomplete validation coverage')
        values=torch.tensor([by_index[i] for i in range(len(data))])
        initial={'total':len(data),'correct_05':int((values>=.5).sum()),
                 'correct_07':int((values>=.7).sum()),'miou':float(values.mean())}
        independence=batch_independence(model,images.to(device),phrases)
        if max(independence.values())>1e-5:raise AssertionError(f'Batch mixing: {independence}')
        training=Subset(dataset('train',a),list(range(192)))
        sampler=DistributedSampler(training,shuffle=True)
        train_loader=DataLoader(training,batch_sampler=BatchSampler(sampler,a.batch_size,drop_last=True),
                                num_workers=2,collate_fn=collate_fn)
        wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
        val_loader=DataLoader(data,batch_size=12,sampler=DistributedSampler(data,shuffle=False),
                              num_workers=2,collate_fn=collate_fn)
        native_validation=validate(a,wrapped,val_loader,device)
        for key,expected in (('accu',initial['correct_05']/initial['total']),
                             ('accu_07',initial['correct_07']/initial['total']),
                             ('miou',initial['miou'])):
            if abs(native_validation[key]-expected)>1e-6:
                raise AssertionError(f'Native validation mismatch for {key}')
        before=frozen_digest(model,a)
        opt,sched=r.optimizer_for(model,a)
        ema=ParameterEMA(model,[n for n,p in model.named_parameters() if p.requires_grad],a.ema_decay)
        eh=opt.register_step_post_hook(lambda *_:ema.update(model))
        tracked={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        gradients={};hooks=[]
        for name,p in model.named_parameters():
            if name in tracked:
                def record(grad,name=name):
                    gradients[name]=gradients.get(name,0.)+(0. if grad is None else grad.detach().abs().sum().item())
                hooks.append(p.register_hook(record))
        stats=train_one_epoch(a,wrapped,train_loader,opt,device,0,1.)
        sched.step()
        for hook in hooks:hook.remove()
        if 'loss_tcr_target' not in stats or ema.updates!=2:
            raise AssertionError('Target supervision or two effective updates missing')
        r.assert_scope(model,a)
        if frozen_digest(model,a)!=before:raise AssertionError('Frozen model parameters changed')
        branches=('tcr_target.rgb.','tcr_target.tir.','tcr_target.bias_strength',
                  'student.base.differential_exchange.rgb.0.output.',
                  'student.base.differential_exchange.ir.0.output.')
        sums={branch:sum(v for name,v in gradients.items() if branch in name) for branch in branches}
        if not all(math.isfinite(x) and x>0 for x in sums.values()):
            raise AssertionError(f'Required gradients missing: {sums}')
        changes={branch:sum(not torch.equal(before_value,dict(model.named_parameters())[name])
                            for name,before_value in tracked.items() if branch in name) for branch in branches}
        if not all(changes.values()):raise AssertionError(f'Required branches did not update: {changes}')
        tcr_effect=measure_tcr_attention_effect(model,images.to(device),phrases)
        query_effect=measure_query_gate_effect(model,images.to(device),phrases)
        states=[None]*2
        dist.all_gather_object(states,r.rng_state())
        probe=root/'preflight'/f'{variant}_recovery_probe.pth'
        if rank==0:
            ck=r.delta_record(model,a,0,{'accu':.5,'accu_07':.3,'miou':.4},.5,{'preflight':True})
            ck.update(optimizer=opt.state_dict(),lr_scheduler=sched.state_dict(),rng_by_rank=states,ema_state=ema.state_dict())
            r.atomic_checkpoint(probe,ck)
        dist.barrier()
        ck=torch.load(probe,map_location='cpu',weights_only=False)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if r.selected_name(name,a):p.add_(1.)
        r.restore_delta(model,a,ck)
        opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['lr_scheduler'])
        saved_ema=ema.state_dict();ema.load_state_dict(ck['ema_state'])
        if ema.updates!=2 or any(not torch.equal(saved_ema['shadow'][n],v.cpu()) for n,v in ema.shadow.items()):
            raise AssertionError('EMA restore failed')
        r.restore_rng(ck['rng_by_rank'][rank]);draw=torch.rand(16,device=device)
        r.restore_rng(states[rank])
        if not torch.equal(draw,torch.rand(16,device=device)):raise AssertionError('RNG restore failed')
        if any(not torch.equal(v,model.state_dict()[n].cpu()) for n,v in ck['model_delta'].items()):
            raise AssertionError('Model restore failed')
        rank_results=[None]*2
        dist.all_gather_object(rank_results,{'rank':rank,'gradient_l1':sums,'updated_tensors':changes,
            'tcr_attention_effect':tcr_effect,'query_gate_effect':query_effect,'updates':2,
            'frozen_unchanged':True,'optimizer_ema_rng_recovered':True})
        all_results[variant]={'dataset':a.dataset,'initial_validation':initial,'initialization':initialization,
                              'batch_independence':independence,'native_validation':native_validation,
                              'training_metrics':stats,'ranks':rank_results}
        if rank==0:
            r.write_json(root/'preflight/progress.json',{'completed_variants':list(all_results)})
            probe.unlink()
            print(f'Preflight passed: {variant}, initial validation={initial}',flush=True)
        dist.barrier()
        eh.remove()
        del model,wrapped,opt,sched,ema,ck,tracked,training,loader,val_loader,train_loader,data,indexed
        gc.collect();torch.cuda.empty_cache()
    if rank==0:r.write_json(root/'preflight.json',{'passed':True,'variants':all_results,
                                                   'test_evaluated':False})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root.resolve())
