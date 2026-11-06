"""Verify matched warm starts and one true double-GPU training update."""
import gc,math
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler,Subset,BatchSampler
from . import runtime as r
from script_train.clip_lavs_replacement_control.data import dataset
from utils import misc
from utils.misc import collate_fn
from engine import train_one_epoch,validate


def run(root):
    a=r.recipe(root/'preflight','flir','joint');misc.init_distributed_mode(a)
    if misc.get_world_size()!=2:raise ValueError('Two GPUs required')
    torch.set_num_threads(4);rank=misc.get_rank();device=torch.device('cuda',a.gpu);results={};reference=None
    for method in r.METHODS:
        args=r.recipe(root/'preflight'/method,'flir',method);args.gpu=a.gpu;args.rank=rank;args.distributed=True
        torch.manual_seed(r.SEED+rank)
        model=r.build(args,initialize=True);model.to(device)
        wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[a.gpu],find_unused_parameters=True)
        # A zero-initialized detail and boundary path must preserve the source
        # function. Every candidate is checked on the same complete validation set.
        val=dataset('val',args)
        vloader=DataLoader(val,batch_size=12,sampler=DistributedSampler(val,shuffle=False),num_workers=2,collate_fn=collate_fn)
        scores=validate(args,wrapped,vloader,device)
        if abs(float(scores['accu'])-479/608)>1e-12 or abs(float(scores['miou'])-.6659335337187114)>1e-5:
            raise ValueError(f'Warm-start validation differs: {method} {scores}')
        if reference is None:reference={k:float(scores[k]) for k in ('accu','accu_07','miou')}
        elif any(abs(float(scores[k])-reference[k])>(1e-5 if k=='miou' else 1e-12) for k in reference):
            raise ValueError('Models do not share the same initial predictions')
        params=[p for n,p in model.named_parameters() if 'visumodel' not in n and 'textmodel' not in n
                and (p.requires_grad or '.lora_A.' in n or '.lora_B.' in n)]
        opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
        sub=Subset(dataset('train',args),list(range(96)))
        train_loader=DataLoader(sub,batch_sampler=BatchSampler(DistributedSampler(sub,shuffle=True),args.batch_size,drop_last=True),
                                num_workers=2,collate_fn=collate_fn)
        grads={};hooks=[]
        for name,p in model.named_parameters():
            if name.startswith(('tcr_target.rgb.','tcr_target.tir.','tir_detail.','tir_injector.',
                                'tir_region_head.','boundary_refiner.')):
                def record(grad,name=name):grads[name]=grads.get(name,0.)+(0. if grad is None else grad.detach().abs().sum().item())
                hooks.append(p.register_hook(record))
        stats=train_one_epoch(args,wrapped,train_loader,opt,device,0,args.clip_max_norm)
        for h in hooks:h.remove()
        required=['loss_tcr_target']
        prefixes=['tcr_target.rgb.','tcr_target.tir.']
        if args.enable_tir_detail:required+=['loss_tir_region'];prefixes+=['tir_detail.','tir_injector.','tir_region_head.']
        if args.enable_boundary_refiner:required+=['loss_boundary_b0','loss_boundary_b1'];prefixes+=['boundary_refiner.']
        if not set(required).issubset(stats):raise ValueError(f'Missing loss: {required}')
        if not all(math.isfinite(float(v)) for k,v in stats.items() if k.startswith('loss')):raise ValueError('Nonfinite loss')
        for prefix in prefixes:
            if sum(v for k,v in grads.items() if k.startswith(prefix))<=0:raise ValueError(f'Missing gradient {prefix}')
        results[method]={'initial_validation':{k:float(v) for k,v in scores.items()},
                         'losses':{k:float(v) for k,v in stats.items()},'required_gradient_prefixes':prefixes}
        if rank==0:
            r.write_json(root/'preflight/progress.json',{'passed_cases':list(results)})
            print('Preflight passed',method,flush=True)
        dist.barrier()
        del model,wrapped,params,opt,sub,train_loader,val,vloader
        gc.collect();torch.cuda.empty_cache()
    if rank==0:r.write_json(root/'preflight.json',{'passed':True,'cases':results,'test_evaluated':False})
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root.resolve())
