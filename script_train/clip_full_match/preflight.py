"""Exercise true C1 losses and full optimizer update on both datasets and models."""
import gc
import math
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler,Subset,BatchSampler
from . import runtime as r
from script_train.infmae_experiments.train_clip_control import load_common_initialization
from script_train.clip_lavs_replacement_control.data import dataset
from utils import misc
from utils.misc import collate_fn
from engine import train_one_epoch,validate


def run(root):
    a=r.recipe(root/'preflight','m3fd','c1');misc.init_distributed_mode(a)
    if misc.get_world_size()!=2:raise ValueError('Two GPUs required')
    torch.set_num_threads(4);rank=misc.get_rank();device=torch.device('cuda',a.gpu);results={}
    for ds in r.DATASETS:
        for method in r.METHODS:
            args=r.recipe(root/'preflight'/f'{ds}_{method}',ds,method)
            args.gpu=a.gpu;args.rank=rank;args.distributed=True
            torch.manual_seed(r.SEED+rank)
            model=r.build(args,public_migration=method=='tcr_query')
            loading=load_common_initialization(model,args.retrain)
            migration=getattr(model,'_public_migration',None)
            if method=='tcr_query' and (not migration or not migration['attention']['foreground_tensors_exact']):
                raise AssertionError('Public attention migration incomplete')
            model.to(device)
            wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[a.gpu],find_unused_parameters=True)
            params=[p for n,p in model.named_parameters() if 'visumodel' not in n and 'textmodel' not in n
                    and (p.requires_grad or '.lora_A.' in n or '.lora_B.' in n)]
            opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
            training=Subset(dataset('train',args),list(range(96)))
            loader=DataLoader(training,batch_sampler=BatchSampler(DistributedSampler(training,shuffle=True),
                              args.batch_size,drop_last=True),num_workers=2,collate_fn=collate_fn)
            checks={};hooks=[]
            if method=='tcr_query':
                for name,p in model.named_parameters():
                    if name.startswith(('tcr_target.rgb.','tcr_target.tir.','differential_exchange.rgb.0.output.',
                                        'differential_exchange.ir.0.output.')):
                        def audit(grad,name=name):
                            checks[name]=checks.get(name,0.)+(0. if grad is None else grad.detach().abs().sum().item())
                        hooks.append(p.register_hook(audit))
            stats=train_one_epoch(args,wrapped,loader,opt,device,0,args.clip_max_norm)
            for hook in hooks:hook.remove()
            if not all(math.isfinite(float(v)) for k,v in stats.items() if k.startswith('loss')):
                raise ValueError('Non-finite loss')
            if method=='tcr_query':
                if 'loss_tcr_target' not in stats:raise ValueError('Target loss missing')
                for prefix in ('tcr_target.rgb.','tcr_target.tir.','differential_exchange.rgb.0.output.',
                               'differential_exchange.ir.0.output.'):
                    if sum(v for k,v in checks.items() if k.startswith(prefix))<=0:
                        raise ValueError(f'Missing gradient {prefix}')
            # Exercise native validation with the exact C1 loss/model interface.
            val=Subset(dataset('val',args),list(range(24)))
            val_loader=DataLoader(val,batch_size=12,sampler=DistributedSampler(val,shuffle=False),
                                  num_workers=2,collate_fn=collate_fn)
            scores=validate(args,wrapped,val_loader,device)
            if not all(math.isfinite(float(scores[k])) for k in ('accu','accu_07','miou')):
                raise ValueError('Non-finite validation')
            results[f'{ds}_{method}']={'losses':{k:float(v) for k,v in stats.items()},
                'validation':{k:float(v) for k,v in scores.items()},'common_loading':loading,
                'migration':migration,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
            if rank==0:
                r.write_json(root/'preflight/progress.json',{'passed_cases':list(results)})
                print('Preflight passed:',ds,method,flush=True)
            dist.barrier()
            del model,wrapped,opt,params,training,loader,val,val_loader
            gc.collect();torch.cuda.empty_cache()
    if rank==0:r.write_json(root/'preflight.json',{'passed':True,'cases':results,'test_evaluated':False})
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root.resolve())
