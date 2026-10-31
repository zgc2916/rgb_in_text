"""Evaluate a compact fusion checkpoint after reconstructing its frozen C1."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .runtime import initialize, restore_delta, seed_all, write_json, sha256
from script_train.clip_soft_control.evaluate import IndexedDataset, collate_indexed, summarize
from utils import misc
from utils.box_utils import bbox_iou, xywh2xyxy
from datasets import build_dataset
from models import build_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--splits',nargs='+',choices=('val','test'),default=['val'])
    cli=parser.parse_args()
    checkpoint=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    args=checkpoint['args']
    torch.set_num_threads(4)
    misc.init_distributed_mode(args)
    seed_all(args.seed+misc.get_rank())
    model=build_model(args)
    initialize(model,args)
    restore_delta(model,args,checkpoint)
    source={'checkpoint':str(cli.checkpoint.resolve()),'sha256':sha256(cli.checkpoint),
            'epoch':checkpoint['epoch'],'recorded_val_acc05':checkpoint['val_accu'],
            'base_checkpoint':checkpoint['base_checkpoint'],'base_sha256':checkpoint['base_sha256']}
    del checkpoint
    device=torch.device('cuda',args.gpu)
    model.to(device).eval()
    wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
    for split in cli.splits:
        out=cli.output_root/f'{split}.json'
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite evaluation: {out}')
        dataset=build_dataset(split,args)
        indexed=IndexedDataset(dataset)
        sampler=DistributedSampler(indexed,shuffle=False)
        loader=DataLoader(indexed,batch_size=12,sampler=sampler,num_workers=args.num_workers,collate_fn=collate_indexed)
        local=[]
        with torch.no_grad():
            for indices,(images,phrases,targets,_) in loader:
                boxes=wrapped(images.to(device),phrases)[0]
                values=bbox_iou(xywh2xyxy(boxes).clamp(0,1),xywh2xyxy(targets.to(device)))
                for index,phrase,target,box,value in zip(indices,phrases,targets,boxes.cpu(),values.cpu()):
                    local.append({'index':index,'image':str(dataset.images[index][0]),'phrase':phrase,
                        'target_xywh':target.tolist(),'pred_xywh':box.tolist(),'iou':value.item()})
        gathered=[None]*misc.get_world_size()
        dist.all_gather_object(gathered,local)
        if misc.get_rank()==0:
            all_rows=[row for part in gathered for row in part]
            by_index={row['index']:row for row in all_rows}
            rows=[by_index[index] for index in range(len(dataset))]
            write_json(out,{'source':source,'split':split,'summary':summarize(rows),
                'protocol':{'world_size':misc.get_world_size(),'batch_size_per_gpu':12,
                    'grad_context':'torch.no_grad','duplicate_padding_removed':len(all_rows)-len(rows),
                    'corruption_enabled':False,'ground_truth_in_forward':False},'samples':rows})
            print(json.dumps({'split':split,**summarize(rows)}),flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
