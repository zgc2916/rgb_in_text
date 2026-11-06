"""Evaluate frozen validation-selected C1 and TCR models on their dataset."""
import argparse
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
from script_train.clip_full_match.runtime import build,sha256,write_json
from script_train.clip_soft_control.evaluate import IndexedDataset,summarize
from script_train.clip_lavs_replacement_control.data import dataset as make_dataset
from utils import misc
from utils.misc import collate_fn
from utils.box_utils import bbox_iou,xywh2xyxy


def collate(batch):
    indices,values=zip(*batch)
    return indices,collate_fn(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--splits',nargs='+',choices=('val','test'),default=['val','test'])
    cli=p.parse_args()
    checkpoint=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    args=checkpoint['args'];torch.set_num_threads(4);misc.init_distributed_mode(args)
    model=build(args,public_migration=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    device=torch.device('cuda',args.gpu);model.to(device).eval()
    wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
    source={'checkpoint':str(cli.checkpoint.resolve()),'sha256':sha256(cli.checkpoint),
            'epoch':checkpoint['epoch'],'weight_view':'raw','validation':checkpoint['val_stats'],
            'dataset':args.dataset,'method':args.full_match_method,'initialization':'public MMVG'}
    for split in cli.splits:
        out=cli.output_root/f'{split}.json'
        if out.exists():
            old=json.loads(out.read_text())
            if old['source']['sha256']!=source['sha256'] or old['split']!=split:raise ValueError('Existing result differs')
            dist.barrier();continue
        data=make_dataset(split,args);indexed=IndexedDataset(data)
        loader=DataLoader(indexed,batch_size=args.batch_size,sampler=DistributedSampler(indexed,shuffle=False),
                          num_workers=args.num_workers,collate_fn=collate)
        local=[]
        with torch.no_grad():
            for indices,(images,phrases,targets,_) in loader:
                boxes=wrapped(images.to(device),phrases)[0]
                values=bbox_iou(xywh2xyxy(boxes).clamp(0,1),xywh2xyxy(targets.to(device)))
                for i,phrase,target,box,value in zip(indices,phrases,targets,boxes.cpu(),values.cpu()):
                    local.append({'index':i,'image':str(data.images[i][0]),'phrase':phrase,
                                  'target_xywh':target.tolist(),'pred_xywh':box.tolist(),'iou':value.item()})
        parts=[None]*misc.get_world_size();dist.all_gather_object(parts,local)
        if misc.get_rank()==0:
            all_rows=[row for part in parts for row in part];by_index={r['index']:r for r in all_rows}
            if set(by_index)!=set(range(len(data))):raise ValueError('Incomplete sample coverage')
            rows=[by_index[i] for i in range(len(data))];summary=summarize(rows)
            if split=='val':
                for current,key in (('acc_05','accu'),('acc_07','accu_07'),('miou','miou')):
                    if abs(summary[current]-source['validation'][key])>(1e-5 if key=='miou' else 1e-12):
                        raise ValueError('Validation did not reproduce selected checkpoint')
            groups={'small':[r for r in rows if str(data.images[r['index']][5]).lower()=='small'],
                    'weak':[r for r in rows if str(data.images[r['index']][4]) in ('1','2')]}
            write_json(out,{'source':source,'split':split,'summary':summary,
                           'groups':{k:summarize(v) for k,v in groups.items()},'samples':rows,
                           'protocol':{'single_model':True,'ground_truth_in_forward':False,
                                       'duplicate_padding_removed':len(all_rows)-len(rows)}})
            print(json.dumps({'split':split,**summary}),flush=True)
        dist.barrier()
    dist.destroy_process_group()

if __name__=='__main__':main()
