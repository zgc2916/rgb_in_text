"""Use the original C1 data pipeline; all controls share augmentation and losses."""
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
from datasets import build_dataset
from utils.misc import collate_fn


def dataset(split,args):
    return build_dataset(split,args)


def loaders(args,train=True):
    val=dataset('val',args)
    val_loader=DataLoader(val,batch_size=12,sampler=DistributedSampler(val,shuffle=False),
                          num_workers=args.num_workers,collate_fn=collate_fn)
    if not train:
        return val_loader
    data=dataset('train',args)
    sampler=DistributedSampler(data,shuffle=True)
    loader=DataLoader(data,batch_sampler=BatchSampler(sampler,args.batch_size,drop_last=True),
                      num_workers=args.num_workers,collate_fn=collate_fn)
    return loader,val_loader,sampler
