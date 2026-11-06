"""Resize original augmented pixels, rather than upsampling a 224 canvas."""
import copy

from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
from datasets import transforms as T
from script_train.clip_reobserve_control.data import NativeDataset, collate_native, MEAN, STD


class ResolutionDataset(NativeDataset):
    def __init__(self, split, args):
        if args.imsize not in (224, 336):
            raise ValueError('Only preregistered 224/336 inputs are supported')
        construction = copy.copy(args)
        construction.imsize = 224
        super().__init__(split, construction)
        self.args = args
        self.sizes = ([args.imsize - args.imsize//8*i for i in range(7)]
                      if args.aug_scale and split == 'train' else [args.imsize])
        self.finish = T.Compose([T.RandomResize(self.sizes), T.ToTensor(),
            T.NormalizeAndPad(mean=MEAN, std=STD, size=args.imsize,
                              aug_translate=bool(args.aug_translate and split == 'train'))])


def loaders(args, train=True):
    val = ResolutionDataset('val', args)
    val_loader = DataLoader(val, batch_size=12, sampler=DistributedSampler(val, shuffle=False),
                            num_workers=args.num_workers, collate_fn=collate_native)
    if not train:
        return val_loader
    data = ResolutionDataset('train', args)
    sampler = DistributedSampler(data, shuffle=True)
    loader = DataLoader(data, batch_sampler=BatchSampler(sampler, args.batch_size, drop_last=True),
                        num_workers=args.num_workers, collate_fn=collate_native)
    return loader, val_loader, sampler
