"""Protect target scale only in training; keep the held-out C1 transform."""
from datasets import transforms as T
from script_train.clip_reobserve_control.data import (
    NativeDataset, collate_native, MEAN, STD,
)


class ScaleGeometryDataset(NativeDataset):
    def __init__(self, split, args):
        super().__init__(split, args)
        floor = float(args.target_scale_floor_pixels) if split == 'train' else 0.
        self.finish = T.Compose([T.RandomResize(self.sizes, minimum_target_side=floor),
            T.ToTensor(), T.NormalizeAndPad(mean=MEAN, std=STD, size=224,
                aug_translate=bool(args.aug_translate and split == 'train'))])


def loaders(args, train=True):
    from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
    val = ScaleGeometryDataset('val', args)
    val_loader = DataLoader(val, batch_size=12, sampler=DistributedSampler(val, shuffle=False),
                            num_workers=args.num_workers, collate_fn=collate_native)
    if not train:
        return val_loader
    data = ScaleGeometryDataset('train', args)
    sampler = DistributedSampler(data, shuffle=True)
    loader = DataLoader(data, batch_sampler=BatchSampler(sampler, args.batch_size, drop_last=True),
                        num_workers=args.num_workers, collate_fn=collate_native)
    return loader, val_loader, sampler
