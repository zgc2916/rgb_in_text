"""Keep a native-resolution view alongside the unchanged C1 global canvas."""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import transforms as T
from datasets.data_loader import MDETRCLIP
from utils.misc import NestedTensor

MEAN = (.631, .6401, .632, .5337)
STD = (.2152, .227, .2439, .2562)


class NativeBatch(NestedTensor):
    def __init__(self, tensors, mask, native, sizes, rectangles):
        super().__init__(tensors, mask)
        self.native, self.sizes, self.rectangles = native, sizes, rectangles

    def to(self, device):
        return NativeBatch(self.tensors.to(device), self.mask.to(device), self.native.to(device),
                           self.sizes.to(device), self.rectangles.to(device))


def valid_rectangle(mask, native_size=None):
    y, x = torch.where(~mask.bool())
    if not len(x):
        raise ValueError('An image canvas must contain valid pixels')
    height, width = mask.shape
    valid_w, valid_h = float(x.max() + 1 - x.min()), float(y.max() + 1 - y.min())
    if native_size is not None:
        native_h, native_w = native_size
        ratio = max(valid_h, valid_w) / max(native_h, native_w)
        valid_w, valid_h = native_w * ratio, native_h * ratio
    return torch.tensor([x.min(), y.min(), valid_w, valid_h], dtype=torch.float32) / torch.tensor([width, height, width, height])


class NativeDataset(Dataset):
    def __init__(self, split, args):
        if args.dataset != 'rgbtvg_flir' or args.modality != 'rgbt' or args.imsize != 224:
            raise ValueError('Native re-observation is registered for 224 RefFLIR RGB-T only')
        self.args, self.split = args, split
        self.source = MDETRCLIP(args, args.data_root, args.split_root, args.dataset,
                               transform=lambda x: x, split=split, max_query_len=args.max_query_len)
        self.images = self.source.images
        self.photo = T.ColorJitter(.4, .4, .4)
        self.blur = T.GaussianBlur(aug_blur=args.aug_blur)
        self.sizes = [224 - 28 * i for i in range(7)] if args.aug_scale and split == 'train' else [224]
        self.finish = T.Compose([T.RandomResize(self.sizes), T.ToTensor(),
            T.NormalizeAndPad(mean=MEAN, std=STD, size=224,
                              aug_translate=bool(args.aug_translate and split == 'train'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, phrase, target = self.source.pull_item(index)
        value = {'img': image, 'text': phrase, 'box': target.clone()}
        if self.split == 'train':
            directional = any(word in phrase for word in ('left', 'right', 'top', 'bottom', 'middle'))
            if self.args.aug_crop and not directional and random.random() < .5:
                # Crop the native image before any downsampling. All controls
                # share this recipe; it is not identical to the old resize/crop recipe.
                lower = min(384, image.width, image.height)
                value = T.TargetPreservingSizeCrop(lower, 600)(value)
            value = self.blur(self.photo(value))
        native = torch.from_numpy(np.array(value['img'], copy=True)).permute(2, 0, 1).contiguous()
        finished = self.finish(value)
        return {'image': finished['img'], 'mask': finished['mask'], 'phrase': phrase,
                'target': finished['box'], 'native': native,
                'rectangle': valid_rectangle(finished['mask'], native.shape[-2:])}


def collate_native(rows):
    sizes = torch.tensor([row['native'].shape[-2:] for row in rows], dtype=torch.float32)
    height, width = (int(x) for x in sizes.max(0).values)
    native = torch.zeros(len(rows), 4, height, width, dtype=torch.uint8)
    for index, row in enumerate(rows):
        h, w = row['native'].shape[-2:]
        native[index, :, :h, :w] = row['native']
    batch = NativeBatch(torch.stack([r['image'] for r in rows]), torch.stack([r['mask'] for r in rows]),
                        native, sizes, torch.stack([r['rectangle'] for r in rows]))
    return batch, tuple(r['phrase'] for r in rows), torch.stack([r['target'] for r in rows]), torch.zeros(len(rows), 1, 224, 224)


def loaders(args, train=True):
    from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
    val = NativeDataset('val', args)
    val_loader = DataLoader(val, batch_size=12, sampler=DistributedSampler(val, shuffle=False),
                            num_workers=args.num_workers, collate_fn=collate_native)
    if not train:
        return val_loader
    data = NativeDataset('train', args)
    sampler = DistributedSampler(data, shuffle=True)
    loader = DataLoader(data, batch_sampler=BatchSampler(sampler, args.batch_size, drop_last=True),
                        num_workers=args.num_workers, collate_fn=collate_native)
    return loader, val_loader, sampler
