"""Collect validation/test boxes using the training validator's DDP batches."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from datasets import build_dataset
from models import build_model
from utils.box_utils import bbox_iou, xywh2xyxy
from utils import misc as utils


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def collate_indexed(batch):
    indices, values = zip(*batch)
    return indices, utils.collate_fn(values)


def summarize(rows):
    values = torch.tensor([row["iou"] for row in rows], dtype=torch.float64)
    return {
        "total": len(rows), "correct_05": int((values >= 0.5).sum()),
        "correct_07": int((values >= 0.7).sum()),
        "acc_05": float((values >= 0.5).double().mean()),
        "acc_07": float((values >= 0.7).double().mean()), "miou": float(values.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument('--blur-detail', action='store_true')
    parser.add_argument('--disable-independent-memory', action='store_true')
    cli = parser.parse_args()
    from utils.checkpoint_storage import load_checkpoint
    checkpoint = load_checkpoint(cli.checkpoint)
    args = checkpoint["args"]
    from .runtime import ORIGINAL
    args.dataset_bert_model_path = str(ORIGINAL/'dataset_and_pretrain_model/pretrain_model/pretrained_weights/Bert')
    args.pretrained_model_path = str(ORIGINAL/'dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip-vit-base-patch16')
    if cli.blur_detail or cli.disable_independent_memory:
        if cli.splits != ['val']:
            raise ValueError('Structure interventions are restricted to validation')
        if cli.blur_detail and not getattr(args, 'enable_tir_detail', False):
            raise ValueError('No detail branch to blur')
        if cli.disable_independent_memory and not getattr(args, 'enable_boundary_refiner', False):
            raise ValueError('No independent memory reader to disable')
    args.clip_structure_blur_detail = cli.blur_detail
    args.clip_structure_disable_memory = cli.disable_independent_memory
    if args.dataset != "rgbtvg_flir":
        raise ValueError("Expected RefFLIR")
    utils.init_distributed_mode(args)
    rank = utils.get_rank()
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = build_model(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    source = {"checkpoint": str(cli.checkpoint.resolve()), "epoch": checkpoint["epoch"],
              "recorded_val_acc05": checkpoint["val_accu"]}
    del checkpoint
    device = torch.device("cuda", args.gpu)
    model.to(device).eval()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    for split in cli.splits:
        path = cli.output_root / f"{split}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
        dataset = build_dataset(split, args)
        indexed = IndexedDataset(dataset)
        sampler = DistributedSampler(indexed, shuffle=False)
        loader = DataLoader(indexed, batch_size=args.batch_size, sampler=sampler,
                            drop_last=False, num_workers=args.num_workers, collate_fn=collate_indexed)
        local_rows = []
        # Match engine.validate's no_grad context; inference_mode can select
        # different attention kernels and has differed on threshold samples.
        with torch.no_grad():
            for batch_number, (indices, (images, phrases, targets, _)) in enumerate(loader):
                images = images.to(device)
                targets = targets.to(device)
                boxes = model(images, phrases)[0]
                values = bbox_iou(xywh2xyxy(boxes).clamp(0, 1), xywh2xyxy(targets))
                for index, phrase, box, target, value in zip(
                    indices, phrases, boxes.cpu(), targets.cpu(), values.cpu()
                ):
                    local_rows.append({"index": index, "image": str(dataset.images[index][0]),
                                       "phrase": phrase, "pred_xywh": box.tolist(),
                                       "target_xywh": target.tolist(), "iou": value.item()})
                if rank == 0 and (batch_number + 1) % 30 == 0:
                    print(f"{split}: rank0 evaluated {len(local_rows)}/{len(sampler)}", flush=True)
        gathered = [None] * utils.get_world_size()
        dist.all_gather_object(gathered, local_rows)
        if rank == 0:
            all_rows = [row for rows in gathered for row in rows]
            by_index = {row["index"]: row for row in all_rows}
            rows = [by_index[index] for index in range(len(dataset))]
            report = {"source": source, "split": split, "summary": summarize(rows),
                      "protocol": {"world_size": utils.get_world_size(), "batch_size_per_gpu": args.batch_size,
                                   "grad_context": "torch.no_grad", "sampler": "DistributedSampler(shuffle=False)",
                                   "duplicate_padding_removed": len(all_rows) - len(rows)}, "samples": rows}
            cli.output_root.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(path)
            print(json.dumps({"split": split, **report["summary"]}), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
