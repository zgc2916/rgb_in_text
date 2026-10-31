"""Evaluate one InfMAE checkpoint on RefFLIR with optional inference probes."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn import functional as F

from datasets import build_dataset
from models import build_model
from utils.box_utils import bbox_iou, xywh2xyxy
from utils.misc import NestedTensor, collate_fn


def unflip_xywh(boxes):
    """Map normalized horizontal-flip predictions to the original canvas."""
    boxes = boxes.clone()
    boxes[:, 0] = 1.0 - boxes[:, 0]
    return boxes


def official_box_iou(pred, target):
    return bbox_iou(xywh2xyxy(pred).clamp(0, 1), xywh2xyxy(target))


def mask_to_box(mask_logits, fallback):
    """Turn the auxiliary box-mask prediction into normalized xywh boxes."""
    mask = F.interpolate(
        mask_logits, size=(224, 224), mode="bilinear", align_corners=False
    )[:, 0] > 0
    boxes = fallback.clone()
    for row, binary in enumerate(mask):
        ys, xs = torch.where(binary)
        if xs.numel() == 0:
            continue
        x0, x1 = xs.min() / 224, (xs.max() + 1) / 224
        y0, y1 = ys.min() / 224, (ys.max() + 1) / 224
        boxes[row] = torch.stack(
            ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
        )
    return boxes


def summarize(ious):
    values = torch.tensor(ious, dtype=torch.float64)
    return {
        "correct_05": int((values >= 0.5).sum()),
        "acc_05": float((values >= 0.5).double().mean()),
        "correct_07": int((values >= 0.7).sum()),
        "acc_07": float((values >= 0.7).double().mean()),
        "miou": float(values.mean()),
        "total": values.numel(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--average_checkpoint", type=Path)
    parser.add_argument("--average_alpha", type=float, default=0.5)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("val", "test", "testA", "testB", "testC"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.average_checkpoint and not 0.0 <= cli.average_alpha <= 1.0:
        raise ValueError("--average_alpha must be in [0, 1]")

    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args.FusionMethod not in {
        "InfMAEA5BridgeFTF2", "InfMAEA5BridgeFTF2Peak", "InfMAEA5BridgeFTF2MRP"
    } or args.dataset != "rgbtvg_flir":
        raise ValueError("Expected an A5-F2 InfMAE RefFLIR checkpoint")
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = build_model(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    source = {
        "checkpoint": str(cli.checkpoint.resolve()),
        "epoch": int(checkpoint["epoch"]),
        "recorded_val_acc_05": float(checkpoint["val_accu"]),
    }
    del checkpoint

    if cli.average_checkpoint:
        other = torch.load(cli.average_checkpoint, map_location="cpu", weights_only=False)
        state = model.state_dict()
        if set(state) != set(other["model"]):
            raise ValueError("Checkpoint state keys differ; cannot average models")
        with torch.no_grad():
            for name, target in state.items():
                value = other["model"][name]
                if target.shape != value.shape:
                    raise ValueError(f"Shape mismatch for {name}")
                if target.is_floating_point():
                    target.lerp_(value, cli.average_alpha)
                elif not torch.equal(target, value):
                    raise ValueError(f"Nonfloating buffer mismatch for {name}")
        source["average_checkpoint"] = str(cli.average_checkpoint.resolve())
        source["average_epoch"] = int(other["epoch"])
        source["average_alpha"] = cli.average_alpha
        del other

    device = torch.device(cli.device)
    model.to(device).eval()
    dataset = build_dataset(cli.split, args)
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collate_fn,
        num_workers=cli.num_workers,
    )
    results = []
    index = 0
    with torch.inference_mode():
        for batch_number, (image, phrases, target, _) in enumerate(loader):
            pixels = image.tensors.to(device)
            mask = image.mask.to(device)
            target = target.to(device)
            model_output = model(NestedTensor(pixels, mask), phrases)
            original = model_output[0]
            predictions = {"original": original}
            if cli.mask:
                mask_box = mask_to_box(model_output[4], original)
                predictions["mask_box"] = mask_box
                predictions["mask_mean"] = 0.5 * (original + mask_box)
            if cli.flip:
                mirrored = model(
                    NestedTensor(pixels.flip(-1), mask.flip(-1)), phrases
                )[0]
                mirrored = unflip_xywh(mirrored)
                predictions["flip"] = mirrored
                predictions["flip_mean"] = 0.5 * (original + mirrored)
            ious = {
                name: official_box_iou(boxes, target).cpu()
                for name, boxes in predictions.items()
            }
            cpu_boxes = {name: boxes.cpu() for name, boxes in predictions.items()}
            for local_index, phrase in enumerate(phrases):
                sample = dataset.images[index]
                results.append(
                    {
                        "index": index,
                        "image": str(sample[0]),
                        "phrase": phrase,
                        "target_xywh": target[local_index].cpu().tolist(),
                        "pred_xywh": {
                            name: boxes[local_index].tolist()
                            for name, boxes in cpu_boxes.items()
                        },
                        "iou": {
                            name: float(scores[local_index])
                            for name, scores in ious.items()
                        },
                    }
                )
                index += 1
            if (batch_number + 1) % 20 == 0:
                print(f"evaluated {index}/{len(dataset)}", flush=True)

    summary = {
        name: summarize([row["iou"][name] for row in results])
        for name in results[0]["iou"]
    }
    report = {"source": source, "split": cli.split, "summary": summary, "samples": results}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {cli.output}", flush=True)


if __name__ == "__main__":
    main()
