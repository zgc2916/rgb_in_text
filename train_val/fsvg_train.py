import os
import time
import math
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate


def get_args_parser():
    parser = argparse.ArgumentParser("FSVG Training Args", add_help=False)

    # model selector
    parser.add_argument("--model_name", type=str, default="FSVG")

    # optimization
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_text", default=1e-5, type=float)
    parser.add_argument("--lr_visu", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--clip_max_norm", default=0.0, type=float, help="gradient clipping max norm")
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--lr_scheduler", default="step", type=str)
    parser.add_argument("--lr_drop", default=60, type=int)
    parser.add_argument("--lr_power", default=0.9, type=float)
    parser.add_argument("--lr_exponential", default=0.9, type=float)

    # augmentation
    parser.add_argument("--aug_blur", action="store_true")
    parser.add_argument("--aug_crop", action="store_true")
    parser.add_argument("--aug_scale", action="store_true")
    parser.add_argument("--aug_translate", action="store_true")

    # CLIP backbone
    # Note: `models/clip_mdetr` does not provide `ViT-L/14@336px` in some environments.
    # Default to ViT-B/16 to be maximally compatible.
    parser.add_argument("--model", type=str, default="ViT-B/16", help="CLIP backbone name")
    parser.add_argument("--imsize", default=224, type=int)

    # Candidate Elimination / FS
    parser.add_argument("--fs", action="store_true", help="Enable Candidate Elimination keep-rate schedule")
    parser.add_argument("--fs_keep", default=0.7, type=float)
    parser.add_argument("--fs_start", default=20, type=int)
    parser.add_argument("--fs_warm", default=30, type=int)

    # dataset
    # Default to RGBT datasets used in this codebase
    parser.add_argument("--modality", default="rgbt", type=str)
    parser.add_argument("--sup_type", default="full", type=str)
    parser.add_argument("--data_root", type=str, default="../dataset_and_pretrain_model/datasets/VG/image_data")
    parser.add_argument("--split_root", type=str, default="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled")
    parser.add_argument("--dataset", default="rgbtvg_flir", type=str)
    parser.add_argument("--max_query_len", default=77, type=int)
    parser.add_argument("--prompt", type=str, default="", help="Prompt template")
    parser.add_argument("--num_workers", default=8, type=int)

    # io
    parser.add_argument("--output_dir", default="./output_training/FSVG", help="path where to save")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--clipvg_pretrained",
        default="",
        type=str,
        help="optional CLIP-VG checkpoint path for partial weight initialization",
    )
    parser.add_argument("--start_epoch", default=0, type=int)

    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    return parser


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
    return checkpoint


def _load_clipvg_pretrained(args, model_without_ddp: torch.nn.Module):
    ckpt_path = getattr(args, "clipvg_pretrained", "")
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"clipvg_pretrained not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    pretrained = _extract_state_dict(checkpoint)
    if not isinstance(pretrained, dict):
        raise ValueError(f"Unsupported checkpoint format for {ckpt_path}")

    model_state = model_without_ddp.state_dict()
    matched = {}
    shape_mismatch = 0
    for k, v in pretrained.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                matched[k] = v
            else:
                shape_mismatch += 1

    missing_in_pretrained = len(model_state) - len(matched)
    model_without_ddp.load_state_dict(matched, strict=False)
    if utils.is_main_process():
        print(
            "[FSVG] Loaded CLIP-VG pretrained weights from {} | matched: {} / {} | "
            "shape_mismatch: {} | missing_after_load: {}".format(
                ckpt_path, len(matched), len(model_state), shape_mismatch, missing_in_pretrained
            )
        )


def _build_optimizer(args, model_without_ddp: torch.nn.Module):
    visu_param = []
    text_param = []
    rest_param = []

    for n, p in model_without_ddp.named_parameters():
        if not p.requires_grad:
            continue
        if (("clip" in n and "visual" in n) or "text_resize" in n) and p.requires_grad:
            visu_param.append(p)
        elif ("clip" in n) and (
            "transformer" in n
            or "ln_final" in n
            or "token_embedding" in n
            or "text_projection" in n
            or "positional_embedding" in n
        ) and p.requires_grad:
            text_param.append(p)
        else:
            rest_param.append(p)

    param_list = [
        {"params": rest_param, "lr": args.lr},
        {"params": visu_param, "lr": args.lr_visu},
        {"params": text_param, "lr": args.lr_text},
    ]

    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError("optimizer type not supported")
    return optimizer


def _build_scheduler(args, optimizer):
    if args.lr_scheduler == "poly":
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    if args.lr_scheduler == "halfdecay":
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    if args.lr_scheduler == "cosine":
        lr_func = lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if args.lr_scheduler == "exponential":
        lr_func = lambda epoch: args.lr_exponential ** epoch
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    raise ValueError("lr scheduler type not supported")


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    _load_clipvg_pretrained(args, model)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print("number of requires_grad params: ", n_parameters_grad)
    print("number of all params: ", n_parameters)

    optimizer = _build_optimizer(args, model_without_ddp)
    lr_scheduler = _build_scheduler(args, optimizer)

    # build dataset
    print("build dataset...")
    if args.sup_type == "full":
        dataset_train = build_dataset("train", args)
    else:
        dataset_train = build_dataset("train_pseudo", args)
    dataset_val = build_dataset("val", args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    best_accu = -0.01
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
        # for FSVG, validate uses CE schedule; record current epoch for keep-rate computation
        args.current_eval_epoch = args.start_epoch
        val_stats = validate(args, model, data_loader_val, device)
        best_accu = val_stats["accu"]
        print("best_accu: {}".format(best_accu))

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(str(args) + "\n")

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        args.current_eval_epoch = epoch
        val_stats = validate(args, model, data_loader_val, device)

        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": round(v, 6) for k, v in train_stats.items()},
            **{f"val_{k}": round(v, 6) for k, v in val_stats.items()},
            "n_parameters": n_parameters,
        }
        print(log_stats)

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [os.path.join(args.output_dir, "checkpoint.pth")]
            if val_stats["accu"] > best_accu:
                checkpoint_paths.append(os.path.join(args.output_dir, "best_checkpoint.pth"))
                best_accu = val_stats["accu"]

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "val_accu": val_stats["accu"],
                    },
                    checkpoint_path,
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FSVG training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
