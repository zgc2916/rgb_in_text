"""Verify and launch one frozen-trunk box-head or F2 center screen."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .run_scale_ablation import epoch_results, timestamp, write_json


def preflight(cli):
    import copy
    import random

    import numpy as np
    import torch

    from datasets import build_dataset
    from models import build_model
    from utils.box_utils import generalized_box_iou, xywh2xyxy
    from utils.misc import NestedTensor, collate_fn

    torch.set_num_threads(4)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    random.seed(cli.seed)
    source = torch.load(cli.source, map_location="cpu", weights_only=False)
    warmstart = torch.load(cli.warmstart, map_location="cpu", weights_only=False)
    if set(source["model"]) != set(warmstart["model"]) or any(
        not torch.equal(value, warmstart["model"][key])
        for key, value in source["model"].items()
    ):
        raise ValueError("Warm-start weights are not exactly equal to the fixed source model")
    if "optimizer" in warmstart:
        raise ValueError("Warm-start must contain model weights without optimizer state")
    args = copy.deepcopy(source["args"])
    source_epoch = source["epoch"]
    del source
    args.infmae_local_center_only = not cli.bbox_head_only
    args.infmae_bbox_head_only = cli.bbox_head_only
    args.infmae_bbox_head_last_layer_only = cli.bbox_head_last_layer_only
    args.infmae_local_center_max_shift_pixels = cli.max_shift_pixels
    if cli.no_aug_scale:
        args.aug_scale = False
    model = build_model(args)
    model.load_state_dict(warmstart["model"], strict=True)
    del warmstart
    model.to("cuda").eval()
    dataset = build_dataset('val', args)
    images, texts, targets, _ = collate_fn([dataset[index] for index in range(2)])
    images = images.to("cuda")
    targets = targets.to("cuda")

    def fresh_images():
        return NestedTensor(images.tensors.clone(), images.mask)

    head = model.bbox_embed if cli.bbox_head_only else model.infmae_local_center
    prefix = "bbox_embed." if cli.bbox_head_only else "infmae_local_center."
    if cli.bbox_head_last_layer_only:
        prefix = "bbox_embed.layers.2."
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefix))
    model.eval()
    if cli.bbox_head_last_layer_only:
        head.layers[-1].train()
    else:
        head.train()
    # Bypass only the residual output, keeping identical base mode, PEFT
    # freezing, and autograd context for the paired forward computations.
    bypass = (None if cli.bbox_head_only else
              head.register_forward_hook(lambda module, inputs, output: (inputs[2], {})))
    baseline = model(fresh_images(), texts)[0].detach()
    if bypass is not None:
        bypass.remove()
    shapes = []
    handle = model.infmae_direct_encoder.register_forward_hook(
        lambda module, inputs, output: shapes.append(list(output[0].shape))
    )
    prediction = model(fresh_images(), texts)[0]
    handle.remove()
    if not torch.equal(prediction, baseline):
        print('identity_max_abs_difference:', (prediction - baseline).abs().max().item(), flush=True)
        raise AssertionError("Initial controlled-head predictions changed baseline predictions")
    active = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not active or any(not name.startswith(prefix) for name in active):
        raise AssertionError("A baseline parameter was reopened during adapter switching")
    training_modules = [name for name, module in model.named_modules() if module.training]
    if any(not name.startswith(prefix.rstrip('.')) for name in training_modules):
        raise AssertionError("A frozen baseline module is still in training mode")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cli.lr)
    loss = 2 * torch.nn.functional.l1_loss(prediction, targets, reduction="sum") / len(targets)
    loss = loss + 2 * (1 - torch.diag(generalized_box_iou(
        xywh2xyxy(prediction), xywh2xyxy(targets)
    ))).mean()
    loss.backward()
    if not torch.isfinite(loss) or not all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters
    ):
        raise AssertionError("Refiner regression gradients are missing or non-finite")
    if any(parameter.grad is not None for name, parameter in model.named_parameters()
           if not name.startswith(prefix)):
        raise AssertionError("Regression propagated a gradient into the frozen baseline")
    optimizer.step()
    with torch.no_grad():
        updated = model(fresh_images(), texts)[0]
    if not torch.isfinite(updated).all() or not ((updated >= 0) & (updated <= 1)).all():
        raise AssertionError("Updated head produced invalid normalized boxes")
    if not cli.bbox_head_only and not torch.equal(updated[:, 2:], baseline[:, 2:]):
        raise AssertionError("Center refinement changed widths or heights")
    max_shift = ((updated[:, :2] - baseline[:, :2]).abs() * args.imsize).max().item()
    if not cli.bbox_head_only and max_shift > cli.max_shift_pixels + 1e-5:
        raise AssertionError("Center update exceeded its pixel bound")
    summary = {
        "checked_at": timestamp(), "source_epoch": source_epoch,
        "warmstart_exact_match": True, "validation_samples": len(dataset),
        "checked_samples": 2, "f2_shapes": shapes, "initial_predictions_exact_identity": True,
        "head_kind": "bbox_embed" if cli.bbox_head_only else "infmae_local_center",
        "last_layer_only": cli.bbox_head_last_layer_only,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_names": active, "training_modules": training_modules,
        "baseline_gradients_absent": True, "regression_loss": loss.item(),
        "width_height_exactly_preserved_after_step": bool(torch.equal(updated[:, 2:], baseline[:, 2:])),
        "max_shift_pixels_after_step": max_shift,
        "gpu_memory_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    write_json(cli.output_root / "preflight.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-shift-pixels", type=float, default=4.0)
    parser.add_argument("--no-aug-scale", action="store_true")
    parser.add_argument("--bbox-head-only", action="store_true")
    parser.add_argument("--bbox-head-last-layer-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()
    gpu_ids = cli.gpus.split(",")
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(gpu.isdigit() for gpu in gpu_ids):
        parser.error("--gpus must specify two distinct GPU IDs")
    if not 0 < cli.epochs <= 60 or cli.lr <= 0 or cli.max_shift_pixels <= 0:
        parser.error("Require 1..60 epochs and positive lr/max-shift-pixels")
    if cli.bbox_head_last_layer_only and not cli.bbox_head_only:
        parser.error("--bbox-head-last-layer-only requires --bbox-head-only")
    cli.source = cli.source.resolve(strict=True)
    cli.warmstart = cli.warmstart.resolve(strict=True)
    cli.output_root = cli.output_root.resolve()
    if cli.preflight_only:
        preflight(cli)
        return
    root = cli.output_root
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    directory = root / "train"
    if status_path.exists() or directory.exists():
        raise FileExistsError(f"Refusing to overwrite an existing screen: {root}")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=cli.gpus, TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(root / "matplotlib"))
    repository = Path(__file__).resolve().parents[2]
    status = {"started_at": timestamp(), "pid": os.getpid(), "gpus": gpu_ids,
              "source": str(cli.source), "warmstart": str(cli.warmstart),
              "epochs": cli.epochs, "lr": cli.lr, "seed": cli.seed,
              "max_shift_pixels": cli.max_shift_pixels, "phase": "preflight"}
    status["no_aug_scale"] = cli.no_aug_scale
    status["bbox_head_only"] = cli.bbox_head_only
    status["bbox_head_last_layer_only"] = cli.bbox_head_last_layer_only
    write_json(status_path, status)
    common = ["--source", str(cli.source), "--warmstart", str(cli.warmstart),
              "--epochs", str(cli.epochs), "--lr", str(cli.lr), "--seed", str(cli.seed)]
    if cli.no_aug_scale:
        common.append("--no-aug-scale")
    if cli.bbox_head_last_layer_only:
        common.append("--bbox-head-last-layer-only")
    check_command = [sys.executable, "-m", "script_train.infmae_experiments.run_local_center_screen",
                     *common, "--output-root", str(root), "--gpus", cli.gpus,
                     "--max-shift-pixels", str(cli.max_shift_pixels), "--preflight-only"]
    if cli.bbox_head_only:
        check_command.append("--bbox-head-only")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               "-m", "script_train.infmae_experiments.continue_a5f2", *common,
               "--output_dir", str(directory), "--bbox-head-only" if cli.bbox_head_only else "--local-center",
               "--local-center-max-shift-pixels", str(cli.max_shift_pixels)]
    write_json(root / "commands.json", {"preflight": check_command, "train": command})
    with (root / "preflight.log").open("w", encoding="utf-8") as output:
        result = subprocess.run(check_command, cwd=repository, env=environment,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        status.update(phase="failed_preflight", returncode=result.returncode, finished_at=timestamp())
        write_json(status_path, status)
        raise SystemExit(result.returncode)
    directory.mkdir()
    with (directory / "console.log").open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=repository, env=environment,
                                   stdout=output, stderr=subprocess.STDOUT)
        status.update(phase="train", train_pid=process.pid, train_started_at=timestamp())
        write_json(status_path, status)
        returncode = process.wait()
    rows = epoch_results(directory)
    initial_path = directory / "initial_validation.json"
    initial = json.loads(initial_path.read_text()) if initial_path.is_file() else None
    write_json(root / "results.json", {
        "source": str(cli.source), "epochs": rows, "returncode": returncode,
        "initial_validation": initial,
        "best_acc05": max(rows, key=lambda row: (row["acc_05"], row["acc_07"], row["miou"]) if cli.bbox_head_only
                          else row["acc_05"]) if rows else None,
        "qualified_epochs": [row for row in rows if initial and row["correct_05"] >= 463
                             and row["acc_07"] >= initial['accu_07']
                             and row["miou"] >= initial['miou']],
    })
    status.update(phase="complete" if returncode == 0 else "failed_train",
                  returncode=returncode, finished_at=timestamp())
    write_json(status_path, status)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
