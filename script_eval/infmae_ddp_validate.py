"""Run the project's exact distributed validation for an InfMAE checkpoint.

The normal training entry point also performs a validation pass immediately
after loading ``--resume``.  This wrapper sets ``eval`` and makes the training
range empty, so it reuses that path without updating weights or writing a
training checkpoint.
"""

import argparse
from pathlib import Path

import torch

from train_val.mmvg_train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    cli = parse_args()
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    del checkpoint

    args.resume = str(cli.checkpoint.resolve())
    args.retrain = ""
    args.eval = True
    args.output_dir = str(cli.output_dir.resolve())
    # ``mmvg_train.main`` validates immediately after resume.  Keeping the
    # loop empty makes this an evaluation-only operation.
    args.start_epoch = args.epochs
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    train_main(args)


if __name__ == "__main__":
    main()
