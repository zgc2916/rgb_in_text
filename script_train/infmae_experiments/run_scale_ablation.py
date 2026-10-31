"""Run paired RefFLIR scale-augmentation screens sequentially on two GPUs."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def epoch_results(directory):
    path = directory / "log.txt"
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "validation_accu" in row:
            rows.append({
                "epoch": row["epoch"],
                "correct_05": round(row["validation_accu"] * 608),
                "correct_07": round(row["validation_accu_07"] * 608),
                "miou": row["validation_miou"],
                "acc_05": row["validation_accu"],
                "acc_07": row["validation_accu_07"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=13)
    cli = parser.parse_args()
    gpu_ids = cli.gpus.split(",")
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(x.isdigit() for x in gpu_ids):
        parser.error("--gpus must identify two distinct GPUs, e.g. 3,6")
    if cli.epochs <= 0 or cli.lr <= 0:
        parser.error("--epochs and --lr must be positive")
    source = cli.source.resolve(strict=True)
    warmstart = cli.warmstart.resolve()
    root = cli.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    if status_path.exists() or any((root / name).exists() for name in ("no_scale", "control")):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {root}")
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": cli.gpus,
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "MPLCONFIGDIR": str(root / "matplotlib"),
    })
    status = {
        "started_at": timestamp(), "pid": os.getpid(), "gpus": gpu_ids,
        "source": str(source), "warmstart": str(warmstart),
        "epochs_per_run": cli.epochs, "seed": cli.seed, "lr": cli.lr,
        "validation_samples": 608, "phase": "prepare", "runs": {},
    }
    write_json(status_path, status)
    prepare = [
        sys.executable, "-m", "script_train.infmae_experiments.continue_a5f2",
        "--prepare", "--source", str(source), "--warmstart", str(warmstart),
    ]
    with (root / "prepare.log").open("w", encoding="utf-8") as output:
        result = subprocess.run(prepare, cwd=repository, env=environment,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        status.update(phase="failed_prepare", returncode=result.returncode, finished_at=timestamp())
        write_json(status_path, status)
        raise SystemExit(result.returncode)

    commands = {"prepare": prepare}
    summary = {"source": str(source), "validation_samples": 608, "runs": {}}
    for name in ("no_scale", "control"):
        directory = root / name
        directory.mkdir()
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=2", "-m", "script_train.infmae_experiments.continue_a5f2",
            "--source", str(source), "--warmstart", str(warmstart),
            "--output_dir", str(directory), "--epochs", str(cli.epochs),
            "--lr", str(cli.lr), "--seed", str(cli.seed),
        ]
        if name == "no_scale":
            command.append("--no-aug-scale")
        commands[name] = command
        write_json(root / "commands.json", commands)
        status["phase"] = name
        status["runs"][name] = {"started_at": timestamp(), "output_dir": str(directory)}
        print(f"{timestamp()} starting {name}: {directory}", flush=True)
        with (directory / "console.log").open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=repository, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT)
            status["runs"][name]["pid"] = process.pid
            write_json(status_path, status)
            returncode = process.wait()
        rows = epoch_results(directory)
        summary["runs"][name] = {
            "output_dir": str(directory), "epochs": rows,
            "best_acc05": max(rows, key=lambda row: row["acc_05"]) if rows else None,
            "returncode": returncode,
        }
        write_json(root / "comparison.json", summary)
        status["runs"][name].update(returncode=returncode, finished_at=timestamp())
        if returncode:
            status.update(phase=f"failed_{name}", finished_at=timestamp())
            write_json(status_path, status)
            raise SystemExit(returncode)
        write_json(status_path, status)
        print(f"{timestamp()} finished {name}", flush=True)
    status.update(phase="complete", finished_at=timestamp())
    write_json(status_path, status)
    print(f"Saved paired results: {root / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
