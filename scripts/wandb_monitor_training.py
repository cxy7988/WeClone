#!/usr/bin/env python3
"""Mirror metrics from an already-running training log into Weights & Biases."""

from __future__ import annotations

import argparse
import ast
import re
import time
from pathlib import Path
from typing import Any

import wandb


METRIC_RECORD = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
METRIC_NAMES = {
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/learning_rate",
    "entropy": "train/entropy",
    "num_tokens": "train/num_tokens",
    "mean_token_accuracy": "train/mean_token_accuracy",
    "epoch": "train/epoch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--project", default="WeClone-Qwen3.5-SFT")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    return parser.parse_args()


def parse_metrics(text: str) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for match in METRIC_RECORD.finditer(text):
        raw = ast.literal_eval(match.group(0))
        metrics: dict[str, float] = {}
        for source, destination in METRIC_NAMES.items():
            value: Any = raw.get(source)
            if value is not None:
                metrics[destination] = float(value)
        if "train/loss" in metrics:
            records.append(metrics)
    return records


def main() -> None:
    args = parse_args()
    if not args.log_file.is_file():
        raise FileNotFoundError(args.log_file)

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        job_type="live-log-monitor",
        tags=["qwen3.5-4b", "lora", "sft", "sidecar"],
        config={"source_log": str(args.log_file), "deepspeed": False},
    )
    print(f"W&B run: {run.url}", flush=True)

    step = 0
    position = 0
    remainder = ""
    while True:
        with args.log_file.open(encoding="utf-8", errors="replace") as handle:
            handle.seek(position)
            chunk = handle.read()
            position = handle.tell()

        if chunk:
            combined = remainder + chunk
            last_newline = combined.rfind("\n")
            if last_newline < 0:
                remainder = combined
            else:
                complete = combined[: last_newline + 1]
                remainder = combined[last_newline + 1 :]
                for metrics in parse_metrics(complete):
                    step += 1
                    metrics["train/global_step"] = float(step)
                    run.log(metrics, step=step)
                if "Training exited with code" in complete:
                    break
        time.sleep(args.poll_interval)

    run.finish()


if __name__ == "__main__":
    main()
