#!/usr/bin/env python3
"""Persistently quantize a Hugging Face causal LM to compressed-tensors FP8."""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Linear weights to FP8 with dynamic FP8 activations.",
    )
    parser.add_argument("source", type=Path, help="Source Hugging Face model directory")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument(
        "--gpu-memory-gib",
        type=int,
        default=22,
        help="Maximum model memory per visible GPU (default: 22 GiB)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="Quantization device placement (default: auto; use cpu when GPUs are busy)",
    )
    return parser.parse_args()


def quantize(source: Path, output: Path, gpu_memory_gib: int, device: str) -> None:
    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM

    source = source.resolve()
    output = output.resolve()
    incomplete = output.with_name(f"{output.name}.incomplete")

    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Not a Hugging Face model directory: {source}")
    for path in (output, incomplete):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    if gpu_memory_gib <= 0:
        raise ValueError("--gpu-memory-gib must be positive")

    if device == "auto":
        cuda_driver = ctypes.CDLL("libcuda.so.1")
        cuda_result = cuda_driver.cuInit(0)
        if cuda_result != 0:
            raise RuntimeError(f"CUDA driver initialization failed with error {cuda_result}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; rerun with --device cpu")
        max_memory: dict[int | str, str] = dict.fromkeys(
            range(torch.cuda.device_count()), f"{gpu_memory_gib}GiB"
        )
        max_memory["cpu"] = "96GiB"
        device_map: str | dict[str, str] = "auto"
    else:
        max_memory = {"cpu": "96GiB"}
        device_map = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )
    oneshot(
        model=model,
        processor=str(source),
        recipe=recipe,
        pipeline="datafree",
        output_dir=str(incomplete),
        save_compressed=True,
    )

    for filename in ("merge_info.json", "training_loss.png"):
        source_file = source / filename
        if source_file.is_file() and not (incomplete / filename).exists():
            shutil.copy2(source_file, incomplete / filename)

    info = {
        "source": str(source),
        "format": "compressed-tensors",
        "scheme": "FP8_DYNAMIC",
        "weights": "float8_e4m3fn, per-channel static",
        "input_activations": "float8_e4m3fn, per-token dynamic",
        "ignored_modules": ["lm_head"],
        "quantization_device": device,
        "torch_version": torch.__version__,
    }
    (incomplete / "quantization_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    incomplete.rename(output)
    print(f"FP8 model saved to: {output}")


if __name__ == "__main__":
    args = parse_args()
    quantize(args.source, args.output, args.gpu_memory_gib, args.device)
