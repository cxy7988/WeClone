#!/usr/bin/env python3
"""Recover a ZeRO-3 LoRA checkpoint and merge it into its base model.

This is intended for checkpoints whose ``adapter_model.safetensors`` is an
empty placeholder because ``stage3_gather_16bit_weights_on_model_save`` was
disabled. The trainable LoRA tensors are recovered from the DeepSpeed shards,
converted to PEFT's portable adapter format, and then merged into the base
model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a DeepSpeed ZeRO-3 LoRA checkpoint into a Hugging Face base model."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/Qwen3-14B"),
        help="Base model directory (default: models/Qwen3-14B).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_output/checkpoint-800"),
        help="Checkpoint directory containing latest, adapter_config.json and ZeRO shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_output/Qwen3-14B-checkpoint-800-merged"),
        help="New directory for the merged model. It must be empty or absent.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
        help="Dtype used to load and save the merged model (default: bfloat16).",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="Transformers device map. Use 'cpu' for low GPU-memory risk or 'auto' for GPU/CPU placement.",
    )
    parser.add_argument(
        "--adapter-name",
        default="default",
        help="Adapter name used while training (default: default).",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum size of each merged safetensors shard (default: 5GB).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Parent directory for temporary recovered adapter files (default: system temp).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from the local base model directory.",
    )
    return parser.parse_args()


def resolve_and_validate(args: argparse.Namespace) -> None:
    args.base_model = args.base_model.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.work_dir is not None:
        args.work_dir = args.work_dir.expanduser().resolve()

    required = (
        args.base_model / "config.json",
        args.checkpoint / "adapter_config.json",
        args.checkpoint / "latest",
        args.checkpoint / "zero_to_fp32.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n  " + "\n  ".join(missing))

    tag = (args.checkpoint / "latest").read_text(encoding="utf-8").strip()
    shard_dir = args.checkpoint / tag
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"DeepSpeed shard directory does not exist: {shard_dir}")
    if not list(shard_dir.glob("*_optim_states.pt")):
        raise FileNotFoundError(f"No DeepSpeed optimizer shards found under: {shard_dir}")

    if args.output_dir in (args.base_model, args.checkpoint):
        raise ValueError("output-dir must not overwrite the base model or checkpoint directory")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {args.output_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}\n"
            "Choose a new --output-dir so existing model files are not overwritten."
        )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.output_dir.parent).free / 1024**3
    if free_gib < 32:
        print(
            f"WARNING: only {free_gib:.1f} GiB is free under {args.output_dir.parent}; "
            "a merged 14B bfloat16 model normally needs about 30 GiB.",
            file=sys.stderr,
        )


def recover_lora_adapter(
    checkpoint: Path,
    recovered_dir: Path,
    adapter_name: str,
) -> tuple[int, int]:
    """Recover trainable tensors and rewrite their keys to PEFT save format."""
    raw_dir = recovered_dir / "raw_zero_state"
    raw_dir.mkdir(parents=True)

    command = [
        sys.executable,
        str(checkpoint / "zero_to_fp32.py"),
        str(checkpoint),
        str(raw_dir),
        "--safe_serialization",
        "--exclude_frozen_parameters",
        "--max_shard_size",
        "100GB",
    ]
    print("[1/4] Recovering trainable LoRA tensors from ZeRO-3 shards...")
    subprocess.run(command, check=True)

    raw_weights = raw_dir / "model.safetensors"
    if not raw_weights.is_file():
        raise FileNotFoundError(f"Recovered weights were not created: {raw_weights}")

    with safe_open(raw_weights, framework="pt", device="cpu") as source:
        raw_keys = list(source.keys())
        if not raw_keys:
            raise RuntimeError("The recovered ZeRO state dict contains no trainable parameters")

        portable_state: dict[str, torch.Tensor] = {}
        marker = f".{adapter_name}"
        for raw_key in raw_keys:
            # PEFT removes the adapter name (usually '.default') when saving an
            # adapter and restores it while loading.
            portable_key = raw_key.replace(marker, "")
            if portable_key in portable_state:
                raise RuntimeError(f"Duplicate PEFT key after conversion: {portable_key}")
            portable_state[portable_key] = source.get_tensor(raw_key).contiguous()

    tensor_count = len(portable_state)
    parameter_count = sum(tensor.numel() for tensor in portable_state.values())
    if not all("lora_" in key for key in portable_state):
        unexpected = [key for key in portable_state if "lora_" not in key][:5]
        raise RuntimeError(f"Recovered non-LoRA trainable parameters: {unexpected}")

    shutil.copy2(checkpoint / "adapter_config.json", recovered_dir / "adapter_config.json")
    save_file(
        portable_state,
        recovered_dir / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    del portable_state

    print(f"      Recovered {tensor_count} tensors / {parameter_count:,} LoRA parameters.")
    return tensor_count, parameter_count


def merge_model(args: argparse.Namespace, adapter_dir: Path) -> None:
    dtype = DTYPES[args.dtype]
    print(f"[2/4] Loading base model from {args.base_model} ({args.dtype}, device_map={args.device_map})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    print("[3/4] Loading the recovered adapter and merging it into the base model...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    merged_model = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
    merged_model.config.use_cache = True

    print(f"[4/4] Saving merged model to {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer_source = args.checkpoint if (args.checkpoint / "tokenizer_config.json").is_file() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(args.output_dir)

    merge_info = {
        "base_model": str(args.base_model),
        "checkpoint": str(args.checkpoint),
        "dtype": args.dtype,
        "format": "merged_full_model",
    }
    (args.output_dir / "merge_info.json").write_text(
        json.dumps(merge_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    resolve_and_validate(args)

    temp_parent = str(args.work_dir) if args.work_dir is not None else None
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)

    print("This operation needs roughly 35-40 GiB of available CPU RAM and 30 GiB of disk space.")
    with tempfile.TemporaryDirectory(prefix="weclone-lora-recovery-", dir=temp_parent) as temp_dir:
        adapter_dir = Path(temp_dir) / "adapter"
        adapter_dir.mkdir()
        recover_lora_adapter(args.checkpoint, adapter_dir, args.adapter_name)
        merge_model(args, adapter_dir)

    print(f"Merge completed successfully: {args.output_dir}")


if __name__ == "__main__":
    main()
