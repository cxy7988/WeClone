#!/usr/bin/env python3
"""Merge a Qwen3.5 LoRA adapter into a text-only Hugging Face model.

Qwen3.5 base checkpoints use a multimodal wrapper even for text-only training.
This exporter loads only ``text_config`` and ``model.language_model`` before
merging the adapter, so the saved model never instantiates or stores the vision
tower.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a Qwen3.5 LoRA adapter and export only its language model."
    )
    parser.add_argument("--base-model", type=Path, required=True, help="Local Qwen3.5 base model.")
    parser.add_argument("--adapter", type=Path, required=True, help="LoRA adapter or checkpoint directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New text-only model directory.")
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
        help="Dtype used to load and save the model (default: bfloat16).",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="Transformers device map (default: cpu; use auto to allow GPU placement).",
    )
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum output shard size.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Parent directory for temporary files.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow local custom model code.")
    return parser.parse_args()


def resolve_and_validate(args: argparse.Namespace) -> None:
    args.base_model = args.base_model.expanduser().resolve()
    args.adapter = args.adapter.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.work_dir is not None:
        args.work_dir = args.work_dir.expanduser().resolve()

    required = (
        args.base_model / "config.json",
        args.adapter / "adapter_config.json",
        args.adapter / ADAPTER_WEIGHTS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n  " + "\n  ".join(missing))
    if args.output_dir in (args.base_model, args.adapter):
        raise ValueError("output-dir must not overwrite the base model or adapter directory")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {args.output_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)


def make_text_config(base_model: Path, trust_remote_code: bool):
    full_config = AutoConfig.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    if full_config.model_type != "qwen3_5":
        raise ValueError(f"Expected model_type='qwen3_5', found {full_config.model_type!r}")
    if not hasattr(full_config, "text_config"):
        raise ValueError("Qwen3.5 config does not contain text_config")

    text_config = full_config.text_config
    if text_config.model_type != "qwen3_5_text":
        raise ValueError(
            f"Expected text model_type='qwen3_5_text', found {text_config.model_type!r}"
        )
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    text_config.use_cache = True
    # Qwen3_5ForCausalLM does not contain the multimodal wrapper's MTP module.
    # Keeping a non-zero declaration makes some GGUF converters expect absent weights.
    text_config.mtp_num_hidden_layers = 0
    return text_config


def rewrite_adapter_for_text_model(adapter_dir: Path, destination: Path) -> tuple[int, int]:
    """Remove the multimodal ``language_model`` level from PEFT tensor keys."""
    source_path = adapter_dir / ADAPTER_WEIGHTS
    destination.mkdir()

    rewritten: dict[str, torch.Tensor] = {}
    with safe_open(source_path, framework="pt", device="cpu") as source:
        keys = list(source.keys())
        if not keys:
            raise RuntimeError(f"Adapter contains no tensors: {source_path}")
        for source_key in keys:
            if ".language_model." not in source_key:
                raise RuntimeError(
                    "The adapter is not language-model-only; refusing to discard this tensor: "
                    f"{source_key}"
                )
            target_key = source_key.replace(".language_model.", ".", 1)
            if target_key in rewritten:
                raise RuntimeError(f"Duplicate adapter key after rewriting: {target_key}")
            rewritten[target_key] = source.get_tensor(source_key).contiguous()

    tensor_count = len(rewritten)
    parameter_count = sum(tensor.numel() for tensor in rewritten.values())
    shutil.copy2(adapter_dir / "adapter_config.json", destination / "adapter_config.json")
    save_file(rewritten, destination / ADAPTER_WEIGHTS, metadata={"format": "pt"})
    return tensor_count, parameter_count


def merge(args: argparse.Namespace) -> None:
    text_config = make_text_config(args.base_model, args.trust_remote_code)
    temp_parent = str(args.work_dir) if args.work_dir is not None else None

    with tempfile.TemporaryDirectory(prefix="weclone-qwen35-text-", dir=temp_parent) as temp_dir:
        rewritten_adapter = Path(temp_dir) / "adapter"
        print("[1/4] Rewriting LoRA keys for the text-only architecture...")
        tensor_count, parameter_count = rewrite_adapter_for_text_model(
            args.adapter, rewritten_adapter
        )
        print(f"      Prepared {tensor_count} tensors / {parameter_count:,} LoRA parameters.")

        print(
            f"[2/4] Loading only Qwen3.5 language weights ({args.dtype}, "
            f"device_map={args.device_map})..."
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=text_config,
            dtype=DTYPES[args.dtype],
            device_map=args.device_map,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )

        print("[3/4] Merging the rewritten LoRA adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            rewritten_adapter,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
        merged_model = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
        merged_model.config.use_cache = True
        merged_model.eval()

        print(f"[4/4] Saving text-only model to {args.output_dir}...")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(
            args.output_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )

    tokenizer_source = (
        args.adapter if (args.adapter / "tokenizer_config.json").is_file() else args.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(args.output_dir)

    source_size = sum(path.stat().st_size for path in args.base_model.glob("*.safetensors"))
    output_size = sum(path.stat().st_size for path in args.output_dir.glob("*.safetensors"))
    merge_info = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "dtype": args.dtype,
        "format": "merged_qwen3_5_text_only",
        "source_weight_bytes": source_size,
        "output_weight_bytes": output_size,
        "removed_weight_bytes": max(0, source_size - output_size),
    }
    (args.output_dir / "merge_info.json").write_text(
        json.dumps(merge_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    resolve_and_validate(args)
    print("The default CPU device map does not consume GPU memory during merging.")
    try:
        merge(args)
    except Exception:
        if args.output_dir.exists() and not any(args.output_dir.iterdir()):
            args.output_dir.rmdir()
        raise
    print(f"Text-only merge completed successfully: {args.output_dir}")


if __name__ == "__main__":
    main()
