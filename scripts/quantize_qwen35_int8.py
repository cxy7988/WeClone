#!/usr/bin/env python3
"""Quantize a merged Qwen3.5 model from BF16 to bitsandbytes LLM.int8()."""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)


DEFAULT_MODEL = Path("model_output/Qwen3.5-4B-SFT-1500-merged")
DEFAULT_OUTPUT = Path("model_output/Qwen3.5-4B-SFT-1500-merged-int8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize Qwen3.5 Linear weights to bitsandbytes LLM.int8 and save a reloadable "
            "Hugging Face model directory. No calibration dataset is required."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Merged BF16 model directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New INT8 model directory.")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index (default: 0).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="LLM.int8 outlier threshold (default: 6.0).",
    )
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum output shard size.")
    parser.add_argument(
        "--skip-module",
        action="append",
        default=None,
        help="Module name to retain in BF16; repeat the option for multiple modules.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow local custom model code.")
    return parser.parse_args()


def require_bitsandbytes() -> str:
    try:
        installed = version("bitsandbytes")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "bitsandbytes is not installed. Run: uv sync --group main"
        ) from error

    major_minor = tuple(int(part) for part in installed.split("+")[0].split(".")[:2])
    if major_minor < (0, 46):
        raise RuntimeError(f"bitsandbytes>=0.46.1 is required, found {installed}")
    return installed


def validate_paths(model_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Model config is missing: {model_dir / 'config.json'}")
    if model_dir == output_dir:
        raise ValueError("--output must not overwrite --model")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return model_dir, output_dir


def quantize(args: argparse.Namespace) -> None:
    model_dir, output_dir = validate_paths(args.model, args.output)
    bnb_version = require_bitsandbytes()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bitsandbytes INT8 quantization")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device {args.device}; found {torch.cuda.device_count()} device(s)")

    config = AutoConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=args.trust_remote_code
    )
    if config.model_type != "qwen3_5":
        raise ValueError(f"Expected model_type='qwen3_5', found {config.model_type!r}")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=args.threshold,
        llm_int8_skip_modules=args.skip_module,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
    )

    print(f"Source: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: cuda:{args.device} ({torch.cuda.get_device_name(args.device)})")
    print(f"bitsandbytes: {bnb_version}; threshold: {args.threshold}")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map={"": args.device},
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.save_pretrained(output_dir, max_shard_size=args.max_shard_size)

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=args.trust_remote_code
        )
        tokenizer.save_pretrained(output_dir)
        processor = AutoProcessor.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=args.trust_remote_code
        )
        processor.save_pretrained(output_dir)

        source_modelfile = model_dir / "Modelfile"
        if source_modelfile.is_file():
            shutil.copy2(source_modelfile, output_dir / source_modelfile.name)

        linear8bit_count = sum(
            module.__class__.__name__ == "Linear8bitLt" for module in model.modules()
        )
        metadata = {
            "source_model": str(model_dir),
            "format": "bitsandbytes_llm_int8",
            "bitsandbytes_version": bnb_version,
            "transformers_version": version("transformers"),
            "torch_version": torch.__version__,
            "source_dtype": "bfloat16",
            "threshold": args.threshold,
            "skip_modules": args.skip_module or [],
            "linear8bit_modules": linear8bit_count,
        }
        (output_dir / "quantization_info.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if linear8bit_count == 0:
            raise RuntimeError("No Linear modules were quantized; refusing to report success")
        print(f"Quantized Linear modules: {linear8bit_count}")
        print("INT8 model saved successfully.")
    except Exception:
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise


if __name__ == "__main__":
    quantize(parse_args())
