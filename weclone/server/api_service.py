"""Start WeClone's Hugging Face or native vLLM API server."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, cast

import uvicorn

from weclone.utils.config import load_config
from weclone.utils.config_models import VllmArgs, WCInferConfig
from weclone.utils.log import logger


_RESERVED_VLLM_ARGS = {
    "api_key",
    "enable_lora",
    "host",
    "lora_modules",
    "model",
    "model_tag",
    "override_generation_config",
    "port",
    "root_path",
    "served_model_name",
}


def _local_model_path(model: str) -> str:
    """Resolve existing local model paths while preserving Hub model IDs."""
    candidate = Path(model).expanduser()
    return str(candidate.resolve()) if candidate.exists() else model


def _uses_bitsandbytes(model: str) -> bool:
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return False

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    quantization_config = config.get("quantization_config") or {}
    return quantization_config.get("quant_method") == "bitsandbytes"


def _append_vllm_option(command: list[str], name: str, value: Any) -> None:
    if value is None or value is False:
        return

    flag = f"--{name.replace('_', '-')}"
    if value is True:
        command.append(flag)
    elif isinstance(value, (dict, list)):
        command.extend((flag, json.dumps(value, ensure_ascii=False)))
    else:
        command.extend((flag, str(value)))


def build_native_vllm_command(
    infer_config: WCInferConfig,
    vllm_config: VllmArgs,
    environ: Mapping[str, str] | None = None,
    executable: str = "vllm",
) -> list[str]:
    """Translate WeClone settings into a vLLM 0.27 native serve command."""
    env = os.environ if environ is None else environ
    model = _local_model_path(infer_config.model_name_or_path)
    command = [executable, "serve", model]
    requested_model_name = env.get("API_MODEL_NAME", "gpt-3.5-turbo")
    served_base_name = (
        f"{requested_model_name}-base"
        if infer_config.adapter_name_or_path
        else requested_model_name
    )

    command.extend(("--host", env.get("API_HOST", "0.0.0.0")))
    command.extend(("--port", env.get("API_PORT", "8005")))
    command.extend(("--served-model-name", served_base_name))

    api_key = env.get("API_KEY")
    if api_key:
        command.extend(("--api-key", api_key))

    root_path = env.get("FASTAPI_ROOT_PATH")
    if root_path:
        command.extend(("--root-path", root_path))

    generation_config = {
        "temperature": infer_config.temperature,
        "top_p": infer_config.top_p,
        "repetition_penalty": infer_config.repetition_penalty,
        "max_new_tokens": infer_config.max_length,
    }
    command.extend(
        ("--override-generation-config", json.dumps(generation_config, ensure_ascii=False))
    )

    engine_options = vllm_config.model_dump(exclude_none=True)
    conflicting = _RESERVED_VLLM_ARGS.intersection(engine_options)
    if conflicting:
        names = ", ".join(sorted(conflicting))
        raise ValueError(f"Reserved vllm_args cannot be overridden: {names}")

    if _uses_bitsandbytes(model):
        engine_options.setdefault("quantization", "bitsandbytes")
        engine_options.setdefault("load_format", "bitsandbytes")

    for name, value in engine_options.items():
        _append_vllm_option(command, name, value)

    if infer_config.adapter_name_or_path:
        adapter = _local_model_path(infer_config.adapter_name_or_path)
        command.extend(
            ("--enable-lora", "--lora-modules", f"{requested_model_name}={adapter}")
        )

    return command


def _run_huggingface_server(config: WCInferConfig) -> None:
    from llamafactory.api.app import create_app
    from llamafactory.chat import ChatModel

    chat_model = ChatModel(config.model_dump(mode="json"))
    app = create_app(chat_model)
    port = int(os.environ.get("API_PORT", "8005"))
    print(f"Visit http://localhost:{port}/docs for API documentation.")
    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=port,
        workers=1,
    )


def _run_native_vllm_server(config: WCInferConfig) -> None:
    vllm_config = cast(VllmArgs, load_config("vllm"))
    executable = shutil.which("vllm") or str(Path(sys.executable).with_name("vllm"))
    if not Path(executable).is_file():
        raise FileNotFoundError("vLLM executable is missing. Run: uv sync")

    command = build_native_vllm_command(config, vllm_config, executable=executable)
    port = os.environ.get("API_PORT", "8005")
    logger.info("Starting the native vLLM 0.27 OpenAI-compatible server.")
    print(f"Visit http://localhost:{port}/docs for API documentation.", flush=True)

    # vLLM's CUDA kernels are bundled in its wheel. Hide an independently
    # installed flash-attn package from the service process so an extension
    # compiled against another PyTorch ABI cannot break engine startup.
    environment = os.environ.copy()
    shim_root = Path(__file__).resolve().parents[1] / "_vllm_shims"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{shim_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(shim_root)
    )
    os.execve(executable, command, environment)


def main() -> None:
    config = cast(WCInferConfig, load_config("api_service"))
    if config.infer_backend == "vllm":
        _run_native_vllm_server(config)
    else:
        _run_huggingface_server(config)


if __name__ == "__main__":
    main()
