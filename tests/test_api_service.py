import json
import os
from pathlib import Path

from weclone.server import api_service
from weclone.server.api_service import build_native_vllm_command
from weclone.utils.config_models import VllmArgs, WCInferConfig


def _infer_config(**overrides) -> WCInferConfig:
    values = {
        "model_name_or_path": "Qwen/Qwen3.5-4B",
        "adapter_name_or_path": None,
        "infer_backend": "vllm",
        "repetition_penalty": 1.2,
        "temperature": 0.5,
        "top_p": 0.65,
        "max_length": 256,
        "template": "qwen",
        "default_system": "You are a helpful assistant.",
    }
    values.update(overrides)
    return WCInferConfig(**values)


def test_native_vllm_command_uses_openai_server_defaults() -> None:
    command = build_native_vllm_command(
        _infer_config(),
        VllmArgs(gpu_memory_utilization=0.8),
        environ={},
    )

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3.5-4B"]
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert command[command.index("--port") + 1] == "8005"
    assert command[command.index("--served-model-name") + 1] == "gpt-3.5-turbo"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.8"

    generation_config = json.loads(command[command.index("--override-generation-config") + 1])
    assert generation_config == {
        "temperature": 0.5,
        "top_p": 0.65,
        "repetition_penalty": 1.2,
        "max_new_tokens": 256,
    }


def test_native_vllm_command_enables_lora() -> None:
    command = build_native_vllm_command(
        _infer_config(adapter_name_or_path="org/adapter"),
        VllmArgs(),
        environ={"API_MODEL_NAME": "weclone"},
    )

    assert "--enable-lora" in command
    assert command[command.index("--lora-modules") + 1] == "weclone=org/adapter"
    assert command[command.index("--served-model-name") + 1] == "weclone-base"


def test_native_vllm_process_hides_external_flash_attn(monkeypatch) -> None:
    captured = {}
    vllm_executable = str(Path(api_service.sys.executable).with_name("vllm"))
    monkeypatch.setattr(api_service.shutil, "which", lambda _name: vllm_executable)
    monkeypatch.setattr(
        api_service.os,
        "execve",
        lambda executable, command, environment: captured.update(
            executable=executable,
            command=command,
            environment=environment,
        ),
    )
    monkeypatch.setattr(
        api_service,
        "load_config",
        lambda _name: VllmArgs(gpu_memory_utilization=0.8),
    )

    api_service._run_native_vllm_server(_infer_config())

    shim_root = Path(captured["environment"]["PYTHONPATH"].split(os.pathsep)[0])
    assert (shim_root / "flash_attn" / "__init__.py").is_file()
