import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_qwen35_text_lora.py"
SPEC = importlib.util.spec_from_file_location("merge_qwen35_text_lora", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rewrite_adapter_removes_language_model_level(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    source_key = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    save_file({source_key: torch.ones(2, 3)}, adapter / MODULE.ADAPTER_WEIGHTS)

    destination = tmp_path / "rewritten"
    tensor_count, parameter_count = MODULE.rewrite_adapter_for_text_model(adapter, destination)

    assert (tensor_count, parameter_count) == (1, 6)
    with safe_open(destination / MODULE.ADAPTER_WEIGHTS, framework="pt", device="cpu") as weights:
        assert list(weights.keys()) == [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        ]


def test_rewrite_adapter_rejects_non_language_tensor(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    save_file(
        {"base_model.model.model.visual.proj.lora_A.weight": torch.ones(1)},
        adapter / MODULE.ADAPTER_WEIGHTS,
    )

    with pytest.raises(RuntimeError, match="not language-model-only"):
        MODULE.rewrite_adapter_for_text_model(adapter, tmp_path / "rewritten")


def test_make_text_config_disables_multimodal_mtp(monkeypatch, tmp_path):
    text_config = SimpleNamespace(
        model_type="qwen3_5_text", architectures=None, use_cache=False, mtp_num_hidden_layers=1
    )
    full_config = SimpleNamespace(model_type="qwen3_5", text_config=text_config)
    monkeypatch.setattr(MODULE.AutoConfig, "from_pretrained", lambda *args, **kwargs: full_config)

    result = MODULE.make_text_config(tmp_path, trust_remote_code=False)

    assert result is text_config
    assert result.architectures == ["Qwen3_5ForCausalLM"]
    assert result.use_cache is True
    assert result.mtp_num_hidden_layers == 0


def test_resolve_rejects_non_empty_output(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "output"
    for directory in (base, adapter, output):
        directory.mkdir()
    (base / "config.json").write_text(json.dumps({}), encoding="utf-8")
    (adapter / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
    save_file({"weight": torch.ones(1)}, adapter / MODULE.ADAPTER_WEIGHTS)
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    args = SimpleNamespace(
        base_model=base, adapter=adapter, output_dir=output, work_dir=None
    )

    with pytest.raises(FileExistsError, match="not empty"):
        MODULE.resolve_and_validate(args)
