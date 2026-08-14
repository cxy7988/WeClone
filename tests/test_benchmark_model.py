import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from weclone.eval.benchmark_model import (
    BenchmarkDataset,
    BenchmarkRunner,
    LocalChatCompletionClient,
    load_benchmark_dataset,
    render_report,
    save_results,
    summarize_results,
)
from weclone.utils.config_models import WCBenchmarkConfig


class FakeCompletionClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = []

    def complete(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.responses.pop(0)


def make_config(**overrides) -> WCBenchmarkConfig:
    values = {
        "template": "qwen",
        "default_system": "act human",
        "data_path": "dataset/benchmark/benchmark.sample.json",
        "output_dir": "benchmark_results",
        "run_name": "checkpoint-100",
        "candidate": {
            "base_url": "http://candidate/v1",
            "model": "candidate-model",
            "max_retries": 0,
            "retry_delay": 0,
        },
        "judge": {
            "base_url": "http://judge/v1",
            "model": "judge-model",
            "max_retries": 0,
            "retry_delay": 0,
        },
        "max_workers": 1,
    }
    values.update(overrides)
    return WCBenchmarkConfig.model_validate(values)


def minimal_dataset() -> BenchmarkDataset:
    return BenchmarkDataset.model_validate(
        {
            "version": 1,
            "style_examples": [
                {
                    "messages": [{"role": "user", "content": "忙完了吗"}],
                    "reference": "刚忙完",
                }
            ],
            "samples": [
                {
                    "id": "held-out-1",
                    "category": "daily",
                    "messages": [{"role": "user", "content": "吃了吗"}],
                    "reference": "还没呢",
                }
            ],
        }
    )


def test_sample_dataset_is_valid() -> None:
    dataset, digest = load_benchmark_dataset("dataset/benchmark/benchmark.sample.json")

    assert len(dataset.style_examples) == 4
    assert len(dataset.samples) == 6
    assert len(digest) == 64
    assert all(sample.messages[-1].role == "user" for sample in dataset.samples)


def test_dataset_rejects_style_sample_overlap() -> None:
    duplicate = {
        "messages": [{"role": "user", "content": "same"}],
        "reference": "same response",
    }

    with pytest.raises(ValidationError, match="style examples must be separate"):
        BenchmarkDataset.model_validate(
            {
                "version": 1,
                "style_examples": [duplicate],
                "samples": [{"id": "duplicate", **duplicate}],
            }
        )


def test_runner_generates_one_candidate_and_uses_judge_scores() -> None:
    candidate = FakeCompletionClient(["我刚吃完"])
    judge_payload = {
        "scores": {
            "style_similarity": 4,
            "relevance": 5,
            "coherence": 5,
            "naturalness": 4,
            "persona_consistency": 4,
            "overall": 4,
        },
        "verdict": "pass",
        "issues": [],
        "reason": "回复简短自然且切题。",
    }
    judge = FakeCompletionClient([f"```json\n{json.dumps(judge_payload, ensure_ascii=False)}\n```"])
    config = make_config()

    results = BenchmarkRunner(
        config,
        candidate_client=candidate,
        judge_client=judge,
    ).run(minimal_dataset())

    assert len(candidate.calls) == 1
    assert len(judge.calls) == 1
    assert results[0]["candidate_response"] == "我刚吃完"
    assert results[0]["verdict"] == "pass"
    assert results[0]["scores"]["overall"] == 4.0
    assert candidate.calls[0]["messages"][0] == {"role": "system", "content": "act human"}
    assert "<evaluation_data_json>" in judge.calls[0]["messages"][1]["content"]
    assert '"reference": "还没呢"' in judge.calls[0]["messages"][1]["content"]
    assert judge.calls[0]["kwargs"]["json_response_format"] is True


def test_local_candidate_loads_model_once_and_supports_adapter(tmp_path) -> None:
    model_path = tmp_path / "base-model"
    adapter_path = tmp_path / "checkpoint-100"
    model_path.mkdir()
    adapter_path.mkdir()
    captured = {}

    class FakeChatModel:
        def __init__(self, args):
            captured["args"] = args
            captured["loads"] = captured.get("loads", 0) + 1

        def chat(self, messages, system=None, **kwargs):
            captured["messages"] = messages
            captured["system"] = system
            captured["kwargs"] = kwargs
            return [SimpleNamespace(response_text="本地回复")]

    config = make_config(
        local_model_path=str(model_path),
        local_adapter_path=str(adapter_path),
        local_infer_backend="huggingface",
    )
    client = LocalChatCompletionClient(config, FakeChatModel)

    response = client.complete(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=64,
    )

    assert response == "本地回复"
    assert captured["loads"] == 1
    assert captured["args"]["model_name_or_path"] == str(model_path.resolve())
    assert captured["args"]["adapter_name_or_path"] == str(adapter_path.resolve())
    assert captured["args"]["infer_backend"] == "huggingface"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["system"] == "system prompt"
    assert captured["kwargs"]["max_new_tokens"] == 64


def test_local_vllm_candidate_bypasses_llamafactory_chat_model(tmp_path) -> None:
    model_path = tmp_path / "merged-model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    captured = {}

    class FakeVllm:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["chat_kwargs"] = kwargs
            return [SimpleNamespace(outputs=[SimpleNamespace(text="vLLM 本地回复")])]

    def fake_sampling_params(**kwargs):
        captured["sampling_params"] = kwargs
        return kwargs

    config = make_config(
        local_model_path=str(model_path),
        local_adapter_path=None,
        local_infer_backend="vllm",
        local_vllm_config={"gpu_memory_utilization": 0.8},
    )
    client = LocalChatCompletionClient(
        config,
        vllm_factory=FakeVllm,
        sampling_params_factory=fake_sampling_params,
    )

    response = client.complete(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.5,
        top_p=0.65,
        max_tokens=128,
        seed=42,
    )

    assert response == "vLLM 本地回复"
    assert captured["engine_args"]["model"] == str(model_path.resolve())
    assert captured["engine_args"]["gpu_memory_utilization"] == 0.8
    assert captured["engine_args"]["enable_lora"] is False
    assert captured["sampling_params"]["temperature"] == 0.5
    assert captured["sampling_params"]["top_p"] == 0.65
    assert captured["chat_kwargs"]["use_tqdm"] is False
    assert captured["messages"][0] == {"role": "system", "content": "system prompt"}


def test_local_vllm_remaps_qwen35_language_model_prefix_without_copying_weights(tmp_path) -> None:
    model_path = tmp_path / "merged-text"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "architectures": ["Qwen3_5ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.layers.0.input_layernorm.weight": (
                        "model-00001-of-00001.safetensors"
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeRegistry:
        @staticmethod
        def register_model(architecture, model_class):
            captured["registration"] = (architecture, model_class)

    class FakeVllm:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    config = make_config(
        local_model_path=str(model_path),
        local_adapter_path=None,
        local_infer_backend="vllm",
    )
    LocalChatCompletionClient(
        config,
        vllm_factory=FakeVllm,
        sampling_params_factory=lambda **kwargs: kwargs,
        vllm_model_registry=FakeRegistry,
    )

    assert captured["registration"] == (
        "WeCloneQwen3_5ForCausalLM",
        "weclone.eval.vllm_qwen35:WeCloneQwen3_5ForCausalLM",
    )
    assert captured["engine_args"]["model"] == str(model_path.resolve())
    assert captured["engine_args"]["hf_overrides"]["architectures"] == [
        "WeCloneQwen3_5ForCausalLM"
    ]


def test_summary_and_report_only_aggregate_judge_output(tmp_path) -> None:
    config = make_config(output_dir=str(tmp_path))
    results = [
        {
            "id": "one",
            "category": "daily",
            "status": "ok",
            "verdict": "fail",
            "scores": {
                "style_similarity": 2.0,
                "relevance": 4.0,
                "coherence": 4.0,
                "naturalness": 3.0,
                "persona_consistency": 2.0,
                "overall": 3.0,
            },
            "judge_decisions": [{"reason": "风格不像。"}],
        }
    ]

    summary = summarize_results(results, config, "a" * 64)
    report = render_report(summary, results)
    output_dir = save_results(results, summary, tmp_path, "checkpoint/100")

    assert summary["pass_rate"] == 0.0
    assert summary["scores"]["style_similarity"] == 2.0
    assert "Judge 通过率：0.00%" in report
    assert "风格不像" in report
    assert output_dir.name.endswith("-checkpoint_100")
    assert (output_dir / "samples.jsonl").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "report.md").is_file()
