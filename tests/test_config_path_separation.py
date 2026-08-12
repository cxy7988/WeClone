from copy import deepcopy

from weclone.utils.config import create_config_by_arg_type
from weclone.utils.config_models import WcConfig


def make_config() -> WcConfig:
    return WcConfig.model_validate(
        {
            "version": "0.3.04",
            "common_args": {
                "template": "qwen",
                "default_system": "system",
            },
            "cli_args": {},
            "make_dataset_args": {"platform": "chat"},
            "train_sft_args": {
                "model_name_or_path": "/models/base",
                "output_dir": "/outputs/train",
                "dataset": "train-data",
                "lora_target": "all",
            },
            "infer_args": {
                "model_name_or_path": "/models/merged",
                "adapter_name_or_path": None,
                "temperature": 0.5,
                "top_p": 0.7,
                "max_length": 256,
            },
            "vllm_args": {},
            "test_model_args": {},
        }
    )


def test_training_and_inference_paths_are_independent() -> None:
    config = make_config()

    training = create_config_by_arg_type("train_sft", config)
    inference = create_config_by_arg_type("web_demo", config)
    dataset = create_config_by_arg_type("make_dataset", config)

    assert training.model_name_or_path == "/models/base"
    assert training.output_dir == "/outputs/train"
    assert not hasattr(training, "adapter_name_or_path")
    assert inference.model_name_or_path == "/models/merged"
    assert inference.adapter_name_or_path is None
    assert dataset.model_name_or_path == "/models/base"


def test_resume_and_inference_adapters_are_independent() -> None:
    raw = make_config().model_dump()
    raw["train_sft_args"]["resume_adapter_name_or_path"] = "/adapters/resume"
    raw["infer_args"]["model_name_or_path"] = "/models/base-for-inference"
    raw["infer_args"]["adapter_name_or_path"] = "/adapters/inference"
    config = WcConfig.model_validate(deepcopy(raw))

    training = create_config_by_arg_type("train_sft", config)
    inference = create_config_by_arg_type("api_service", config)

    assert training.adapter_name_or_path == "/adapters/resume"
    assert training.output_dir == "/outputs/train"
    assert inference.model_name_or_path == "/models/base-for-inference"
    assert inference.adapter_name_or_path == "/adapters/inference"
