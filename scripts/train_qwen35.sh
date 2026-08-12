#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONFIG_PATH="${WECLONE_CONFIG_PATH:-${PROJECT_DIR}/settings.jsonc}"
readonly MODEL_DIR="/home/hdd/2/cxy/FujiLLM_finetuning/models/Qwen3.5-4B"
readonly TRAIN_DATA="${PROJECT_DIR}/dataset/res_csv/sft/sft-my-cleaned-train.json"
readonly EVAL_DATA="${PROJECT_DIR}/dataset/res_csv/sft/sft-my-cleaned-val.json"
readonly DATASET_INFO="${PROJECT_DIR}/dataset/res_csv/sft/dataset_info.json"
readonly PYTHON_BIN="${WECLONE_PYTHON:-${PROJECT_DIR}/.venv/bin/python}"
readonly LOG_DIR="${PROJECT_DIR}/logs"
readonly LOG_FILE="${LOG_DIR}/train_qwen35_$(date '+%Y%m%d_%H%M%S').log"

require_file() {
    local file_path="$1"
    if [[ ! -f "${file_path}" ]]; then
        echo "缺少必需文件：${file_path}" >&2
        exit 1
    fi
}

require_file "${CONFIG_PATH}"
require_file "${MODEL_DIR}/config.json"
require_file "${TRAIN_DATA}"
require_file "${EVAL_DATA}"
require_file "${DATASET_INFO}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "找不到可执行的 Python：${PYTHON_BIN}" >&2
    echo "请用 WECLONE_PYTHON=/path/to/compatible/python 指定训练环境。" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${MODEL_DIR}" <<'PY'
import importlib.util
import sys

required_modules = (
    "causal_conv1d",
    "fla",
    "flash_attn",
    "llamafactory",
    "pyjson5",
    "torch",
    "transformers",
    "wandb",
)
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("训练环境缺少依赖：" + ", ".join(missing))

import torch
from transformers import AutoConfig

if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用，已停止启动训练。")

if not torch.cuda.is_bf16_supported():
    raise SystemExit("当前 GPU 不支持 BF16，已停止启动训练。")

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.ops.gated_delta_rule import (  # noqa: F401
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
except ImportError as error:
    raise SystemExit(
        "Qwen3.5 所需的 FLA 内核不可用；"
        "请安装 flash-linear-attention[cuda]==0.5.2 和 causal-conv1d==1.6.2.post1。"
    ) from error

missing_kernels = [
    name
    for name, kernel in (
        ("causal_conv1d_fn", causal_conv1d_fn),
        ("causal_conv1d_update", causal_conv1d_update),
        ("chunk_gated_delta_rule", chunk_gated_delta_rule),
        ("fused_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule),
    )
    if kernel is None
]
if missing_kernels:
    raise SystemExit("Qwen3.5 快速线性注意力内核不完整：" + ", ".join(missing_kernels))

try:
    model_config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
except Exception as error:
    raise SystemExit(
        "当前 Transformers 无法识别本地 Qwen3.5 模型；"
        "请通过 WECLONE_PYTHON 指定支持 qwen3_5 的训练环境。"
    ) from error

if model_config.model_type != "qwen3_5":
    raise SystemExit(f"模型类型不匹配：{model_config.model_type!r}")
PY

mkdir -p "${LOG_DIR}" "${LOG_DIR}/wandb"
cd "${PROJECT_DIR}"

export WECLONE_CONFIG_PATH="${CONFIG_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ACCELERATE_USE_DEEPSPEED="false"
export WANDB_PROJECT="${WANDB_PROJECT:-WeClone-Qwen3.5-SFT}"
export WANDB_DIR="${WANDB_DIR:-${LOG_DIR}/wandb}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "模型：${MODEL_DIR}"
echo "配置：${CONFIG_PATH}"
echo "训练集：${TRAIN_DATA}"
echo "验证集：${EVAL_DATA}"
echo "GPU：${CUDA_VISIBLE_DEVICES}"
echo "DeepSpeed：关闭"
echo "W&B project：${WANDB_PROJECT}"
echo "日志：${LOG_FILE}"

set +e
"${PYTHON_BIN}" -m weclone.cli --config-path "${CONFIG_PATH}" train-sft 2>&1 | tee "${LOG_FILE}"
readonly TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e

if (( TRAIN_EXIT_CODE == 0 )); then
    echo "训练完成，日志保存在：${LOG_FILE}"
else
    echo "训练退出，状态码 ${TRAIN_EXIT_CODE}；日志保存在：${LOG_FILE}" >&2
fi

exit "${TRAIN_EXIT_CODE}"
