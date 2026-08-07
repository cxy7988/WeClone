#!/usr/bin/env bash

set -u -o pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_DIR="${PROJECT_DIR}/logs"
readonly LOG_FILE="${LOG_DIR}/train_sft_$(date '+%Y%m%d_%H%M%S').log"
readonly CUDA_TOOLKIT_DIR="/usr/local/cuda-12.6"

if [[ ! -x "${CUDA_TOOLKIT_DIR}/bin/nvcc" ]]; then
    echo "CUDA compiler not found: ${CUDA_TOOLKIT_DIR}/bin/nvcc" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_TOOLKIT_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export TORCH_EXTENSIONS_DIR="/tmp/weclone_torch_extensions_${UID}"

if [[ -x "${PROJECT_DIR}/.venv/bin/deepspeed" ]]; then
    readonly DEEPSPEED_BIN="${PROJECT_DIR}/.venv/bin/deepspeed"
else
    readonly DEEPSPEED_BIN="deepspeed"
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

echo "Training log: ${LOG_FILE}"
PYTHONUNBUFFERED=1 "${DEEPSPEED_BIN}" --num_gpus=2 weclone/train/train_sft.py "$@" 2>&1 | tee "${LOG_FILE}"
readonly TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "Training exited with code ${TRAIN_EXIT_CODE}. Log saved to: ${LOG_FILE}"
exit "${TRAIN_EXIT_CODE}"
