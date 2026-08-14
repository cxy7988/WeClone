#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SOURCE_DIR="${PROJECT_DIR}/model_output/Qwen3.5-4B-SFT-900-merged"
OUTPUT_GGUF="${PROJECT_DIR}/model_output/Qwen3.5-4B-SFT-900-ollama/qwen3.5-4b-sft-900-f16.gguf"
OLLAMA_MODEL=""
FORCE=false

STAGING_DIR=""
TEMP_MODEL=""
TEMP_OUTPUT=""
TEMP_MODELFILE=""

usage() {
    cat <<'EOF'
将 Hugging Face Safetensors 模型转换成 Ollama 可用的未量化 F16 GGUF。

用法：
  convert_safetensors_to_ollama_gguf.sh [选项]

选项：
  --source DIR     Safetensors 模型目录。
  --output FILE    输出 GGUF 文件。
  --model NAME     转换完成后注册为指定的 Ollama 模型；不传则只导出文件。
  --force          允许覆盖输出文件以及已存在的同名 Ollama 模型。
  -h, --help       显示帮助。

默认值：
  --source model_output/Qwen3.5-4B-SFT-900-merged
  --output model_output/Qwen3.5-4B-SFT-900-ollama/qwen3.5-4b-sft-900-f16.gguf

示例：
  ./scripts/convert_safetensors_to_ollama_gguf.sh \
    --source model_output/Qwen3.5-4B-SFT-900-merged \
    --output model_output/Qwen3.5-4B-SFT-900-ollama/qwen3.5-4b-sft-900-f16.gguf \
    --model qwen3.5-4b-sft-900

说明：
  1. 脚本调用 Ollama 的标准 GGUF 转换器，不使用 --experimental。
  2. 脚本不传 --quantize；Ollama 通常将结果显示为 F16（16 BPW）。
  3. 导出的是语言模型 GGUF，不附带多模态 projector。
  4. 如果 Qwen3.5 配置声明了 MTP 层但权重中不存在，脚本只在临时目录中
     将 mtp_num_hidden_layers 修正为 0，原模型不会被修改。
EOF
}

fail() {
    echo "错误：$*" >&2
    exit 1
}

require_command() {
    local command_name="$1"
    command -v "${command_name}" >/dev/null 2>&1 || fail "缺少命令：${command_name}"
}

cleanup() {
    local exit_code=$?

    if [[ -n "${TEMP_MODEL}" ]]; then
        ollama rm "${TEMP_MODEL}" >/dev/null 2>&1 || true
    fi

    if [[ -n "${STAGING_DIR}" && -d "${STAGING_DIR}" ]]; then
        local staging_parent
        staging_parent="$(dirname -- "${STAGING_DIR}")"
        if [[ "${STAGING_DIR}" == "${staging_parent}/.ollama-gguf."* ]]; then
            rm -rf -- "${STAGING_DIR}"
        else
            echo "警告：拒绝清理非预期临时目录：${STAGING_DIR}" >&2
        fi
    fi

    if [[ -n "${TEMP_OUTPUT}" && -f "${TEMP_OUTPUT}" ]]; then
        rm -f -- "${TEMP_OUTPUT}"
    fi
    if [[ -n "${TEMP_MODELFILE}" && -f "${TEMP_MODELFILE}" ]]; then
        rm -f -- "${TEMP_MODELFILE}"
    fi

    exit "${exit_code}"
}

trap cleanup EXIT

while (($# > 0)); do
    case "$1" in
        --source)
            (($# >= 2)) || fail "--source 缺少参数"
            SOURCE_DIR="$2"
            shift 2
            ;;
        --output)
            (($# >= 2)) || fail "--output 缺少参数"
            OUTPUT_GGUF="$2"
            shift 2
            ;;
        --model)
            (($# >= 2)) || fail "--model 缺少参数"
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "未知参数：$1"
            ;;
    esac
done

require_command ollama
require_command python3
require_command cp
require_command find
require_command mktemp

[[ -d "${SOURCE_DIR}" ]] || fail "模型目录不存在：${SOURCE_DIR}"
SOURCE_DIR="$(cd -- "${SOURCE_DIR}" && pwd -P)"

[[ -f "${SOURCE_DIR}/config.json" ]] || fail "缺少 config.json：${SOURCE_DIR}"
if ! find "${SOURCE_DIR}" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .; then
    fail "目录中没有 .safetensors 权重：${SOURCE_DIR}"
fi

OUTPUT_GGUF="$(python3 - "${OUTPUT_GGUF}" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
readonly OUTPUT_DIR="$(dirname -- "${OUTPUT_GGUF}")"
readonly OUTPUT_NAME="$(basename -- "${OUTPUT_GGUF}")"
readonly PORTABLE_MODELFILE="${OUTPUT_GGUF}.Modelfile"

[[ "${OUTPUT_NAME}" == *.gguf ]] || fail "--output 必须以 .gguf 结尾：${OUTPUT_GGUF}"
if [[ "${FORCE}" != true ]]; then
    [[ ! -e "${OUTPUT_GGUF}" ]] || fail "输出文件已存在；如需覆盖请传 --force：${OUTPUT_GGUF}"
    [[ ! -e "${PORTABLE_MODELFILE}" ]] || fail "Modelfile 已存在；如需覆盖请传 --force：${PORTABLE_MODELFILE}"
fi

if ! ollama list >/dev/null 2>&1; then
    fail "无法连接 Ollama。请先运行：ollama serve"
fi

if [[ -n "${OLLAMA_MODEL}" && "${FORCE}" != true ]]; then
    if ollama list | awk 'NR > 1 { print $1 }' | grep -Fxq "${OLLAMA_MODEL}" ||
        ollama list | awk 'NR > 1 { print $1 }' | grep -Fxq "${OLLAMA_MODEL}:latest"; then
        fail "Ollama 模型已存在；如需覆盖请传 --force：${OLLAMA_MODEL}"
    fi
fi

mkdir -p -- "${OUTPUT_DIR}"

readonly SOURCE_PARENT="$(dirname -- "${SOURCE_DIR}")"
STAGING_DIR="$(mktemp -d "${SOURCE_PARENT}/.ollama-gguf.XXXXXX")"

# 临时目录与源目录位于同一文件系统。硬链接不会重复占用约 9 GB 的权重空间。
cp -al -- "${SOURCE_DIR}/." "${STAGING_DIR}/"

# 原子替换临时目录中的 config.json/Modelfile，以先断开硬链接，避免修改源文件。
python3 - "${STAGING_DIR}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


staging_dir = Path(sys.argv[1])
config_path = staging_dir / "config.json"
index_path = staging_dir / "model.safetensors.index.json"
config = json.loads(config_path.read_text(encoding="utf-8"))

text_config = config.get("text_config")
if not isinstance(text_config, dict):
    text_config = config

base_layers = text_config.get("num_hidden_layers")
mtp_layers = text_config.get("mtp_num_hidden_layers", 0)
fixed_mtp = False

if (
    isinstance(base_layers, int)
    and isinstance(mtp_layers, int)
    and mtp_layers > 0
    and index_path.is_file()
):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_names = index.get("weight_map", {})
    layer_pattern = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.")
    present_layers = {
        int(match.group(1))
        for name in weight_names
        if (match := layer_pattern.search(name)) is not None
    }
    has_mtp_weights = any(layer >= base_layers for layer in present_layers)
    if present_layers and not has_mtp_weights:
        text_config["mtp_num_hidden_layers"] = 0
        fixed_mtp = True


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


atomic_write(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")

modelfile_path = staging_dir / "Modelfile"
if modelfile_path.is_file():
    lines = modelfile_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith("FROM "):
            lines[index] = "FROM ."
            replaced = True
            break
    if not replaced:
        lines.insert(0, "FROM .")
    modelfile = "\n".join(lines).rstrip() + "\n"
else:
    modelfile = "FROM .\n"
atomic_write(modelfile_path, modelfile)

if fixed_mtp:
    print(
        "检测到配置声明了 MTP 层，但 Safetensors 中没有相应权重；"
        "临时设置 mtp_num_hidden_layers=0。"
    )
PY

TEMP_MODEL="weclone-gguf-export-$(date '+%Y%m%d%H%M%S')-$$"

echo "源模型：${SOURCE_DIR}"
echo "输出 GGUF：${OUTPUT_GGUF}"
echo "临时 Ollama 模型：${TEMP_MODEL}"
echo "转换方式：标准 Ollama GGUF 转换，不量化"

# 关键约束：不要添加 --experimental 或 --quantize。
(
    cd -- "${STAGING_DIR}"
    ollama create "${TEMP_MODEL}" -f "${STAGING_DIR}/Modelfile"
)

readonly GENERATED_MODELFILE="${STAGING_DIR}/Generated.Modelfile"
ollama show --modelfile "${TEMP_MODEL}" > "${GENERATED_MODELFILE}"

MODEL_BLOB="$(python3 - "${GENERATED_MODELFILE}" <<'PY'
from pathlib import Path
import shlex
import sys

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if line.strip().startswith("FROM "):
        values = shlex.split(line.strip())
        if len(values) == 2:
            print(values[1])
            break
else:
    raise SystemExit("生成的 Modelfile 中没有 FROM 路径")
PY
)"

[[ -f "${MODEL_BLOB}" ]] || fail "找不到 Ollama 生成的 GGUF blob：${MODEL_BLOB}"

TEMP_OUTPUT="${OUTPUT_GGUF}.tmp.$$"
cp --reflink=auto --sparse=always -- "${MODEL_BLOB}" "${TEMP_OUTPUT}"
mv -f -- "${TEMP_OUTPUT}" "${OUTPUT_GGUF}"
TEMP_OUTPUT=""

TEMP_MODELFILE="${PORTABLE_MODELFILE}.tmp.$$"
python3 - "${GENERATED_MODELFILE}" "${TEMP_MODELFILE}" "${OUTPUT_NAME}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
gguf_name = sys.argv[3]
lines = source.read_text(encoding="utf-8").splitlines()

for index, line in enumerate(lines):
    if line.strip().startswith("FROM "):
        lines[index] = f"FROM ./{gguf_name}"
        break
else:
    raise SystemExit("生成的 Modelfile 中没有 FROM 指令")

destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
PY
mv -f -- "${TEMP_MODELFILE}" "${PORTABLE_MODELFILE}"
TEMP_MODELFILE=""

if [[ -n "${OLLAMA_MODEL}" ]]; then
    (
        cd -- "${OUTPUT_DIR}"
        ollama create "${OLLAMA_MODEL}" -f "${PORTABLE_MODELFILE}"
    )
fi

echo
echo "转换完成。"
echo "GGUF：${OUTPUT_GGUF}"
echo "Modelfile：${PORTABLE_MODELFILE}"
if [[ -n "${OLLAMA_MODEL}" ]]; then
    echo "Ollama 模型：${OLLAMA_MODEL}"
    echo "运行命令：ollama run ${OLLAMA_MODEL}"
else
    echo "注册命令：ollama create <模型名> -f '${PORTABLE_MODELFILE}'"
fi
