> [!IMPORTANT]
> - WeClone仍在快速迭代期，当前效果不代表最终效果。  
> - 微调LLM效果很大程度取决于模型大小、聊天数据的数量和质量，理论上模型越大，数据越多，效果越好。
> - 7B模型效果一般，14B及以上的模型效果会更好。   
> - Windows环境未进行严格测试，可以使用WSL作为运行环境。


### 硬件要求

项目默认使用Qwen2.5-7B-Instruct模型，LoRA方法对sft阶段微调，大约需要16GB显存。也可以使用[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md#%E6%A8%A1%E5%9E%8B)支持的其他模型和方法。

需要显存的估算值：
| 方法                             | 精度 |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | ---- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |  32  | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              |  16  |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |   8  |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |   4  |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |   2  |   4GB |   8GB |  16GB |   24GB | `x/4`GB |


## 环境搭建
1.cuda安装(已安装可跳过，**要求版本12.6及以上**)：[LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#cuda) 

2.建议使用 [uv](https://docs.astral.sh/uv/)安装依赖，这是一个非常快速的 Python 环境管理器。安装uv后，您可以使用以下命令创建一个新的Python环境并安装依赖项，速度较慢可以开启代理：
```bash
git clone https://github.com/xming521/WeClone.git && cd WeClone
uv venv .venv --python=3.12
source .venv/bin/activate # windows下执行 .venv\Scripts\activate
uv sync --frozen
uv pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl
```

3.将配置文件模板复制一份并重命名为`settings.jsonc`，后续配置修改在此文件进行：
```bash
cp settings.template.jsonc settings.jsonc
```
- 微调**多模态模型**时，请使用[examples/mllm.template.jsonc](https://github.com/xming521/WeClone/blob/master/examples/mllm.template.jsonc)作为配置文件。

> [!NOTE]
> 训练以及推理相关配置统一在文件`settings.jsonc`

4.使用以下命令测试CUDA环境是否正确配置并可被PyTorch识别，Mac不需要：
```bash
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available());"
```

5.（可选）安装FlashAttention，加速训练和推理：`uv pip install flash-attn --no-build-isolation` 版本问题可以使用[prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)的预编译包安装。

## 模型下载
中国境内推荐使用[ModelScope](https://www.modelscope.cn/docs/models/download)下载模型。例如下载WeClone默认模型：
```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct
```

## 数据准备

### Telegram
请使用[Telegram Desktop](https://desktop.telegram.org/)导出聊天记录，点击右上角点击导出聊天记录，选择照片类型，格式选择JSON。可以导出多个联系人（不建议使用群聊记录），然后将导出的`ChatExport_*`文件夹放在`./dataset/telegram`目录即可，也就是不同人聊天记录的文件夹一起放在 `./dataset/telegram`。


## 数据预处理
- 首先根据需要修改配置文件中的`language`、`platform`、`include_type`。
- 项目默认通过Microsoft Presidio去除了数据中的`电话号码、电子邮件地址、信用卡号码（12-19位数字）、IP地址、地理位置名称、国际银行账户号码、加密货币钱包地址、年龄信息、通用身份证号码`,但是不能保证100%过滤识别。
- 所以在`settings.jsonc`中提供了一个禁用词词库`blocked_words`，可以自行添加需要过滤的词句（会默认去掉包括禁用词的整句）。
> [!IMPORTANT]
> 🚨 请一定注意保护个人隐私，不要泄露个人信息！

- 执行以下命令对数据进行处理，可以先根据自己的聊天风格修改settings.jsonc的`make_dataset_args`。
```bash
weclone-cli make-dataset
```
数据处理更多参数说明：[数据预处理](https://docs.weclone.love/zh/docs/deploy/data_preprocessing.html#%E7%9B%B8%E5%85%B3%E5%8F%82%E6%95%B0)

## 配置参数并微调模型

当前本地训练配置用于对
`/home/hdd/2/cxy/FujiLLM_finetuning/models/Qwen3.5-4B` 进行 LoRA SFT，主要设置如下：

- 使用 BF16；全注意力层使用 FlashAttention 2，线性注意力层使用 FLA，关闭 FP16。
- 关闭 DeepSpeed，默认使用单张 GPU。
- 跳过本轮数据清洗，直接复用上一轮已经清洗的数据。
- 按时间将已清洗数据的前 95% 用于训练，时间最新的 5% 用于验证。
- 每 100 steps 记录验证损失并保存 checkpoint。
- 将训练和验证指标上报到 Weights & Biases。
- 训练结束后载入 `eval_loss` 最低的 checkpoint。

### 训练集与验证集

已清洗的 15,217 条对话按 `time` 字段严格升序划分：

| 用途 | 数据集名称 | 文件 | 数量 |
|------|------------|------|-----:|
| 训练 | `chat-sft-cleaned-train` | `dataset/res_csv/sft/sft-my-cleaned-train.json` | 14,456 |
| 验证 | `chat-sft-cleaned-val` | `dataset/res_csv/sft/sft-my-cleaned-val.json` | 761 |

两个数据集均在 `dataset/res_csv/sft/dataset_info.json` 中注册。原始的
`sft-my-cleaned.json` 保持不变。不要同时设置 `val_size`，否则会与显式的
`eval_dataset` 冲突。

训练过程中可以在 W&B 中比较 `train/loss` 和 `eval/loss`。如果训练损失继续下降，验证损失却开始上升，通常表示模型开始过拟合。

### Qwen3.5 运行环境

> [!IMPORTANT]
> 仓库现有 `.venv` 使用 Transformers 4.53.2，不能识别 `qwen3_5`，并且当前未安装
> `wandb`。启动训练前必须准备同时支持 Qwen3.5、LLaMA-Factory、FlashAttention 2
> 和 W&B 的 Python 环境。启动脚本只做兼容性检查，不会自动联网安装或修改依赖。

兼容环境至少需要能够导入以下模块：

```text
torch
transformers
llamafactory
flash_attn
fla
causal_conv1d
wandb
pyjson5
```

使用 W&B 在线监控前还需在该环境中完成一次登录：

```bash
/path/to/compatible/python -m wandb login
```

### 一键启动单卡训练

从项目根目录执行：

```bash
WECLONE_PYTHON=/path/to/compatible/python ./scripts/train_qwen35.sh
```

如果兼容环境就是项目的 `.venv`，可以直接执行：

```bash
./scripts/train_qwen35.sh
```

脚本启动前会检查模型、配置、训练集、验证集、CUDA、BF16 和依赖，然后执行
`weclone-cli train-sft`。终端输出同时写入：

```text
logs/train_qwen35_YYYYMMDD_HHMMSS.log
```

可通过环境变量选择 GPU 或覆盖 W&B 项目名：

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=WeClone-Qwen3.5-SFT \
WECLONE_PYTHON=/path/to/compatible/python \
./scripts/train_qwen35.sh
```

脚本会显式设置 `ACCELERATE_USE_DEEPSPEED=false`，与 `settings.jsonc` 中的配置一致。

### 显存与训练速度

有效 batch size 的计算方式为：

```text
per_device_train_batch_size × gradient_accumulation_steps × GPU 数量
```

在保持有效 batch size 不变的情况下，可以增大单卡 batch 并等比例降低梯度累积。例如单卡的 `1 × 16` 可以改为 `2 × 8`，通常能提高 GPU 利用率且不改变整体 batch size。

LLaMA-Factory 使用反向命名的 `disable_gradient_checkpointing` 控制梯度检查点：

```json5
"disable_gradient_checkpointing": true
```

`true` 表示关闭梯度检查点，可以加快训练，但会增加显存占用。

### W&B、验证与 checkpoint

相关的 `train_sft_args` 配置如下：

```json5
"report_to": "wandb",
"do_eval": true,
"eval_strategy": "steps",
"eval_steps": 100,
"save_strategy": "steps",
"save_steps": 100,
"load_best_model_at_end": true,
"metric_for_best_model": "eval_loss",
"greater_is_better": false
```

`eval_steps` 与 `save_steps` 需要保持兼容，才能正确载入最佳 checkpoint。

训练的输入、续训 adapter 和输出目录均在 `train_sft_args` 中独立设置：

```json5
"model_name_or_path": "/path/to/base-model",
"resume_adapter_name_or_path": null,
"output_dir": "./model_output/Qwen3.5-4B-SFT"
```

`resume_adapter_name_or_path` 只用于从已有 LoRA 继续训练；从头训练时保持 `null`。

按照当前保存频率，会在该目录下生成 `checkpoint-100`、`checkpoint-200` 等目录。需要恢复完整的优化器和学习率调度器状态时，可以在 `train_sft_args` 中设置：

```json5
"resume_from_checkpoint": "./model_output/Qwen3.5-4B-SFT/checkpoint-100"
```

从头重新训练时不要设置 `resume_from_checkpoint`，并使用一个新的或空的输出目录。

### 合并为纯文本 Qwen3.5 模型

Qwen3.5 的基础 checkpoint 带有视觉编码器。仅做文本聊天时，可以在合并 LoRA 时直接导出
`qwen3_5_text` 模型，输出不包含视觉塔，也不会在 Transformers/vLLM 加载时实例化视觉模块：

```bash
.venv/bin/python scripts/merge_qwen35_text_lora.py \
  --base-model /home/hdd/2/cxy/FujiLLM_finetuning/models/Qwen3.5-4B \
  --adapter model_output/Qwen3.5-4B-SFT/checkpoint-900 \
  --output-dir model_output/Qwen3.5-4B-SFT-900-merged-text
```

脚本默认使用 `--device-map cpu`，合并过程不占用显存；如需让 Transformers 自动使用
GPU/CPU，可显式传 `--device-map auto`。它会先确认 adapter 中只有语言模型 LoRA 权重，随后
把多模态权重路径转换为原生 `Qwen3_5ForCausalLM` 路径。输出只支持文本输入，不能再处理图片
或视频。后续 INT8 量化脚本同时支持这种 `qwen3_5_text` 输出。

### 导出未量化 Ollama GGUF

完成 LoRA 合并后，可以使用
[`scripts/convert_safetensors_to_ollama_gguf.sh`](scripts/convert_safetensors_to_ollama_gguf.sh)
将 Hugging Face Safetensors 模型转换为 Ollama 可用的 F16 GGUF。该脚本使用 Ollama
的标准 GGUF 转换路径，不传 `--experimental` 或 `--quantize`，因此不会执行 INT4/INT8
量化。Ollama 将 16 位浮点 GGUF 显示为 `F16`（16 BPW）。

> [!IMPORTANT]
> 输入目录必须是已经合并基础模型与 LoRA 的完整 Safetensors 模型，不能直接传入只包含
> adapter 的 checkpoint。运行脚本前还需要先启动 Ollama 服务。

先在一个终端启动 Ollama：

```bash
ollama serve
```

另开终端只导出 GGUF 和配套 Modelfile：

```bash
./scripts/convert_safetensors_to_ollama_gguf.sh \
  --source model_output/Qwen3.5-4B-SFT-900-merged \
  --output model_output/Qwen3.5-4B-SFT-900-ollama/qwen3.5-4b-sft-900-f16.gguf
```

如需在导出后直接注册到 Ollama，再传入 `--model`：

```bash
./scripts/convert_safetensors_to_ollama_gguf.sh \
  --source model_output/Qwen3.5-4B-SFT-900-merged \
  --output model_output/Qwen3.5-4B-SFT-900-ollama/qwen3.5-4b-sft-900-f16.gguf \
  --model qwen3.5-4b-sft-900
```

脚本默认拒绝覆盖已有 GGUF、Modelfile 或同名 Ollama 模型。确认需要覆盖时显式添加
`--force`。转换完成后会得到：

```text
qwen3.5-4b-sft-900-f16.gguf
qwen3.5-4b-sft-900-f16.gguf.Modelfile
```

对于配置中声明了 MTP 层、但 Safetensors 中没有对应 MTP 权重的 Qwen3.5 模型，脚本会
在临时硬链接目录中自动将 `mtp_num_hidden_layers` 修正为 `0`，不会修改原模型。当前导出
仅包含语言模型 GGUF，不附带多模态 projector，因此用于文本聊天，不支持图片输入。

将 Ollama 固定到物理 GPU 1 时，可以这样启动服务：

```bash
CUDA_VISIBLE_DEVICES=1 \
OLLAMA_VULKAN=false \
OLLAMA_CONTEXT_LENGTH=4096 \
OLLAMA_HOST=127.0.0.1:11434 \
ollama serve
```

另开终端运行已注册的模型：

```bash
ollama run qwen3.5-4b-sft-900
```

服务默认监听 `http://127.0.0.1:11434`。由于 `CUDA_VISIBLE_DEVICES` 会重新编号可见显卡，
Ollama 日志中的 `CUDA0` 在这里对应物理 GPU 1。

### 使用浏览器demo简单推理
推理输入独立放在 `infer_args` 中。使用已合并的完整模型时：

```json5
"model_name_or_path": "./model_output/Qwen3.5-4B-SFT-1500-merged",
"adapter_name_or_path": null
```

使用基础模型加未合并 LoRA 时：

```json5
"model_name_or_path": "/path/to/base-model",
"adapter_name_or_path": "./model_output/Qwen3.5-4B-SFT/checkpoint-1500"
```

测试出合适的 temperature、top_p 值，修改 `settings.jsonc` 的 `infer_args` 后启动：
```bash
weclone-cli webchat-demo
```

### 使用接口进行推理

```bash
weclone-cli server
```

当 `infer_backend` 为 `vllm` 时，该命令直接启动当前锁定版本的 vLLM 原生 OpenAI 兼容服务；
服务地址默认为 `http://127.0.0.1:8005/v1`，对外模型名默认为 `gpt-3.5-turbo`。
可通过 `API_HOST`、`API_PORT`、`API_MODEL_NAME` 和 `API_KEY` 环境变量覆盖这些设置。

### 使用常见聊天问题测试
不包含询问个人信息的问题，仅有日常聊天。测试结果在test_result-my.txt。
```bash
weclone-cli server
weclone-cli test-model
```

### 使用 Judge LLM 运行验证 benchmark

`benchmark-model` 每次只评测一个手动指定的生成模型，不自动进行 Base/LoRA
成对生成或 A/B 对比。生成模型和 Judge 都通过 OpenAI-compatible API 调用，所有质量判断均由
Judge LLM 完成，程序只负责校验数据、生成回复以及汇总 Judge 返回的分数。

先在 `settings.jsonc` 中配置 `benchmark_args`。完整配置示例见
[`settings.template.jsonc`](settings.template.jsonc)，其中关键字段如下：

```json5
"benchmark_args": {
    "data_path": "dataset/benchmark/benchmark.sample.json",
    "output_dir": "benchmark_results",
    "run_name": "qwen-lora-checkpoint-1500",
    "candidate": {
        "base_url": "http://127.0.0.1:8005/v1",
        "api_key": "sk-test",
        "model": "gpt-3.5-turbo"
    },
    "judge": {
        "base_url": "https://your-judge-api.example.com/v1",
        "api_key": "${oc.env:JUDGE_API_KEY,}",
        "model": "your-judge-model"
    },
    "judge_repetitions": 1,
    "max_workers": 4
}
```

通过 API 评测时，先启动待测模型的服务，然后运行：

```bash
weclone-cli server
JUDGE_API_KEY=your-key weclone-cli benchmark-model
```

也可以直接传入本地模型目录，不需要先启动待测模型 API。评测合并后的完整模型：

```bash
CUDA_VISIBLE_DEVICES=1 \
JUDGE_API_KEY=your-key weclone-cli benchmark-model \
  --model-path model_output/Qwen3-14B-checkpoint-800-merged-fp8 \
  --run-name 14b
```


本地模型只会加载一次。默认沿用 `infer_args.infer_backend`，也可以通过
`--local-backend vllm` 或 `--local-backend huggingface` 临时覆盖；vLLM 模式同时沿用
`vllm_args`。`--model-path` 和 API 模式的 `--model` 不能同时使用。也可以把本地路径固定写入
`benchmark_args.local_model_path` 和 `benchmark_args.local_adapter_path`。使用本地模式时
`benchmark_args.candidate` 会被忽略，也可以从配置中删除；Judge 配置仍然必须保留。

本地 `vllm` benchmark 会直接调用项目锁定的 vLLM 0.27.1，不经过 LLaMA-Factory 的
`ChatModel` vLLM 包装层。LLaMA-Factory 0.9.5 的包装层只接受 vLLM ≤0.11，直接调用可以避免
错误的版本限制，同时与 `weclone-cli server` 使用的运行时保持一致。`huggingface` 后端仍使用
LLaMA-Factory `ChatModel`。加载 vLLM 前还会优先使用项目内的 FlashAttention 兼容 shim，避免环境中
单独安装的 `flash_attn` 与当前 PyTorch/CUDA ABI 不一致而导致引擎进程在导入阶段退出。
