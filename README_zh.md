![download](https://github.com/user-attachments/assets/cd4a87c6-1649-4ce5-bce8-bd5b08b278de)
<h3 align="center">🚀 One-stop solution for creating your digital avatar from chat history 💡</h3>  
<h3 align="center">🚀从聊天记录创造数字分身的一站式解决方案💡</h3>  


<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/xming521/WeClone?style=for-the-badge&logo=github&label=Stars&logoColor=white&color=ffda65)](https://github.com/xming521/WeClone/stargazers)
[![GitHub release](https://img.shields.io/github/v/release/xming521/WeClone?style=for-the-badge&logo=github&label=Release&logoColor=white&color=06d094)](https://github.com/xming521/WeClone/releases)
<a href="https://qm.qq.com/cgi-bin/qm/qr?k=wNdgbOVT6oFOJ2wlMLsolUXErW9ESLpk&jump_from=webapi&authKey=z/reOp6YLyvR4Tl2k2nYMsLoMC3w9/99ucgKMX0oRGlxDV/WbYnvq2QxODoIkfxn" target="_blank" style="text-decoration: none;">
  <img src="https://img.shields.io/badge/QQ群-708067078-12B7F5?style=for-the-badge&logo=qq&logoColor=white" alt="WeClone①" title="WeClone①">
</a>
[![小红书](https://img.shields.io/badge/WeClone-FE2C55?style=for-the-badge&logo=xiaohongshu&logoColor=white)](https://www.xiaohongshu.com/user/profile/628109730000000021029de4)
[![Twitter](https://img.shields.io/badge/Twitter-@weclone567-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/weclone567)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/+JEdak4m0XEQ3NGNl)

<a href="https://hellogithub.com/repository/12ab209b56cb4cfd885c8cfd4cfdd53e" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=12ab209b56cb4cfd885c8cfd4cfdd53e&claim_uid=RThlPDoGrFvdMY5" alt="Featured｜HelloGitHub" style="width: 150px; height: 28px;" /></a>
<a href="https://trendshift.io/repositories/13759" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13759" alt="xming521%2FWeClone | Trendshift" style="width: 220px; height: 50px;" /></a>
<a href="https://deepwiki.com/xming521/WeClone"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"  style="width: 134px; height: 23px;margin-bottom: 3px;"></a>
</div>

<p align="center">
简体中文｜
  <a href="https://github.com/xming521/WeClone/blob/master/README.md" target="_blank">English</a>｜
  <a href="https://www.weclone.love/" target="_blank"> 项目主页 </a> ｜
  <a href="https://docs.weclone.love/docs/introduce/what-is-weclone.html" target="_blank"> 项目文档 </a>
  
</p>


## ✨核心功能
- 💫 涵盖打造数字分身的全链路方案，包括聊天数据导出、预处理、模型训练、部署
- 💬 使用聊天记录微调LLM，支持图片模态数据，让大模型有"那味儿"
- 🔗 绑定到Discord, Telegram, Slack, Feishu等，实现自己的数字分身
- 🛡️ 隐私信息过滤，本地化微调部署，数据安全可控

## 📋特性与说明

### 数据源平台适配

| 平台 | 文字 | 图片 | 语音 | 视频 | 动画表情 | 链接(分享) | 引用 | 转发 | 位置 | 文件 |
|------|------|------|------|------|----------|-----------|------|------|------|------|
| Telegram | ✅ | ✅ | ❌ | ❌ | ⚠️转为Emjoy | ❌ | ❌ | ✅ | ✅ | ❌ |
| WhatsApp | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Discord | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Slack | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

### 部署平台支持
| 平台 | 部署支持 |
|------|------|
| 个人微信 |✅ (基于 **openclaw-weixin**)|
| Telegram | ✅ | 
| WhatsApp | 🚧 | 
| Discord | ✅ | 
| Slack | ✅ | 

> [!IMPORTANT]
> - WeClone仍在快速迭代期，当前效果不代表最终效果。  
> - 微调LLM效果很大程度取决于模型大小、聊天数据的数量和质量，理论上模型越大，数据越多，效果越好。
> - 7B模型效果一般，14B及以上的模型效果会更好。   
> - Windows环境未进行严格测试，可以使用WSL作为运行环境。

### 近期更新
[25/06/05]支持图片模态数据微调   
[25/07/10]数据源增加Telegram

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
JUDGE_API_KEY=your-key weclone-cli benchmark-model \
  --model-path ./model_output/Qwen3.5-4B-SFT-1500-merged \
  --run-name merged-1500
```

评测未合并的 LoRA 时，`--model-path` 填基础模型目录，`--adapter-path` 填 LoRA checkpoint：

```bash
JUDGE_API_KEY=your-key weclone-cli benchmark-model \
  --model-path ./models/Qwen3.5-4B \
  --adapter-path ./model_output/Qwen3.5-4B-SFT/checkpoint-1500 \
  --run-name lora-checkpoint-1500
```

本地模型只会加载一次。默认沿用 `infer_args.infer_backend`，也可以通过
`--local-backend vllm` 或 `--local-backend huggingface` 临时覆盖；vLLM 模式同时沿用
`vllm_args`。`--model-path` 和 API 模式的 `--model` 不能同时使用。也可以把本地路径固定写入
`benchmark_args.local_model_path` 和 `benchmark_args.local_adapter_path`。使用本地模式时
`benchmark_args.candidate` 会被忽略，也可以从配置中删除；Judge 配置仍然必须保留。

使用 API 模式时，可以在每次运行中直接覆盖待测 API 模型名和结果标签，而不修改配置文件：

```bash
JUDGE_API_KEY=your-key weclone-cli benchmark-model \
  --model gpt-3.5-turbo \
  --run-name qwen-lora-checkpoint-1500
```

`--run-name` 只是本次评测的结果标签，用于区分输出目录和报告，不会传给生成模型，也不会
影响 Judge 评分。例如可以使用 `qwen-base`、`lora-checkpoint-500`、
`lora-checkpoint-1500` 或 `merged-int8`。如果不传该参数，则使用
`settings.jsonc` 中的 `benchmark_args.run_name`。对应的输出目录类似：

```text
benchmark_results/20260814T131500Z-qwen-lora-checkpoint-1500/
```

示例验证集位于
[`dataset/benchmark/benchmark.sample.json`](dataset/benchmark/benchmark.sample.json)。数据格式中：

- `assistant` 表示被模仿的本人，`user` 表示聊天对象；
- 每段 `messages` 必须以 `user` 结尾；
- `reference` 是本人未参与训练的真实下一条回复；
- `style_examples` 只用于向 Judge 展示整体风格，不能与 `samples` 重复；
- 新聊天应按完整会话或日期留出，不要把 benchmark 数据重新加入训练或调参。

每次运行会在 `benchmark_results/<时间>-<run_name>/` 下产生 `samples.jsonl`、
`summary.json` 和 `report.md`。评测另一个模型时修改（或通过命令行覆盖）`run_name` 和 `candidate.model`
（必要时也修改接口地址）后再次运行。只有数据集 SHA-256、Judge 模型、Judge 重复次数和生成参数
一致时，两份报告才适合直接比较。聊天内容会发送到所配置的 Judge API，敏感数据请使用可信的
本地 Judge 或先进行脱敏。

## 🖼️ 微调效果
> [!TIP] 
> **社群内有部署好的Qwen2.5VL 32B Bot，可以体验效果。** 


## 🤖 部署到聊天机器人

### AstrBot

[AstrBot](https://github.com/AstrBotDevs/AstrBot) 是易上手的多平台 LLM 聊天机器人及开发框架 ✨ 平台支持Telegram、飞书等。      

使用步骤：
1. 部署 AstrBot
2. 在 AstrBot 中部署消息平台
3. 执行 `weclone-cli server` 启动api服务
4. 在 AstrBot 中新增服务提供商，类型选择OpenAI，API Base URL 根据AstrBot部署方式填写（例如docker部署可能为http://172.17.0.1:8005/v1） ，模型填写gpt-3.5-turbo,API Key随意填写一个
5. 微调后不支持工具调用，请先关掉默认的工具，消息平台发送指令： `/tool off_all`，否则会没有微调后的效果。 
6. 根据微调时使用的default_system，在 AstrBot 中设置系统提示词。
![5](https://github.com/user-attachments/assets/19de7072-076a-4cdf-8ae6-46b9b89f536a)
> [!IMPORTANT]
> 检查api_service的日志，尽量保证大模型服务请求的参数和微调时一致，tool插件能力都关掉。

### LangBot

[LangBot](https://github.com/RockChinQ/LangBot) 是一个开源的接入全球多种即时通信平台的 LLM 机器人平台，适合各种场景使用。

<img width="450px" alt="image" src="https://github.com/user-attachments/assets/04ceeacf-8a14-40a9-b07a-2f03f257eee6" />


1. [部署 LangBot](https://github.com/RockChinQ/LangBot#-%E5%BC%80%E5%A7%8B%E4%BD%BF%E7%94%A8)
2. 执行 `weclone-cli server` 启动 WeClone API 服务
3. 在 LangBot 中添加一个机器人
4. 在模型页添加新模型，名称`gpt-3.5-turbo`，供应商选择 OpenAI，填写 请求 URL 为 WeClone 的地址，详细连接方式可以参考[文档](https://docs.langbot.app/zh/workshop/network-details.html)，API Key 任意填写。

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/fc167dea-7c93-4d94-9c5f-db709d0320ba" />

6. 在流水线配置中选择刚才添加的模型，或修改提示词配置

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/dbb0fd0a-f760-42db-acd0-bb99c859b52e" />

## 📌 路线图
- [ ] 支持更多数据源
- [ ] 更丰富的上下文：包括上下文对话、聊天对象信息、时间等 
- [ ] Memory 支持
- [ ] 支持多模态:已支持图片
- [ ] 数据增强
- [ ] 支持GUI
- [ ] 支持COT思考


## 问题解决
#### [官方文档FAQ](https://docs.weclone.love/docs/introduce/FAQ.html)    
同时建议使用[DeepWiki](https://deepwiki.com/xming521/WeClone)解决问题。

## ❤️ 贡献代码

欢迎任何 Issues/Pull Requests！

你可以通过查看Issues或帮助审核 PR（拉取请求）来贡献。对于新功能的添加，请先通过 Issue 讨论。   
开发环境：
```bash
uv pip install --group dev -e .
pre-commit install
```

项目使用`pytest`测试，`pyright`检查类型，`ruff`检查代码格式。  
提交代码前你应该先运行`pytest tests`确保所有测试通过。

## 🙏 致谢
BUPT VCIS Lab的支持
感谢以下代码贡献者和社区里其他成员的贡献

<a href="https://github.com/xming521/WeClone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xming521/WeClone" />
</a>

同时本项目受益于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[AstrBot](https://github.com/AstrBotDevs/AstrBot)、[LangBot](https://github.com/RockChinQ/LangBot)等优秀开源项目。

## ⚠️ 免责声明
> [!CAUTION]
> **本项目仅供学习、研究和实验用途，用于生产环境存在较大风险，请谨慎评估。请勿用于非法用途，后果自负。**   
> [针对违规获取及利用微信终端用户数据行为的打击公告](https://mp.weixin.qq.com/s/A6h4ZLTE2EPrY7kJ5fHE2g)


> [!IMPORTANT]
> #### WeClone 目前未与任何平台合作，未发行任何数字货币。唯一官方网站：[weclone.love](https://www.weclone.love)，谨防仿冒。
<details>
<summary>点击查看免责条款</summary>

### 1. 使用风险自担
- 用户在使用本项目时，应充分理解并承担所有相关风险
- **本项目作者不对因使用本项目而产生的任何直接或间接损失承担责任**
- 包括但不限于：数据丢失、经济损失、法律纠纷、个人名誉损害、社会关系影响、心理创伤、职业发展受阻、商业信誉受损等

### 2. 生产环境风险警告
- **用于商业用途或对外提供服务需自行承担全部风险**
- 生产环境使用可能导致的所有后果（包括但不限于服务中断、数据安全问题、用户投诉、法律责任等）完全由用户承担
- **建议在生产环境使用前进行充分的测试、验证和风险评估**

### 3. 模型输出不可靠性
- 微调后的模型可能产生不准确、有害或误导性的内容
- 模型输出不代表真实人物的观点或意图
- 用户应对模型输出进行人工审核和验证

### 4. 数据安全与隐私
- 用户应确保上传的聊天记录等数据符合相关法律法规
- 用户应获得**数据相关人员的适当授权**
- 本项目不对**数据泄露或隐私侵犯**承担责任

### 5. 法律合规
- **用户应确保使用本项目符合当地法律法规**
- 涉及人工智能、数据保护、知识产权等相关法律
- **违法使用造成的后果由用户承担**

### 6. 技术支持限制
- 本项目按"现状"提供，不提供任何明示或暗示的保证
- 作者不承诺提供持续的技术支持或维护
- 不保证项目的稳定性、可靠性或适用性

## 使用建议

### 强制性Bot身份标识
**使用本项目生成的数字分身时，强烈建议：**
- 在每次对话开始时明确标识为"AI Bot"或"数字分身"
- 在用户界面显著位置标注"此为AI生成内容"
- 避免让用户误认为是真实人类在对话，从而造成风险

### 风险评估建议

如确需在生产环境使用，建议：
1. 进行全面的安全性测试
2. 建立完善的内容审核机制
3. 制定应急响应预案
4. 购买相应的保险保障
5. 咨询法律专业人士意见


本免责声明可能随项目更新而修订，用户应定期查看最新版本。继续使用本项目即表示同意最新的免责声明条款。

**一旦您下载、克隆、修改、分发或以任何方式使用本项目的代码或模型，即表示您已完整阅读、理解并同意无条件接受本免责声明的全部条款。**

</details>

**请用户慎重阅读并理解本免责声明的所有内容，确保在使用本项目时严格遵守相关规定。**
<br>  

## ⭐ Star History
> [!TIP] 
> 如果本项目对您有帮助，或者您关注本项目的未来发展，请给项目 Star，谢谢 

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=xming521/WeClone&type=Date)](https://www.star-history.com/#xming521/WeClone&Date)

</div>


<div align="center"> 克隆我们，保留灵魂的芬芳 </div>
