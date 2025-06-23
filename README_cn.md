# 子曰3-数学
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="assets/confucius_logo.png" width="20%" alt="Confucius3-Math" />
</div>

<div align="center" style="line-height: 1;">
💜 <a href="https://confucius.youdao.com/"><b>Confucius Demo</b></a>&nbsp;&nbsp; | &nbsp;&nbsp;🤗 <a href="https://huggingface.co/netease-youdao">Hugging Face</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤖 <a href="https://modelscope.cn/organization/netease-youdao">ModelScope</a>&nbsp;&nbsp; | &nbsp;&nbsp;⌨️ <a href="https://github.com/netease-youdao/Confucius3-Math">GitHub</a>&nbsp;&nbsp; | &nbsp;&nbsp;⌨💬 <a href="https://github.com/netease-youdao/Confucius3-Math/blob/dev/assets/wechat.png">Wechat</a>
<br> 


</div>

<h4 align="center">
    <p>
        <b>中文</b> | <a href="README.md">English</a>
    <p>
</h4>

## 📑 Confucius3-Math: A Lightweight High-Performance Reasoning LLM for Chinese K-12 Mathematics Learning

<div align="center">
  <img src="assets/performance.png" width="100%" alt="Confucius3-Math Performance" />
</div>
 

## 目录

- [1 模型下载](#1-模型下载)
- [2 模型介绍](#2-模型介绍)
- [3 评测结果](#3-评测结果)
- [4 模型推理](#4-模型推理)
- [5 快速开始](#5-快速开始)

## 1 模型下载
| **模型** | **HuggingFace** | **ModelScope** | **WiseModel** |
| :------------: | :------------: | :------------: | :------------: |
| 子曰3-数学 | [🤗 HuggingFace](https://huggingface.co/netease-youdao/Confucius3-Math) | [ModelScope](https://modelscope.cn/models/netease-youdao/Confucius3-Math) | [WiseModel](https://www.wisemodel.cn/models/Netease_Youdao/Confucius3-Math) |

## 2 模型介绍

Confucius3-Math 是由网易有道 AI 团队开发的**140 亿参数开源推理大语言模型**，专门针对 K-12 数学教育场景进行优化。与通用模型不同，Confucius3-Math 具有以下特点：

✅ **数学任务上的顶尖性能**  
通过专门的强化学习训练，在中文 K-12 数学问题上的表现超越了参数规模更大的模型

✅ **高性价比的部署方案**  
可在单张消费级 GPU（如 RTX 4090D）上高效运行

✅ **文化与课程体系的深度契合**  
针对中国国家数学课程标准和解题方法论进行了优化

Confucius3-Math 采用纯强化学习的后期训练流程，结合创新的数据调度策略和改进的组相对优势估计器开发而成。具体技术细节请参考我们的技术报告。

## 3 评测结果

<div align="center">


| Benchmark | DeepSeek-R1 | Qwen3-14B | QwQ-32B | DeepSeek-R1-Distill-Qwen-14B | Confucius3-Math |
|-------------------|----------------------|------------|--------------|----------------|------------|
| CK12-MATH | 92.74 | 94.04 | 93.60 | 82.86 | 96.24 |
| GAOKAO-Bench(math) | 93.27 | 94.44 | 94.93 | 86.75 | 98.46 |
| MathBench(K12) | 89.99 | 96.51 | 96.57 | 88.40 | 95.10 |
| CMATH | 95.81 | 95.90 | 95.95 | 77.41 | 96.13 |
| MATH-500 | 97.30 | 96.80 | 98.00 | 93.90 | 98.80 |
| AIME 2024 | 79.80 | 79.30 | 79.50 | 69.70 | 81.15 |
| AIME 2025 | 70.00 | 70.40 | 69.50 | 42.97 | 69.95 |

</div>


## 4 模型推理
运行该模型的环境要求与 [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) 模型的环境要求完全相同。因此，你可以直接使用 Transformers 或 vLLM 来加载并运行该模型进行推理，进而部署你的服务。

请注意使用下面代码中提供的预定义系统消息和用户消息模板来向模型发出请求。其他模板可能可用，但我们尚未对其进行测试。
```python
SYSTEM_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""

USER_PROMPT_TEMPLATE = """{question}"""
```

然后你可以按如下方式创建你的messages，并使用它们来请求模型结果。你只需在 “question” 字段中填入你的指令即可。
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "netease-youdao/Confucius3-Math"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT_TEMPLATE},
    {'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(question=question)},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
> [!NOTE]
> **采样参数**: 我们推荐使用 Temperature=1.0, TopP=0.7 进行采样.

在获取模型结果后，您可以按如下方式解析出 “思考” 和 “总结” 部分。
```python
def parse_result_nostep(result):
     think_pattern = r"<think>(.*?)</think>(.*)"

    think_list = re.findall(think_pattern, result, re.DOTALL)

    assert len(think_list) == 1, \
        f"The parsing results do not meet the expectations.\n{result}"

    think = think_list[0][0].strip()
    summary = think_list[0][1].strip()
    return think, summary

thinking, summary = parse_result_nostep(response)
```
完整代码参见 `model/hf_demo.py`
## 5 快速开始
本项目采用vLLM后端启动Confucius3-Math服务，并基于Gradio搭建了交互式Web界面。
我们推荐你使用Docker启动Demo，使用Docker的方式是：
```
# 生成镜像
docker build -t confucius3 .
# 启动容器服务
docker run -e ARK_API_KEY=xxx -p 8827:8827 confucius3
```
`ARK_API_KEY`可以在[火山引擎](https://console.volcengine.com/)创建，主要用于输入图片的OCR，如果你不需要支持图片的输入，可以不用设置这里的环境变量。

服务启动后，你可以访问：http://127.0.0.1:8827/ 体验Confucius3-Math的能力

你也可以手动通过如下命令进行安装：
```
pip install -r requirements.txt
```
安装好相关依赖后，你可以使用如下脚本启动Demo服务：
```
export ARK_API_KEY=xxx 
bash run_service_stream.sh
```

## 引用
如果你觉得我们的工作有帮助，欢迎引用我们的成果。
```
@misc{confucius3-math,
   author = {NetEase Youdao Team},
   title = {Confucius3-Math: A Lightweight High-Performance Reasoning LLM for Chinese K-12 Mathematics Learning},
   url = {https://huggingface.co/netease-youdao/Confucius3-Math},
   month = {June},
   year = {2025}
 }
```
