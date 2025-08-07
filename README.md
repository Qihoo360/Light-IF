<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

# Light-IF: Endowing LLMs with Generalizable Reasoning via Preview and Self-Checking for Complex Instruction Following

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64eeb81ad0ceda46832e0160/b2_eQV04B8xSdYJZnB2FD.png" width="95%" alt="Light-IF-32B" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  🤗 <a href="https://huggingface.co/qihoo360/Light-IF-32B">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.03178">Paper Link</a> &nbsp&nbsp|&nbsp&nbsp  📑 <a href="https://zhuanlan.zhihu.com/p/1936535948360918628">Blog</a> &nbsp&nbsp 
<br>
  </a>
</div>

## Evaluation 
|Model|SuperClue|IFEval|CFBench|IFBench|
| ---- | ---- | ---- | ---- | ---- |
|Qwen3-4B|0.225|0.888|0.787|0.382|
|Qwen3-8B|0.225|0.888|0.813|0.417|
|Qwen3-32B|0.234|0.877|0.823|0.384|
|Qwen3-235B-A22B|0.244|0.882|0.834|0.423|
|Qwen3-235B-A22B-Thinking-2507|0.434|0.916|0.843|0.475|
|DeepSeek-R1-0528|0.436|0.863|0.827|0.415|
|Doubao-seed-1-6-thinking-250615|0.362|0.832|0.82|0.477|
|Doubao-seed-1-6-thinking-250715|0.345|0.856|0.84|0.366|
|ChatGPT-4o-latest|0.260|0.836|0.807|0.365|
|Deepseek-v3-250324|0.306|0.859|0.833|0.405|
|Doubao-1.5-pro-32k-250115|0.285|0.889|0.797|0.375|
|Kimi-K2|0.227|0.921|0.820|0.395|
|GLM-4.5|0.395|0.893|0.833|0.466|
| [**Light-IF-4B (ours)** 🤗](https://huggingface.co/qihoo360/Light-IF-4B) |**0.445**|**0.916**|**0.80**|**0.443**| 
| [**Light-IF-8B (ours)** 🤗](https://huggingface.co/qihoo360/Light-IF-8B) |**0.471**|**0.932**|**0.82**|**0.473**| 
| [**Light-IF-32B (ours)** 🤗](https://huggingface.co/qihoo360/Light-IF-32B) |**0.575**|**0.938**|**0.85**|**0.575**| 

## Introduction
**Instruction following** is a core ability of large language models (LLMs), but performance remains inconsistent, especially on complex tasks.

We identify **lazy reasoning** during the thinking stage as a key cause of poor instruction adherence.

To address this, we propose a framework that promotes rigorous reasoning through **previewing and self-checking**.

Our method begins by generating instruction data with **complex constraints**, filtering out samples that are too easy or too difficult. We then use rejection sampling to build a small but high-quality dataset for model adaptation.

Training involves entropy-preserving supervised fine-tuning (**Entropy-SFT**) and token-wise entropy-adaptive reinforcement learning (**TEA-RL**), guided by rule-based multidimensional rewards.

This approach encourages models to plan ahead and verify their outputs, fostering more generalizable reasoning abilities.

Experiments show consistent improvements across model sizes. Notably, our 32B model outperforms both larger open-source models like **DeepSeek-R1** and closed-source models like **ChatGPT-4o** on challenging instruction-following benchmarks.

<p align="left"><b></b> The overall framework of the proposed method:</p>
<p align="left">
  <img src="https://github.com/user-attachments/assets/8a714c31-cad7-475d-936c-300742f8d4bc" alt="The overall framework of the proposed method" width="600"/>
</p>

## Quickstart

The following contains a code snippet illustrating how to use the model generate content based on given inputs. 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "qihoo360/Light-IF-32B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "Please help me write a poem with a total of 15 lines and no more than 300 words. The poem should be divided into 4 stanzas, each beginning with a **highlighted subtitle**."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True 
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
print("thinking content:", thinking_content)
print("content:", content)
```

## License & Acknowledgements

All released materials of this project follow the open-source license Apache 2.0.


## Citation

```bibtex
@article{Light-IF,
  title={Light-IF: Endowing LLMs with Generalizable Reasoning via Preview and Self-Checking for Complex Instruction Following},
  author={Light-IF Team},
  journal={arXiv preprint arXiv:2508.03178},
  year={2025}
}
```
