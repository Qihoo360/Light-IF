<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

# Light-IF: Endowing LLMs with Generalizable Reasoning via Preview and Self-Checking

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64eeb81ad0ceda46832e0160/b2_eQV04B8xSdYJZnB2FD.png" width="95%" alt="Light-IF-32B" />
</div>
<hr>
<div align="center">
  
[![Paper](https://img.shields.io/badge/arXiv-2508.03178-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2508.03178)
[![collections](https://img.shields.io/badge/HFCollections-Light--IF-FFD21E?logo=huggingface&style=for-the-badge)](https://huggingface.co/collections/qihoo360/light-if-688753563c174e9d14d6978f)

[![dataset](https://img.shields.io/badge/HFData-Light--IF--SFTData-FFD21E?logo=huggingface&style=for-the-badge)](https://huggingface.co/datasets/qihoo360/Light-IF-SFTData)
[![Blog](https://img.shields.io/badge/Blog-Light--IF-lightgrey?style=for-the-badge)](https://zhuanlan.zhihu.com/p/1936535948360918628)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
</div>

### Aug. 13 update: [SOTA 14B](https://huggingface.co/qihoo360/Light-IF-14B) 

## 🧪 Benchmarks
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
| [**Light-IF-14B (ours)** 🤗](https://huggingface.co/qihoo360/Light-IF-14B) |**0.589**|**0.962**|**0.833**|**0.697**| 

> **Light-IF** is a powerful instruction-following large language model (LLM) series that leverages *Preview-Checking* reasoning to handle complex instructions with generalizable behavior — all trained with less than $3,000 in compute.

[**Light-IF-14B** 🤗](https://huggingface.co/qihoo360/Light-IF-14B) is the most powerful 14B instruction-following model we have open-sourced, even outperforming Light-IF-32B.

This remarkable performance is largely attributed to our carefully designed curriculum learning strategy.

---

## 📌 Highlights

- 🔍 Identifies and overcomes *lazy reasoning* in LLMs.
- 🧩 Integrates Preview + Self-Checking mechanisms.
- 🚀 Combines Entropy-SFT and TEA-RL for robust training.
- 💡 Achieves state-of-the-art results on instruction benchmarks.
- 💰 Trained efficiently on A800 GPUs at very low cost.

---

## 🔨 Technical Overview

Light-IF addresses the challenge of poor instruction-following due to lazy reasoning. Its pipeline includes:

### 1. Hardness-aware Prompt Synthesis
- Construct prompts with complex verifiable constraints.
- Filter invalid outputs using LLMs to form high-quality datasets.

### 2. Zero-RL Training
- Train a base model to reject lazy thinking with length-based and correctness-based rewards.

### 3. Entropy-Preserving SFT
- Select tokens by balancing NLL and entropy.
- Prevents overfitting and preserves model diversity.

### 4. TEA-RL (Token-wise Entropy-Adaptive RL)
- Dense rewards for partially satisfying constraints.
- Entropy-regularized policy gradient for stable learning.

<p align="left"><b></b> The overall framework of the proposed method:</p>
<p align="left">
  <img src="https://github.com/user-attachments/assets/8a714c31-cad7-475d-936c-300742f8d4bc" alt="The overall framework of the proposed method" width="600"/>
</p>

---

## 💻 Quick Usage

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

---

## ⚙️ Training Cost

| Model | GPUs | Hours | Cost (USD) |
|-------|------|-------|------------|
| Light-IF-1.7B | A800×4 | 10 | ~$342 |
| Light-IF-32B  | A800×88 | 30 | ~$2,800 |

---

## 📜 License

This repository is licensed under the **Apache 2.0 License**.

---

## Citation

```bibtex
@article{Light-IF,
  title={Light-IF: Endowing LLMs with Generalizable Reasoning via Preview and Self-Checking for Complex Instruction Following},
  author={Light-IF Team},
  journal={arXiv preprint arXiv:2508.03178},
  year={2025}
}
```
