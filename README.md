
# 🧠 Fine-Tuning DeepSeek R1 with Unsloth on Alpaca-GPT4 Dataset

This project demonstrates how to fine-tune the [`unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit`](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit) model using the [`vicgalle/alpaca-gpt4`](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) dataset with LoRA and Unsloth's efficient training interface.

The fine tuned model file is on my HuggingFace : https://huggingface.co/junaid17/finetuned-DeepSeek_r1/tree/main

---

## 🚀 Model & Tokenizer Setup

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype=None,
    load_in_4bit = True,
)
```

---

## 🔧 LoRA PEFT Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_rslora = True,
)
```

---

## 📚 Dataset: Alpaca-GPT4

We use a subset (5,000 rows) of the Alpaca-GPT4 dataset for quick fine-tuning:

```python
from datasets import load_dataset
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train[:5000]")
```

---

## 📂 Output Directory

All model checkpoints and logs are saved in the `outputs/` directory.

---

## 🧠 Notes

- Fine-tuning used **LoRA** with **r=16** on 4-bit quantized weights.
- **Only 5k** rows were used for fast iteration.
- `apply_chat_template()` helped match conversational finetuning structure.

---
