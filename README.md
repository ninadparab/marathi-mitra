# 🌸 Marathi Mitra

Fine-tuned LLM for Marathi vocabulary learning — built using Supervised Fine-Tuning (SFT) with QLoRA on Phi-3 Mini.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/02_finetune.ipynb)
[![HF Spaces](https://img.shields.io/badge/🤗-Live%20Demo-orange)](https://huggingface.co/spaces/ninadp/marathi-mitra)
[![HF Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/ninadp/marathi-mitra-phi3-v2)

---

## What It Does

Type any English word and get a Marathi lesson:

```
Input:  butterfly
Output: 🌟 BUTTERFLY in Marathi is फुलपाखरू
        📢 How to say it: Phul-pakh-roo
        📖 फुलपाखरू फुलांवर बसते.
           (The butterfly sits on flowers.)
        🎉 फूल = flower + पाखरू = bird — flower-bird! 🦋
```

**[Try Live Demo →](https://huggingface.co/spaces/ninadp/marathi-mitra)**

---

## Evaluation Results

| Model | Training Data | Seen Words | Unseen Words | Overall |
|-------|--------------|------------|--------------|---------|
| Base Phi-3 | 0 examples | 11.2% | 8.0% | 9.6% |
| v1 Fine-tuned | 30 examples | 36.4% | 25.6% | 31.0% |
| **v2 Fine-tuned** | **250 examples** | **100.0%** | **78.8%** | **89.4%** |

**Total improvement: +79.8% over base model**

Evaluation criteria:
- Field presence (40%) — all 5 sections present in output
- Exact match (60%) — correct Marathi word + pronunciation

Seen words: `butterfly` `mother` `rain` `elephant` `school`

Unseen words: `apple` `star` `tiger` `ocean` `dance`

Generalisation gap (v2): 21.2% → ✅ Good

---

## Model Versions

| Version | Examples | Seen | Unseen | Repo |
|---------|----------|------|--------|------|
| v1 | 30 | 36.4% | 25.6% | [ninadp/marathi-mitra-phi3](https://huggingface.co/ninadp/marathi-mitra-phi3) |
| v2 | 250 | 100.0% | 78.8% | [ninadp/marathi-mitra-phi3-v2](https://huggingface.co/ninadp/marathi-mitra-phi3-v2) |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Base model | Phi-3 Mini 4k Instruct |
| Fine-tuning | QLoRA — PEFT + TRL SFTTrainer |
| Quantization | 4-bit (bitsandbytes) |
| Training hardware | Google Colab T4 GPU |
| UI | Gradio |
| TTS | gTTS (Marathi) |
| Hosting | Hugging Face Spaces + Hub |

---

## Training

Best configuration (Exp4):

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-4 |
| Epochs | 25 |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| Quantization | 4-bit QLoRA |
| Train/eval split | 80/20 |

---

## Project Structure

```
marathi-mitra/
├── notebooks/
│   ├── 01_dataset_prep.ipynb    # data exploration
│   ├── 02_finetune.ipynb        # SFT experiments
│   ├── 03_evaluate.ipynb        # model evaluation
│   └── 04_optuna_hpo.ipynb      # automated HPO
├── data/
│   ├── vocabulary.json          # 250 word vocabulary
│   ├── create_dataset.py        # dataset generation
│   └── vocabulary_dataset.json  # training data
├── src/
│   └── train.py                 # reproducible training
├── app/
│   └── app.py                   # Gradio demo
├── config.yaml                  # best hyperparameters
└── requirements.txt
```

---

## Notebooks

| Notebook | Description | Open |
|----------|-------------|------|
| 01_dataset_prep | Explore and validate dataset | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/01_dataset_prep.ipynb) |
| 02_finetune | QLoRA fine-tuning + experiments | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/02_finetune.ipynb) |
| 03_evaluate | Evaluate base vs v1 vs v2 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/03_evaluate.ipynb) |
| 04_optuna_hpo | Automated hyperparameter search | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/04_optuna_hpo.ipynb) |

---

## Run Locally

```bash
git clone https://github.com/ninadparab/marathi-mitra.git
cd marathi-mitra
pip install -r requirements.txt
python app/app.py
```

---

## Reproduce Training

```bash
# Configure config.yaml with best hyperparameters
# Add HF credentials to .env
python src/train.py
```

---

## Use the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "ninadp/marathi-mitra-phi3-v2")
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
)
```

---

## Roadmap

- [x] Fine-tune Phi-3 Mini with QLoRA
- [x] Hyperparameter experiments (4 configs)
- [x] Expand dataset 30 → 250 examples
- [x] Evaluate base vs v1 vs v2
- [x] Deploy Gradio app on HF Spaces
- [ ] Run Optuna HPO (20 trials on A100)
- [ ] Add quiz mode to app
- [ ] MCP server for Claude Desktop

---

## License

MIT
