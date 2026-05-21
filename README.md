# 🌸 Marathi Mitra

> Fine-tuned LLM for Marathi vocabulary learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/02_finetune.ipynb)
[![HF Spaces](https://img.shields.io/badge/🤗-Live%20Demo-orange)](https://huggingface.co/spaces/ninadp/marathi-mitra)
[![HF Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/ninadp/marathi-mitra-phi3)

---

## What It Does

Type any English word → get a Marathi lesson with:
- Marathi word in Devanagari script
- Pronunciation guide
- Example sentence
- Fun fact

```
Input:  butterfly
Output: 🌟 BUTTERFLY in Marathi is फुलपाखरू
        📢 How to say it: Phul-pakh-roo
        📖 फुलपाखरू फुलांवर बसते.
        🎉 फूल = flower + पाखरू = bird — flower-bird!
```

---

## Live Demo

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/ninadp/marathi-mitra)**

> Note: Running on CPU — responses take 2-5 minutes

---

## Architecture

```
English word
     ↓
Fine-tuned Phi-3 Mini (QLoRA)
     ↓
Marathi lesson + gTTS pronunciation
```

---

## Fine-Tuning Results

| Experiment | LR | Epochs | r | Loss | Score |
|---|---|---|---|---|---|
| Baseline | — | — | — | — | 11.2% |
| Exp1 | 2e-4 | 5 | 16 | 1.29 | 12.8% |
| Exp2 | 2e-4 | 25 | 16 | 0.20 | 28.8% |
| Exp3 | 1e-4 | 25 | 16 | 0.37 | 16.0% |
| **Exp4** | **2e-4** | **25** | **32** | **0.22** | **36.4% ✅** |

**Evaluation criteria:**
- Field presence (40%) — all sections in output
- Exact match (60%) — correct Marathi word + pronunciation

**Key finding:** SFT successfully taught output format. Vocabulary accuracy improves with more training data.

---

## Tech Stack

| Component | Tool |
|---|---|
| Base model | Phi-3 Mini 4k Instruct |
| Fine-tuning | QLoRA (PEFT + TRL) |
| Quantization | 4-bit (bitsandbytes) |
| Training | Google Colab T4 GPU |
| UI | Gradio |
| TTS | gTTS (Marathi) |
| Model hosting | Hugging Face Hub |

---

## Project Structure

```
marathi-mitra/
├── notebooks/
│   ├── 01_dataset_prep.ipynb   # data exploration
│   ├── 02_finetune.ipynb       # SFT experiments
│   ├── 03_evaluate.ipynb       # model evaluation
│   └── 04_optuna_hpo.ipynb     # automated HPO
├── data/
│   ├── vocabulary.json         # raw vocabulary
│   ├── create_dataset.py       # dataset generation
│   └── vocabulary_dataset.json # training data
├── src/
│   └── train.py                # reproducible training
├── app/
│   └── app.py                  # Gradio demo
├── config.yaml                 # best hyperparameters
└── requirements.txt
```

---

## Notebooks

| Notebook | Description | Run |
|---|---|---|
| 01_dataset_prep | Explore and validate dataset | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/01_dataset_prep.ipynb) |
| 02_finetune | QLoRA fine-tuning + experiments | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/02_finetune.ipynb) |
| 03_evaluate | Evaluate on unseen words | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/03_evaluate.ipynb) |
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
# Fill config.yaml with best hyperparameters
# Add HF credentials to .env
python src/train.py
```

---

## Model

**[ninadp/marathi-mitra-phi3 on Hugging Face →](https://huggingface.co/ninadp/marathi-mitra-phi3)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "ninadp/marathi-mitra-phi3")
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
)
```

---

## Roadmap

- [ ] Expand dataset to 200+ words
- [ ] Retrain — expected score improvement to 70%+
- [ ] Run Optuna HPO (20 trials on A100)
- [ ] Add quiz mode
- [ ] MCP server for Claude Desktop integration

---

## License

MIT
