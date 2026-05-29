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

> Note: Running on CPU — responses take 2-5 minutes

---

## Evaluation Results

| Model | Training Data | Method | Seen Words | Unseen Words | Overall |
|-------|--------------|--------|------------|--------------|---------|
| Base Phi-3 | 0 examples | None | 11.2% | 8.0% | 9.6% |
| v1 | 30 examples | Manual HPO | 36.4% | 25.6% | 31.0% |
| **v2** | **250 examples** | **Manual HPO** | **100.0%** | **78.8%** | **89.4%** |
| v3 | 250 examples | Optuna HPO | 76.0% | 82.0% | 79.0% |

**Seen words tested:** `butterfly` `mother` `rain` `elephant` `school`

**Unseen words tested:** `apple` `star` `tiger` `ocean` `dance`

**Evaluation criteria:**
- Field presence (40%) — all 5 sections present in output
- Exact match (60%) — correct Marathi word + pronunciation

---

## Key Findings

### 1. Data Quantity Matters Most
```
Base → v1:  +21.4%  (30 examples, manual HPO)
v1   → v2:  +58.4%  (250 examples, same config)
v2   → v3:  -10.4%  (same data, Optuna HPO)
```
Expanding from 30 → 250 examples (+58.4%) had far greater
impact than automated hyperparameter optimization alone.

### 2. Metric-Objective Misalignment in Optuna
v3 was optimized purely for **unseen word score**. This produced
an unexpected result — unseen score exceeded seen score (82% vs 76%),
creating a **negative generalisation gap of -6%**.

```
v2: Seen=100%  Unseen=78.8%  Gap=+21.2%  (seen > unseen)
v3: Seen=76%   Unseen=82.0%  Gap= -6.0%  (unseen > seen!)
```

The model generalised better to new words but partially forgot
specific training examples. This demonstrates that optimizing
for a single metric in HPO can hurt overall performance —
the objective function should reflect the true goal.

**Lesson learned:** Optuna should optimize for overall score
`(seen + unseen) / 2` rather than unseen score alone.

### 3. Generalisation Analysis
```
Model  Gap      Verdict
────────────────────────────
base   +3.2%  → Excellent (but low absolute scores)
v1     +10.8% → Good
v2     +21.2% → Good      ← best overall performance
v3     -6.0%  → Excellent ← best generalisation
```

### 4. v2 Recommended for Production
v2 achieves the highest overall score (89.4%) and perfect
accuracy on training vocabulary. v3's generalisation advantage
does not outweigh its drop in overall performance.

---

## Model Versions

| Version | Examples | Seen | Unseen | Overall | Repo |
|---------|----------|------|--------|---------|------|
| v1 | 30 | 36.4% | 25.6% | 31.0% | [marathi-mitra-phi3](https://huggingface.co/ninadp/marathi-mitra-phi3) |
| v2 ⭐ | 250 | 100.0% | 78.8% | 89.4% | [marathi-mitra-phi3-v2](https://huggingface.co/ninadp/marathi-mitra-phi3-v2) |
| v3 | 250 | 76.0% | 82.0% | 79.0% | [marathi-mitra-phi3-v3](https://huggingface.co/ninadp/marathi-mitra-phi3-v3) |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Base model | Phi-3 Mini 4k Instruct |
| Fine-tuning | QLoRA — PEFT + TRL SFTTrainer |
| Quantization | 4-bit (bitsandbytes) |
| HPO | Optuna (TPE sampler, 20 trials) |
| Training hardware | Google Colab T4 / A100 |
| UI | Gradio |
| TTS | gTTS (Marathi) |
| Hosting | Hugging Face Spaces + Hub |

---

## Training

### Hyperparameter Experiments

| Experiment | LR | Epochs | r | Loss | Score |
|---|---|---|---|---|---|
| Baseline | — | — | — | — | 9.6% |
| Exp1 | 2e-4 | 5 | 16 | 1.29 | 12.8% |
| Exp2 | 2e-4 | 25 | 16 | 0.20 | 28.8% |
| Exp3 | 1e-4 | 25 | 16 | 0.37 | 16.0% |
| **Exp4 (v2)** | **2e-4** | **25** | **32** | **0.22** | **36.4%** |

### Optuna Best Config (v3)

| Parameter | Value |
|-----------|-------|
| Learning rate | 2.36e-4 |
| Epochs | 32 |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| Quantization | 4-bit |
| Train/eval split | 80/20 |
| Sampler | TPE (warm started) |
| Trials | 20 |
| Hardware | A100 (40GB) |

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
| 03_evaluate | Evaluate base vs v1 vs v2 vs v3 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/03_evaluate.ipynb) |
| 04_optuna_hpo | Automated hyperparameter search | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninadparab/marathi-mitra/blob/main/notebooks/04_optuna_hpo.ipynb) |

---

## Run Locally

```bash
git clone https://github.com/ninadparab/marathi-mitra.git
cd marathi-mitra
pip install -r requirements.txt
python app/app.py
```

## Reproduce Training

```bash
# Configure config.yaml with best hyperparameters
# Add HF credentials to .env
python src/train.py
```

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

## License

MIT
