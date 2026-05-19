# ═══════════════════════════════════════════════════════════
# src/train.py — Marathi Mitra
#
# Reproduces best model using config.yaml settings
# Run: python src/train.py
#
# Prerequisites:
# 1. pip install -r requirements.txt
# 2. .env file with HF_TOKEN and HF_USERNAME
# 3. data/vocabulary.json exists
# ═══════════════════════════════════════════════════════════

import os
import sys
import json
import yaml
import torch
from dotenv import load_dotenv
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# ── Add data folder to path ──────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from create_dataset import create_dataset


# ── Load config ──────────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded from {path}")
    return config


# ── Load credentials ─────────────────────────────────────────
def load_credentials():
    load_dotenv(".env")
    token    = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    assert token,    "❌ HF_TOKEN not found in .env"
    assert username, "❌ HF_USERNAME not found in .env"
    print(f"✅ Credentials loaded for: {username}")
    return token, username


# ── Prepare dataset ──────────────────────────────────────────
def prepare_dataset(config):
    # Regenerate dataset from vocabulary.json
    print("Preparing dataset...")
    examples = create_dataset()
    dataset  = Dataset.from_list(examples)
    print(f"✅ Dataset ready: {len(dataset)} examples")
    return dataset


# ── Load base model ──────────────────────────────────────────
def load_model(config):
    model_name = config["model"]["name"]
    print(f"Loading base model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"✅ Model loaded")
    print(f"✅ Memory: {model.get_memory_footprint() / 1e9:.2f} GB")
    return model, tokenizer


# ── Apply LoRA ───────────────────────────────────────────────
def apply_lora(model, config):
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Train ────────────────────────────────────────────────────
def train(model, tokenizer, dataset, config):
    train_cfg = config["training"]

    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        fp16=train_cfg["fp16"],
        logging_steps=train_cfg["logging_steps"],
        max_seq_length=config["model"]["max_seq_length"],
        dataset_text_field=config["data"]["text_field"],
        warmup_ratio=train_cfg["warmup_ratio"],
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print(f"\n🚀 Training with best config:")
    print(f"   learning_rate: {train_cfg['learning_rate']}")
    print(f"   epochs:        {train_cfg['epochs']}")
    print(f"   r:             {config['lora']['r']}")
    print(f"   lora_alpha:    {config['lora']['alpha']}")

    trainer.train()

    # Print loss progression
    print("\nLoss progression:")
    for log in trainer.state.log_history:
        if "loss" in log:
            print(f"  Step {log['step']:3d} → Loss: {log['loss']:.4f}")

    final_loss = trainer.state.log_history[-1].get("train_loss", None)
    print(f"\n✅ Training complete!")
    print(f"✅ Final loss: {final_loss:.4f}")

    return trainer


# ── Save to Hugging Face Hub ─────────────────────────────────
def save_model(model, tokenizer, config, token, username):
    repo = config["huggingface"]["model_repo"]

    # Replace placeholder username if needed
    repo = repo.replace("your-username", username)

    print(f"\nSaving model to: {repo}")
    model.push_to_hub(repo, token=token)
    tokenizer.push_to_hub(repo, token=token)

    print(f"✅ Model saved!")
    print(f"✅ View at: https://huggingface.co/{repo}")


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Marathi Mitra — Reproducing Best Model")
    print("=" * 60)

    # Step 1 — Load config and credentials
    config         = load_config()
    token, username = load_credentials()
    login(token=token)

    # Step 2 — Prepare dataset
    dataset = prepare_dataset(config)

    # Step 3 — Load model
    model, tokenizer = load_model(config)

    # Step 4 — Apply LoRA
    model = apply_lora(model, config)

    # Step 5 — Train
    trainer = train(model, tokenizer, dataset, config)

    # Step 6 — Save to Hub
    save_model(model, tokenizer, config, token, username)

    print("\n" + "=" * 60)
    print("✅ Done! Model reproduced and saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()