import json
import os

# ── Load raw vocabulary from separate JSON ───────────────────────
def load_vocabulary(path="data/vocabulary.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Prompt template ──────────────────────────────────────────────
INSTRUCTION = (
    "You are Marathi Mitra, a friendly Marathi teacher for kids. "
    "When given an English word, teach it in Marathi with the word "
    "in Devanagari script, pronunciation, a simple story sentence, "
    "and a fun fact. Always be encouraging and kid-friendly."
)


def format_output(item):
    """Format the structured output the model should produce."""
    return f"""🌟 **{item['word'].upper()}** in Marathi is **{item['marathi']}**

📢 **How to say it:** {item['pronunciation']}

📖 **Example sentence:**
{item['sentence']}
*({item['translation']})*

🎉 **Fun Fact:** {item['fun_fact']}"""


def format_training_example(item):
    """Format one vocabulary item into a training example."""
    input_text = f"Teach me the Marathi word for: {item['word']}"
    output_text = format_output(item)

    return {
        "word":        item["word"],
        "marathi":     item["marathi"],
        "instruction": INSTRUCTION,
        "input":       input_text,
        "output":      output_text,
        "text": f"""### Instruction:
{INSTRUCTION}

### Input:
{input_text}

### Response:
{output_text}"""
    }


# ── Build and save dataset ───────────────────────────────────────
def create_dataset():
    vocab = load_vocabulary()
    print(f"Loaded {len(vocab)} words from vocabulary.json")

    examples = [format_training_example(item) for item in vocab]

    output_path = "data/vocabulary_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"✅ Dataset saved → {output_path}")
    print(f"✅ Total examples: {len(examples)}")
    print(f"\nSample output:")
    print("─" * 50)
    print(examples[0]["text"])

    return examples


if __name__ == "__main__":
    create_dataset()