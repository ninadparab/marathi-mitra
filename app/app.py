# ═══════════════════════════════════════════════════════════
# app/app.py — Marathi Mitra
# Kid-friendly Marathi vocabulary learning app
#
# Run locally:  python app/app.py
# Deploy:       Hugging Face Spaces (Gradio SDK)
#
# Local note:   Runs on CPU — slow (~2-3 mins per word)
#               Use HF Spaces for fast GPU inference
# ═══════════════════════════════════════════════════════════

import os
import re
import torch
import tempfile
import gradio as gr
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from gtts import gTTS
from peft import PeftModel

# ── Load credentials ──────────────────────────────────────────
# Works in all environments:
# Colab: uses userdata, HF Spaces: uses os.getenv, Local: uses .env
try:
    from google.colab import userdata
    HF_TOKEN    = userdata.get("HF_TOKEN")
    HF_USERNAME = userdata.get("HF_USERNAME")
except ImportError:
    load_dotenv("../.env")
    HF_TOKEN    = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME", "ninadp")

# ── Fix 1: Use v2 model ───────────────────────────────────────
MODEL_REPO = f"{HF_USERNAME}/marathi-mitra-phi3-v2"

# ── Quick word categories ─────────────────────────────────────
CATEGORIES = {
    "🌿 Nature":  ["sun", "moon", "rain", "flower", "tree",
                   "river", "sky", "water", "mountain"],
    "🐾 Animals": ["cat", "dog", "bird", "fish", "elephant",
                   "cow", "monkey", "parrot", "butterfly"],
    "👨‍👩‍👧 Family": ["mother", "father", "sister", "brother",
                   "grandmother", "grandfather"],
    "🏫 Daily":   ["school", "book", "pencil", "food",
                   "house", "friend"],
}


# ── Load model ────────────────────────────────────────────────
def load_model():
    BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    ADAPTER    = MODEL_REPO

    use_gpu = torch.cuda.is_available()
    print(f"Step 1: Loading base model...")
    print(f"        {BASE_MODEL}")

    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

    print(f"Step 2: Loading adapter...")
    print(f"        {ADAPTER}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER,
        token=HF_TOKEN,
    )
    model.eval()

    print(f"Step 3: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = "GPU ✅" if use_gpu else "CPU ⚠️"
    print(f"✅ Model ready on {device}")
    return model, tokenizer


# Load once at startup
model, tokenizer = load_model()


# ── Generate Marathi lesson ───────────────────────────────────
def generate_lesson(word: str) -> str:
    word = word.strip().lower()
    if not word:
        return ""

    prompt = f"""### Instruction:
You are Marathi Mitra, a friendly Marathi teacher for kids. When given an English word, teach it in Marathi with the word in Devanagari script, pronunciation, a simple story sentence, and a fun fact. Always be encouraging and kid-friendly.

### Input:
Teach me the Marathi word for: {word}

### Response:
"""
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text.split("### Response:")[-1].strip()


# ── Text to Speech ────────────────────────────────────────────
def text_to_speech(lesson: str):
    if not lesson:
        return None
    try:
        devanagari = re.findall(r"[\u0900-\u097F]+", lesson)
        if not devanagari:
            return None
        marathi_word = " ".join(devanagari[:3])
        tts = gTTS(text=marathi_word, lang="mr", slow=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"TTS error: {e}")
        return None


# ── Score and streak ──────────────────────────────────────────
def format_status(score, streak):
    fire  = "🔥" * min(streak, 5)
    stars = "⭐" * min(score, 10)
    return (
        f'<div style="text-align:center; font-weight:bold; '
        f'color:#E85D04; padding:8px;">'
        f'{fire} Streak: {streak} &nbsp;|&nbsp; {stars} Words learned: {score}'
        f'</div>'
    )


# ── Core learn function ───────────────────────────────────────
def learn_word(word, score, streak, learned_words):
    word = word.strip().lower()
    if not word:
        return (
            "Please enter a word! 😊",
            None,
            score,
            streak,
            learned_words,
            format_status(score, streak),
        )

    lesson = generate_lesson(word)
    audio  = text_to_speech(lesson)

    if word not in learned_words:
        score         += 1
        streak        += 1
        learned_words  = learned_words + [word]

    return lesson, audio, score, streak, learned_words, format_status(score, streak)


# ── Fix 2: Button factory fixes closure bug in loops ──────────
def make_handler(w):
    """
    Factory function captures each word correctly.
    Without this all buttons share the last loop value.
    """
    def handler(score, streak, learned_words):
        return learn_word(w, score, streak, learned_words)
    return handler


# ── CSS ───────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;800&family=Nunito:wght@400;600;700&display=swap');

:root {
    --saffron:     #FF6B35;
    --gold:        #FFD60A;
    --cream:       #FFF8F0;
    --deep-orange: #E85D04;
    --text-dark:   #1A1A2E;
    --shadow:      0 4px 20px rgba(255, 107, 53, 0.15);
}

body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'Nunito', sans-serif !important;
}

.quick-btn {
    background: white !important;
    color: var(--text-dark) !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
    border: 2px solid #FFD60A !important;
    border-radius: 50px !important;
    font-size: 0.9rem !important;
    margin: 3px !important;
}

.quick-btn:hover {
    background: #FFD60A !important;
}

textarea, input[type="text"] {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    border: 2px solid #FFD60A !important;
}
"""


# ── Build UI ──────────────────────────────────────────────────
with gr.Blocks(title="🌸 Marathi Mitra", css=CSS) as app:

    # ── State ─────────────────────────────────────────────────
    score_state         = gr.State(0)
    streak_state        = gr.State(0)
    learned_words_state = gr.State([])

    # ── Header ────────────────────────────────────────────────
    gr.HTML("""
        <div style='text-align:center; padding:1.5rem;
                    background:linear-gradient(135deg,#FF6B35,#FFD60A);
                    border-radius:16px; margin-bottom:1rem;'>
            <h1 style='color:white; font-size:2.5rem;
                        font-family:"Baloo 2",cursive; margin:0;'>
                🌸 Marathi Mitra 🌸
            </h1>
            <p style='color:rgba(255,255,255,0.9); margin:0.3rem 0 0;'>
                माझा मराठी मित्र — Your Marathi Learning Friend!
            </p>
        </div>
    """)

    # ── CPU warning ───────────────────────────────────────────
    gr.HTML("""
        <div style='text-align:center; color:#E85D04; padding:8px;
                    background:#FFF3E0; border-radius:8px; margin-bottom:8px;'>
            ⏱️ Running on CPU — responses take 2-5 minutes. Please be patient!
        </div>
    """)

    # ── Status bar ────────────────────────────────────────────
    status_display = gr.HTML(
        '<div style="text-align:center; font-weight:bold; '
        'color:#E85D04; padding:8px;">'
        'Start learning to build your streak! 🌟'
        '</div>'
    )

    # ── Fix 3: Define outputs BEFORE quick word buttons ───────
    with gr.Row():

        # Left — Input
        with gr.Column(scale=1):
            word_input = gr.Textbox(
                label="✏️ Enter an English word",
                placeholder="e.g. butterfly, sun, mother...",
                lines=1,
            )
            learn_btn = gr.Button(
                "🚀 Learn in Marathi!",
                variant="primary",
                size="lg",
            )

        # Right — Output (defined BEFORE quick buttons reference them)
        with gr.Column(scale=2):
            lesson_output = gr.Textbox(
                label="📖 Your Marathi Lesson",
                lines=12,
            )
            audio_output = gr.Audio(
                label="🔊 Hear the Marathi pronunciation",
                type="filepath",
                autoplay=True,
            )

    # ── Quick word buttons (after outputs are defined) ─────────
    gr.Markdown("### ⚡ Quick Words — Click to Learn!")

    with gr.Tabs():
        for category, words in CATEGORIES.items():
            with gr.Tab(category):
                for i in range(0, len(words), 3):
                    chunk = words[i:i+3]
                    with gr.Row():
                        for word in chunk:
                            gr.Button(
                                word,
                                size="sm",
                                elem_classes=["quick-btn"],
                            ).click(
                                fn=make_handler(word),   # Fix 2: factory fn
                                inputs=[
                                    score_state,
                                    streak_state,
                                    learned_words_state,
                                ],
                                outputs=[
                                    lesson_output,
                                    audio_output,
                                    score_state,
                                    streak_state,
                                    learned_words_state,
                                    status_display,
                                ],
                            )

    # ── Footer ────────────────────────────────────────────────
    gr.HTML("""
        <div style='text-align:center; margin-top:1rem;
                    color:#666; font-size:0.9rem;'>
            Built with ❤️ using Fine-tuned Phi-3 Mini + QLoRA + Gradio<br>
            <a href='https://github.com/ninadparab/marathi-mitra'
               target='_blank'>📦 View on GitHub</a>
            &nbsp;|&nbsp;
            <a href='https://huggingface.co/ninadp/marathi-mitra-phi3-v2'
               target='_blank'>🤗 Model on HF Hub</a>
        </div>
    """)

    # ── Wire main button ───────────────────────────────────────
    learn_btn.click(
        fn=learn_word,
        inputs=[
            word_input,
            score_state,
            streak_state,
            learned_words_state,
        ],
        outputs=[
            lesson_output,
            audio_output,
            score_state,
            streak_state,
            learned_words_state,
            status_display,
        ],
    )

    word_input.submit(
        fn=learn_word,
        inputs=[
            word_input,
            score_state,
            streak_state,
            learned_words_state,
        ],
        outputs=[
            lesson_output,
            audio_output,
            score_state,
            streak_state,
            learned_words_state,
            status_display,
        ],
    )


# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        show_error=True,
        server_port=7860,
    )