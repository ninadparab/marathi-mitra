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

# ── Load credentials ─────────────────────────────────────────
load_dotenv("../.env")
HF_TOKEN    = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "your-username")
MODEL_REPO  = f"{HF_USERNAME}/marathi-mitra-phi3"


# ── Quick word categories ─────────────────────────────────────
CATEGORIES = {
    "🌿 Nature":    ["sun", "moon", "rain", "flower", "tree",
                     "river", "sky", "water", "mountain"],
    "🐾 Animals":   ["cat", "dog", "bird", "fish", "elephant",
                     "cow", "monkey", "parrot", "butterfly"],
    "👨‍👩‍👧 Family":   ["mother", "father", "sister", "brother",
                     "grandmother", "grandfather"],
    "🏫 Daily":     ["school", "book", "pencil", "food",
                     "house", "friend"],
}

# Flatten for streak tracking
ALL_QUICK_WORDS = [w for words in CATEGORIES.values() for w in words]


# ── Load model ────────────────────────────────────────────────
def load_model():
    """Load fine-tuned model from Hugging Face Hub."""
    print(f"Loading model from {MODEL_REPO}...")

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
    else:
        # CPU fallback — slow but works for local testing
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            token=HF_TOKEN,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()

    device = "GPU ✅" if use_gpu else "CPU ⚠️ (slow)"
    print(f"✅ Model loaded on {device}")
    return model, tokenizer


# Load once at startup
model, tokenizer = load_model()


# ── Generate Marathi lesson ───────────────────────────────────
def generate_lesson(word: str) -> str:
    """Run word through fine-tuned model."""
    word = word.strip().lower()
    if not word:
        return ""

    prompt = f"""### Instruction:
You are Marathi Mitra, a friendly Marathi teacher for kids. \
When given an English word, teach it in Marathi with the word \
in Devanagari script, pronunciation, a simple story sentence, \
and a fun fact. Always be encouraging and kid-friendly.

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
    response  = full_text.split("### Response:")[-1].strip()
    return response


# ── Text to Speech ────────────────────────────────────────────
def text_to_speech(lesson: str) -> str | None:
    """
    Extract Marathi word from lesson and generate audio.
    Returns path to audio file or None if failed.
    """
    if not lesson:
        return None

    try:
        # Extract Devanagari text from lesson
        devanagari = re.findall(r"[\u0900-\u097F]+", lesson)

        if not devanagari:
            return None

        # Use first Devanagari word found (the Marathi word)
        marathi_word = " ".join(devanagari[:3])

        # Also extract pronunciation for English TTS fallback
        pronun_match = re.search(r"How to say it:\*?\*?\s*([A-Za-z\-]+)", lesson)
        pronun_text  = pronun_match.group(1) if pronun_match else None

        # Generate Marathi TTS
        tts = gTTS(text=marathi_word, lang="mr", slow=True)

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        )
        tts.save(tmp.name)
        return tmp.name

    except Exception as e:
        print(f"TTS error: {e}")
        return None


# ── Main app function ─────────────────────────────────────────
def learn_word(word, score, streak, learned_words):
    """
    Core function — generates lesson, TTS, updates score.
    Returns: lesson, audio, score, streak, learned_words, status
    """
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

    # Generate lesson
    lesson = generate_lesson(word)

    # Generate audio
    audio = text_to_speech(lesson)

    # Update score and streak
    if word not in learned_words:
        score        += 1
        streak       += 1
        learned_words = learned_words + [word]   # gradio state is immutable
    
    status = format_status(score, streak)
    return lesson, audio, score, streak, learned_words, status


def format_status(score, streak):
    """Format score and streak display."""
    fire = "🔥" * min(streak, 5)   # max 5 fire emojis
    stars = "⭐" * min(score, 10)   # max 10 stars
    return f"{fire} Streak: {streak}  |  {stars} Words learned: {score}"


def quick_word_click(word, score, streak, learned_words):
    """Handle quick word button click."""
    return learn_word(word, score, streak, learned_words)


# ── Custom CSS ────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;800&family=Nunito:wght@400;600;700&display=swap');

/* ── Root variables ── */
:root {
    --saffron:     #FF6B35;
    --gold:        #FFD60A;
    --cream:       #FFF8F0;
    --deep-orange: #E85D04;
    --soft-green:  #52B788;
    --light-blue:  #ADE8F4;
    --text-dark:   #1A1A2E;
    --text-mid:    #4A4A6A;
    --card-bg:     #FFFFFF;
    --shadow:      0 4px 20px rgba(255, 107, 53, 0.15);
}

/* ── Global ── */
body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #FF6B35 0%, #FFD60A 100%);
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}

.app-title {
    font-family: 'Baloo 2', cursive !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    margin: 0 !important;
    line-height: 1.1 !important;
}

.app-subtitle {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.1rem !important;
    color: rgba(255,255,255,0.9) !important;
    margin-top: 0.3rem !important;
}

/* ── Status bar ── */
.status-bar {
    text-align: center;
    font-family: 'Baloo 2', cursive;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--deep-orange);
    background: white;
    border-radius: 50px;
    padding: 0.6rem 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

/* ── Input section ── */
.input-section {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

/* ── Learn button ── */
.learn-btn {
    background: linear-gradient(135deg, #FF6B35, #E85D04) !important;
    color: white !important;
    font-family: 'Baloo 2', cursive !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    cursor: pointer !important;
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    width: 100% !important;
}

.learn-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(255, 107, 53, 0.5) !important;
}

/* ── Quick word buttons ── */
.quick-btn {
    background: white !important;
    color: var(--text-dark) !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
    border: 2px solid #FFD60A !important;
    border-radius: 50px !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin: 3px !important;
}

.quick-btn:hover {
    background: #FFD60A !important;
    transform: translateY(-1px) !important;
}

/* ── Output section ── */
.output-section {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: var(--shadow);
    font-family: 'Nunito', sans-serif;
    font-size: 1.05rem;
    line-height: 1.8;
    min-height: 200px;
    border-left: 5px solid var(--saffron);
}

/* ── Audio player ── */
.audio-player {
    background: var(--cream) !important;
    border-radius: 12px !important;
    border: 2px solid #FFD60A !important;
}

/* ── Category tabs ── */
.tab-nav {
    font-family: 'Baloo 2', cursive !important;
    font-weight: 600 !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: var(--text-mid);
    font-size: 0.9rem;
    margin-top: 1rem;
    padding: 1rem;
}

/* ── Text input ── */
textarea, input[type="text"] {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    border: 2px solid #FFD60A !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.2) !important;
}
"""


# ── Build Gradio UI ───────────────────────────────────────────
with gr.Blocks(
    css=CSS,
    title="🌸 Marathi Mitra",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="yellow",
        font=gr.themes.GoogleFont("Nunito"),
    ),
) as app:

    # ── State ──────────────────────────────────────────────────
    score_state        = gr.State(0)
    streak_state       = gr.State(0)
    learned_words_state = gr.State([])

    # ── Header ─────────────────────────────────────────────────
    gr.HTML("""
        <div class="app-header">
            <p class="app-title">🌸 Marathi Mitra 🌸</p>
            <p class="app-subtitle">
                माझा मराठी मित्र — Your Marathi Learning Friend!
            </p>
        </div>
    """)

    # ── Status bar ─────────────────────────────────────────────
    status_display = gr.HTML(
        f'<div class="status-bar">'
        f'Start learning to build your streak! 🌟'
        f'</div>'
    )

    with gr.Row():

        # ── Left column — Input ─────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML('<div class="input-section">')

            word_input = gr.Textbox(
                label="✏️ Enter an English word",
                placeholder="e.g. butterfly, sun, mother...",
                lines=1,
            )

            learn_btn = gr.Button(
                "🚀 Learn in Marathi!",
                elem_classes=["learn-btn"],
            )

            gr.HTML('<hr style="border-color: #FFD60A; margin: 1rem 0;">')

            # ── Quick word category tabs ────────────────────────
            gr.HTML("<p style='font-family: Baloo 2; font-weight: 700; color: #E85D04;'>⚡ Quick Words</p>")

            with gr.Tabs():
                for category, words in CATEGORIES.items():
                    with gr.Tab(category):
                        with gr.Row(equal_height=True):
                            # 3 buttons per row
                            for i in range(0, len(words), 3):
                                chunk = words[i:i+3]
                                with gr.Row():
                                    for word in chunk:
                                        btn = gr.Button(
                                            word,
                                            elem_classes=["quick-btn"],
                                            size="sm",
                                        )
                                        # Wire each button
                                        btn.click(
                                            fn=quick_word_click,
                                            inputs=[
                                                gr.Textbox(
                                                    value=word,
                                                    visible=False,
                                                ),
                                                score_state,
                                                streak_state,
                                                learned_words_state,
                                            ],
                                            outputs=[
                                                gr.Textbox(visible=False),
                                                gr.Audio(visible=False),
                                                score_state,
                                                streak_state,
                                                learned_words_state,
                                                status_display,
                                            ],
                                        )

            gr.HTML('</div>')

        # ── Right column — Output ───────────────────────────────
        with gr.Column(scale=2):

            lesson_output = gr.Textbox(
                label="📖 Your Marathi Lesson",
                lines=12,
                elem_classes=["output-section"],
                show_copy_button=True,
            )

            audio_output = gr.Audio(
                label="🔊 Hear the Marathi pronunciation",
                type="filepath",
                elem_classes=["audio-player"],
                autoplay=True,
            )

    # ── Footer ─────────────────────────────────────────────────
    gr.HTML("""
        <div class="app-footer">
            <p>
                Built with ❤️ using
                <strong>Fine-tuned Phi-3 Mini</strong> +
                <strong>QLoRA</strong> +
                <strong>Gradio</strong>
            </p>
            <p style="font-size: 0.8rem; color: #999;">
                Model fine-tuned on Marathi vocabulary dataset
                using Supervised Fine-Tuning (SFT)
            </p>
        </div>
    """)

    # ── Wire learn button ───────────────────────────────────────
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

    # Also trigger on Enter key
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
        share=False,      # True = public link (useful for Colab)
        server_port=7860,
        show_error=True,
    )