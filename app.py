# ====================================================
# AI RULE-BASED NUMBER TRANSLATOR FOR FULANI LANGUAGE
# By Musharaf Agoh (2025)
# Offline Edition for Project Defense
# ====================================================

import gradio as gr
import os
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from num2words import num2words
from gtts import gTTS
import tempfile
import base64

# ---------------------------
# Load Model and Tokenizers
# ---------------------------
MODEL_PATH = "english_fulfulde_final.h5"
ENG_TOKENIZER_PATH = "eng_tokenizer.pkl"
FUL_TOKENIZER_PATH = "ful_tokenizer.pkl"
TRAIN_DATA_PATH = "train_data.pkl"

print("🔄 Loading model and tokenizers...")
model = load_model(MODEL_PATH)

with open(ENG_TOKENIZER_PATH, "rb") as f:
    eng_tokenizer = pickle.load(f)
with open(FUL_TOKENIZER_PATH, "rb") as f:
    ful_tokenizer = pickle.load(f)
with open(TRAIN_DATA_PATH, "rb") as f:
    X_train, y_train, X_test, y_test, english_vocab_size, fulfulde_vocab_size, max_len = pickle.load(f)

index_to_word = {idx: word for word, idx in ful_tokenizer.word_index.items()}
print("✅ Model and tokenizers loaded successfully!")


# ---------------------------
# Translation Function
# ---------------------------
def translate_number(english_text):
    english_text = english_text.strip().lower()
    if not english_text:
        return "Please enter a valid number."

    if re.fullmatch(r"\d+", english_text):
        english_text = num2words(int(english_text), lang="en").replace("-", " ")

    seq = eng_tokenizer.texts_to_sequences([english_text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")

    pred = model.predict(seq, verbose=0)[0]
    pred_indices = np.argmax(pred, axis=-1)
    words = [index_to_word.get(idx, "") for idx in pred_indices if idx != 0]
    return " ".join(words).strip() if words else "unknown"


# ---------------------------
# Lovable.ai Compatible Audio
# ---------------------------
def translate_and_speak(english_text):
    translation = translate_number(english_text)
    if translation == "unknown" or "valid number" in translation.lower():
        return translation, "<p style='color:red;'>No audio generated.</p>"

    try:
        # Generate Hausa audio
        tts = gTTS(text=translation, lang="ha")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)

        # Convert to Base64
        with open(tmp.name, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # HTML audio player (works inside Lovable iframe)
        audio_html = f"""
        <audio controls style="width:100%; margin-top:10px;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Your browser does not support audio.
        </audio>
        """

        return translation, audio_html

    except Exception as e:
        return f"Error generating audio: {e}", "<p>Error generating audio.</p>"


# ---------------------------
# INTERFACE
# ---------------------------
with gr.Blocks(
    title="AI RULE-BASED NUMBER TRANSLATOR FOR FULANI LANGUAGE",
    theme=gr.themes.Soft(primary_hue="cyan", text_size="md"),
    css="""
    body {background: #0b0f19; color: #e6fbff;}
    .gr-button {background: #00e6e6 !important; color: black !important; font-weight: bold;}
    .gr-textbox {border: 1px solid #00e6e6 !important;}
    #profile_pic {border-radius: 50%; box-shadow: 0 0 12px #00e6e6;}
    """
) as demo:

    gr.Markdown("<h2 style='text-align:center; color:#00e6e6;'>🤖 AI RULE-BASED NUMBER TRANSLATOR — Fulfulde</h2>")

    with gr.Row():

        # LEFT SIDEBAR
        with gr.Column(scale=1):
            if os.path.exists("profile.jpg"):
                gr.Image("profile.jpg", show_label=False, width=150, elem_id="profile_pic")

            gr.Markdown("### **Musharaf Agoh**")
            gr.Markdown("AI Fulfulde Number Translator<br>Federal University Wukari — 2025")

            gr.Markdown("""
            **Supervisor:** Dr. Siman Emmanuel  
            **Department:** Computer Science  
            **Institution:** Federal University Wukari  
            **Year:** 2025
            """)

            gr.Markdown("---")
            gr.Markdown("This translator converts English numbers into Fulfulde with Hausa-accented speech output.")

        # RIGHT PANEL
        with gr.Column(scale=2):

            input_text = gr.Textbox(
                label="Enter English Number or Words",
                placeholder="e.g., 25 or 'twenty five'",
                lines=3
            )

            output_text = gr.Textbox(
                label="Fulfulde Translation",
                interactive=False
            )

            audio_output = gr.HTML(label="Hausa Voice")  # Important change

            with gr.Row():
                run_btn = gr.Button("Translate 🔁")
                clear_btn = gr.ClearButton([input_text, output_text, audio_output])

            run_btn.click(translate_and_speak, input_text, [output_text, audio_output])

    gr.Markdown("---")
    gr.Markdown("<p style='text-align:center;'>Developed by <b>Musharaf Agoh (2025)</b></p>")


# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    demo.launch(share=False)
