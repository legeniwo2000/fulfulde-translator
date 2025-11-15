from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import re
import tempfile
import base64
from num2words import num2words
from gtts import gTTS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------
# CREATE FASTAPI APP FIRST
# -------------------------------------------------------
app = FastAPI()

# CORS for Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# LOAD MODEL + TOKENIZERS
# -------------------------------------------------------
model = load_model("english_fulfulde_final.h5")
eng_tokenizer = pickle.load(open("eng_tokenizer.pkl", "rb"))
ful_tokenizer = pickle.load(open("ful_tokenizer.pkl", "rb"))
X_train, y_train, X_test, y_test, eng_vocab, ful_vocab, max_len = pickle.load(open("train_data.pkl", "rb"))

# Map predicted tokens back to words
index_to_word = {idx: word for word, idx in ful_tokenizer.word_index.items()}

# -------------------------------------------------------
# Pydantic model for input
# -------------------------------------------------------
class InputNumber(BaseModel):
    text: str

# -------------------------------------------------------
# Translation function
# -------------------------------------------------------
def translate_number(english_text):
    english_text = english_text.strip().lower()

    # If digits, convert to English words
    if re.fullmatch(r"\d+", english_text):
        english_text = num2words(int(english_text), lang="en").replace("-", " ")

    seq = eng_tokenizer.texts_to_sequences([english_text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")

    pred = model.predict(seq, verbose=0)[0]
    pred_indices = np.argmax(pred, axis=-1)

    words = [index_to_word.get(i, "") for i in pred_indices if i != 0]
    return " ".join(words).strip()

# -------------------------------------------------------
# /translate endpoint (TRANSLATION + AUDIO BASE64)
# -------------------------------------------------------
@app.post("/translate")
def translate_api(data: InputNumber):
    ful = translate_number(data.text)

    # Generate audio using gTTS
    tts = gTTS(text=ful, lang="ha")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)

    # Convert MP3 to Base64
    with open(tmp.name, "rb") as f:
        mp3_bytes = f.read()

    audio_base64 = base64.b64encode(mp3_bytes).decode("utf-8")

    # Prepended Data URL (required for browser)
    audio_data_url = f"data:audio/mp3;base64,{audio_base64}"

    # Return translation + audio
    return {
        "translation": ful,
        "audio_base64": audio_data_url
    }
