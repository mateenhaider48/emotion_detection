import streamlit as st
import pickle
import string
import re
import emoji

from nltk.corpus import stopwords

import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "emotion_model.joblib")

tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")

labels = joblib.load(BASE_DIR / "label_mapping.joblib")

# -----------------------------
# Preprocessing
# -----------------------------

def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_number(txt):
    return ''.join(ch for ch in txt if not ch.isdigit())

def remove_url(txt):
    return re.sub(r'https?://\S+|www\.\S+', '', txt)

def remove_html(txt):
    return re.sub(r'<.*?>', '', txt)

def remove_emoji(txt):
    return emoji.replace_emoji(txt, replace='')

def remove_stopwords(txt):
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in txt.split() if w.lower() not in stop_words])

def preprocess(text):

    text = text.lower()
    text = remove_punc(text)
    text = remove_number(text)
    text = remove_url(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)

    return text

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Emotion Detection", page_icon="😊")

st.title("😊 Emotion Detection")

text = st.text_area("Enter Your Text")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter text.")

    else:

        clean = preprocess(text)

        vector = tfidf.transform([clean])

        prediction = model.predict(vector)[0]

        emotion = labels[prediction]

        st.success(f"Predicted Emotion: **{emotion.upper()}**")