import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import string
import re
import emoji
import pickle

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(
    r"C:\Users\User\OneDrive\Desktop\All-Projects\Python\NLP\Emotion_Detection\train.txt",
    sep=";",
    header=None,
    names=("text", "emotion")
)

# -----------------------------
# Encode Labels
# -----------------------------

emotion_labels = {}

for i, emotion in enumerate(df["emotion"].unique()):
    emotion_labels[emotion] = i

reverse_labels = {v: k for k, v in emotion_labels.items()}

df["emotion"] = df["emotion"].map(emotion_labels)

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

df["text"] = df["text"].apply(preprocess)

X = df["text"]
y = df["emotion"]

# -----------------------------
# TF-IDF
# -----------------------------

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(X)

# -----------------------------
# Train Model
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

# -----------------------------
# Save Files
# -----------------------------

import joblib

# Save model
joblib.dump(model, "emotion_model.joblib")

# Save vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.joblib")

# Save label mapping
joblib.dump(reverse_labels, "label_mapping.joblib")

print("Model Saved Successfully!")

print("Model Saved Successfully!")