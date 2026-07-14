import streamlit as st
import joblib
import string
import re
import emoji
from pathlib import Path
from nltk.corpus import stopwords

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="AI Emotion Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

/* Hide Streamlit default menu */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Background */

.stApp{

background:linear-gradient(
135deg,
#0f172a,
#1e293b,
#312e81
);

}

/* Main */

.block-container{

padding-top:2rem;
padding-bottom:2rem;
max-width:1200px;

}

/* Hero Card */

.hero{

background:rgba(255,255,255,.08);

backdrop-filter:blur(18px);

border:1px solid rgba(255,255,255,.12);

border-radius:25px;

padding:35px;

text-align:center;

margin-bottom:30px;

box-shadow:0px 8px 30px rgba(0,0,0,.25);

}

.hero h1{

color:white;

font-size:46px;

font-weight:700;

margin-bottom:10px;

}

.hero p{

color:#d1d5db;

font-size:18px;

}

/* Input Card */

.card{

background:rgba(255,255,255,.08);

backdrop-filter:blur(16px);

padding:25px;

border-radius:20px;

border:1px solid rgba(255,255,255,.15);

box-shadow:0px 10px 30px rgba(0,0,0,.25);

}

/* Text Area */

textarea{

background:#111827 !important;

color:white !important;

border-radius:15px !important;

font-size:18px !important;

}

/* Buttons */

.stButton>button{

width:100%;

height:55px;

font-size:18px;

font-weight:600;

border-radius:14px;

border:none;

background:linear-gradient(90deg,#2563eb,#7c3aed);

color:white;

transition:.3s;

}

.stButton>button:hover{

transform:translateY(-3px);

box-shadow:0px 8px 25px rgba(37,99,235,.45);

}

/* Labels */

label{

color:white !important;

font-size:17px !important;

font-weight:600;

}

/* Metric */

[data-testid="stMetric"]{

background:rgba(255,255,255,.08);

padding:15px;

border-radius:15px;

}

</style>

""",unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR/"emotion_model.joblib")

tfidf = joblib.load(BASE_DIR/"tfidf_vectorizer.joblib")

labels = joblib.load(BASE_DIR/"label_mapping.joblib")

STOP_WORDS = set(stopwords.words("english"))

# ============================================
# PREPROCESSING
# ============================================

def preprocess(text):

    text=text.lower()

    text=re.sub(r'https?://\\S+|www\\.\\S+','',text)

    text=re.sub(r'<.*?>','',text)

    text=emoji.replace_emoji(text,replace='')

    text=text.translate(str.maketrans('','',string.punctuation))

    text=''.join(i for i in text if not i.isdigit())

    text=" ".join(
        word
        for word in text.split()
        if word not in STOP_WORDS
    )

    return text

# ============================================
# HERO
# ============================================

st.markdown("""

<div class="hero">

<h1>🤖 AI Emotion Analyzer</h1>

<p>

Understand emotions hidden inside your text using Machine Learning.

</p>

</div>

""",unsafe_allow_html=True)

# ============================================
# INPUT CARD
# ============================================

st.markdown('<div class="card">',unsafe_allow_html=True)

text=st.text_area(

"✍️ Enter your text",

height=220,

placeholder="Example: Today is one of the happiest days of my life..."

)

col1,col2=st.columns(2)

predict=col1.button("🚀 Analyze Emotion")

clear=col2.button("🗑 Clear")

st.markdown("</div>",unsafe_allow_html=True)

if clear:

    st.rerun()

# ============================================
# EMOTION STYLES
# ============================================

emotion_style = {
    "joy": ("😍", "#22c55e"),
    "happy": ("😊", "#22c55e"),
    "sadness": ("😢", "#3b82f6"),
    "sad": ("😢", "#3b82f6"),
    "anger": ("😡", "#ef4444"),
    "fear": ("😨", "#a855f7"),
    "love": ("❤️", "#ec4899"),
    "surprise": ("😲", "#f59e0b"),
}

# ============================================
# PREDICTION
# ============================================

if predict:

    if text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:

        with st.spinner("🧠 AI is analyzing your text..."):

            clean = preprocess(text)

            vector = tfidf.transform([clean])

            prediction = model.predict(vector)[0]

            emotion = labels[prediction]

        name = emotion.lower()

        icon, color = emotion_style.get(
            name,
            ("🤖", "#2563eb")
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Beautiful Prediction Card

        st.markdown(f"""
        <div style="
            background:rgba(255,255,255,.08);
            backdrop-filter:blur(15px);
            padding:30px;
            border-radius:20px;
            border-left:8px solid {color};
            box-shadow:0px 8px 30px rgba(0,0,0,.25);
        ">

        <h1 style="
            color:white;
            text-align:center;
            margin-bottom:5px;
        ">
        {icon}
        </h1>

        <h2 style="
            color:{color};
            text-align:center;
            margin-top:0;
        ">
        {emotion.upper()}
        </h2>

        </div>

        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # Confidence (if available)
        # =====================================

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(vector)[0]

            confidence = probs.max()

            st.subheader("Confidence")

            st.progress(float(confidence))

            st.metric(
                "Prediction Confidence",
                f"{confidence*100:.2f}%"
            )

            st.markdown("---")

            st.subheader("Emotion Probability")

            probability = {}

            for i, p in enumerate(probs):

                probability[
                    labels[i]
                ] = float(p)

            st.bar_chart(probability)

        # =====================================
        # Processed Text
        # =====================================

        with st.expander("📝 Processed Text"):

            st.code(clean)

        # =====================================
        # Model Info
        # =====================================

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Algorithm",
            type(model).__name__
        )

        col2.metric(
            "Features",
            tfidf.max_features if hasattr(tfidf, "max_features") else "TF-IDF"
        )

        col3.metric(
            "Classes",
            len(labels)
        )    