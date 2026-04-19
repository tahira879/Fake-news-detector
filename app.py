import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score


# ================================
# PAGE CONFIG (MUST BE FIRST)
# ================================

st.set_page_config(
    page_title="Fake News Prediction",
    page_icon="📰",
    layout="centered"
)


# ================================
# LOAD DATASET
# ================================

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1)


# ================================
# PREPARE DATA
# ================================

X = data["text"]
y = data["label"]


def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text


X = X.apply(clean_text)


# ================================
# VECTORIZATION
# ================================

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X = vectorizer.fit_transform(X)


# ================================
# TRAIN MODEL
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


# ================================
# CUSTOM UI CSS
# ================================

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#020617);
color:white;
}

.title{
text-align:center;
font-size:40px;
font-weight:bold;
}

.subtitle{
text-align:center;
font-size:24px;
margin-bottom:30px;
}

textarea{
border-radius:12px !important;
}

button{
border-radius:25px !important;
height:45px;
font-size:18px;
}

.result{
text-align:center;
font-size:35px;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


# ================================
# UI TITLE
# ================================

st.markdown("<div class='title'>Fake News Prediction</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Check if news is real or fake!</div>", unsafe_allow_html=True)

st.write(f"Model Accuracy: **{accuracy*100:.2f}%**")


# ================================
# INPUT BOX
# ================================

news = st.text_area("Enter the news here...")


# ================================
# PREDICTION
# ================================

if st.button("Predict"):

    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    cleaned = clean_text(news)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    probability = model.predict_proba(vector)

    real_prob = probability[0][1] * 100


    if prediction[0] == 0:

        st.markdown("<div class='result'>FAKE</div>", unsafe_allow_html=True)

    else:

        st.markdown("<div class='result'>REAL</div>", unsafe_allow_html=True)


    st.progress(int(real_prob))

    st.write(f"The news is **{real_prob:.2f}% real**")


    st.write("Give feedback if prediction is wrong 🙂")

    col1, col2 = st.columns(2)

    with col1:
        st.success("Real")

    with col2:
        st.error("Fake")
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie
