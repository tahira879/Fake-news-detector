import streamlit as st
import pandas as pd
import re
import joblib
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================================
# PAGE CONFIG — MUST BE FIRST
# ================================================
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# GLOBAL CSS
# ================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #dce8ff !important;
}

.stApp {
    background: #04060f !important;
    background-image:
        radial-gradient(ellipse 700px 450px at 85% -5%, rgba(0,229,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 500px 400px at -5% 85%, rgba(0,255,157,0.05) 0%, transparent 60%),
        linear-gradient(rgba(0,229,255,0.013) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.013) 1px, transparent 1px) !important;
    background-size: auto, auto, 48px 48px, 48px 48px !important;
}

[data-testid="stSidebar"] {
    background: #090e1d !important;
    border-right: 1px solid #1a2540 !important;
}
[data-testid="stSidebar"] * { color: #dce8ff !important; }

.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    background: linear-gradient(135deg, #00e5ff, #0070f3) !important;
    color: #000 !important;
    border: none !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px rgba(0,229,255,0.5), 0 8px 24px rgba(0,0,0,0.4) !important;
}

textarea, .stTextArea textarea {
    background: #0e1428 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 14px !important;
    color: #dce8ff !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}
textarea:focus, .stTextArea textarea:focus {
    border-color: rgba(0,229,255,0.45) !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,0.08) !important;
}

[data-testid="stMetric"] {
    background: #090e1d !important;
    border: 1px solid #1a2540 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] {
    color: #5a6e9a !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 26px !important;
    color: #00e5ff !important;
}

[data-testid="stExpander"] {
    background: #090e1d !important;
    border: 1px solid #1a2540 !important;
    border-radius: 14px !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #00e5ff, #00ff9d) !important;
    border-radius: 100px !important;
}
.stProgress > div > div {
    background: #0e1428 !important;
    border-radius: 100px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #090e1d !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid #1a2540 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #5a6e9a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.1) !important;
    color: #00e5ff !important;
}

.stRadio label { color: #dce8ff !important; }
hr { border-color: #1a2540 !important; }
#MainMenu, footer, header { visibility: hidden; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #04060f; }
::-webkit-scrollbar-thumb { background: #1a2540; border-radius: 100px; }

@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.15;} }
@keyframes pulse {
    0%,100% { box-shadow: 0 0 30px var(--gc), 0 0 70px var(--gc2); }
    50%      { box-shadow: 0 0 50px var(--gc), 0 0 100px var(--gc2); }
}
</style>
""", unsafe_allow_html=True)


# ================================================
# HELPERS
# ================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def neon_metric(label, value, color="#00e5ff", suffix=""):
    return (
        f"<div style='background:#090e1d;border:1px solid #1a2540;border-radius:14px;"
        f"padding:18px 20px;text-align:center;'>"
        f"<div style='font-size:11px;letter-spacing:2px;text-transform:uppercase;"
        f"color:#5a6e9a;margin-bottom:6px;'>{label}</div>"
        f"<div style='font-family:Syne,sans-serif;font-size:26px;font-weight:700;"
        f"color:{color};text-shadow:0 0 14px {color}66;'>{value}{suffix}</div>"
        f"</div>"
    )


def result_card(label, emoji, color, prob, glow):
    return (
        f"<div style='background:linear-gradient(135deg,{color}10,{color}05);"
        f"border:2px solid {color}55;border-radius:20px;"
        f"padding:36px 28px;text-align:center;"
        f"--gc:{color}44;--gc2:{glow}22;"
        f"animation:pulse 2.5s ease-in-out infinite;'>"
        f"<div style='font-size:58px;margin-bottom:10px;'>{emoji}</div>"
        f"<div style='font-family:Syne,sans-serif;font-size:44px;font-weight:800;"
        f"color:{color};letter-spacing:-1px;"
        f"text-shadow:0 0 22px {glow}99,0 0 55px {glow}44;'>{label}</div>"
        f"<div style='color:#5a6e9a;font-size:13px;margin-top:8px;letter-spacing:1px;"
        f"text-transform:uppercase;'>Confidence: "
        f"<span style='color:{color};font-weight:600;'>{prob:.1f}%</span></div>"
        f"<div style='margin-top:20px;background:#0e1428;border-radius:100px;"
        f"height:8px;overflow:hidden;'>"
        f"<div style='width:{prob}%;height:100%;"
        f"background:linear-gradient(90deg,{color},{glow});"
        f"border-radius:100px;box-shadow:0 0 10px {color}88;'></div>"
        f"</div></div>"
    )


def prob_bar(label, prob, color):
    return (
        f"<div style='background:#090e1d;border:1px solid #1a2540;"
        f"border-radius:14px;padding:18px;'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:10px;'>"
        f"<span style='color:#5a6e9a;font-size:13px;'>{label}</span>"
        f"<span style='color:{color};font-weight:600;'>{prob:.1f}%</span></div>"
        f"<div style='background:#0e1428;border-radius:100px;height:6px;'>"
        f"<div style='width:{prob}%;height:100%;background:{color};"
        f"border-radius:100px;box-shadow:0 0 8px {color}77;'></div>"
        f"</div></div>"
    )


def plotly_base(margin=None):
    """Return a Plotly layout dict. Pass margin as a separate dict to avoid key conflicts."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,14,29,0.85)",
        font=dict(family="DM Sans", color="#dce8ff", size=12),
    )
    if margin:
        layout["margin"] = margin
    return layout


# ================================================
# MODEL TRAINING (cached)
# ================================================
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    MODEL_PATH = "fake_news_model.pkl"
    VEC_PATH   = "vectorizer.pkl"
    ACC_PATH   = "accuracy.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        acc        = joblib.load(ACC_PATH) if os.path.exists(ACC_PATH) else 0.0
        return model, vectorizer, acc, None, None

    for paths in [("Fake.csv.zip", "True.csv.zip"), ("Fake.csv", "True.csv")]:
        try:
            fake = pd.read_csv(paths[0])
            true = pd.read_csv(paths[1])
            break
        except FileNotFoundError:
            continue
    else:
        return None, None, 0.0, None, None

    fake["label"] = 0
    true["label"] = 1
    data = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

    if "title" in data.columns:
        data["full_text"] = data["text"].fillna("") + " " + data["title"].fillna("")
    else:
        data["full_text"] = data["text"].fillna("")

    X = data["full_text"].apply(clean_text)
    y = data["label"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=3,
        ngram_range=(1, 2),
        max_features=60000,
        sublinear_tf=True
    )
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    pred     = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    rep      = classification_report(y_test, pred, output_dict=True)
    cm_mat   = confusion_matrix(y_test, pred)

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(accuracy,   ACC_PATH)

    return model, vectorizer, accuracy, rep, cm_mat


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    pred    = int(model.predict(vec)[0])
    proba   = model.predict_proba(vec)[0]
    return pred, float(proba[1] * 100), float(proba[0] * 100)


# ================================================
# LOAD MODEL
# ================================================
with st.spinner("Loading model..."):
    model, vectorizer, accuracy, report, cm = load_or_train_model()

model_ready = model is not None


# ================================================
# SIDEBAR
# ================================================
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:24px 0 12px;'>"
        "<div style='font-family:Syne,sans-serif;font-size:28px;font-weight:800;"
        "background:linear-gradient(135deg,#ffffff,#00e5ff,#00ff9d);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "background-clip:text;letter-spacing:-1px;'>TruthLens</div>"
        "<div style='font-size:11px;color:#5a6e9a;letter-spacing:2px;"
        "text-transform:uppercase;margin-top:4px;'>Fake News Detector</div>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#5a6e9a;letter-spacing:2px;"
        "text-transform:uppercase;margin-bottom:10px;'>Navigation</div>",
        unsafe_allow_html=True
    )
    page = st.radio("", ["🔍  Analyze News", "📊  Model Stats", "📖  About"],
                    label_visibility="collapsed")
    page = page.split("  ")[1]
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;color:#5a6e9a;line-height:1.9;'>"
        "<div style='color:#dce8ff;font-weight:500;margin-bottom:8px;'>How it works</div>"
        "Uses <span style='color:#00e5ff;'>TF-IDF bigrams</span> with "
        "<span style='color:#00ff9d;'>Logistic Regression</span> "
        "trained on thousands of real &amp; fake news articles."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    if model_ready:
        st.markdown(
            f"<div style='text-align:center;font-size:12px;color:#5a6e9a;'>"
            f"Model accuracy<br/>"
            f"<span style='font-family:Syne,sans-serif;font-size:22px;font-weight:700;"
            f"color:#00e5ff;'>{accuracy*100:.2f}%</span></div>",
            unsafe_allow_html=True
        )


# ================================================
# GLOBAL HEADER
# ================================================
st.markdown(
    "<div style='text-align:center;padding:44px 0 32px;'>"
    "<div style='display:inline-flex;align-items:center;gap:8px;"
    "background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.2);"
    "border-radius:100px;padding:6px 18px;font-size:11px;letter-spacing:2px;"
    "color:#00e5ff;text-transform:uppercase;margin-bottom:18px;'>"
    "<span style='width:6px;height:6px;background:#00e5ff;border-radius:50%;"
    "display:inline-block;animation:blink 2s infinite;'></span>"
    "AI-Powered News Verification"
    "</div>"
    "<div style='font-family:Syne,sans-serif;"
    "font-size:clamp(34px,6vw,58px);font-weight:800;"
    "letter-spacing:-2px;line-height:1;margin-bottom:12px;"
    "background:linear-gradient(135deg,#ffffff 0%,#00e5ff 50%,#00ff9d 100%);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "background-clip:text;'>TruthLens</div>"
    "<div style='color:#5a6e9a;font-size:16px;font-weight:300;"
    "max-width:440px;margin:0 auto;'>"
    "Detect misinformation with machine-learning precision"
    "</div></div>",
    unsafe_allow_html=True
)


# ================================================
# PAGE: ANALYZE NEWS
# ================================================
if page == "Analyze News":

    if model_ready:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(neon_metric("Accuracy",  f"{accuracy*100:.2f}", "#00e5ff", "%"), unsafe_allow_html=True)
        with c2: st.markdown(neon_metric("Algorithm", "LogReg",              "#a855f7"),        unsafe_allow_html=True)
        with c3: st.markdown(neon_metric("Features",  "60K",                 "#00ff9d"),        unsafe_allow_html=True)
        with c4: st.markdown(neon_metric("N-Grams",   "1 – 2",              "#ffb300"),        unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#090e1d;border:1px solid #1a2540;border-radius:18px;"
        "padding:28px 28px 8px;position:relative;overflow:hidden;margin-bottom:4px;'>"
        "<div style='position:absolute;top:0;left:0;right:0;height:1px;"
        "background:linear-gradient(90deg,transparent,#00e5ff66,transparent);'></div>"
        "<div style='font-size:11px;letter-spacing:2px;text-transform:uppercase;"
        "color:#5a6e9a;margin-bottom:10px;'>Paste News Article</div>"
        "</div>",
        unsafe_allow_html=True
    )

    news_text = st.text_area(
        "",
        height=190,
        placeholder="Paste a news headline, paragraph, or full article here...",
        label_visibility="collapsed",
        key="news_input"
    )

    word_count = len(news_text.split()) if news_text.strip() else 0
    char_count = len(news_text)

    info_col, btn_col = st.columns([3, 2])
    with info_col:
        st.markdown(
            f"<div style='color:#5a6e9a;font-size:13px;padding-top:6px;'>"
            f"<span style='color:#00e5ff;'>{word_count}</span> words &nbsp;·&nbsp; "
            f"<span style='color:#00e5ff;'>{char_count}</span> chars</div>",
            unsafe_allow_html=True
        )
    with btn_col:
        b1, b2 = st.columns(2)
        with b1:
            analyze_clicked = st.button("🔍 Analyze", use_container_width=True)
        with b2:
            clear_clicked = st.button("✕ Clear", use_container_width=True, key="clear_btn")

    if clear_clicked:
        st.rerun()

    with st.expander("📋 Try Sample Articles"):
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                "<div style='color:#00ff9d;font-size:12px;font-weight:600;margin-bottom:6px;'>"
                "✓ REAL NEWS SAMPLE</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<div style='background:#0e1428;border:1px solid #1a2540;border-radius:10px;"
                "padding:14px;font-size:13px;color:#8a9fc0;line-height:1.65;'>"
                "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
                "marking the tenth consecutive increase as policymakers continue their effort "
                "to bring inflation down from its four-decade high. The decision was unanimous "
                "among the Federal Open Market Committee members."
                "</div>",
                unsafe_allow_html=True
            )
        with s2:
            st.markdown(
                "<div style='color:#ff3d6e;font-size:12px;font-weight:600;margin-bottom:6px;'>"
                "✗ FAKE NEWS SAMPLE</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<div style='background:#0e1428;border:1px solid #1a2540;border-radius:10px;"
                "padding:14px;font-size:13px;color:#8a9fc0;line-height:1.65;'>"
                "SHOCKING: Scientists CONFIRM that drinking bleach cures all diseases! "
                "The mainstream media is HIDING this miracle cure! Government is suppressing "
                "this to keep big pharma profits flowing! SHARE before they DELETE THIS!!!"
                "</div>",
                unsafe_allow_html=True
            )

    st.markdown("<br/>", unsafe_allow_html=True)

    if analyze_clicked:
        if not news_text.strip():
            st.markdown(
                "<div style='background:rgba(255,179,0,0.07);border:1px solid rgba(255,179,0,0.3);"
                "border-radius:14px;padding:18px 24px;color:#ffb300;text-align:center;'>"
                "⚠️ &nbsp; Please enter a news article to analyze.</div>",
                unsafe_allow_html=True
            )
        elif not model_ready:
            st.markdown(
                "<div style='background:rgba(255,61,110,0.07);border:1px solid rgba(255,61,110,0.3);"
                "border-radius:14px;padding:18px 24px;color:#ff3d6e;text-align:center;'>"
                "❌ &nbsp; Model not loaded. Ensure Fake.csv &amp; True.csv are in the project folder.</div>",
                unsafe_allow_html=True
            )
        else:
            prog   = st.progress(0)
            status = st.empty()
            for pct, msg in [
                (15, "Preprocessing text…"),
                (40, "TF-IDF vectorizing…"),
                (65, "Running inference…"),
                (90, "Computing probabilities…"),
                (100, "Done ✓")
            ]:
                prog.progress(pct)
                status.markdown(
                    f"<div style='color:#5a6e9a;font-size:13px;text-align:center;'>{msg}</div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.16)
            prog.empty()
            status.empty()

            pred, real_prob, fake_prob = predict(news_text, model, vectorizer)

            st.markdown("<br/>", unsafe_allow_html=True)
            left, right = st.columns([1.15, 1])

            with left:
                if pred == 1:
                    st.markdown(result_card("REAL", "✅", "#00ff9d", real_prob, "#00e5ff"), unsafe_allow_html=True)
                else:
                    st.markdown(result_card("FAKE", "🚨", "#ff3d6e", fake_prob, "#ff6b35"), unsafe_allow_html=True)

            with right:
                # Donut chart — margin passed only once inside update_layout
                fig = go.Figure(go.Pie(
                    values=[fake_prob, real_prob],
                    labels=["Fake", "Real"],
                    hole=0.65,
                    marker=dict(
                        colors=["#ff3d6e", "#00ff9d"],
                        line=dict(color="#04060f", width=3)
                    ),
                    textinfo="percent",
                    textfont=dict(size=13, family="DM Sans", color="white"),
                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
                ))
                fig.add_annotation(
                    text="REAL" if pred == 1 else "FAKE",
                    x=0.5, y=0.54,
                    font=dict(size=19, family="Syne", color="#00e5ff"),
                    showarrow=False
                )
                fig.add_annotation(
                    text=f"{real_prob if pred==1 else fake_prob:.0f}%",
                    x=0.5, y=0.38,
                    font=dict(size=13, family="DM Sans", color="#5a6e9a"),
                    showarrow=False
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(9,14,29,0.85)",
                    font=dict(family="DM Sans", color="#dce8ff", size=12),
                    margin=dict(l=10, r=10, t=10, b=40),
                    showlegend=True,
                    legend=dict(
                        orientation="h", x=0.5, xanchor="center", y=-0.05,
                        font=dict(size=12, color="#dce8ff")
                    ),
                    height=258,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:11px;letter-spacing:2px;text-transform:uppercase;"
                "color:#5a6e9a;margin-bottom:12px;'>Probability Breakdown</div>",
                unsafe_allow_html=True
            )
            pb1, pb2 = st.columns(2)
            with pb1:
                st.markdown(prob_bar("Real News", real_prob, "#00ff9d"), unsafe_allow_html=True)
            with pb2:
                st.markdown(prob_bar("Fake News", fake_prob, "#ff3d6e"), unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:11px;letter-spacing:2px;text-transform:uppercase;"
                "color:#5a6e9a;margin-bottom:10px;'>Was this prediction correct?</div>",
                unsafe_allow_html=True
            )
            fb1, fb2, _ = st.columns([1, 1, 3])
            with fb1:
                if st.button("👍  Yes"):
                    st.markdown(
                        "<div style='color:#00ff9d;font-size:13px;'>Thank you! ✓</div>",
                        unsafe_allow_html=True
                    )
            with fb2:
                if st.button("👎  No"):
                    st.markdown(
                        "<div style='color:#ff3d6e;font-size:13px;'>Thanks for the feedback!</div>",
                        unsafe_allow_html=True
                    )


# ================================================
# PAGE: MODEL STATS
# ================================================
elif page == "Model Stats":
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:28px;font-weight:700;"
        "margin-bottom:6px;color:#dce8ff;'>Model Performance</div>"
        "<div style='color:#5a6e9a;font-size:14px;margin-bottom:28px;'>"
        "Metrics evaluated on 20% held-out test set</div>",
        unsafe_allow_html=True
    )

    if not model_ready:
        st.markdown(
            "<div style='background:rgba(255,179,0,0.07);border:1px solid rgba(255,179,0,0.3);"
            "border-radius:14px;padding:24px;text-align:center;color:#ffb300;'>"
            "⚠️ Model not trained yet. Add Fake.csv and True.csv and restart.</div>",
            unsafe_allow_html=True
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Accuracy",  f"{accuracy*100:.2f}%")
        with m2:
            prec = (report["weighted avg"]["precision"] * 100) if report else 0
            st.metric("Precision", f"{prec:.2f}%")
        with m3:
            rec = (report["weighted avg"]["recall"] * 100) if report else 0
            st.metric("Recall", f"{rec:.2f}%")
        with m4:
            f1 = (report["weighted avg"]["f1-score"] * 100) if report else 0
            st.metric("F1-Score", f"{f1:.2f}%")

        st.markdown("<br/>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📊 Metrics Chart", "🔲 Confusion Matrix", "⚙️ Model Config"])

        with tab1:
            if report:
                fig2 = go.Figure()
                for metric, color in [("precision","#ff3d6e"),("recall","#00ff9d"),("f1-score","#00e5ff")]:
                    fig2.add_trace(go.Bar(
                        name=metric.capitalize(),
                        x=["Fake (0)", "Real (1)"],
                        y=[report["0"][metric]*100, report["1"][metric]*100],
                        marker_color=color,
                        marker_line_color="rgba(0,0,0,0)",
                        opacity=0.85,
                    ))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(9,14,29,0.85)",
                    font=dict(family="DM Sans", color="#dce8ff", size=12),
                    margin=dict(l=20, r=20, t=50, b=20),
                    barmode="group",
                    xaxis=dict(gridcolor="#1a2540"),
                    yaxis=dict(gridcolor="#1a2540", range=[0, 108], ticksuffix="%"),
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                    height=380,
                    title=dict(text="Per-Class Performance Metrics",
                               font=dict(size=15, color="#dce8ff"))
                )
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Retrain model to view metrics.")

        with tab2:
            if cm is not None:
                fig3 = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Fake", "Real"],
                    y=["Fake", "Real"],
                    color_continuous_scale=[[0,"#090e1d"],[0.5,"#1e3a5f"],[1,"#00e5ff"]],
                    text_auto=True,
                )
                fig3.update_traces(textfont=dict(size=22, family="Syne", color="white"))
                fig3.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(9,14,29,0.85)",
                    font=dict(family="DM Sans", color="#dce8ff", size=12),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=380,
                    title=dict(text="Confusion Matrix", font=dict(size=15, color="#dce8ff")),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Retrain model to view confusion matrix.")

        with tab3:
            params = [
                ("Algorithm",         "Logistic Regression"),
                ("Regularization (C)","5.0"),
                ("Max Iterations",    "1000"),
                ("Vectorizer",        "TF-IDF"),
                ("N-Gram Range",      "(1, 2)"),
                ("Max Features",      "60,000"),
                ("Min Doc Freq",      "3"),
                ("Max Doc Freq",      "70%"),
                ("TF-IDF Sublinear",  "Yes"),
                ("Test Split",        "20%"),
            ]
            rows = "".join(
                f"<tr>"
                f"<td style='padding:12px 16px;color:#8a9fc0;"
                f"border-bottom:1px solid #1a2540;'>{p}</td>"
                f"<td style='padding:12px 16px;color:#00e5ff;"
                f"font-family:Syne,sans-serif;font-weight:600;"
                f"border-bottom:1px solid #1a2540;'>{v}</td>"
                f"</tr>"
                for p, v in params
            )
            st.markdown(
                f"<div style='background:#090e1d;border:1px solid #1a2540;"
                f"border-radius:16px;overflow:hidden;'>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr style='background:#0e1428;'>"
                f"<th style='padding:14px 16px;text-align:left;color:#5a6e9a;"
                f"font-size:11px;letter-spacing:2px;text-transform:uppercase;"
                f"border-bottom:1px solid #1a2540;'>Parameter</th>"
                f"<th style='padding:14px 16px;text-align:left;color:#5a6e9a;"
                f"font-size:11px;letter-spacing:2px;text-transform:uppercase;"
                f"border-bottom:1px solid #1a2540;'>Value</th>"
                f"</tr></thead><tbody>{rows}</tbody></table></div>",
                unsafe_allow_html=True
            )


# ================================================
# PAGE: ABOUT
# ================================================
elif page == "About":
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:28px;font-weight:700;"
        "margin-bottom:6px;color:#dce8ff;'>About TruthLens</div>"
        "<div style='color:#5a6e9a;font-size:14px;margin-bottom:28px;'>"
        "Pipeline overview and methodology</div>",
        unsafe_allow_html=True
    )

    steps = [
        ("01", "#00e5ff", "Text Input",
         "User submits a news article, headline, or paragraph for analysis."),
        ("02", "#a855f7", "Preprocessing",
         "Text is lowercased; URLs, HTML tags, and punctuation are removed, leaving clean alphabetic tokens."),
        ("03", "#00ff9d", "TF-IDF Vectorization",
         "Cleaned text → 60,000-feature sparse vector using unigrams + bigrams. sublinear_tf=True boosts rare informative terms."),
        ("04", "#ffb300", "Logistic Regression",
         "Model computes log-odds from TF-IDF features. C=5.0 controls regularization strength for best generalization."),
        ("05", "#ff3d6e", "Result & Confidence",
         "predict_proba returns Real/Fake probabilities. The higher class is returned with its confidence score."),
    ]
    for num, color, title, desc in steps:
        st.markdown(
            f"<div style='display:flex;gap:20px;margin-bottom:14px;"
            f"background:#090e1d;border:1px solid #1a2540;border-radius:16px;padding:20px 24px;'>"
            f"<div style='font-family:Syne,sans-serif;font-size:26px;font-weight:800;"
            f"color:{color};opacity:0.35;flex-shrink:0;min-width:40px;line-height:1;'>{num}</div>"
            f"<div>"
            f"<div style='font-family:Syne,sans-serif;font-size:16px;font-weight:700;"
            f"color:{color};margin-bottom:5px;'>{title}</div>"
            f"<div style='color:#8a9fc0;font-size:14px;line-height:1.7;'>{desc}</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.2);"
        "border-radius:16px;padding:24px;'>"
        "<div style='font-family:Syne,sans-serif;font-size:16px;font-weight:700;"
        "color:#00e5ff;margin-bottom:10px;'>⚠️ Disclaimer</div>"
        "<div style='color:#8a9fc0;font-size:14px;line-height:1.8;'>"
        "TruthLens is an AI-assisted educational tool. "
        "It should not be used as the sole source of truth for verifying news. "
        "Always cross-reference with trusted sources and fact-checking organizations. "
        "The model may produce incorrect results on very short text or content "
        "outside its training distribution."
        "</div></div>",
        unsafe_allow_html=True
    )
