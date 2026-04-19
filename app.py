import streamlit as st
import pandas as pd
import numpy as np
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
# GLOBAL CSS — Dark Neon Cyber Theme
# ================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg:        #04060f;
    --surface:   #090e1d;
    --surface2:  #0e1428;
    --border:    #1a2540;
    --text:      #dce8ff;
    --muted:     #5a6e9a;
    --cyan:      #00e5ff;
    --green:     #00ff9d;
    --red:       #ff3d6e;
    --amber:     #ffb300;
    --purple:    #a855f7;
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── APP BACKGROUND ── */
.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 800px 500px at 80% -10%, rgba(0,229,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 600px 400px at -10% 80%, rgba(0,255,157,0.05) 0%, transparent 60%),
        linear-gradient(rgba(0,229,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.015) 1px, transparent 1px);
    background-size: auto, auto, 48px 48px, 48px 48px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── BUTTONS ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
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
    box-shadow: 0 0 35px rgba(0,229,255,0.45), 0 8px 24px rgba(0,0,0,0.4) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* Secondary button style */
.secondary-btn > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    box-shadow: none !important;
}
.secondary-btn > button:hover {
    border-color: rgba(255,255,255,0.25) !important;
    color: var(--text) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── TEXT AREA ── */
textarea, .stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
textarea:focus, .stTextArea textarea:focus {
    border-color: rgba(0,229,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,0.08), 0 0 20px rgba(0,229,255,0.2) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: border-color 0.3s, transform 0.3s !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,229,255,0.3) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12px !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 26px !important; color: var(--cyan) !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

/* ── SELECTBOX / RADIO ── */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stRadio label { color: var(--text) !important; }

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--green)) !important;
    border-radius: 100px !important;
}
.stProgress > div > div {
    background: var(--surface2) !important;
    border-radius: 100px !important;
}

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; }

/* ── PLOTLY CHART ── */
.js-plotly-plot { border-radius: 14px !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.1) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.2) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 100px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── MARKDOWN TEXT ── */
.stMarkdown p, .stMarkdown li { color: var(--text) !important; }

/* ── HIDE STREAMLIT BRANDING ── */
#MainMenu, footer, header { visibility: hidden; }
.viewerBadge_container__1QSob { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ================================================
# UTILITY FUNCTIONS
# ================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def neon_metric(label, value, color="#00e5ff", suffix=""):
    return f"""
    <div style="background:#090e1d;border:1px solid #1a2540;border-radius:14px;
                padding:18px 20px;text-align:center;transition:all 0.3s;">
      <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                  color:#5a6e9a;margin-bottom:6px;">{label}</div>
      <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:700;
                  color:{color};text-shadow:0 0 15px {color}55;">{value}{suffix}</div>
    </div>
    """


def result_card(label, emoji, color, prob, glow_color):
    bar_color = color
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}12, {color}06);
        border: 2px solid {color}55;
        border-radius: 20px;
        padding: 36px 28px;
        text-align: center;
        box-shadow: 0 0 40px {glow_color}33, 0 0 80px {glow_color}11;
        animation: pulse 2s ease-in-out infinite;
        position: relative; overflow: hidden;
    ">
      <div style="font-size:64px;margin-bottom:12px;">{emoji}</div>
      <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:800;
                  color:{color};letter-spacing:-1px;
                  text-shadow:0 0 20px {glow_color}88, 0 0 60px {glow_color}44;">
          {label}
      </div>
      <div style="color:#5a6e9a;font-size:14px;margin-top:8px;letter-spacing:1px;
                  text-transform:uppercase;">
          Confidence: <span style="color:{color};font-weight:600;">{prob:.1f}%</span>
      </div>
      <div style="margin-top:20px;background:#0e1428;border-radius:100px;height:8px;overflow:hidden;">
        <div style="width:{prob}%;height:100%;
             background:linear-gradient(90deg,{color},{glow_color});
             border-radius:100px;
             box-shadow:0 0 10px {color}88;
             transition:width 1s ease;"></div>
      </div>
    </div>
    <style>
    @keyframes pulse {{
      0%,100% {{ box-shadow: 0 0 40px {glow_color}33, 0 0 80px {glow_color}11; }}
      50% {{ box-shadow: 0 0 60px {glow_color}55, 0 0 100px {glow_color}22; }}
    }}
    </style>
    """


# ================================================
# MODEL TRAINING (cached)
# ================================================
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    MODEL_PATH = "fake_news_model.pkl"
    VEC_PATH   = "vectorizer.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        # load accuracy from disk if available
        acc = joblib.load("accuracy.pkl") if os.path.exists("accuracy.pkl") else 0.0
        return model, vectorizer, acc, None, None

    # Try to load datasets
    try:
        fake = pd.read_csv("Fake.csv.zip")
        true = pd.read_csv("True.csv.zip")
    except FileNotFoundError:
        try:
            fake = pd.read_csv("Fake.csv")
            true = pd.read_csv("True.csv")
        except FileNotFoundError:
            return None, None, 0, None, None

    fake["label"] = 0
    true["label"] = 1
    data = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

    data["text"] = data["text"].fillna("") + " " + data.get("title", pd.Series([""] * len(data))).fillna("")
    X = data["text"].apply(clean_text)
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

    model = LogisticRegression(
        C=5.0, max_iter=1000, solver="lbfgs", n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred     = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    report   = classification_report(y_test, pred, output_dict=True)
    cm       = confusion_matrix(y_test, pred)

    joblib.dump(model,     MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(accuracy,  "accuracy.pkl")

    return model, vectorizer, accuracy, report, cm


# ================================================
# PREDICT
# ================================================
def predict(text, model, vectorizer):
    cleaned   = clean_text(text)
    vec       = vectorizer.transform([cleaned])
    pred      = model.predict(vec)[0]
    proba     = model.predict_proba(vec)[0]
    real_prob = proba[1] * 100
    fake_prob = proba[0] * 100
    return pred, real_prob, fake_prob


# ================================================
# PLOTLY THEME HELPER
# ================================================
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(9,14,29,0.8)",
    font=dict(family="DM Sans", color="#dce8ff", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ================================================
# SIDEBAR
# ================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 8px;">
      <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;
           background:linear-gradient(135deg,#fff,#00e5ff,#00ff9d);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text;letter-spacing:-1px;">TruthLens</div>
      <div style="font-size:11px;color:#5a6e9a;letter-spacing:2px;
           text-transform:uppercase;margin-top:4px;">Fake News Detector</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div style='font-size:11px;color:#5a6e9a;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🔍  Analyze News", "📊  Model Stats", "📖  About"],
        label_visibility="collapsed"
    )
    page = page.split("  ")[1]

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:#5a6e9a;line-height:1.8;">
      <div style="color:#dce8ff;font-weight:500;margin-bottom:8px;">How it works</div>
      Uses <span style="color:#00e5ff;">TF-IDF</span> vectorization with
      <span style="color:#00ff9d;">Logistic Regression</span> trained on
      thousands of real & fake news articles.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size:11px;color:#5a6e9a;text-align:center;">
      Built with Streamlit & scikit-learn<br/>
      <span style="color:#00e5ff;">v2.0 Pro</span>
    </div>
    """, unsafe_allow_html=True)


# ================================================
# LOAD MODEL
# ================================================
with st.spinner(""):
    model, vectorizer, accuracy, report, cm = load_or_train_model()

model_ready = model is not None


# ================================================
# HEADER
# ================================================
st.markdown("""
<div style="text-align:center;padding:48px 0 36px;">
  <div style="display:inline-flex;align-items:center;gap:8px;
       background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.2);
       border-radius:100px;padding:6px 18px;font-size:11px;letter-spacing:2px;
       color:#00e5ff;text-transform:uppercase;margin-bottom:20px;">
    <span style="width:6px;height:6px;background:#00e5ff;border-radius:50%;
         display:inline-block;animation:blink 2s infinite;"></span>
    AI-Powered Verification
  </div>
  <div style="font-family:'Syne',sans-serif;font-size:clamp(36px,6vw,60px);
       font-weight:800;letter-spacing:-2px;line-height:1;margin-bottom:14px;
       background:linear-gradient(135deg,#fff 0%,#00e5ff 50%,#00ff9d 100%);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
       background-clip:text;">
    TruthLens
  </div>
  <div style="color:#5a6e9a;font-size:16px;font-weight:300;max-width:460px;margin:0 auto;">
    Detect misinformation with machine-learning precision
  </div>
</div>
<style>
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.15;} }
</style>
""", unsafe_allow_html=True)


# ================================================
# PAGE: ANALYZE NEWS
# ================================================
if page == "Analyze News":

    # STATS ROW
    if model_ready:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(neon_metric("Model Accuracy", f"{accuracy*100:.2f}", "#00e5ff", "%"), unsafe_allow_html=True)
        with c2:
            st.markdown(neon_metric("Algorithm", "LogReg", "#a855f7"), unsafe_allow_html=True)
        with c3:
            st.markdown(neon_metric("Features", "60K", "#00ff9d"), unsafe_allow_html=True)
        with c4:
            st.markdown(neon_metric("N-Grams", "1–2", "#ffb300"), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

    # INPUT CARD
    st.markdown("""
    <div style="background:#090e1d;border:1px solid #1a2540;border-radius:20px;
         padding:32px;position:relative;overflow:hidden;margin-bottom:24px;">
      <div style="position:absolute;top:0;left:0;right:0;height:1px;
           background:linear-gradient(90deg,transparent,#00e5ff,transparent);opacity:0.5;"></div>
      <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
           color:#5a6e9a;margin-bottom:12px;">Paste News Article</div>
    </div>
    """, unsafe_allow_html=True)

    # Keep card open with the textarea inside using st components
    news_text = st.text_area(
        "",
        height=200,
        placeholder="Paste a news headline, paragraph, or full article here...",
        label_visibility="collapsed",
        key="news_input"
    )

    char_count = len(news_text)
    word_count = len(news_text.split()) if news_text.strip() else 0

    col_info, col_btns = st.columns([3, 2])
    with col_info:
        st.markdown(
            f"<div style='color:#5a6e9a;font-size:13px;padding-top:8px;'>"
            f"<span style='color:#00e5ff;'>{word_count}</span> words &nbsp;·&nbsp; "
            f"<span style='color:#00e5ff;'>{char_count}</span> characters</div>",
            unsafe_allow_html=True
        )

    with col_btns:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            analyze_clicked = st.button("🔍 Analyze", use_container_width=True)
        with btn_col2:
            with st.container():
                st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
                clear_clicked = st.button("✕ Clear", use_container_width=True, key="clear_btn")
                st.markdown('</div>', unsafe_allow_html=True)

    if clear_clicked:
        st.rerun()

    # SAMPLE ARTICLES
    with st.expander("📋 Try Sample Articles"):
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            st.markdown("<div style='color:#00ff9d;font-size:12px;font-weight:600;margin-bottom:6px;'>✓ REAL NEWS SAMPLE</div>", unsafe_allow_html=True)
            st.markdown("""<div style='background:#0e1428;border:1px solid #1a2540;border-radius:10px;padding:14px;font-size:13px;color:#8a9fc0;line-height:1.6;'>
            The Federal Reserve raised interest rates by 25 basis points on Wednesday, 
            marking the tenth consecutive increase as policymakers continue their effort 
            to bring inflation down from its four-decade high. The decision was unanimous 
            among the Federal Open Market Committee members.</div>""", unsafe_allow_html=True)
        with sample_col2:
            st.markdown("<div style='color:#ff3d6e;font-size:12px;font-weight:600;margin-bottom:6px;'>✗ FAKE NEWS SAMPLE</div>", unsafe_allow_html=True)
            st.markdown("""<div style='background:#0e1428;border:1px solid #1a2540;border-radius:10px;padding:14px;font-size:13px;color:#8a9fc0;line-height:1.6;'>
            SHOCKING: Scientists CONFIRM that drinking bleach cures all known diseases! 
            The mainstream media is HIDING this miracle cure! Government is suppressing 
            this information to keep big pharma profits flowing! SHARE before they 
            DELETE THIS!!!</div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ANALYSIS RESULT
    if analyze_clicked:
        if not news_text.strip():
            st.markdown("""
            <div style="background:rgba(255,179,0,0.08);border:1px solid rgba(255,179,0,0.3);
                 border-radius:14px;padding:18px 24px;color:#ffb300;text-align:center;">
              ⚠️ &nbsp; Please enter a news article to analyze.
            </div>
            """, unsafe_allow_html=True)

        elif not model_ready:
            st.markdown("""
            <div style="background:rgba(255,61,110,0.08);border:1px solid rgba(255,61,110,0.3);
                 border-radius:14px;padding:18px 24px;color:#ff3d6e;text-align:center;">
              ❌ &nbsp; Model not loaded. Please ensure Fake.csv and True.csv are in the project folder.
            </div>
            """, unsafe_allow_html=True)

        else:
            with st.spinner(""):
                # Loading animation
                progress_bar = st.progress(0)
                status_text  = st.empty()
                steps = [
                    (15, "Preprocessing text..."),
                    (35, "Vectorizing with TF-IDF..."),
                    (60, "Running model inference..."),
                    (85, "Computing probabilities..."),
                    (100, "Analysis complete ✓")
                ]
                for pct, msg in steps:
                    progress_bar.progress(pct)
                    status_text.markdown(f"<div style='color:#5a6e9a;font-size:13px;text-align:center;'>{msg}</div>", unsafe_allow_html=True)
                    time.sleep(0.18)

                progress_bar.empty()
                status_text.empty()

                pred, real_prob, fake_prob = predict(news_text, model, vectorizer)

            st.markdown("<br/>", unsafe_allow_html=True)

            # RESULT CARD
            r_col, l_col = st.columns([1.2, 1])

            with r_col:
                if pred == 1:
                    st.markdown(
                        result_card("REAL", "✅", "#00ff9d", real_prob, "#00ff9d"),
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        result_card("FAKE", "🚨", "#ff3d6e", fake_prob, "#ff3d6e"),
                        unsafe_allow_html=True
                    )

            with l_col:
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

                # Probability donut chart
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
                    text=f"{'REAL' if pred == 1 else 'FAKE'}",
                    x=0.5, y=0.52, font=dict(size=20, family="Syne", color="#00e5ff"),
                    showarrow=False
                )
                fig.add_annotation(
                    text=f"{real_prob if pred==1 else fake_prob:.0f}%",
                    x=0.5, y=0.38, font=dict(size=14, family="DM Sans", color="#5a6e9a"),
                    showarrow=False
                )
                fig.update_layout(
                    **PLOT_LAYOUT,
                    showlegend=True,
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05,
                                font=dict(size=12, color="#dce8ff")),
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=40)
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # PROBABILITY BREAKDOWN
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                 color:#5a6e9a;margin-bottom:14px;">Probability Breakdown</div>
            """, unsafe_allow_html=True)

            pb_c1, pb_c2 = st.columns(2)
            with pb_c1:
                st.markdown(f"""
                <div style="background:#090e1d;border:1px solid #1a2540;border-radius:14px;padding:18px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <span style="color:#5a6e9a;font-size:13px;">Real News</span>
                    <span style="color:#00ff9d;font-weight:600;">{real_prob:.1f}%</span>
                  </div>
                  <div style="background:#0e1428;border-radius:100px;height:6px;">
                    <div style="width:{real_prob}%;height:100%;background:linear-gradient(90deg,#00ff9d,#00e5ff);
                         border-radius:100px;box-shadow:0 0 8px rgba(0,255,157,0.5);"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with pb_c2:
                st.markdown(f"""
                <div style="background:#090e1d;border:1px solid #1a2540;border-radius:14px;padding:18px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <span style="color:#5a6e9a;font-size:13px;">Fake News</span>
                    <span style="color:#ff3d6e;font-weight:600;">{fake_prob:.1f}%</span>
                  </div>
                  <div style="background:#0e1428;border-radius:100px;height:6px;">
                    <div style="width:{fake_prob}%;height:100%;background:linear-gradient(90deg,#ff3d6e,#ff6b35);
                         border-radius:100px;box-shadow:0 0 8px rgba(255,61,110,0.5);"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # FEEDBACK
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                 color:#5a6e9a;margin-bottom:12px;">Was this prediction correct?</div>
            """, unsafe_allow_html=True)
            fb1, fb2, fb3 = st.columns([1, 1, 3])
            with fb1:
                if st.button("👍  Yes, correct"):
                    st.markdown("""<div style="color:#00ff9d;font-size:13px;padding-top:4px;">
                    Thank you for the feedback! ✓</div>""", unsafe_allow_html=True)
            with fb2:
                if st.button("👎  No, wrong"):
                    st.markdown("""<div style="color:#ff3d6e;font-size:13px;padding-top:4px;">
                    Thanks! We'll improve the model.</div>""", unsafe_allow_html=True)


# ================================================
# PAGE: MODEL STATS
# ================================================
elif page == "Model Stats":
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:700;
         margin-bottom:8px;color:#dce8ff;">Model Performance</div>
    <div style="color:#5a6e9a;font-size:14px;margin-bottom:32px;">
        Detailed metrics from validation set evaluation
    </div>
    """, unsafe_allow_html=True)

    if not model_ready:
        st.markdown("""
        <div style="background:rgba(255,179,0,0.08);border:1px solid rgba(255,179,0,0.3);
             border-radius:14px;padding:24px;text-align:center;color:#ffb300;">
          ⚠️ Model not trained yet. Add Fake.csv and True.csv to the project folder and restart.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Accuracy", f"{accuracy*100:.2f}%")
        with m2:
            prec = report["weighted avg"]["precision"] * 100 if report else 0
            st.metric("Precision", f"{prec:.2f}%")
        with m3:
            rec = report["weighted avg"]["recall"] * 100 if report else 0
            st.metric("Recall", f"{rec:.2f}%")
        with m4:
            f1 = report["weighted avg"]["f1-score"] * 100 if report else 0
            st.metric("F1-Score", f"{f1:.2f}%")

        st.markdown("<br/>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Metrics Chart", "🔲 Confusion Matrix", "⚙️ Model Config"])

        with tab1:
            if report:
                classes = ["Fake (0)", "Real (1)"]
                metrics = ["precision", "recall", "f1-score"]
                vals = {
                    m: [report["0"][m]*100, report["1"][m]*100]
                    for m in metrics
                }
                fig = go.Figure()
                colors = ["#ff3d6e", "#00ff9d", "#00e5ff"]
                for i, metric in enumerate(metrics):
                    fig.add_trace(go.Bar(
                        name=metric.capitalize(),
                        x=classes,
                        y=vals[metric],
                        marker_color=colors[i],
                        marker_line_color="rgba(0,0,0,0)",
                        opacity=0.85
                    ))
                fig.update_layout(
                    **PLOT_LAYOUT,
                    barmode="group",
                    xaxis=dict(gridcolor="#1a2540"),
                    yaxis=dict(gridcolor="#1a2540", range=[0, 105], ticksuffix="%"),
                    legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                    height=380,
                    title=dict(text="Per-Class Performance Metrics", font=dict(size=15, color="#dce8ff"))
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Retrain the model to view detailed metrics.")

        with tab2:
            if cm is not None:
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Fake", "Real"], y=["Fake", "Real"],
                    color_continuous_scale=[[0, "#090e1d"], [0.5, "#1e3a5f"], [1, "#00e5ff"]],
                    text_auto=True,
                )
                fig_cm.update_traces(textfont=dict(size=22, family="Syne", color="white"))
                fig_cm.update_layout(
                    **PLOT_LAYOUT,
                    height=380,
                    title=dict(text="Confusion Matrix", font=dict(size=15, color="#dce8ff")),
                    coloraxis_showscale=False,
                    xaxis=dict(side="bottom"),
                )
                st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Retrain the model to view confusion matrix.")

        with tab3:
            config_data = {
                "Parameter": ["Algorithm", "Regularization (C)", "Max Iterations", "Vectorizer",
                               "N-Gram Range", "Max Features", "Min Doc Freq", "Max Doc Freq",
                               "TF-IDF Sublinear", "Test Split"],
                "Value": ["Logistic Regression", "5.0", "1000", "TF-IDF",
                           "(1, 2)", "60,000", "3", "70%", "Yes", "20%"]
            }
            df_config = pd.DataFrame(config_data)

            # Custom table
            rows_html = ""
            for _, row in df_config.iterrows():
                rows_html += f"""
                <tr>
                  <td style="padding:12px 16px;color:#8a9fc0;border-bottom:1px solid #1a2540;">{row['Parameter']}</td>
                  <td style="padding:12px 16px;color:#00e5ff;font-family:'Syne',sans-serif;
                       font-weight:600;border-bottom:1px solid #1a2540;">{row['Value']}</td>
                </tr>
                """
            st.markdown(f"""
            <div style="background:#090e1d;border:1px solid #1a2540;border-radius:16px;overflow:hidden;">
              <table style="width:100%;border-collapse:collapse;">
                <thead>
                  <tr style="background:#0e1428;">
                    <th style="padding:14px 16px;text-align:left;color:#5a6e9a;
                         font-size:11px;letter-spacing:2px;text-transform:uppercase;
                         border-bottom:1px solid #1a2540;">Parameter</th>
                    <th style="padding:14px 16px;text-align:left;color:#5a6e9a;
                         font-size:11px;letter-spacing:2px;text-transform:uppercase;
                         border-bottom:1px solid #1a2540;">Value</th>
                  </tr>
                </thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)


# ================================================
# PAGE: ABOUT
# ================================================
elif page == "About":
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:700;
         margin-bottom:8px;color:#dce8ff;">About TruthLens</div>
    <div style="color:#5a6e9a;font-size:14px;margin-bottom:32px;">
        How this application works
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "#00e5ff", "Text Input", "User submits a news article, headline, or paragraph for analysis."),
        ("02", "#a855f7", "Preprocessing", "Text is lowercased, URLs and HTML tags removed, then cleaned to pure alphabetic tokens."),
        ("03", "#00ff9d", "TF-IDF Vectorization", "Cleaned text is converted to a 60,000-feature sparse vector using TF-IDF with bigrams, capturing word importance and co-occurrence patterns."),
        ("04", "#ffb300", "Model Inference", "Logistic Regression model computes a probability score for each class (Real vs. Fake)."),
        ("05", "#ff3d6e", "Result & Confidence", "The class with the higher probability is returned along with the confidence score."),
    ]

    for num, color, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex;gap:20px;margin-bottom:16px;
             background:#090e1d;border:1px solid #1a2540;border-radius:16px;padding:20px 24px;
             transition:all 0.3s;" onmouseover="this.style.borderColor='rgba(0,229,255,0.25)'"
             onmouseout="this.style.borderColor='#1a2540'">
          <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;
               color:{color};opacity:0.4;flex-shrink:0;min-width:40px;">{num}</div>
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                 color:{color};margin-bottom:6px;">{title}</div>
            <div style="color:#8a9fc0;font-size:14px;line-height:1.7;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.2);
         border-radius:16px;padding:24px;">
      <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
           color:#00e5ff;margin-bottom:12px;">⚠️ Disclaimer</div>
      <div style="color:#8a9fc0;font-size:14px;line-height:1.8;">
        TruthLens is an AI-assisted tool for educational purposes. 
        It should not be used as the sole source of truth for verifying news.
        Always cross-reference with trusted news sources and fact-checking organizations.
        The model may produce incorrect results on very short text or highly domain-specific content.
      </div>
    </div>
    """, unsafe_allow_html=True)
