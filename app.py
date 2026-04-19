import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

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
    color: #5a6e9a !important; font-size: 12px !important;
    text-transform: uppercase !important; letter-spacing: 1.5px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 26px !important;
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
    background: #0e1428 !important; border-radius: 100px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #090e1d !important; border-radius: 12px !important;
    padding: 4px !important; border: 1px solid #1a2540 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 8px !important;
    color: #5a6e9a !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.1) !important; color: #00e5ff !important;
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
        f"border:2px solid {color}55;border-radius:20px;padding:36px 28px;"
        f"text-align:center;--gc:{color}44;--gc2:{glow}22;"
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


def section_header(title, subtitle=""):
    sub = f"<div style='color:#5a6e9a;font-size:14px;margin-bottom:24px;'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div style='font-family:Syne,sans-serif;font-size:11px;letter-spacing:2px;"
        f"text-transform:uppercase;color:#5a6e9a;margin-bottom:6px;'>{title}</div>{sub}"
    )


# ================================================
# MODEL TRAINING  (saves ALL artifacts to disk)
# ================================================
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    MODEL_PATH  = "fake_news_model.pkl"
    VEC_PATH    = "vectorizer.pkl"
    ACC_PATH    = "accuracy.pkl"
    REPORT_PATH = "report.pkl"
    CM_PATH     = "cm.pkl"
    ROC_PATH    = "roc.pkl"
    LC_PATH     = "lc.pkl"

    # ── Load from disk if everything exists ──
    if all(os.path.exists(p) for p in [MODEL_PATH, VEC_PATH, ACC_PATH,
                                        REPORT_PATH, CM_PATH, ROC_PATH, LC_PATH]):
        return (
            joblib.load(MODEL_PATH),
            joblib.load(VEC_PATH),
            joblib.load(ACC_PATH),
            joblib.load(REPORT_PATH),
            joblib.load(CM_PATH),
            joblib.load(ROC_PATH),
            joblib.load(LC_PATH),
        )

    # ── Load CSVs ──
    for paths in [("Fake.csv.zip", "True.csv.zip"), ("Fake.csv", "True.csv")]:
        try:
            fake = pd.read_csv(paths[0])
            true = pd.read_csv(paths[1])
            break
        except FileNotFoundError:
            continue
    else:
        return None, None, 0.0, None, None, None, None

    fake["label"] = 0
    true["label"] = 1
    data = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

    if "title" in data.columns:
        data["full_text"] = data["text"].fillna("") + " " + data["title"].fillna("")
    else:
        data["full_text"] = data["text"].fillna("")

    X_raw = data["full_text"].apply(clean_text)
    y     = data["label"]

    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.7, min_df=3,
        ngram_range=(1, 2), max_features=60000, sublinear_tf=True
    )
    X_vec = vectorizer.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    rep      = classification_report(y_test, y_pred, output_dict=True)
    cm_mat   = confusion_matrix(y_test, y_pred)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc      = float(auc(fpr, tpr))
    roc_data     = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

    # Learning curve (subsample for speed)
    train_sizes = np.linspace(0.1, 1.0, 8)
    lc_sizes, lc_train, lc_val = learning_curve(
        LogisticRegression(C=5.0, max_iter=500, solver="lbfgs"),
        X_vec, y, cv=3, train_sizes=train_sizes,
        scoring="accuracy", n_jobs=-1
    )
    lc_data = {
        "sizes":      lc_sizes.tolist(),
        "train_mean": lc_train.mean(axis=1).tolist(),
        "train_std":  lc_train.std(axis=1).tolist(),
        "val_mean":   lc_val.mean(axis=1).tolist(),
        "val_std":    lc_val.std(axis=1).tolist(),
    }

    # Save everything
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(accuracy,   ACC_PATH)
    joblib.dump(rep,        REPORT_PATH)
    joblib.dump(cm_mat,     CM_PATH)
    joblib.dump(roc_data,   ROC_PATH)
    joblib.dump(lc_data,    LC_PATH)

    return model, vectorizer, accuracy, rep, cm_mat, roc_data, lc_data


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    pred    = int(model.predict(vec)[0])
    proba   = model.predict_proba(vec)[0]
    return pred, float(proba[1] * 100), float(proba[0] * 100)


# ── Plotly dark theme shortcut ──
def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,14,29,0.85)",
        font=dict(family="DM Sans", color="#dce8ff", size=12),
    )
    base.update(kwargs)
    return base


# ================================================
# LOAD MODEL
# ================================================
with st.spinner("🔄 Loading / training model — please wait..."):
    result = load_or_train_model()

model, vectorizer, accuracy, report, cm, roc_data, lc_data = result
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
        if roc_data:
            st.markdown(
                f"<div style='text-align:center;font-size:12px;color:#5a6e9a;margin-top:10px;'>"
                f"ROC-AUC Score<br/>"
                f"<span style='font-family:Syne,sans-serif;font-size:22px;font-weight:700;"
                f"color:#a855f7;'>{roc_data['auc']:.4f}</span></div>",
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


# ╔══════════════════════════════════════════════╗
# ║              PAGE: ANALYZE NEWS              ║
# ╚══════════════════════════════════════════════╝
if page == "Analyze News":

    if model_ready:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(neon_metric("Accuracy",  f"{accuracy*100:.2f}", "#00e5ff", "%"), unsafe_allow_html=True)
        with c2: st.markdown(neon_metric("Algorithm", "LogReg",   "#a855f7"), unsafe_allow_html=True)
        with c3: st.markdown(neon_metric("Features",  "60K",      "#00ff9d"), unsafe_allow_html=True)
        with c4:
            auc_val = f"{roc_data['auc']:.4f}" if roc_data else "N/A"
            st.markdown(neon_metric("ROC-AUC", auc_val, "#ffb300"), unsafe_allow_html=True)
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
        "", height=190,
        placeholder="Paste a news headline, paragraph, or full article here...",
        label_visibility="collapsed", key="news_input"
    )

    word_count = len(news_text.split()) if news_text.strip() else 0
    info_col, btn_col = st.columns([3, 2])
    with info_col:
        st.markdown(
            f"<div style='color:#5a6e9a;font-size:13px;padding-top:6px;'>"
            f"<span style='color:#00e5ff;'>{word_count}</span> words &nbsp;·&nbsp; "
            f"<span style='color:#00e5ff;'>{len(news_text)}</span> chars</div>",
            unsafe_allow_html=True
        )
    with btn_col:
        b1, b2 = st.columns(2)
        with b1: analyze_clicked = st.button("🔍 Analyze", use_container_width=True)
        with b2: clear_clicked   = st.button("✕ Clear",   use_container_width=True)

    if clear_clicked:
        st.rerun()

    with st.expander("📋 Try Sample Articles"):
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                "<div style='color:#00ff9d;font-size:12px;font-weight:600;margin-bottom:6px;'>"
                "✓ REAL NEWS SAMPLE</div>"
                "<div style='background:#0e1428;border:1px solid #1a2540;border-radius:10px;"
                "padding:14px;font-size:13px;color:#8a9fc0;line-height:1.65;'>"
                "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
                "marking the tenth consecutive increase as policymakers continue their effort "
                "to bring inflation down from its four-decade high. The decision was unanimous "
                "among the Federal Open Market Committee members.</div>",
                unsafe_allow_html=True
            )
        with s2:
            st.markdown(
                "<div style='color:#ff3d6e;font-size:12px;font-weight:600;margin-bottom:6px;'>"
                "✗ FAKE NEWS SAMPLE</div>"
                "<div style='background:#0e1428;border:1px solid #1a2540;border-radius:10px;"
                "padding:14px;font-size:13px;color:#8a9fc0;line-height:1.65;'>"
                "SHOCKING: Scientists CONFIRM that drinking bleach cures all diseases! "
                "The mainstream media is HIDING this miracle cure! Government is suppressing "
                "this to keep big pharma profits flowing! SHARE before they DELETE THIS!!!</div>",
                unsafe_allow_html=True
            )

    st.markdown("<br/>", unsafe_allow_html=True)

    if analyze_clicked:
        if not news_text.strip():
            st.markdown(
                "<div style='background:rgba(255,179,0,0.07);border:1px solid rgba(255,179,0,0.3);"
                "border-radius:14px;padding:18px 24px;color:#ffb300;text-align:center;'>"
                "⚠️ Please enter a news article to analyze.</div>",
                unsafe_allow_html=True
            )
        elif not model_ready:
            st.markdown(
                "<div style='background:rgba(255,61,110,0.07);border:1px solid rgba(255,61,110,0.3);"
                "border-radius:14px;padding:18px 24px;color:#ff3d6e;text-align:center;'>"
                "❌ Model not loaded. Ensure Fake.csv &amp; True.csv are in the project folder.</div>",
                unsafe_allow_html=True
            )
        else:
            prog = st.progress(0)
            stat = st.empty()
            for pct, msg in [(15,"Preprocessing text…"),(40,"TF-IDF vectorizing…"),
                             (65,"Running inference…"),(90,"Computing probabilities…"),(100,"Done ✓")]:
                prog.progress(pct)
                stat.markdown(
                    f"<div style='color:#5a6e9a;font-size:13px;text-align:center;'>{msg}</div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.15)
            prog.empty(); stat.empty()

            pred, real_prob, fake_prob = predict(news_text, model, vectorizer)

            st.markdown("<br/>", unsafe_allow_html=True)
            left, right = st.columns([1.15, 1])
            with left:
                if pred == 1:
                    st.markdown(result_card("REAL","✅","#00ff9d", real_prob,"#00e5ff"), unsafe_allow_html=True)
                else:
                    st.markdown(result_card("FAKE","🚨","#ff3d6e", fake_prob,"#ff6b35"), unsafe_allow_html=True)

            with right:
                fig = go.Figure(go.Pie(
                    values=[fake_prob, real_prob], labels=["Fake","Real"], hole=0.65,
                    marker=dict(colors=["#ff3d6e","#00ff9d"], line=dict(color="#04060f", width=3)),
                    textinfo="percent",
                    textfont=dict(size=13, family="DM Sans", color="white"),
                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
                ))
                fig.add_annotation(text="REAL" if pred==1 else "FAKE",
                                   x=0.5, y=0.54, showarrow=False,
                                   font=dict(size=19, family="Syne", color="#00e5ff"))
                fig.add_annotation(text=f"{real_prob if pred==1 else fake_prob:.0f}%",
                                   x=0.5, y=0.38, showarrow=False,
                                   font=dict(size=13, family="DM Sans", color="#5a6e9a"))
                fig.update_layout(**dark_layout(
                    margin=dict(l=10,r=10,t=10,b=40), height=258,
                    showlegend=True,
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05,
                                font=dict(size=12, color="#dce8ff"))
                ))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:11px;letter-spacing:2px;text-transform:uppercase;"
                "color:#5a6e9a;margin-bottom:12px;'>Probability Breakdown</div>",
                unsafe_allow_html=True
            )
            pb1, pb2 = st.columns(2)
            with pb1: st.markdown(prob_bar("Real News", real_prob, "#00ff9d"), unsafe_allow_html=True)
            with pb2: st.markdown(prob_bar("Fake News", fake_prob, "#ff3d6e"), unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:11px;letter-spacing:2px;text-transform:uppercase;"
                "color:#5a6e9a;margin-bottom:10px;'>Was this prediction correct?</div>",
                unsafe_allow_html=True
            )
            fb1, fb2, _ = st.columns([1,1,3])
            with fb1:
                if st.button("👍  Yes"):
                    st.markdown("<div style='color:#00ff9d;font-size:13px;'>Thank you! ✓</div>", unsafe_allow_html=True)
            with fb2:
                if st.button("👎  No"):
                    st.markdown("<div style='color:#ff3d6e;font-size:13px;'>Thanks for the feedback!</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════╗
# ║             PAGE: MODEL STATS                ║
# ╚══════════════════════════════════════════════╝
elif page == "Model Stats":
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:28px;font-weight:700;"
        "margin-bottom:6px;color:#dce8ff;'>Model Performance</div>"
        "<div style='color:#5a6e9a;font-size:14px;margin-bottom:24px;'>"
        "Complete evaluation on 20% held-out test set</div>",
        unsafe_allow_html=True
    )

    if not model_ready:
        st.markdown(
            "<div style='background:rgba(255,179,0,0.07);border:1px solid rgba(255,179,0,0.3);"
            "border-radius:14px;padding:24px;text-align:center;color:#ffb300;'>"
            "⚠️ Model not trained yet. Add Fake.csv and True.csv and restart.</div>",
            unsafe_allow_html=True
        )
        st.stop()

    # ── Top KPI strip ──
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Accuracy",  f"{accuracy*100:.2f}%")
    with m2:
        prec = report["weighted avg"]["precision"]*100 if report else 0
        st.metric("Precision", f"{prec:.2f}%")
    with m3:
        rec = report["weighted avg"]["recall"]*100 if report else 0
        st.metric("Recall", f"{rec:.2f}%")
    with m4:
        f1 = report["weighted avg"]["f1-score"]*100 if report else 0
        st.metric("F1-Score", f"{f1:.2f}%")
    with m5:
        auc_val = roc_data["auc"] if roc_data else 0
        st.metric("ROC-AUC", f"{auc_val:.4f}")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── 6-Tab layout ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Class Metrics",
        "🔲 Confusion Matrix",
        "📈 ROC Curve",
        "🎓 Learning Curve",
        "🥧 Class Distribution",
        "⚙️ Model Config",
    ])

    # ── TAB 1: Per-class bar chart ──
    with tab1:
        if report:
            st.markdown(section_header("Per-Class Performance"), unsafe_allow_html=True)
            fig_bar = go.Figure()
            for metric, color in [("precision","#ff3d6e"),("recall","#00ff9d"),("f1-score","#00e5ff")]:
                fig_bar.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=["Fake (0)", "Real (1)"],
                    y=[round(report["0"][metric]*100,2), round(report["1"][metric]*100,2)],
                    marker_color=color, marker_line_color="rgba(0,0,0,0)", opacity=0.88,
                    text=[f"{report['0'][metric]*100:.1f}%", f"{report['1'][metric]*100:.1f}%"],
                    textposition="outside",
                    textfont=dict(size=13, color="#dce8ff"),
                ))
            fig_bar.update_layout(**dark_layout(
                barmode="group",
                margin=dict(l=20,r=20,t=50,b=20),
                xaxis=dict(gridcolor="#1a2540"),
                yaxis=dict(gridcolor="#1a2540", range=[0,115], ticksuffix="%"),
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                height=400,
                title=dict(text="Precision · Recall · F1-Score by Class",
                           font=dict(size=15, color="#dce8ff"))
            ))
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # Support counts
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(section_header("Support (sample count per class)"), unsafe_allow_html=True)
            fake_sup = int(report["0"]["support"])
            real_sup = int(report["1"]["support"])
            total    = fake_sup + real_sup
            sa, sb, sc = st.columns(3)
            with sa: st.markdown(neon_metric("Fake Samples", f"{fake_sup:,}", "#ff3d6e"), unsafe_allow_html=True)
            with sb: st.markdown(neon_metric("Real Samples", f"{real_sup:,}", "#00ff9d"), unsafe_allow_html=True)
            with sc: st.markdown(neon_metric("Total Test",   f"{total:,}",    "#00e5ff"), unsafe_allow_html=True)
        else:
            st.info("Retrain model to view metrics.")

    # ── TAB 2: Confusion Matrix ──
    with tab2:
        if cm is not None:
            st.markdown(section_header("Confusion Matrix"), unsafe_allow_html=True)

            # Normalised + raw side-by-side
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(
                    "<div style='font-size:12px;color:#5a6e9a;text-align:center;"
                    "margin-bottom:6px;'>Raw Counts</div>",
                    unsafe_allow_html=True
                )
                fig_cm_raw = px.imshow(
                    cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Fake","Real"], y=["Fake","Real"],
                    color_continuous_scale=[[0,"#090e1d"],[0.5,"#1e3a5f"],[1,"#00e5ff"]],
                    text_auto=True,
                )
                fig_cm_raw.update_traces(textfont=dict(size=20, family="Syne", color="white"))
                fig_cm_raw.update_layout(**dark_layout(
                    margin=dict(l=20,r=20,t=20,b=20), height=320,
                    coloraxis_showscale=False,
                ))
                st.plotly_chart(fig_cm_raw, use_container_width=True, config={"displayModeBar": False})

            with col_b:
                st.markdown(
                    "<div style='font-size:12px;color:#5a6e9a;text-align:center;"
                    "margin-bottom:6px;'>Normalised (%)</div>",
                    unsafe_allow_html=True
                )
                cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
                fig_cm_norm = px.imshow(
                    cm_norm, labels=dict(x="Predicted", y="Actual", color="%"),
                    x=["Fake","Real"], y=["Fake","Real"],
                    color_continuous_scale=[[0,"#090e1d"],[0.5,"#1a3a20"],[1,"#00ff9d"]],
                    text_auto=".1f",
                )
                fig_cm_norm.update_traces(textfont=dict(size=20, family="Syne", color="white"))
                fig_cm_norm.update_layout(**dark_layout(
                    margin=dict(l=20,r=20,t=20,b=20), height=320,
                    coloraxis_showscale=False,
                ))
                st.plotly_chart(fig_cm_norm, use_container_width=True, config={"displayModeBar": False})

            # Interpretation cards
            tn, fp, fn, tp = cm.ravel()
            st.markdown("<br/>", unsafe_allow_html=True)
            i1, i2, i3, i4 = st.columns(4)
            with i1: st.markdown(neon_metric("True Negatives",  f"{tn:,}", "#00ff9d"), unsafe_allow_html=True)
            with i2: st.markdown(neon_metric("False Positives", f"{fp:,}", "#ffb300"), unsafe_allow_html=True)
            with i3: st.markdown(neon_metric("False Negatives", f"{fn:,}", "#ff3d6e"), unsafe_allow_html=True)
            with i4: st.markdown(neon_metric("True Positives",  f"{tp:,}", "#00e5ff"), unsafe_allow_html=True)
        else:
            st.info("Retrain model to view confusion matrix.")

    # ── TAB 3: ROC Curve ──
    with tab3:
        if roc_data:
            st.markdown(section_header("ROC Curve — Receiver Operating Characteristic"), unsafe_allow_html=True)

            fpr_list = roc_data["fpr"]
            tpr_list = roc_data["tpr"]
            roc_auc  = roc_data["auc"]

            fig_roc = go.Figure()
            # Diagonal baseline
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(color="#1a2540", width=1.5, dash="dash"),
                name="Random Classifier",
                hoverinfo="skip"
            ))
            # ROC curve with gradient fill
            fig_roc.add_trace(go.Scatter(
                x=fpr_list, y=tpr_list, mode="lines",
                line=dict(color="#00e5ff", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,229,255,0.08)",
                name=f"TruthLens (AUC = {roc_auc:.4f})",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
            ))
            fig_roc.update_layout(**dark_layout(
                margin=dict(l=30,r=20,t=60,b=30),
                height=420,
                xaxis=dict(title="False Positive Rate", gridcolor="#1a2540", range=[0,1]),
                yaxis=dict(title="True Positive Rate",  gridcolor="#1a2540", range=[0,1.02]),
                legend=dict(x=0.55, y=0.05, bgcolor="rgba(0,0,0,0)",
                            font=dict(size=13, color="#dce8ff")),
                title=dict(text=f"ROC Curve  ·  AUC = {roc_auc:.4f}",
                           font=dict(size=15, color="#dce8ff"))
            ))
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

            ra, rb = st.columns(2)
            with ra: st.markdown(neon_metric("AUC Score", f"{roc_auc:.4f}", "#a855f7"), unsafe_allow_html=True)
            with rb: st.markdown(neon_metric("Interpretation", "Excellent" if roc_auc > 0.97 else "Good", "#00ff9d"), unsafe_allow_html=True)
        else:
            st.info("Retrain model to generate ROC curve.")

    # ── TAB 4: Learning Curve ──
    with tab4:
        if lc_data:
            st.markdown(section_header("Learning Curve — Bias vs Variance Analysis"), unsafe_allow_html=True)

            sizes      = lc_data["sizes"]
            train_mean = lc_data["train_mean"]
            train_std  = lc_data["train_std"]
            val_mean   = lc_data["val_mean"]
            val_std    = lc_data["val_std"]

            fig_lc = go.Figure()

            # Train std band
            fig_lc.add_trace(go.Scatter(
                x=sizes + sizes[::-1],
                y=[m+s for m,s in zip(train_mean, train_std)] +
                  [m-s for m,s in zip(train_mean[::-1], train_std[::-1])],
                fill="toself", fillcolor="rgba(0,229,255,0.07)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"
            ))
            # Val std band
            fig_lc.add_trace(go.Scatter(
                x=sizes + sizes[::-1],
                y=[m+s for m,s in zip(val_mean, val_std)] +
                  [m-s for m,s in zip(val_mean[::-1], val_std[::-1])],
                fill="toself", fillcolor="rgba(0,255,157,0.07)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"
            ))
            # Train line
            fig_lc.add_trace(go.Scatter(
                x=sizes, y=[round(v*100,2) for v in train_mean],
                mode="lines+markers", name="Training Score",
                line=dict(color="#00e5ff", width=2.5),
                marker=dict(size=7, color="#00e5ff"),
                hovertemplate="Samples: %{x}<br>Accuracy: %{y:.2f}%<extra>Train</extra>"
            ))
            # Val line
            fig_lc.add_trace(go.Scatter(
                x=sizes, y=[round(v*100,2) for v in val_mean],
                mode="lines+markers", name="Validation Score",
                line=dict(color="#00ff9d", width=2.5),
                marker=dict(size=7, color="#00ff9d"),
                hovertemplate="Samples: %{x}<br>Accuracy: %{y:.2f}%<extra>Val</extra>"
            ))
            fig_lc.update_layout(**dark_layout(
                margin=dict(l=30,r=20,t=60,b=30),
                height=420,
                xaxis=dict(title="Training Samples", gridcolor="#1a2540"),
                yaxis=dict(title="Accuracy (%)", gridcolor="#1a2540"),
                legend=dict(x=0.65, y=0.05, bgcolor="rgba(0,0,0,0)",
                            font=dict(size=13, color="#dce8ff")),
                title=dict(text="Learning Curve  ·  Training vs Validation Accuracy",
                           font=dict(size=15, color="#dce8ff"))
            ))
            st.plotly_chart(fig_lc, use_container_width=True, config={"displayModeBar": False})

            gap = abs(train_mean[-1] - val_mean[-1]) * 100
            verdict = "Low overfitting ✓" if gap < 3 else ("Moderate overfitting" if gap < 8 else "High overfitting ✗")
            st.markdown(
                f"<div style='background:#090e1d;border:1px solid #1a2540;border-radius:14px;"
                f"padding:16px 20px;font-size:14px;color:#8a9fc0;'>"
                f"Train–Val gap: <span style='color:#00e5ff;font-weight:600;'>{gap:.2f}%</span> &nbsp;·&nbsp; "
                f"Verdict: <span style='color:#00ff9d;font-weight:600;'>{verdict}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("Retrain model to generate learning curve.")

    # ── TAB 5: Class Distribution ──
    with tab5:
        if report:
            st.markdown(section_header("Class Balance in Test Set"), unsafe_allow_html=True)

            fake_sup = int(report["0"]["support"])
            real_sup = int(report["1"]["support"])
            total    = fake_sup + real_sup

            col_pie, col_radar = st.columns(2)

            # Pie chart
            with col_pie:
                fig_pie = go.Figure(go.Pie(
                    values=[fake_sup, real_sup],
                    labels=["Fake News", "Real News"],
                    hole=0.5,
                    marker=dict(colors=["#ff3d6e","#00ff9d"],
                                line=dict(color="#04060f", width=3)),
                    textinfo="percent+label",
                    textfont=dict(size=13, family="DM Sans", color="white"),
                    hovertemplate="%{label}: %{value:,} samples<extra></extra>"
                ))
                fig_pie.add_annotation(
                    text=f"{total:,}<br><span style='font-size:11px;'>total</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=18, family="Syne", color="#dce8ff")
                )
                fig_pie.update_layout(**dark_layout(
                    margin=dict(l=10,r=10,t=30,b=10), height=320,
                    showlegend=False,
                    title=dict(text="Test Set Distribution", font=dict(size=14, color="#dce8ff"))
                ))
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            # Radar chart of all metrics
            with col_radar:
                cats = ["Precision","Recall","F1-Score","Accuracy","AUC"]
                fake_vals = [
                    report["0"]["precision"]*100,
                    report["0"]["recall"]*100,
                    report["0"]["f1-score"]*100,
                    accuracy*100,
                    roc_data["auc"]*100 if roc_data else 0,
                ]
                real_vals = [
                    report["1"]["precision"]*100,
                    report["1"]["recall"]*100,
                    report["1"]["f1-score"]*100,
                    accuracy*100,
                    roc_data["auc"]*100 if roc_data else 0,
                ]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=fake_vals + [fake_vals[0]], theta=cats + [cats[0]],
                    fill="toself", fillcolor="rgba(255,61,110,0.12)",
                    line=dict(color="#ff3d6e", width=2), name="Fake (0)"
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=real_vals + [real_vals[0]], theta=cats + [cats[0]],
                    fill="toself", fillcolor="rgba(0,255,157,0.12)",
                    line=dict(color="#00ff9d", width=2), name="Real (1)"
                ))
                fig_radar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    polar=dict(
                        bgcolor="rgba(9,14,29,0.85)",
                        radialaxis=dict(visible=True, range=[0,105],
                                        gridcolor="#1a2540", tickcolor="#5a6e9a",
                                        tickfont=dict(color="#5a6e9a", size=10)),
                        angularaxis=dict(gridcolor="#1a2540",
                                         tickfont=dict(color="#dce8ff", size=12))
                    ),
                    showlegend=True,
                    legend=dict(x=0.5, xanchor="center", y=-0.12,
                                orientation="h", font=dict(color="#dce8ff", size=12)),
                    margin=dict(l=40,r=40,t=40,b=40),
                    height=320,
                    font=dict(family="DM Sans", color="#dce8ff"),
                    title=dict(text="Metric Radar Chart", font=dict(size=14, color="#dce8ff"))
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Retrain model to view class distribution.")

    # ── TAB 6: Model Config ──
    with tab6:
        st.markdown(section_header("Hyperparameters & Configuration"), unsafe_allow_html=True)
        params = [
            ("Algorithm",          "Logistic Regression"),
            ("Regularization (C)", "5.0"),
            ("Max Iterations",     "1000"),
            ("Solver",             "lbfgs"),
            ("Vectorizer",         "TF-IDF"),
            ("N-Gram Range",       "(1, 2)"),
            ("Max Features",       "60,000"),
            ("Min Doc Frequency",  "3"),
            ("Max Doc Frequency",  "70%"),
            ("TF-IDF Sublinear",   "Yes"),
            ("Test Split",         "20%"),
            ("Stratified Split",   "Yes"),
            ("Random State",       "42"),
        ]
        rows = "".join(
            f"<tr>"
            f"<td style='padding:12px 16px;color:#8a9fc0;border-bottom:1px solid #1a2540;'>{p}</td>"
            f"<td style='padding:12px 16px;color:#00e5ff;font-family:Syne,sans-serif;"
            f"font-weight:600;border-bottom:1px solid #1a2540;'>{v}</td>"
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


# ╔══════════════════════════════════════════════╗
# ║              PAGE: ABOUT                     ║
# ╚══════════════════════════════════════════════╝
elif page == "About":
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:28px;font-weight:700;"
        "margin-bottom:6px;color:#dce8ff;'>About TruthLens</div>"
        "<div style='color:#5a6e9a;font-size:14px;margin-bottom:28px;'>"
        "Pipeline overview and methodology</div>",
        unsafe_allow_html=True
    )
    steps = [
        ("01","#00e5ff","Text Input",
         "User submits a news article, headline, or paragraph for analysis."),
        ("02","#a855f7","Preprocessing",
         "Text is lowercased; URLs, HTML tags, numbers and punctuation are stripped, leaving clean alphabetic tokens."),
        ("03","#00ff9d","TF-IDF Vectorization",
         "Cleaned text is converted to a 60,000-feature sparse vector using unigrams + bigrams. sublinear_tf=True boosts rare informative terms."),
        ("04","#ffb300","Logistic Regression",
         "Model computes log-odds from TF-IDF features. C=5.0 with lbfgs solver balances regularization and convergence."),
        ("05","#ff3d6e","Result & Confidence",
         "predict_proba returns Real/Fake probabilities. The higher class is returned with its full confidence score and visualizations."),
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
