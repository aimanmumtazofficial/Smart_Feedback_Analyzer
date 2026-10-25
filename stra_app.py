"""
================================================================================
 CUSTOMER FEEDBACK INTELLIGENCE DASHBOARD
 A professional, industry-grade Sentiment Analysis platform built on top of
 an Amazon Reviews dataset, TF-IDF feature extraction and a trained
 Logistic Regression / Random Forest classifier.

 Folder structure expected (relative to this file):
   data/
     quality_summary.json
     reviews_clean.csv
   model/
     sentiment_model.pkl
     tfidf_vectorizer.pkl
     model_metadata.json
     model_comparison.csv
     classification_report.csv
     confusion_matrix.csv
     top_features.csv

 If a file is missing, the relevant section shows a friendly placeholder
 instead of crashing, so the app always runs even with a partial backend.
================================================================================
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Feedback Intelligence Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Neutral": "#f1c40f",
    "Negative": "#e74c3c",
}

# ------------------------------------------------------------------------------
# GLOBAL STYLE — dark navy sidebar + pink/purple/blue gradient theme
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* ---- App background ---- */
    .stApp {
        background: linear-gradient(135deg, #f5c6e8 0%, #b892e8 35%, #7fb3f0 100%);
        background-attachment: fixed;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1225 0%, #1b1f3b 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #eef0ff !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }

    /* ---- Headings ---- */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        color: #1b1f3b;
    }

    /* ---- Glass cards ---- */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin-bottom: 18px;
    }

    /* ---- KPI metric cards ---- */
    .kpi-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(31, 38, 135, 0.12);
        border: 1px solid rgba(255,255,255,0.5);
    }
    .kpi-value {
        font-family: 'Poppins', sans-serif;
        font-size: 34px;
        font-weight: 800;
        background: linear-gradient(90deg, #d946ef, #6366f1, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* ---- Section badge ---- */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    /* ---- Header banner ---- */
    .hero {
        background: linear-gradient(120deg, #ec4899, #8b5cf6, #3b82f6);
        border-radius: 20px;
        padding: 34px 38px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.35);
    }
    .hero h1 { color: white !important; margin-bottom: 4px;}
    .hero p { color: rgba(255,255,255,0.9); font-size: 15px; margin: 0;}

    /* ---- Buttons ---- */
    .stButton>button {
        background: linear-gradient(90deg, #ec4899, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(139, 92, 246, 0.4);
    }

    /* ---- Dataframe container ---- */
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.85);
        border-radius: 14px;
        padding: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# DATA / MODEL LOADERS (cached, fail-safe)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(model_path) and os.path.exists(vec_path):
        return joblib.load(model_path), joblib.load(vec_path)
    return None, None


quality_summary = load_json(os.path.join(DATA_DIR, "quality_summary.json"))
reviews_df = load_csv(os.path.join(DATA_DIR, "reviews_clean.csv"))
model_metadata = load_json(os.path.join(MODEL_DIR, "model_metadata.json"))
model_comparison = load_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))
classification_report_df = load_csv(os.path.join(MODEL_DIR, "classification_report.csv"))
confusion_matrix_df = load_csv(os.path.join(MODEL_DIR, "confusion_matrix.csv"))
top_features_df = load_csv(os.path.join(MODEL_DIR, "top_features.csv"))
model, tfidf = load_model()


def missing_file_notice(what):
    st.info(f"⚠️ **{what}** was not found in the `data/` or `model/` folder yet. "
            f"Run your training/export script to generate it, then refresh this page.")


def clean_text_basic(text):
    import re
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 Feedback Intelligence")
    st.caption("Sentiment Analytics Platform")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🔍 Data Quality Report",
            "📊 Exploratory Analytics",
            "🧠 Model Performance",
            "✍️ Predict Sentiment",
            "📁 Bulk Prediction",
            "📘 Methodology",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · Plotly")
    st.caption(f"Session date: {datetime.now().strftime('%d %b %Y')}")


# ------------------------------------------------------------------------------
# HERO HEADER (shown on every page)
# ------------------------------------------------------------------------------
def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of (label, value)"""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 1 — OVERVIEW
# ==============================================================================
if page == "🏠 Overview":
    hero(
        "Customer Feedback Intelligence Dashboard",
        "End-to-end sentiment analysis pipeline — from raw reviews to a production-ready classifier.",
    )

    total_reviews = None
    if reviews_df is not None:
        total_reviews = len(reviews_df)
    elif quality_summary:
        total_reviews = quality_summary.get("cleaned_full_shape", [None])[0]

    best_acc = None
    best_model_name = "—"
    if model_metadata:
        best_acc = model_metadata.get("best_accuracy")
        best_model_name = model_metadata.get("best_model_name", "—")

    kpi_row(
        [
            ("Reviews Analyzed", f"{total_reviews:,}" if total_reviews else "N/A"),
            ("Best Model", best_model_name),
            ("Model Accuracy", f"{best_acc*100:.1f}%" if best_acc else "N/A"),
            ("Sentiment Classes", "3 (Pos / Neu / Neg)"),
        ]
    )

    st.write("")
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📌 Project Summary")
        st.markdown(
            """
This dashboard analyzes customer product reviews and classifies each one as
**Positive**, **Neutral**, or **Negative**. The pipeline covers:

- **Data Cleaning** — deduplication, missing-value handling, text normalization
- **Feature Engineering** — TF-IDF vectorization (5,000 features)
- **Modeling** — Logistic Regression vs. Random Forest, best model auto-selected
- **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix
- **Explainability** — top predictive words driving each sentiment
- **Deployment** — live single-review prediction + bulk CSV scoring
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if reviews_df is not None and "sentiment" in reviews_df.columns:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Sentiment Split")
            counts = reviews_df["sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            fig = px.pie(
                counts,
                names="Sentiment",
                values="Count",
                hole=0.55,
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.1),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            missing_file_notice("data/reviews_clean.csv")


# ==============================================================================
# PAGE 2 — DATA QUALITY REPORT
# ==============================================================================
elif page == "🔍 Data Quality Report":
    hero("Data Quality Report", "Transparency into how the raw dataset was cleaned before modeling.")

    if quality_summary:
        orig_shape = quality_summary.get("original_shape", [0, 0])
        clean_shape = quality_summary.get("cleaned_full_shape", [0, 0])
        dup_count = quality_summary.get("duplicate_text_count", 0)

        kpi_row(
            [
                ("Raw Rows", f"{orig_shape[0]:,}"),
                ("Cleaned Rows", f"{clean_shape[0]:,}"),
                ("Duplicates Removed", f"{dup_count:,}"),
                ("Rows Dropped (%)", f"{(1 - clean_shape[0]/orig_shape[0])*100:.1f}%" if orig_shape[0] else "N/A"),
            ]
        )

        st.write("")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🧩 Missing Values by Column")
            missing = quality_summary.get("missing_values", {})
            missing_df = pd.DataFrame(list(missing.items()), columns=["Column", "Missing Count"])
            missing_df = missing_df.sort_values("Missing Count", ascending=False)
            fig = px.bar(
                missing_df,
                x="Missing Count",
                y="Column",
                orientation="h",
                color="Missing Count",
                color_continuous_scale=["#a78bfa", "#ec4899"],
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🗂️ Column Types")
            num_cols = quality_summary.get("numerical_columns", [])
            cat_cols = quality_summary.get("categorical_columns", [])
            st.markdown(f"**Numerical columns ({len(num_cols)}):**")
            st.write(", ".join(num_cols) if num_cols else "—")
            st.markdown(f"**Categorical columns ({len(cat_cols)}):**")
            st.write(", ".join(cat_cols) if cat_cols else "—")

            st.markdown("---")
            st.markdown("**Cleaning steps applied:**")
            st.markdown(
                """
1. Dropped rows with missing `ProfileName` or `Summary`
2. Removed duplicate reviews (same `Text`)
3. Filtered out reviews shorter than 5 characters
4. Sampled 15,000 rows for efficient model training
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        missing_file_notice("data/quality_summary.json")


# ==============================================================================
# PAGE 3 — EXPLORATORY ANALYTICS
# ==============================================================================
elif page == "📊 Exploratory Analytics":
    hero("Exploratory Analytics", "Visual patterns in customer sentiment, ratings, and review behavior.")

    if reviews_df is None:
        missing_file_notice("data/reviews_clean.csv")
    else:
        df = reviews_df.copy()

        # ---- Filters ----
        with st.expander("🔧 Filters", expanded=False):
            sentiments_available = sorted(df["sentiment"].unique()) if "sentiment" in df.columns else []
            selected_sentiments = st.multiselect(
                "Filter by sentiment", sentiments_available, default=sentiments_available
            )
            if selected_sentiments:
                df = df[df["sentiment"].isin(selected_sentiments)]

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("⭐ Rating Distribution")
            if "Score" in df.columns:
                fig = px.histogram(
                    df, x="Score", nbins=5,
                    color_discrete_sequence=["#8b5cf6"],
                )
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                                   xaxis_title="Star Rating", yaxis_title="Number of Reviews")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with row1_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📝 Review Length by Sentiment")
            if "review_length" in df.columns and "sentiment" in df.columns:
                fig = px.box(
                    df, x="sentiment", y="review_length", color="sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                )
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False,
                                   xaxis_title="Sentiment", yaxis_title="Word Count")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📈 Sentiment Trend Over Time")
        if "year_month" in df.columns and "sentiment" in df.columns:
            trend = df.groupby(["year_month", "sentiment"]).size().reset_index(name="count")
            fig = px.line(
                trend, x="year_month", y="count", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS, markers=True,
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=360,
                               xaxis_title="Month", yaxis_title="Review Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Add a `year_month` column to your cleaned data to unlock this chart.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔎 Sample Reviews")
        show_cols = [c for c in ["Score", "sentiment", "Summary", "Text"] if c in df.columns]
        st.dataframe(df[show_cols].sample(min(10, len(df))), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 4 — MODEL PERFORMANCE
# ==============================================================================
elif page == "🧠 Model Performance":
    hero("Model Performance", "How the sentiment classifier was trained, compared, and evaluated.")

    if model_metadata:
        kpi_row(
            [
                ("Best Model", model_metadata.get("best_model_name", "—")),
                ("Accuracy", f"{model_metadata.get('best_accuracy', 0)*100:.1f}%"),
                ("Training Samples", f"{model_metadata.get('training_samples', 0):,}"),
                ("Test Samples", f"{model_metadata.get('test_samples', 0):,}"),
            ]
        )
        st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚔️ Model Comparison")
        if model_comparison is not None:
            fig = px.bar(
                model_comparison, x="Model", y="Accuracy", color="Model",
                color_discrete_sequence=["#ec4899", "#6366f1"], text_auto=".2%",
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False,
                               yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_file_notice("model/model_comparison.csv")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧮 Confusion Matrix")
        if confusion_matrix_df is not None:
            cm = confusion_matrix_df.set_index(confusion_matrix_df.columns[0])
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale=["#eef2ff", "#8b5cf6", "#ec4899"],
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_file_notice("model/confusion_matrix.csv")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Classification Report")
    if classification_report_df is not None:
        cr = classification_report_df.rename(columns={classification_report_df.columns[0]: "Class"})
        st.dataframe(cr, use_container_width=True, hide_index=True)
    else:
        missing_file_notice("model/classification_report.csv")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔑 Top Predictive Words (Feature Importance)")
    if top_features_df is not None:
        fig = px.bar(
            top_features_df.sort_values("importance", ascending=True).tail(15),
            x="importance", y="word", orientation="h",
            color="importance", color_continuous_scale=["#a78bfa", "#ec4899"],
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420, coloraxis_showscale=False,
                           xaxis_title="Importance", yaxis_title="Word")
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing_file_notice("model/top_features.csv")
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 5 — PREDICT SENTIMENT (single review)
# ==============================================================================
elif page == "✍️ Predict Sentiment":
    hero("Predict Sentiment", "Type or paste a customer review to classify it in real time.")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    review_text = st.text_area(
        "Enter a review",
        placeholder="e.g. This product exceeded my expectations, great quality and fast delivery!",
        height=140,
    )
    predict_btn = st.button("🔮 Predict Sentiment")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        if model is None or tfidf is None:
            missing_file_notice("model/sentiment_model.pkl and model/tfidf_vectorizer.pkl")
        elif not review_text.strip():
            st.warning("Please enter some review text first.")
        else:
            cleaned = clean_text_basic(review_text)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]
            color = SENTIMENT_COLORS.get(pred, "#999")

            proba_html = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                classes = model.classes_
                proba_df = pd.DataFrame({"Sentiment": classes, "Confidence": proba}).sort_values(
                    "Confidence", ascending=False
                )

            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <span class="badge" style="background:{color}22; color:{color}; border:1px solid {color};">
                        Predicted Sentiment
                    </span>
                    <h1 style="color:{color}; margin-top:10px;">{pred}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Confidence Breakdown")
                fig = px.bar(
                    proba_df, x="Confidence", y="Sentiment", orientation="h",
                    color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
                    text_auto=".1%",
                )
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260,
                                   showlegend=False, xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 6 — BULK PREDICTION
# ==============================================================================
elif page == "📁 Bulk Prediction":
    hero("Bulk Prediction", "Upload a CSV of reviews and score them all at once.")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("Your CSV needs at least one text column (e.g. `Text` or `Summary`).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        bulk_df = pd.read_csv(uploaded)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Preview")
        st.dataframe(bulk_df.head(), use_container_width=True, hide_index=True)

        text_col = st.selectbox("Select the column containing review text", bulk_df.columns)
        run_btn = st.button("🚀 Run Bulk Prediction")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_btn:
            if model is None or tfidf is None:
                missing_file_notice("model/sentiment_model.pkl and model/tfidf_vectorizer.pkl")
            else:
                with st.spinner("Scoring reviews..."):
                    cleaned = bulk_df[text_col].astype(str).apply(clean_text_basic)
                    vecs = tfidf.transform(cleaned)
                    preds = model.predict(vecs)
                    bulk_df["predicted_sentiment"] = preds

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Results")
                counts = bulk_df["predicted_sentiment"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Count"]
                fig = px.pie(
                    counts, names="Sentiment", values="Count", hole=0.5,
                    color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
                )
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(bulk_df, use_container_width=True, hide_index=True)
                csv_out = bulk_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_out,
                    file_name="bulk_sentiment_predictions.csv",
                    mime="text/csv",
                )
                st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 7 — METHODOLOGY
# ==============================================================================
elif page == "📘 Methodology":
    hero("Project Methodology", "A transparent walkthrough of the full pipeline, end to end.")

    steps = [
        ("1. Data Collection", "Started with the Amazon Fine Food Reviews dataset (568,454 raw reviews with 10 columns)."),
        ("2. Data Cleaning", "Removed rows with missing ProfileName/Summary, dropped 174,875 duplicate reviews, and filtered out very short text, leaving 393,560 clean rows."),
        ("3. Sampling", "Took a stratified random sample of 15,000 reviews for efficient, reproducible model training."),
        ("4. Text Preprocessing", "Lowercased text, stripped HTML tags and non-alphabetic characters, and collapsed whitespace."),
        ("5. Label Creation", "Converted the 1–5 star Score into three sentiment classes: Positive (≥4), Neutral (=3), Negative (<3)."),
        ("6. Feature Engineering", "Vectorized cleaned text with TF-IDF, capped at 5,000 features, English stop words removed."),
        ("7. Model Training", "Trained and compared Logistic Regression and Random Forest, both with class-balanced weighting to handle sentiment imbalance."),
        ("8. Model Selection", "Selected the higher-accuracy model automatically and persisted it with joblib for deployment."),
        ("9. Evaluation", "Generated a full classification report, confusion matrix, and top-feature importance for interpretability."),
        ("10. Deployment", "Wrapped everything in this Streamlit dashboard for live single-review prediction and bulk CSV scoring."),
    ]

    for title, desc in steps:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin-bottom:6px;">{title}</h4>
                <p style="margin:0; color:#333;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )