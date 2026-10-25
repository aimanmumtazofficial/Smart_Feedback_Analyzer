import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Smart Feedback Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Custom Professional Theme (CSS) ----------------
st.markdown("""
    <style>
        /* Overall background */
        .stApp {
            background-color: #0e1117;
        }

        /* Main title */
        h1 {
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
            color: #ffffff;
            padding-bottom: 0px;
        }

        /* Subtitle / caption */
        .subtitle {
            color: #9ca3af;
            font-size: 16px;
            margin-top: -10px;
            margin-bottom: 25px;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #1c1f26;
            border: 1px solid #2d3139;
            border-radius: 12px;
            padding: 15px 10px;
            text-align: center;
        }
        div[data-testid="stMetricValue"] {
            color: #60a5fa;
            font-size: 28px;
        }
        div[data-testid="stMetricLabel"] {
            color: #9ca3af;
        }

        /* Tabs styling */
        button[data-baseweb="tab"] {
            font-size: 16px;
            font-weight: 600;
            color: #d1d5db;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #60a5fa;
            border-bottom: 3px solid #60a5fa;
        }

        /* Text area / input boxes */
        .stTextArea textarea, .stTextInput input {
            background-color: #1c1f26;
            color: #ffffff;
            border-radius: 8px;
            border: 1px solid #2d3139;
        }

        /* Buttons */
        .stButton button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 24px;
            font-weight: 600;
        }
        .stButton button:hover {
            background-color: #1d4ed8;
            color: white;
        }

        /* Prediction/answer result box */
        .result-box {
            background-color: #1c1f26;
            border-left: 4px solid #60a5fa;
            border-radius: 8px;
            padding: 16px;
            margin-top: 12px;
            color: #e5e7eb;
            font-size: 16px;
        }

        /* Section headers inside tabs */
        .section-header {
            color: #ffffff;
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 10px;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------- Helper Functions ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ask_ai_assistant(question, data):
    """
    This is the 'AI Assistant'. It is currently rule-based (uses the
    dataset's statistics). If you get an OpenAI/Hugging Face API key,
    you can plug the API call inside this function - the rest of the
    app won't need to change.
    """
    q = question.lower()
    total = len(data)
    pos_pct = (data["sentiment"] == "Positive").mean() * 100
    neg_pct = (data["sentiment"] == "Negative").mean() * 100
    avg_score = data["Score"].mean()

    if "trend" in q or "overall" in q:
        return f"Out of {total} reviews, {pos_pct:.1f}% are positive and {neg_pct:.1f}% are negative. The average rating is {avg_score:.2f}/5."
    elif "negative" in q or "complain" in q:
        return f"{neg_pct:.1f}% of reviews are negative."
    elif "positive" in q:
        return f"{pos_pct:.1f}% of reviews are positive."
    elif "average" in q or "rating" in q:
        return f"The average rating is {avg_score:.2f}/5."
    else:
        return "Try asking: overall trend, negative %, positive %, or average rating."


# ---------------- Header ----------------
st.title("🧠 Smart Feedback Analyzer")
st.markdown('<p class="subtitle">AI-powered sentiment analysis and insights for customer feedback</p>', unsafe_allow_html=True)

# ---------------- Load Data ----------------
data = pd.read_csv("data/reviews_sample.csv")
st.success(f"Loaded {len(data)} reviews")

# ---------------- Top Metrics ----------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", f"{len(data):,}")
col2.metric("Average Rating", f"{data['Score'].mean():.2f} / 5")
col3.metric("Positive Sentiment", f"{(data['sentiment'] == 'Positive').mean() * 100:.1f}%")

st.write("")  # spacing

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📊  Charts", "🤖  Predict Sentiment", "💬  AI Assistant"])

# ----- Tab 1: Charts -----
with tab1:
    st.markdown('<p class="section-header">Rating & Sentiment Overview</p>', unsafe_allow_html=True)

    plt.style.use("dark_background")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="Score", data=data, hue="Score", palette="Blues_d", legend=False, ax=ax)
        ax.set_title("Rating Distribution", color="white")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        st.pyplot(fig)

    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.countplot(x="sentiment", data=data, hue="sentiment",
                      order=["Positive", "Neutral", "Negative"],
                      palette=["#22c55e", "#eab308", "#ef4444"], legend=False, ax=ax2)
        ax2.set_title("Sentiment Distribution", color="white")
        ax2.set_facecolor("#0e1117")
        fig2.patch.set_facecolor("#0e1117")
        st.pyplot(fig2)

# ----- Tab 2: Predict -----
with tab2:
    st.markdown('<p class="section-header">Test the Sentiment Classifier</p>', unsafe_allow_html=True)

    model = joblib.load("model/sentiment_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

    user_text = st.text_area("Type a review:", "The product was okay but delivery was slow.")

    if st.button("Predict Sentiment"):
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        color_map = {"Positive": "#22c55e", "Neutral": "#eab308", "Negative": "#ef4444"}
        color = color_map.get(prediction, "#60a5fa")

        st.markdown(
            f'<div class="result-box">Predicted Sentiment: '
            f'<span style="color:{color}; font-weight:700;">{prediction}</span></div>',
            unsafe_allow_html=True
        )

# ----- Tab 3: AI Assistant -----
with tab3:
    st.markdown('<p class="section-header">Ask the AI Assistant</p>', unsafe_allow_html=True)

    question = st.text_input("Your question:", "What is the overall trend?")

    if st.button("Ask"):
        answer = ask_ai_assistant(question, data)
        st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)