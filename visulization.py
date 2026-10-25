import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/reviews_sample.csv")

# ========= Chart One ==========
# ======== Rating Distribution ==========
plt.figure(figsize=(6,6))
sns.countplot(x="Score", data=data, palette="viridis")
plt.title("Rating Distribution")
plt.xlabel("Score (1-5)")
plt.ylabel("Number Of Reviews")
plt.savefig("charts/rating_distribution.png")
plt.close()


# ============ Chart Two ===========
# ========= Sentiment Distribution ==========
plt.figure(figsize=(6,4))
sns.countplot(x="sentiment", data=data, order=["Positive", "Neural", "Negative"])
plt.title("Sentiment Distribution")
plt.savefig("charts/sentiment_distribution.png")
plt.close()


# ========= Chart Three =========
# ========= Review Length vs Rating ===========
plt.figure(figsize=(6,4))
sns.boxplot(x="Score", y="review_length", data=data)
plt.title("Review Length vs Rating")
plt.savefig("charts/length_vs_rating.png")
plt.close()


# ========= Chart Four ========
# ======== Rating trend over time ========
trend = data.groupby("year_month")["Score"].mean().reset_index()
plt.figure(figsize=(10, 4))
plt.plot(trend["year_month"], trend["Score"], marker="o")
plt.xticks(rotation=90, fontsize=6)
plt.title("Average Rating Over Time")
plt.tight_layout()
plt.savefig("charts/rating_trend.png")
plt.close()
