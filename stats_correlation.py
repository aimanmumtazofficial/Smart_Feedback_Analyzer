import pandas as pd 

data = pd.read_csv("data/reviews_sample.csv")

# =========== Descriptive stats ==========

print("Descriptive Stats")
print(data[["Score", "review_length"]].describe())


# ============  Correlation: Rating vs Review Length ===========

correlation = data["Score"].corr(data["review_length"])
print(f"\nCorrelation between Rating and Review Length: {correlation:.3f}")

# ======== According To Sentiment Average Review Length ========
print("\nAverage review length by sentiment")
print(data.groupby("sentiment")["review_length"].mean())

# ======= Sentiment Percentage =======

print("Sentiment Percentage")
print((data["sentiment"].value_counts(normalize=True) * 100).round(1))