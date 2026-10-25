import pandas as pd 
import re 

# ======= Sample Data ======
data = pd.read_csv("data/reviews_sample.csv")
print("\n Loaded Rows:", len(data))

# ====== Remove Missing Values ======
data = data.dropna(subset=["Text", "Score"])


# ======= Remove Duplicates Reviews ========
data = data.dropna(subset=["Text"])

# ====== Text Cleaning Function ========
def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)       
    text = re.sub(r"\s+", " ", text).strip()     
    return text


data["clean_text"] = data["Text"].apply(clean_text)

data = data[data["clean_text"].str.len() > 5]
data["review_length"] = data["clean_text"].apply(lambda t: len(t.split()))


def score_to_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

data["sentiment"] = data["Score"].apply(score_to_sentiment)

data["date"] = pd.to_datetime(data["Time"], unit="s")
data["year_month"] = data["date"].dt.to_period("M").astype(str)

data.to_csv("data/reviews_sample.csv", index=False)
print("Cleaned data saved!")
print(data["sentiment"].value_counts())