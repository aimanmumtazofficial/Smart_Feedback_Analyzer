import pandas as pd

data = pd.read_csv("data/reviews_sample.csv")

def ask_ai_assistant(question, data):
    """
    This function is the 'AI Assistant'. Right now it is rule-based
    (uses the data's statistics). If you get an OpenAI/Hugging Face
    API key, you can plug the API call inside this function -
    the rest of the app won't need to change.
    """
    q = question.lower()
    total = len(data)
    pos_pct = (data["sentiment"] == "Positive").mean() * 100
    neg_pct = (data["sentiment"] == "Negative").mean() * 100
    avg_score = data["Score"].mean()

    if "trend" in q or "overall" in q:
        return f"Out of {total} total reviews, {pos_pct:.1f}% are positive and {neg_pct:.1f}% are negative. The average rating is {avg_score:.2f}/5."
    elif "negative" in q or "complain" in q:
        return f"{neg_pct:.1f}% of reviews are negative. Customers mostly complain about shipping or taste."
    elif "positive" in q:
        return f"{pos_pct:.1f}% of reviews are positive - overall customers seem satisfied."
    elif "average" in q or "rating" in q:
        return f"The average rating is {avg_score:.2f} out of 5."
    else:
        return "You can ask about: overall trend, negative feedback %, positive feedback %, or average rating."

# Test it out in the terminal:
if __name__ == "__main__":
    print(ask_ai_assistant("what is the overall trend?", data))
    print(ask_ai_assistant("tell me about negative reviews", data))