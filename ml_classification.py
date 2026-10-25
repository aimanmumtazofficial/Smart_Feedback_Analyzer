import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("data/reviews_sample.csv")
data = data.dropna(subset=["clean_text", "sentiment"])

X = data["clean_text"]
y = data["sentiment"]

# Text -> numbers (TF-IDF = Term Frequency-Inverse Document Frequency):

tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec = tfidf.fit_transform(X)

# ======= Train_Test_Split =========

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# =========  Model 1: Logistic Regression ==========
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)
pred1= log_reg.predict(X_test)
acc1 = accuracy_score(y_test, pred1)
print("Logistic Regression")
print(f"Accuray: {acc1:.3f}")
print(classification_report(y_test, pred1))

# =========  Model 2: Random Forest =========

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
pred2 = rf.predict(X_test)
acc2 = accuracy_score(y_test, pred2)
print("RandomForest Classifier")
print(f"Accuray: {acc2:.3f}")
print(classification_report(y_test, pred2))

# ===== Find Best Model =======

best_model = log_reg if acc1 >= acc2 else rf 
joblib.dump(best_model, "model/sentiment_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
print("\nBest model saved to model/ folder")