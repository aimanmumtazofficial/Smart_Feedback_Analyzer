"""
train_model_v2.py
-------------------
Same as train_model.py, but fixes the "Neutral almost never predicted"
problem by giving the Neutral class extra weight during training
(on top of scikit-learn's normal balanced weighting).

Run order:
    1. python prepare_data.py          -> creates data/reviews_clean.csv
    2. python train_model_v2.py        -> retrains with boosted Neutral weight

Produces in model/ (overwrites the old files):
    sentiment_model.pkl
    tfidf_vectorizer.pkl
    model_metadata.json
    model_comparison.csv
    classification_report.csv
    confusion_matrix.csv
    top_features.csv
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# How much extra weight to give the Neutral class, on top of the normal
# "balanced" weighting. Try 3-6; higher = model favors Neutral more,
# which can start hurting Positive/Negative accuracy if pushed too far.
NEUTRAL_BOOST = 5

clean_path = os.path.join(DATA_DIR, "reviews_clean.csv")
if not os.path.exists(clean_path):
    raise FileNotFoundError(f"Could not find '{clean_path}'. Run prepare_data.py first.")

df = pd.read_csv(clean_path)
df = df.dropna(subset=["clean_text", "sentiment"])
X = df["clean_text"]
y = df["sentiment"]

print("Class counts in full cleaned dataset:")
print(y.value_counts())

print("\nVectorizing text with TF-IDF ...")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Build custom class weights: start from sklearn's "balanced" weights,
# then multiply the Neutral class by NEUTRAL_BOOST.
# ---------------------------------------------------------------------------
classes = np.unique(y_train)
balanced_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, balanced_weights))

if "Neutral" in class_weight_dict:
    class_weight_dict["Neutral"] *= NEUTRAL_BOOST

print("\nFinal class weights used for training:")
print(class_weight_dict)

print("\nTraining Logistic Regression (boosted Neutral weight) ...")
log_reg = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
log_reg.fit(X_train, y_train)
pred1 = log_reg.predict(X_test)
acc1 = accuracy_score(y_test, pred1)

print("Training Random Forest (boosted Neutral weight) ...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight_dict)
rf.fit(X_train, y_train)
pred2 = rf.predict(X_test)
acc2 = accuracy_score(y_test, pred2)

# ---------------------------------------------------------------------------
# Pick the best model by looking at Neutral recall first, then accuracy,
# since raw accuracy alone would just favor Positive again.
# ---------------------------------------------------------------------------
def neutral_recall(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return report.get("Neutral", {}).get("recall", 0.0)

neutral_recall1 = neutral_recall(y_test, pred1)
neutral_recall2 = neutral_recall(y_test, pred2)

print(f"\nLogistic Regression -> Accuracy: {acc1:.4f}, Neutral recall: {neutral_recall1:.4f}")
print(f"Random Forest       -> Accuracy: {acc2:.4f}, Neutral recall: {neutral_recall2:.4f}")

if neutral_recall1 >= neutral_recall2:
    best_model, best_pred, best_name = log_reg, pred1, "Logistic Regression"
else:
    best_model, best_pred, best_name = rf, pred2, "Random Forest"

print(f"\nSelected model: {best_name} (best Neutral recall)")

joblib.dump(best_model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# Model comparison
comparison = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [acc1, acc2],
        "Neutral_Recall": [neutral_recall1, neutral_recall2],
    }
)
comparison.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)

# Classification report
report_dict = classification_report(y_test, best_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)
report_df.to_csv(os.path.join(MODEL_DIR, "classification_report.csv"))

# Confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(os.path.join(MODEL_DIR, "confusion_matrix.csv"))

# Top features
feature_names = np.array(tfidf.get_feature_names_out())
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    top_features = pd.DataFrame({"word": feature_names[top_idx], "importance": importances[top_idx]})
else:
    coefs = np.abs(best_model.coef_).mean(axis=0)
    top_idx = np.argsort(coefs)[::-1][:20]
    top_features = pd.DataFrame({"word": feature_names[top_idx], "importance": coefs[top_idx]})
top_features.to_csv(os.path.join(MODEL_DIR, "top_features.csv"), index=False)

# Metadata
metadata = {
    "best_model_name": best_name,
    "best_accuracy": float(max(acc1, acc2)),
    "neutral_recall": float(max(neutral_recall1, neutral_recall2)),
    "neutral_boost_factor": NEUTRAL_BOOST,
    "training_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "tfidf_features": int(X_vec.shape[1]),
    "classes": list(labels),
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nDone. Confusion matrix:")
print(cm_df)
print("\nAll model/ files updated. Refresh your Streamlit dashboard to see the new results.")
print(f"\nTip: if Neutral recall is still too low, increase NEUTRAL_BOOST (currently {NEUTRAL_BOOST}) near the top of this file and re-run.")