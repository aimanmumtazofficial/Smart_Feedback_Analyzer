"""
evaluate_existing_model.py
---------------------------
Use this INSTEAD of train_model.py if you already have a trained model
(sentiment_model.pkl + tfidf_vectorizer.pkl) and just want the dashboard's
evaluation files generated from it — no retraining happens here.

Run order:
    1. python prepare_data.py         -> creates data/reviews_clean.csv
    2. Put your existing sentiment_model.pkl and tfidf_vectorizer.pkl in model/
    3. python evaluate_existing_model.py

Produces in model/:
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODEL_DIR = "model"

clean_path = os.path.join(DATA_DIR, "reviews_clean.csv")
model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

for p in [clean_path, model_path, tfidf_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Could not find '{p}'. Make sure prepare_data.py has run and your "
            f"model files are inside the model/ folder."
        )

print("Loading cleaned data ...")
df = pd.read_csv(clean_path)
df = df.dropna(subset=["clean_text", "sentiment"])
X = df["clean_text"]
y = df["sentiment"]

print("Loading existing model + vectorizer (no retraining) ...")
model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)

X_vec = tfidf.transform(X)

# Same split settings used originally, so the test set is evaluated consistently
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

print("Evaluating on the held-out test split ...")
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

model_name = type(model).__name__.replace("Classifier", "")
if model_name == "RandomForest":
    model_name = "Random Forest"
elif model_name == "LogisticRegression":
    model_name = "Logistic Regression"

# ---------------------------------------------------------------------------
# model_comparison.csv (single row, since we only have one existing model)
# ---------------------------------------------------------------------------
comparison = pd.DataFrame({"Model": [model_name], "Accuracy": [acc]})
comparison.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)
print("Saved model/model_comparison.csv")

# ---------------------------------------------------------------------------
# classification_report.csv
# ---------------------------------------------------------------------------
report_dict = classification_report(y_test, preds, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)
report_df.to_csv(os.path.join(MODEL_DIR, "classification_report.csv"))
print("Saved model/classification_report.csv")

# ---------------------------------------------------------------------------
# confusion_matrix.csv
# ---------------------------------------------------------------------------
labels = sorted(y.unique())
cm = confusion_matrix(y_test, preds, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(os.path.join(MODEL_DIR, "confusion_matrix.csv"))
print("Saved model/confusion_matrix.csv")

# ---------------------------------------------------------------------------
# top_features.csv
# ---------------------------------------------------------------------------
feature_names = np.array(tfidf.get_feature_names_out())
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    top_features = pd.DataFrame({"word": feature_names[top_idx], "importance": importances[top_idx]})
elif hasattr(model, "coef_"):
    coefs = np.abs(model.coef_).mean(axis=0)
    top_idx = np.argsort(coefs)[::-1][:20]
    top_features = pd.DataFrame({"word": feature_names[top_idx], "importance": coefs[top_idx]})
else:
    top_features = pd.DataFrame({"word": [], "importance": []})
top_features.to_csv(os.path.join(MODEL_DIR, "top_features.csv"), index=False)
print("Saved model/top_features.csv")

# ---------------------------------------------------------------------------
# model_metadata.json
# ---------------------------------------------------------------------------
metadata = {
    "best_model_name": model_name,
    "best_accuracy": float(acc),
    "training_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "tfidf_features": int(X_vec.shape[1]),
    "classes": list(labels),
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved model/model_metadata.json")

print("\nDone! All model/ evaluation files generated from your existing model. Refresh the dashboard.")