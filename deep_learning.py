import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


data = pd.read_csv("data/reviews_sample.csv")
data = data.dropna(subset=["clean_text", "sentiment"])

tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X= tfidf.fit_transform(data["clean_text"]).toarray()

# ========= Labels Convert Into Numbers ==========

encoder = LabelEncoder()
y = encoder.fit_transform(data["sentiment"])
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# ======== Simple Neural Network =========
model= Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nNeural Network Test Accuracy: {accuracy:.3f}")
print("Compare this with Logistic Regression/Random Forest accuracy from Module 4!")