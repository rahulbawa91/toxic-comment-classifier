import os
import pandas as pd
import joblib

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')   # 👈 ADD THIS
nltk.download('stopwords')
nltk.download('wordnet')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from preprocessing import clean_text
from feature_engineering import create_vectorizer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # If multi-label dataset, convert to binary
    if "is_toxic" not in df.columns:
        label_columns = ["toxic", "severe_toxic", "obscene",
                         "threat", "insult", "identity_hate"]
        if all(col in df.columns for col in label_columns):
            df["is_toxic"] = df[label_columns].sum(axis=1) > 0
            df["is_toxic"] = df["is_toxic"].astype(int)

    print("Cleaning text...")
    df["cleaned_text"] = df["comment_text"].apply(clean_text)

    X = df["cleaned_text"]
    y = df["is_toxic"] if "is_toxic" in df.columns else df["label"]

    print("Creating TF-IDF features...")
    vectorizer = create_vectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))

    # Save model & vectorizer
    joblib.dump(model, os.path.join(MODEL_DIR, "toxic_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    print("\n✅ Model and vectorizer saved successfully!")


if __name__ == "__main__":
    train_model()
