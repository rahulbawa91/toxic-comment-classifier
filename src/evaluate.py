import os
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from preprocessing import clean_text

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxic_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


def evaluate_model():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "is_toxic" not in df.columns:
        label_columns = ["toxic", "severe_toxic", "obscene",
                         "threat", "insult", "identity_hate"]
        if all(col in df.columns for col in label_columns):
            df["is_toxic"] = df[label_columns].sum(axis=1) > 0
            df["is_toxic"] = df["is_toxic"].astype(int)

    df["cleaned_text"] = df["comment_text"].apply(clean_text)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X = vectorizer.transform(df["cleaned_text"])
    y = df["is_toxic"] if "is_toxic" in df.columns else df["label"]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y, y_pred))

    print("\nAUC Score:", roc_auc_score(y, y_proba))


if __name__ == "__main__":
    evaluate_model()
