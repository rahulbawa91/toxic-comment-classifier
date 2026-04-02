import os
import joblib
from preprocessing import clean_text

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxic_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_comment(text: str) -> str:
    """
    Predict whether a comment is toxic or not.
    """
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    return "Toxic" if pred == 1 else "Non-Toxic"


if __name__ == "__main__":
    comment = input("Enter a comment: ")
    print("Prediction:", predict_comment(comment))
