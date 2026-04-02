from django.shortcuts import render

# Create your views here.
import os
import sys
import joblib
from django.shortcuts import render

# Add src folder to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from preprocessing import clean_text

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "toxic_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def home(request):
    result = None
    confidence = None

    if request.method == "POST":
        comment = request.POST.get("comment")

        cleaned = clean_text(comment)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]

        result = "Toxic ❌" if prediction == 1 else "Non-Toxic ✅"
        confidence = round(proba * 100, 2)

    return render(request, "index.html", {
        "result": result,
        "confidence": confidence
    })
