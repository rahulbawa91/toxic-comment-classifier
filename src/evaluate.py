import os
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from preprocessing import clean_text

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data","raw","train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models","toxic_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models","vectorizer.pkl")
