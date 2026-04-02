import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(max_features=10000):
    """
    Create TF-IDF vectorizer with unigrams + bigrams.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    return vectorizer


def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)


def load_vectorizer(path):
    return joblib.load(path)
