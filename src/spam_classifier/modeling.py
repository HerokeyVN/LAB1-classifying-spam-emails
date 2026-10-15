"""Reusable sklearn training and inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ModelName = Literal["logistic_regression", "linear_svm", "naive_bayes"]


def build_pipeline(model_name: ModelName = "logistic_regression") -> Pipeline:
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1_000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
        "naive_bayes": MultinomialNB(),
    }
    if model_name not in classifiers:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(classifiers)}")
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("classifier", classifiers[model_name]),
        ]
    )


def evaluate_model(model: Pipeline, texts, labels) -> dict[str, object]:
    predictions = model.predict(texts)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "spam_precision": round(float(precision), 4),
        "spam_recall": round(float(recall), 4),
        "spam_f1": round(float(f1), 4),
        "report": classification_report(
            labels, predictions, target_names=["ham", "spam"], zero_division=0
        ),
    }


def save_model(model: Pipeline, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
