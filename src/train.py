"""Huấn luyện KNN và Naive Bayes, rồi lưu vectorizer cùng hai model vào models/."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import KNNClassifier, MultinomialNaiveBayes, load_json, load_model, save_knn, save_nb, save_tfidf, transform
from preprocess import prepare


# Train hai model, lưu JSON, rồi kiểm tra dự đoán từ JSON khớp lúc train.
def train():
    x_train, x_test, y_train, y_test, vectorizer, config = prepare()
    knn = KNNClassifier(n_neighbors=config["knn"]["k"])
    nb = MultinomialNaiveBayes(alpha=config["naive_bayes"]["alpha"])
    knn.build_model()
    nb.build_model()
    knn.fit(x_train, y_train)
    nb.fit(x_train, y_train)
    save_tfidf(vectorizer)
    save_knn(knn)
    save_nb(nb)
    test_text = pd.read_csv(SRC.parent / "data" / "processed" / "test.csv")["clean_text"]
    rebuilt = transform(load_json("tfidf.json"), test_text) 
    # Kiểm tra vector TF-IDF trước và sau khi lưu JSON.
    if not np.allclose(x_test.toarray(), rebuilt.toarray()):
        gap = float(np.max(np.abs(x_test.toarray() - rebuilt.toarray())))
        raise AssertionError(f"TF-IDF JSON lệch {gap}")
    # Kiểm tra nhãn KNN trước và sau khi lưu JSON.
    if not np.array_equal(knn.predict(x_test), load_model("knn.json").predict(rebuilt)):
        raise AssertionError("KNN JSON không khớp predict") 
    if not np.array_equal(nb.predict(x_test), load_model("naive_bayes.json").predict(rebuilt)):
        raise AssertionError("Naive Bayes JSON không khớp predict")
    print(f"train {tuple(x_train.shape)} test {tuple(x_test.shape)}")


if __name__ == "__main__":
    train()