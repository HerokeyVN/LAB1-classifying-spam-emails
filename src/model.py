"""KNN và Naive Bayes. Lưu và nạp trạng thái bằng JSON, không dùng pickle."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


# KNN tự tính khoảng cách Euclid và bỏ phiếu theo k láng giềng.
class KNNClassifier:
    # Lưu số láng giềng trước khi nhận dữ liệu huấn luyện.
    def __init__(self, n_neighbors=5):
        self.n_neighbors = int(n_neighbors)
        self._built = False

    # Kiểm tra cấu hình và chuẩn bị trạng thái trước khi huấn luyện.
    def build_model(self):
        if self.n_neighbors < 1:
            raise ValueError("k phải lớn hơn 0")
        for name in ("X_train_", "y_train_", "classes_", "train_norms_"):
            if hasattr(self, name):
                delattr(self, name)
        self._built = True
        return self

    # Lưu vector thưa và nhãn Train để tính khoảng cách khi dự đoán.
    def fit(self, X, y):
        if not self._built:
            raise ValueError("Cần gọi build_model() trước khi fit KNN")
        matrix = csr_matrix(X, dtype=float)
        labels = np.asarray(y)
        if matrix.shape[0] != len(labels) or len(labels) == 0:
            raise ValueError("Số mẫu và nhãn Train không hợp lệ")
        if not 1 <= self.n_neighbors <= len(labels):
            raise ValueError("k phải từ 1 đến số mẫu Train")
        self.X_train_, self.y_train_ = matrix, labels
        self.classes_ = np.unique(labels)
        self.train_norms_ = np.asarray(self.X_train_.multiply(self.X_train_).sum(axis=1)).ravel()
        return self

    # Tính tỷ lệ phiếu từng lớp theo k khoảng cách Euclid nhỏ nhất.
    def predict_proba(self, X):
        if not hasattr(self, "X_train_"):
            raise ValueError("Cần fit KNN trước khi dự đoán")
        query = csr_matrix(X, dtype=float)
        if query.shape[1] != self.X_train_.shape[1]:
            raise ValueError("Số đặc trưng không khớp dữ liệu Train")
        result = np.zeros((query.shape[0], len(self.classes_)))
        for start in range(0, query.shape[0], 128):
            block = query[start:start + 128]
            norms = np.asarray(block.multiply(block).sum(axis=1)).ravel()
            # Bình phương khoảng cách = ||x||² + ||z||² - 2 x·z.
            distances = norms[:, None] + self.train_norms_[None, :] - 2 * (block @ self.X_train_.T).toarray()
            np.maximum(distances, 0, out=distances)
            for row, distance in enumerate(distances):
                # Khi khoảng cách bằng nhau, ưu tiên mẫu có chỉ số Train nhỏ hơn.
                neighbors = np.argsort(distance, kind="stable")[:self.n_neighbors]
                votes = self.y_train_[neighbors]
                result[start + row] = [np.count_nonzero(votes == label) / self.n_neighbors for label in self.classes_]
        return result

    # Chọn nhãn có nhiều phiếu nhất; hòa phiếu ưu tiên lớp đầu theo thứ tự nhãn.
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


# Multinomial Naive Bayes tự ước lượng prior và xác suất đặc trưng.
class MultinomialNaiveBayes:
    # Lưu hệ số làm trơn Laplace trước khi huấn luyện.
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self._built = False

    # Kiểm tra alpha và chuẩn bị trạng thái trước khi huấn luyện.
    def build_model(self):
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha phải là số hữu hạn lớn hơn 0")
        for name in ("classes_", "class_log_prior_", "feature_log_prob_"):
            if hasattr(self, name):
                delattr(self, name)
        self._built = True
        return self

    # Tính log prior và log xác suất đặc trưng từng lớp trên Train.
    def fit(self, X, y):
        if not self._built:
            raise ValueError("Cần gọi build_model() trước khi fit Naive Bayes")
        X = csr_matrix(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != len(y) or len(y) == 0 or X.shape[1] == 0:
            raise ValueError("Kích thước dữ liệu hoặc nhãn không hợp lệ")
        if X.data.size and np.min(X.data) < 0:
            raise ValueError("Multinomial Naive Bayes yêu cầu đặc trưng không âm")
        self.classes_, class_indices = np.unique(y, return_inverse=True)
        counts = np.bincount(class_indices, minlength=len(self.classes_))
        self.class_log_prior_ = np.log(counts / len(y))
        totals = np.vstack([np.asarray(X[class_indices == i].sum(axis=0)).ravel() for i in range(len(self.classes_))])
        smoothed = totals + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    # Tính log prior cộng tổng trọng số đặc trưng nhân log likelihood.
    def predict_log_scores(self, X):
        if not hasattr(self, "feature_log_prob_"):
            raise ValueError("Cần fit Naive Bayes trước khi dự đoán")
        X = csr_matrix(X, dtype=float)
        if X.data.size and np.min(X.data) < 0:
            raise ValueError("Multinomial Naive Bayes yêu cầu đặc trưng không âm")
        return np.asarray(X @ self.feature_log_prob_.T + self.class_log_prior_)

    # Chọn lớp có điểm log lớn nhất; hòa điểm ưu tiên lớp đầu.
    def predict(self, X):
        scores = self.predict_log_scores(X)
        return self.classes_[scores.argmax(axis=1)]


# Ghi một dict ra file JSON. indent=2 thì mỗi phần tử vector nằm một dòng.
def _write(path, payload, indent=None):
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


# Lưu từ vựng và IDF đã học trên train vào models/tfidf.json.
def save_tfidf(vectorizer):
    _write(MODELS / "tfidf.json", {
        "max_features": int(vectorizer.max_features),
        "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf": [float(value) for value in vectorizer.idf_],
    }, indent=2)


# Lưu K, nhãn train và ma trận TF-IDF train dạng sparse vào models/knn.json.
def save_knn(model):
    sparse = model.X_train_
    _write(MODELS / "knn.json", {
        "k": int(model.n_neighbors),
        "classes": [str(label) for label in model.classes_],
        "labels": [str(label) for label in model.y_train_],
        "shape": [int(sparse.shape[0]), int(sparse.shape[1])],
        "data": [float(value) for value in sparse.data],
        "indices": [int(value) for value in sparse.indices],
        "indptr": [int(value) for value in sparse.indptr],
    }, indent=2)


# Lưu alpha, prior và log xác suất từ theo lớp vào models/naive_bayes.json.
def save_nb(model):
    _write(MODELS / "naive_bayes.json", {
        "alpha": float(model.alpha),
        "classes": [str(label) for label in model.classes_],
        "class_log_prior": [float(value) for value in model.class_log_prior_],
        "feature_log_prob": [[float(value) for value in row] for row in model.feature_log_prob_],
    }, indent=2)


# Đọc một file JSON trong thư mục models.
def load_json(name):
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


# Biến thư mới thành TF-IDF bằng từ vựng và IDF đã lưu, không fit lại.
def transform(state, texts):
    counter = CountVectorizer(vocabulary=state["vocabulary"], stop_words="english")
    counts = counter.fit_transform(texts)
    idf = np.asarray(state["idf"], dtype=float)
    weighted = counts @ diags(idf)
    squared = np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel()
    norms = np.sqrt(squared)
    norms[norms == 0] = 1.0
    return diags(1.0 / norms) @ weighted


# Khôi phục mô hình từ JSON; dùng lại các phương thức của class khi dự đoán.
def load_model(name):
    state = load_json(name)
    if name == "knn.json":
        model = KNNClassifier(state["k"]).build_model()
        matrix = csr_matrix((state["data"], state["indices"], state["indptr"]), shape=tuple(state["shape"]))
        return model.fit(matrix, state["labels"])
    if name == "naive_bayes.json":
        model = MultinomialNaiveBayes(state["alpha"]).build_model()
        model.classes_ = np.asarray(state["classes"])
        model.class_log_prior_ = np.asarray(state["class_log_prior"], dtype=float)
        model.feature_log_prob_ = np.asarray(state["feature_log_prob"], dtype=float)
        return model
    raise ValueError(f"Không hỗ trợ model: {name}")


# Biến đổi một lần và dùng chung điểm lớp để lấy nhãn, tùy chọn trả cả điểm ROC.
def predict_both(texts, return_scores=False):
    matrix = transform(load_json("tfidf.json"), texts)
    predictions, scores = {}, {}
    for key, filename in (("knn", "knn.json"), ("naive_bayes", "naive_bayes.json")):
        model = load_model(filename)
        values = model.predict_proba(matrix) if key == "knn" else model.predict_log_scores(matrix)
        predictions[key] = model.classes_[values.argmax(axis=1)]
        if return_scores:
            spam = list(model.classes_).index("spam")
            ham = list(model.classes_).index("ham")
            scores[key] = values[:, spam] if key == "knn" else values[:, spam] - values[:, ham]
    return (predictions, scores) if return_scores else predictions
