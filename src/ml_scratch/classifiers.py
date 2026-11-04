"""Các classifier tự viết, dùng chung cho notebook và file mô hình."""

import numpy as np
from .formulas import (euclidean_distances, nearest_indices, majority_vote,
    class_prior, laplace_probabilities, log_score, normalized_scores)

class KNNSpamClassifier:
    def __init__(self, k=5):
        self.k = k
        self.ready = False
        self.X_train = None
        self.y_train = None

    def build_model(self):
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError("K phải là số nguyên dương.")
        self.X_train = None
        self.y_train = None
        self.ready = True

    def fit(self, X_train, y_train):
        if not self.ready:
            raise ValueError("Hãy gọi build_model() trước fit().")
        if self.k > X_train.shape[0]:
            raise ValueError("K không được vượt quá số mẫu train.")
        if X_train.shape[0] != len(y_train):
            raise ValueError("Số mẫu và số nhãn phải bằng nhau.")
        self.X_train = X_train.toarray()
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        if self.X_train is None:
            raise ValueError("Hãy gọi fit() trước predict().")
        predictions = []
        for row in range(X_test.shape[0]):
            email = X_test[row].toarray().ravel()
            distances = euclidean_distances(self.X_train, email)
            # Khoảng cách bằng nhau: giữ thứ tự mẫu trong train.
            neighbor_indices = nearest_indices(distances, self.k)
            nearest_labels = self.y_train[neighbor_indices]
            labels, votes = np.unique(nearest_labels, return_counts=True)
            # Nếu hòa phiếu, chọn lớp đầu trong thứ tự nhãn (ham trước spam).
            best_label = majority_vote(nearest_labels)
            predictions.append(best_label)
        return np.array(predictions)

    def predict_spam_score(self, X_test):
        if self.X_train is None:
            raise ValueError("Hãy gọi fit() trước khi tính điểm spam.")
        scores = []
        for row in range(X_test.shape[0]):
            email = X_test[row].toarray().ravel()
            distances = euclidean_distances(self.X_train, email)
            nearest = nearest_indices(distances, self.k)
            # Số hàng xóm spam chia cho K. Với K=5, điểm chỉ là 0, 0.2, ..., 1.
            scores.append(np.mean(self.y_train[nearest] == "spam"))
        return np.array(scores)


class NaiveBayesSpamClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.prior = None
        self.word_probability = None

    def build_model(self):
        if self.alpha <= 0:
            raise ValueError("Alpha phải lớn hơn 0.")
        self.classes = None
        self.prior = {}
        self.word_probability = {}

    def fit(self, X_train, y_train):
        if self.prior is None:
            raise ValueError("Hãy gọi build_model() trước fit().")
        y_train = np.array(y_train)
        self.classes = np.unique(y_train)
        number_of_emails, number_of_words = X_train.shape

        for label in self.classes:
            # Chọn các email thuộc lớp đang xét.
            X_class = X_train[y_train == label]
            self.prior[label] = class_prior(X_class.shape[0], number_of_emails)

            # Tổng trọng số TF-IDF của mỗi từ trong lớp.
            word_totals = np.asarray(X_class.sum(axis=0)).ravel()
            self.word_probability[label] = laplace_probabilities(word_totals, self.alpha)

    def predict(self, X_test):
        if self.classes is None:
            raise ValueError("Hãy gọi fit() trước predict().")
        predictions = []
        for row in range(X_test.shape[0]):
            email = X_test[row]
            scores = []
            for label in self.classes:
                # Tổng x[j] * log P(từ j | lớp), cộng log xác suất lớp.
                score = log_score(email, self.prior[label], self.word_probability[label])
                scores.append(score)
            # argmax theo thứ tự classes. Hòa điểm thì lớp đứng trước, ham trước spam.
            best_class = np.argmax(scores)
            predictions.append(self.classes[best_class])
        return np.array(predictions)

    def predict_spam_score(self, X_test):
        if self.classes is None:
            raise ValueError("Hãy gọi fit() trước khi tính điểm spam.")
        spam_index = np.flatnonzero(self.classes == "spam")[0]
        probabilities = []
        for row in range(X_test.shape[0]):
            scores = np.array([
                log_score(X_test[row], self.prior[label], self.word_probability[label])
                for label in self.classes
            ])
            # Thành phần spam sau khi đưa điểm log về trọng số tổng bằng 1.
            probabilities.append(normalized_scores(scores)[spam_index])
        return np.array(probabilities)


