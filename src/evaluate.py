"""Tính Accuracy, Precision, Recall, F1 và ma trận nhầm lẫn. Kết quả ghi vào reports/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, roc_auc_score

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import predict_both, load_json

ROOT = SRC.parent
LABELS = ["ham", "spam"]


# Chấm test bằng model JSON và ghi metric cùng ma trận nhầm lẫn.
def evaluate():
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    y_true = test["label"].to_numpy()
    texts = test["clean_text"].tolist()
    predicted, scores = predict_both(texts, return_scores=True)
    y_binary = (y_true == "spam").astype(int)
    if len(set(y_binary)) != 2:
        raise ValueError("ROC cần tập Test có cả hai nhãn ham và spam")
    notes = {
        "knn": f"k={load_json('knn.json')['k']}",
        "naive_bayes": f"alpha={load_json('naive_bayes.json')['alpha']:g}",
    }
    names = {"knn": "KNN", "naive_bayes": "Naive Bayes"}
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    roc_fig, roc_axis = plt.subplots(figsize=(6, 5))
    for axis, key in zip(axes, ("knn", "naive_bayes")):
        y_pred = predicted[key]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label="spam"
        )
        matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
        tn, fp, fn, tp = matrix.ravel()
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        auc = roc_auc_score(y_binary, scores[key])
        roc_fpr, roc_tpr, _ = roc_curve(y_binary, scores[key])
        roc_axis.plot(roc_fpr, roc_tpr, label=f"{names[key]} (AUC = {auc:.4f})")
        rows.append({
            "Model": names[key],
            "Feature Type": "TF-IDF",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "TPR": tpr,
            "FPR": fpr,
            "AUC": auc,
            "Ghi chú / Hyperparameters": notes[key],
        })
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
            ROOT / "reports" / f"{key}_confusion_matrix.csv"
        )
        axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], LABELS)
        axis.set_yticks([0, 1], LABELS)
        axis.set_xlabel("Dự đoán")
        axis.set_ylabel("Nhãn thật")
        axis.set_title(names[key])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    roc_axis.plot([0, 1], [0, 1], "--", color="gray", label="Ngẫu nhiên (AUC = 0.5000)")
    roc_axis.set(xlabel="FPR", ylabel="TPR", title="ROC — Tập Test", xlim=(0, 1), ylim=(0, 1))
    roc_axis.legend(loc="lower right")
    roc_fig.tight_layout()
    roc_fig.savefig(ROOT / "docs" / "images" / "roc_curve.png", dpi=150)
    plt.close(roc_fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "reports" / "metrics.csv", index=False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "confusion_matrices.png")
    plt.close(fig)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    evaluate()