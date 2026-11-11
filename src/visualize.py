"""Vẽ so sánh từ metrics đã có, không huấn luyện lại."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# Đọc metrics thật và lưu biểu đồ vào docs/images.
def visualize():
    frame = pd.read_csv(ROOT / "reports" / "metrics.csv")
    axis = frame.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot.bar(ylim=(0, 1), rot=0)
    axis.set_ylabel("Score")
    axis.set_title("KNN và Multinomial Naive Bayes — Test")
    axis.figure.tight_layout()
    target = ROOT / "docs" / "images" / "metrics_comparison.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    axis.figure.savefig(target, dpi=150)
    plt.close(axis.figure)
    rates = frame.set_index("Model")[["TPR", "FPR"]].plot.bar(ylim=(0, 1), rot=0)
    rates.set_ylabel("Tỷ lệ")
    rates.set_title("TPR và FPR — Tập Test")
    rates.figure.tight_layout()
    rates.figure.savefig(ROOT / "docs" / "images" / "tpr_fpr_comparison.png", dpi=150)
    plt.close(rates.figure)
    return target


if __name__ == "__main__":
    print(visualize())
