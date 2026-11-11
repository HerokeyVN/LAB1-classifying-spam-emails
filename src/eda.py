"""Kiểm kê dữ liệu gốc và trực quan hóa phân bố nhãn."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


# Kiểm kê dữ liệu gốc và vẽ phân bố ham/spam.
def explore():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / config["data"]["raw_path"])
    report = {
        "source": config["data"]["raw_path"],
        "rows": int(len(data)),
        "columns": data.columns.tolist(),
        "missing": {column: int(data[column].isna().sum()) for column in data.columns},
        "labels": {str(name): int(count) for name, count in data["label"].value_counts().items()},
    }
    total = report["rows"]
    report["label_percentages"] = {
        label: round(count / total * 100, 2) if total else 0.0
        for label, count in report["labels"].items()
    }
    reports = ROOT / "reports"
    images = ROOT / "docs" / "images"
    reports.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    (reports / "collection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = data["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.to_numpy())
    ax.set_title("Số email theo nhãn")
    ax.set_ylabel("Số dòng")
    ax.margins(y=0.15)
    for index, value in enumerate(counts.to_numpy()):
        label = counts.index[index]
        percentage = report["label_percentages"].get(label, 0.0)
        ax.text(index, value, f"{int(value)} ({percentage:.2f}%)", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(images / "label_distribution.png")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    explore()