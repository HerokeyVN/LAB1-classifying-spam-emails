"""Xuất Markdown và HTML từ kết quả đã lưu."""
from html import escape
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# Tạo báo cáo hiển thị metrics với bốn chữ số thập phân.
def generate_report():
    frame = pd.read_csv(ROOT / "reports" / "metrics.csv")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "TPR", "FPR", "AUC"]
    for column in metrics:
        frame[column] = frame[column].map(lambda value: f"{value:.4f}")
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    overlap = len(set(train["clean_text"]) & set(test["clean_text"]))
    split_note = f"Train: {len(train)}; Test: {len(test)}; nội dung sạch trùng giữa hai tập: {overlap}. Nội dung chung được giữ ở Train, bỏ bản trùng khỏi Test."
    columns = list(frame.columns)
    lines = ["# Kết quả phân loại email", "", "Nguồn: reports/metrics.csv; đánh giá tập Test. Không huấn luyện lại khi xuất báo cáo.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines[2] += " " + split_note
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines += ["", "TPR bằng Recall; FPR là tỷ lệ email Ham bị báo nhầm là Spam. AUC đo khả năng xếp hạng trên các ngưỡng; xem xét dữ liệu trùng giữa Train và Test khi diễn giải kết quả.", "", "![So sánh metrics](../docs/images/metrics_comparison.png)", "", "![TPR và FPR](../docs/images/tpr_fpr_comparison.png)", "", "![Đường ROC](../docs/images/roc_curve.png)"]
    markdown = ROOT / "reports" / "summary.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html = '<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kết quả Lab 01</title><body><h1>Kết quả phân loại email</h1><p>' + escape(lines[2]) + '</p>' + frame.to_html(index=False) + '<p>TPR bằng Recall; FPR là tỷ lệ Ham bị báo nhầm. AUC đánh giá xếp hạng trên các ngưỡng; cần kiểm tra dữ liệu trùng giữa Train và Test.</p>' + ''.join(f'<img src="../docs/images/{name}" alt="{alt}" style="max-width:100%">' for name, alt in [("metrics_comparison.png", "So sánh metrics"), ("tpr_fpr_comparison.png", "TPR và FPR"), ("roc_curve.png", "Đường ROC")]) + '</body></html>'
    (ROOT / "reports" / "summary.html").write_text(html, encoding="utf-8")
    return markdown


if __name__ == "__main__":
    print(generate_report())
