"""Dự đoán một email nhập từ terminal bằng model JSON đã lưu."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import predict_both
from preprocess import clean_text


# Nhận một email từ dòng lệnh hoặc bàn phím, rồi in nhãn của hai model.
def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = input("Email: ").strip()
    if not text:
        raise SystemExit("Chưa có nội dung email")
    labels = predict_both([clean_text(text)])
    print(f"KNN: {labels['knn'][0]}")
    print(f"Naive Bayes: {labels['naive_bayes'][0]}")


if __name__ == "__main__":
    main()