# Spam Email Classifier

Skeleton cho bài tập phân loại email **spam / ham (not spam)** bằng scikit-learn.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
spam-classifier train --data data/samples/emails.csv --model logistic_regression
spam-classifier evaluate --data data/samples/emails.csv --model-path artifacts/model.joblib
spam-classifier predict --model-path artifacts/model.joblib --text "You won a free prize! Click now"
```

## Dataset contract

CSV bắt buộc có cột `label` và ít nhất một trong `text`, `subject`, `body`.
`label` hỗ trợ `spam` / `ham`, `1` / `0` (không phân biệt hoa thường). Các trường
`sender` và `recipient` là tùy chọn. Khi có nhiều trường, pipeline ghép chúng thành một
văn bản trước khi vector hóa TF-IDF.

Ví dụ ở [data/samples/emails.csv](data/samples/emails.csv). Không commit email riêng tư hoặc dữ liệu
nhạy cảm vào repository.

## Layout

```text
data/
├── raw/               # Real/raw datasets (gitignored)
├── samples/           # Safe sample datasets (tracked)
│   └── emails.csv
└── processed/         # Generated data (gitignored)
src/spam_classifier/   # preprocessing, model pipeline, CLI
artifacts/             # model và metric sinh ra khi chạy
tests/                 # unit tests
docs/                  # commit convention và hướng dẫn
```

## Models

- `logistic_regression` (mặc định): baseline cân bằng tốt.
- `linear_svm`: hợp với sparse, high-dimensional TF-IDF.
- `naive_bayes`: baseline nhanh.

Lệnh `train` tạo `artifacts/model.joblib` và `artifacts/metrics.json`; `evaluate`
in classification report trên một split hold-out tái lập được. Với dữ liệu nhỏ, nên
so sánh 3 model và ưu tiên F1 của lớp spam thay vì chỉ nhìn accuracy.

## Bài thực hành Lab 1 của nhóm 5

Hướng dẫn notebook Naive Bayes và KNN tự viết: [LAB1.md](LAB1.md).
Hai notebook dùng cùng dữ liệu email Enron1, cách làm sạch, chia train/test và TF-IDF.
CSV dữ liệu gốc cần tải riêng theo hướng dẫn; kết quả đã lưu nằm trong `outputs/`.
