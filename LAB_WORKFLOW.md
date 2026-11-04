# Lab 1 - Phân loại email spam

Bắt đầu tại [docs/DESIGN.md](docs/DESIGN.md): vai trò EDA và feature engineering, thống kê theo từng loại biến, hợp đồng input/output từng bước, và cách đọc chỉ số.

## Cấu trúc chính
```text
docs/
  DESIGN.md
  DATASET.md
  VALIDATION.md
  Algorithms/KNN.md
  Algorithms/naive_bayes.md
data/
  raw/spam_ham_dataset.csv
  train_data/01_collection.ipynb
  train_data/02_EDA.ipynb
  processed/                 # dữ liệu sạch, split, TF-IDF
src/ml_scratch/               # data, formulas, classifiers, json_io, evaluation
notebooks/                   # 03 feature → 04 KNN / 05 NB → 06 evaluation/demo
artifacts/                   # model JSON + TF-IDF
outputs/                     # CSV kết quả, reports/, figures/
```

## Cài đặt và chạy
Python 3.10+; từ thư mục gốc:
```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-scratch-notebook.txt
```
Chạy notebook theo số 01 → 02 → 03 → 04 và 05 → 06, chọn kernel .venv. Notebook 06 nạp JSON và dự đoán email mới. Không cần pip install package local: notebook tự tìm `src` từ vị trí trong dự án.

## Kết quả và tái lập
Mỗi bước ghi đè output tương ứng khi chạy lại. train/test giữ source_row; X_*.npz cùng thứ tự CSV. Model JSON chứa cả TF-IDF; nạp không cần train lại. Không fit TF-IDF trên test.
Collection/EDA dùng dữ liệu local, không tải mạng. Demo không tham gia test. Chia tập, TF-IDF, KNN, Naive Bayes và các chỉ số đều tự viết trong `src/ml_scratch`. Công thức nằm trong formulas.py, tách khỏi luồng fit/predict.
Không chọn K hay alpha trên test. Quy trình mới kiểm tra model sau khi nạp lại JSON. Số liệu của lần chạy nằm trong `outputs/` và `docs/DESIGN.md`.

Dữ liệu raw nằm nguyên trạng ở data/raw.

Để tái lập, đặt spam_ham_dataset.csv của bộ Enron1 tại data/raw/ rồi chạy notebook 01–06 để tạo lại dữ liệu trung gian và model JSON.
