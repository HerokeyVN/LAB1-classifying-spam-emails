# Kiểm tra sau tái cấu trúc

Ngày: 03/11/2026.
- Các hàm trong `src/ml_scratch` chạy thành công từ raw đến demo.
- 6 notebook chạy code cells tuần tự; kết quả và ảnh được lưu vào notebook.
- Công thức Euclidean, bỏ phiếu hòa, Laplace và log score đã kiểm tra bằng ví dụ tính tay.
- SHA-256 dữ liệu raw sau di chuyển khớp bản trước.
- Chia tập, TF-IDF và chỉ số không dùng scikit-learn. Kết quả hiện tại là lần chạy trên 977 email test, không so với điểm của bản vectorizer cũ.
- JSON sau lưu/nạp dự đoán đúng cùng kết quả; thử email spam, ham, rỗng và từ ngoài từ vựng.
- So sánh kiểm tra source_row và nhãn test; train/test không trùng clean_text.
- Biểu đồ dữ liệu, split, feature, ví dụ, confusion matrix, ROC, comparison và demo đã render và kiểm tra bố cục.

## Map mục tiêu
Lab chỉ có KNN và Multinomial Naive Bayes. Mỗi dòng dưới chỉ file dùng để chứng minh mục tiêu, không thay bài giải thích trong DESIGN.

- EDA khác feature engineering: [DESIGN.md](DESIGN.md), `data/train_data/02_EDA.ipynb`, `notebooks/03_feature_engineering.ipynb`.
- Thống kê và biểu đồ theo loại biến: mục cùng tên trong DESIGN; `outputs/reports/label_distribution.csv`, `numeric_summary.csv`; `outputs/figures/data_labels.png`, `data_lengths.png`.
- Quy trình đánh giá và chặn leakage: DESIGN, `notebooks/06_evaluation_demo.ipynb`, `outputs/reports/model_comparison.csv`.
- Hợp đồng từng bước: bảng trong DESIGN; JSON/CSV/NPZ đúng đường dẫn bảng đó.
- Công thức tách khỏi fit/predict: `src/ml_scratch/formulas.py`, gọi từ `classifiers.py`; ví dụ số ở `outputs/reports/knn_toy_distances.csv` và `naive_bayes_toy_*.csv`.
- Năm nhóm hình: data (`data_labels.png`, `data_lengths.png`, `feature_idf.png`); implementation (mermaid trong DESIGN và docs thuật toán); training/testing (`training_testing.png`, `knn_test.png`, `naive_bayes_test.png`); example (`knn_example.png`, `naive_bayes_example.png`); demo (`demo.png`, `outputs/reports/demo.csv`).

Chạy lại bằng notebook 01 → 02 → 03 → 04 và 05 → 06.
