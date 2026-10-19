# Cấu trúc dự án

| Vị trí | Vai trò |
|---|---|
| `Datasets/data_raw/spam_ham_dataset.csv` | Dữ liệu email gốc, không sửa |
| `Datasets/data_processed/EDA.ipynb` | Khám phá, làm sạch, chia tập, TF-IDF |
| `Models/Naive_Bayes_model.ipynb` | Học từng bước, hiện rõ class và demo |
| `outputs/` | Chỉ số, ma trận nhầm lẫn, dự đoán hiện tại |
| `docs/` | Mô tả, thiết kế, kế hoạch, giải thích và tiến độ |
| `README.md` | Điểm bắt đầu, hướng dẫn cài và chạy |
| `DATASET_DOCUMENTATION.md` | Mô tả dữ liệu chi tiết |

## Luồng dữ liệu
CSV gốc → làm sạch/bỏ trùng → chia train/test → TF-IDF → Naive Bayes → nhãn.
TF-IDF chỉ học từ train; dùng cùng vectorizer biến đổi test và email mới.
`text` là chuỗi trước TF-IDF, `X_train/X_test` là ma trận số sau TF-IDF.

## Vai trò class
`__init__` khai báo trạng thái; `build_model` tạo từ điển lưu xác suất;
`fit` học; `predict` dự đoán. Không gọi tự động build_model trong init.
Class chỉ được định nghĩa trong notebook. Phần cuối notebook xuất kết quả CSV.
`__init__.py` trống chỉ đánh dấu package; khác với phương thức `__init__()`.

## Kết quả
- `outputs/naive_bayes_metrics.csv`: bốn chỉ số.
- `outputs/naive_bayes_confusion_matrix.csv`: nhầm lẫn theo thứ tự ham/spam.
- `outputs/naive_bayes_predictions.csv`: source_row, nhãn thật, nhãn dự đoán.
`source_row` là chỉ mục pandas (bắt đầu 0), không phải dòng vật lý của CSV.

## Mục KNN bổ sung
- Models/KNN.ipynb: toàn bộ mã giả, tiền xử lý, class, train, đánh giá và demo KNN.
- outputs/knn_metrics.csv: Accuracy, Precision, Recall, F1.
- outputs/knn_predictions.csv: source_row, nhãn thật và nhãn KNN.
- outputs/knn_confusion_matrix.csv: ma trận nhầm lẫn ham/spam.
Giữ các thư mục hiện có, không tạo script hoặc class dùng chung.

- docs/pseudocode.md: mã giả chi tiết của KNN và Naive Bayes, khớp code notebook.
