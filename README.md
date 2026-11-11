# Lab 01 — Phân loại email spam

Dự án phân loại email thành **ham** (thư bình thường) và **spam** (thư rác) bằng KNN và Multinomial Naive Bayes. Hai thuật toán được tự cài bằng NumPy/SciPy; TF-IDF và chỉ số đánh giá dùng scikit-learn.

## 1. Cài đặt và chạy

Mở PowerShell tại thư mục dự án, dùng Python 3.10 trở lên:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/pipeline.py
```

Pipeline thực hiện EDA, chia và xử lý dữ liệu, huấn luyện, đánh giá, vẽ biểu đồ, xuất báo cáo và ghi nhật ký. Chạy lại sẽ cập nhật dữ liệu xử lý, model và kết quả theo cấu hình hiện tại.

Các lệnh sử dụng khác:

```powershell

# Nhập email để dự đoán bằng các model đã lưu
.venv\Scripts\python.exe src/predict.py

# Mở giao diện trên trình duyệt
.venv\Scripts\python.exe -m streamlit run app.py

# Xuất lại biểu đồ và báo cáo từ metrics đã lưu
.venv\Scripts\python.exe src/visualize.py
.venv\Scripts\python.exe src/report.py
```

Có thể mở `notebooks/lab01.ipynb`, chọn kernel `.venv` rồi **Run All** để chạy toàn bộ pipeline và xem bảng chỉ số cùng biểu đồ.

## 2. Cấu trúc chính

Present_and_Lab1_ML_G5/  
│  
├── README.md                  Giới thiệu và hướng dẫn chạy  
├── requirements.txt           Thư viện cần cài  
├── config.yaml                Cấu hình dữ liệu và mô hình  
├── app.py                     Giao diện phân loại email  
│  
├── data/  
│   ├── raw/                   Dữ liệu gốc  
│   └── processed/             Train/Test sau xử lý  
│  
├── src/  
│   ├── pipeline.py                      Chạy toàn bộ quy trình  
│   ├── eda.py                           Khám phá dữ liệu  
│   ├── preprocess.py                     Chia tập, làm sạch, loại trùng, TF-IDF  
│   ├── model.py                         Class KNN và Naive Bayes; lưu/nạp JSON  
│   ├── train.py                          Huấn luyện và lưu mô hình  
│   ├── evaluate.py                       Tính chỉ số, confusion matrix và ROC  
│   ├── predict.py                        Dự đoán email mới  
│   ├── visualize.py                      Vẽ biểu đồ so sánh  
│   ├── report.py                         Xuất báo cáo  
│   └── tracking.py                       Ghi lịch sử thực nghiệm  
│  
├── models/                              TF-IDF và hai mô hình đã lưu  
├── notebooks/  
│   └── lab01.ipynb                       Chạy pipeline và xem kết quả  
│  
├── reports/                              Metrics, ma trận nhầm lẫn, báo cáo  
│  
├── docs/  
│   ├── Lab01_descript.md                Quy trình, công thức và kết quả  
│   ├── ReadmeForParameters.md           Giải thích tham số  
│   └── images/                         Các biểu đồ  
│  
├── logs/                               Nhật ký chạy  
└── progress_results/                   Trạng thái từng lần chạy

## 3. Dữ liệu và mô hình

Dữ liệu gốc có **5.171 email**: 3.672 ham và 1.499 spam. Đặc trưng lấy từ `text`, nhãn lấy từ `label`; không đưa `label_num` vào đặc trưng.

Pipeline chia Train/Test 80/20 trước, rồi làm sạch và loại trùng trong từng tập. Nội dung sạch xuất hiện ở cả hai tập được giữ ở Train và bỏ bản trùng khỏi Test. Sau xử lý còn 3.949 mẫu Train và 939 mẫu Test (651 ham, 288 spam); không còn nội dung sạch trùng giữa hai tập. Tỷ lệ 80/20 chỉ áp dụng lúc chia ban đầu. TF-IDF chỉ học từ vựng/IDF trên Train; Test và email mới dùng trạng thái đã học.

Mỗi thuật toán thực hiện `__init__() → build_model() → fit() → predict()`. `build_model()` kiểm tra cấu hình và xóa trạng thái đã học; `fit()` nhận Train. Khi đọc JSON, chương trình khôi phục trạng thái và dùng lại các phương thức của class. Bước đánh giá dùng chung điểm lớp để lấy nhãn và ROC, tránh tính khoảng cách KNN hai lần.

Các tham số mặc định: `seed=42`, `max_features=1000`, `k=5`, `alpha=1`. Xem [tham số](docs/ReadmeForParameters.md) và [quy trình, công thức](docs/Lab01_descript.md).

## 4. Kết quả trên Test

Nguồn: [metrics.csv](reports/metrics.csv). Spam là lớp dương; bảng hiển thị bốn chữ số thập phân, CSV giữ độ chính xác gốc.

| Mô hình | Accuracy | Precision | Recall / TPR | F1 | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|
| KNN | 0.9478 | 0.9107 | 0.9201 | 0.9154 | 0.0399 | 0.9804 |
| Naive Bayes | 0.9191 | 0.8397 | 0.9097 | 0.8733 | 0.0768 | 0.9796 |


KNN phù hợp hơn trên tập Test hiện tại. Mô hình báo nhầm 26 email Ham và bỏ sót 23 email Spam, so với 50 và 26 của Naive Bayes.

## 5. Đọc kết quả

- [Báo cáo bài lab](docs/Lab01_descript.md) trình bày quy trình, công thức và phân tích kết quả.

- [Báo cáo tự động](reports/summary.md): metrics và hình từ lần chạy gần nhất.
- `docs/images/`: phân bố nhãn, confusion matrix, metrics, TPR/FPR và ROC.
- `models/`: từ vựng/IDF, dữ liệu Train của KNN và tham số xác suất Naive Bayes.
- `docs/experiment_logs.csv`, `logs/`, `progress_results/`: lịch sử và trạng thái thực nghiệm.
