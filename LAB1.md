# Lab 1 — Phân loại email spam bằng Naive Bayes và KNN
Đọc và chạy notebook theo thứ tự cell,
không cần hiểu Pipeline, kế thừa class hay cách lưu/nạp mô hình.

## 1. Học từ đâu?

| Nội dung | Nguồn trong bài giảng `Machine_Learning_Full_Course .pdf` |
|---|---|
| Đọc dữ liệu và chia train/test | Trang PDF 34–35 |
| Tạo model, fit, predict | Trang 36 |
| Accuracy, Precision, Recall, F1, confusion matrix | Trang 37–42 |
| Naive Bayes: xác suất tiên nghiệm, có điều kiện và định lý Bayes | Trang 58 |
| Multinomial Naive Bayes cho văn bản | Trang 59–60 |

Số trang tính từ đầu PDF. Bài giảng có sử dụng và dẫn tới scikit-learn.
Làm sạch văn bản và TF-IDF là phần đề Lab 1 yêu cầu/gợi ý. Class bốn phương thức là yêu cầu đã cung cấp.

## 2. Bài toán và dữ liệu

Đầu vào: văn bản email tiếng Anh. Đầu ra: `ham` (thư thường) hoặc `spam` (thư rác).
Nguồn: [Spam Mails Dataset, Enron1](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).
Tác giả chuyển email từ các thư mục spam/ham của Enron1 sang CSV.
Dùng `Datasets/data_raw/spam_ham_dataset.csv`, hai cột `text` và `label`.
Không đưa `label_num` hay cột chỉ mục vào đầu vào. Không xử lý file đính kèm.
Đây là email lịch sử, không phải dữ liệu SMS; không bảo đảm hiệu quả trên email hiện đại/tiếng Việt.
Chi tiết: [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md).

## 3. Cấu trúc thư mục

```text
Datasets/
  data_raw/spam_ham_dataset.csv    # Dữ liệu gốc
  data_processed/EDA.ipynb         # Khám phá, chia tập, TF-IDF
docs/                             # Mô tả, cấu trúc, kế hoạch, giải thích, tiến độ
Models/
  KNN.ipynb                       # KNN tự viết trên cùng dữ liệu
  Naive_Bayes_model.ipynb          # Bài thực hành Naive Bayes theo từng bước
outputs/                          # Kết quả hiện tại
archive/                          # Bản cũ giữ local, không dùng báo cáo
requirements.txt                  # Các thư viện chính
requirements-notebook.txt         # Thêm kernel để mở notebook
```

## 4. Cài đặt và cách chạy

Dùng Python 3.10 trở lên. Mở terminal tại thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

Các thư viện chính là **pandas** để đọc/xử lý bảng và **scikit-learn** để vector hóa, chia tập và đánh giá;
**NumPy** hỗ trợ phép tính mảng. Phần Naive Bayes tự viết công thức. `ipykernel` chỉ để mở notebook.
Nếu chỉ chạy script: cài `requirements.txt` thay cho `requirements-notebook.txt`.

**Cách học chính:** mở `Models/Naive_Bayes_model.ipynb` trong VS Code, chọn kernel `.venv` rồi chạy từ cell đầu đến cuối. Không chạy cell ngẫu nhiên.
Notebook hiện rõ clean_text, class, fit, predict và đánh giá; không import code từ scripts.

**Huấn luyện và demo:** chọn Restart Kernel → Run All trong notebook.
Phần class có đủ bốn bước; phần đánh giá hiển thị các chỉ số.
Sửa danh sách `emails` trong phần dự đoán email mới rồi chạy lại ô đó để demo.
Notebook KNN lưu ba file CSV trong `outputs/`. Notebook Naive Bayes hiện hiển thị kết quả trực tiếp; các CSV Naive Bayes trong `outputs/` là kết quả đã lưu từ lần chạy trước.
Demo email mới không được dùng để tính điểm test.

## 5. Các bước cần hiểu

1. Đọc CSV, giữ văn bản và nhãn; bỏ giá trị thiếu.
2. Chuyển chữ thường, bỏ HTML/dấu câu, gộp khoảng trắng; bỏ văn bản rỗng.
3. Bỏ nội dung trùng sau làm sạch trước khi chia tập để hạn chế học lại đáp án.
4. Chia 80% train, 20% test. `random_state=42` giữ lần chia cố định;
   `stratify` giữ tỷ lệ ham/spam gần nhau.
5. TF-IDF biến văn bản thành số: `fit_transform` chỉ trên train, `transform` trên test.
   Giới hạn 1.000 từ và bỏ stop words tiếng Anh. Ma trận thưa chỉ là định dạng lưu nhiều số 0.
6. Khởi tạo class → `build_model()` → `fit(X_train, y_train)` → `predict(X_test)` → đánh giá.
7. Văn bản mới phải được làm sạch và dùng lại vectorizer vừa học, không `fit` lại.

Không dùng StandardScaler cho TF-IDF ở bài này: TF-IDF đã có chuẩn hóa mặc định và
Naive Bayes dạng Multinomial cần đầu vào không âm. Không thêm tuning, ensemble hay tối ưu nâng cao.

## 6. Thiết kế class

| Phương thức | Việc thực hiện |
|---|---|
| `__init__()` | Khai báo alpha và biến lưu xác suất |
| `build_model()` | Tạo nơi lưu xác suất; gọi lại sẽ xóa trạng thái đã học |
| `fit(X_train, y_train)` | Học thống kê và xác suất theo lớp |
| `predict(X_test)` | Trả mảng nhãn; cần gọi fit trước |

Nhóm tự viết công thức fit/predict của Naive Bayes bằng NumPy; không gọi MultinomialNB. Class nhận **ma trận số**, không nhận trực tiếp chuỗi email.

## 7. Pseudocode và lý thuyết

Mã giả đầy đủ trong hai notebook và [docs/pseudocode.md](docs/pseudocode.md).
Có đầu vào/đầu ra, INIT/BUILD/FIT/PREDICT, phép gán, vòng lặp và điều kiện.
Quy tắc hòa phiếu/hòa điểm khớp code hiện tại.
Giải thích công thức: [implementation](docs/implementation.md).

## 8. Đánh giá và giải thích

- Accuracy: tỷ lệ dự đoán đúng toàn bộ test.
- Precision spam: trong số email bị dự đoán spam, bao nhiêu là spam thật?
- Recall spam: trong số spam thật, phát hiện được bao nhiêu?
- F1: trung bình điều hòa của Precision và Recall.

Confusion matrix: hàng là nhãn thật, cột là nhãn dự đoán, thứ tự `[ham, spam]`.
`[[TN, FP], [FN, TP]]`; FP là thư thường bị chặn nhầm, FN là spam bị bỏ sót.
Xem ví dụ sai bằng `source_row` trong file dự đoán để tra dòng tương ứng trong CSV gốc.
Chỉ fit TF-IDF trên train. Kết quả một lần chia chưa phản ánh mọi dữ liệu mới.


## 9. Tham khảo

- Bài giảng được cung cấp: `Machine_Learning_Full_Course (1).pdf`, trang nêu ở mục 1.
- Đề được cung cấp: `ML project_ Classifying Spam Emails (1).pdf`, trang 1.
- [Nguồn dữ liệu Enron1 CSV](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).

## 10. Kết quả Naive Bayes

Sau làm sạch còn 4888 email; train 3910, test 978.

| Mô hình | Accuracy | Precision spam | Recall spam | F1 spam |
|---|---:|---:|---:|---:|
| Naive Bayes | 94.79% | 89.00% | 94.18% | 91.51% |
| KNN tự viết | 95.09% | 93.26% | 90.07% | 91.64% |


Không tuning để tăng điểm test.

Notebook Naive Bayes có đủ bốn phương thức: `__init__`, `build_model`, `fit`, `predict`.
Đã kiểm tra train và demo Naive Bayes; tập train/test không trùng nội dung sau làm sạch.

## 11. Docs được ghi theo yêu cầu

Mỗi file dưới 200 dòng, viết ngắn gọn; khi thay đổi phải cập nhật tiến độ/ kết quả.

| Tài liệu | Mục đích |
|---|---|
| [Description](docs/Description.md) | Bài toán, mục tiêu và phạm vi |
| [Project architecture](docs/project_architecture.md) | Cấu trúc và vai trò từng file |
| [Detail plan](docs/detail_plan.md) | Các bước thực hiện và điều kiện hoàn thành |
| [Implementation](docs/implementation.md) | Giải thích class, mã giả và bốn bước gọi |
| [Progress/results](docs/progress_results.md) | Nhật ký từng công đoạn và kết quả thực tế |

Thứ tự gọi bắt buộc:
```python
model = NaiveBayesSpamClassifier()
model.build_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## 12. Thư viện và notebook

Notebook chỉ dùng thư viện chuẩn Python, pandas, NumPy và scikit-learn.
Nếu thiếu thư viện, chạy `%pip install pandas numpy scikit-learn` trong một ô notebook,
sau đó Restart Kernel → Run All. Không cần import class từ file Python khác.
Mọi bước của bài lab được chạy trong notebook.

Chi tiết công thức và ví dụ tính tay: [docs/implementation.md](docs/implementation.md).

## KNN trên cùng dữ liệu email

Mở [Models/KNN.ipynb](Models/KNN.ipynb), chọn Restart Kernel → Run All.
Notebook độc lập, có mã giả, clean_text và class KNN tự viết bằng NumPy.
Không gọi KNeighborsClassifier trong notebook. Không cần file class hoặc script riêng.
Dùng cùng mẫu train/test, từ vựng và TF-IDF như Naive Bayes hiện tại.

```python
model = KNNSpamClassifier(k=5)
model.build_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

K=5, Euclidean, phiếu đều. Fit lưu dữ liệu; predict tính khoảng cách và đếm phiếu.
Khoảng cách bằng nhau ưu tiên thứ tự train; hòa phiếu ưu tiên nhãn theo alphabet.
Đổi ma trận sang mảng để code dễ học, dùng thêm khoảng 31 MB cho train float64;
không phù hợp dữ liệu rất lớn. Không tuning theo điểm test.
Kết quả: outputs/knn_metrics.csv, knn_predictions.csv, knn_confusion_matrix.csv.

## Dữ liệu khi clone repo

CSV email gốc không được đưa vào Git theo quy định repo. Tải dữ liệu từ nguồn Kaggle ở mục 2 và đặt tại `Datasets/data_raw/spam_ham_dataset.csv` trước khi chạy notebook. Output đã xóa khỏi notebook trước khi đăng để tránh đưa nội dung email mẫu lên Git. Các CSV trong `outputs/` chỉ chứa chỉ số, nhãn và chỉ mục dòng.
