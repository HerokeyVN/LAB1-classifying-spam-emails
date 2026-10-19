# Kế hoạch chi tiết

| Bước | Việc thực hiện | Hoàn thành khi |
|---|---|---|
| 1 | Đọc đề và bài giảng | Xác định bài toán, nguồn kiến thức, phương thức |
| 2 | Kiểm tra dữ liệu | Có nguồn, số mẫu, phân bố nhãn và kiểm tra thiếu/trùng |
| 3 | Làm sạch | Có hàm ngắn, bỏ HTML/dấu câu và nội dung rỗng |
| 4 | Chia tập | Train/test không trùng nội dung sau chuẩn hóa |
| 5 | TF-IDF | Chỉ fit trên train, cùng số cột cho test |
| 6 | Viết class | Init không build; gọi build_model riêng trước fit |
| 7 | Huấn luyện | Fit trên train, dự đoán test thành công |
| 8 | Đánh giá | Có Accuracy, Precision, Recall, F1 và confusion matrix |
| 9 | Demo | Email mới được làm sạch/transform và trả nhãn |
| 10 | Viết docs | Các file có liên kết, mỗi file tối đa 200 dòng |
| 11 | Kiểm tra | Notebook chạy từ đầu và lệnh README dùng được |

## Thứ tự sử dụng class
```python
model = NaiveBayesSpamClassifier()
model.build_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Tham số giữ cố định
Train/test=80/20, random_state=42, stratify theo nhãn.
TF-IDF: stop_words=english, max_features=1000.
Naive Bayes tự viết dùng alpha=1; không chỉnh tham số theo điểm test.

## Cập nhật sau mỗi công đoạn
Ghi việc làm, kiểm tra và kết quả thật vào [progress_results.md](progress_results.md).
Nếu vượt 200 dòng thì tách tài liệu theo chủ đề và bổ sung liên kết trong README.
Không ghi kết quả chưa chạy như kết quả đã kiểm chứng.

## Cập nhật thêm thuật toán KNN - dùng so sánh với naive bayes
1. Dùng cùng dữ liệu, hàm clean_text, chia tập và TF-IDF với notebook Naive Bayes.
2. Viết class KNN: init → build_model → fit → predict.
3. Cố định K=5, Euclidean, uniform; tự tính khoảng cách, sắp xếp và bỏ phiếu.
4. Chạy trên cùng test, lưu ba CSV tiền tố knn_.
5. Kiểm tra ví dụ tính tay, tính nhất quán dữ liệu và cập nhật kết quả thực tế.
