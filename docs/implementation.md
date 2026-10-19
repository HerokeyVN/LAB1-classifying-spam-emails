# Cài đặt Naive Bayes bằng công thức

## 1. Phạm vi tự viết
Class nằm trực tiếp trong `Models/Naive_Bayes_model.ipynb`.
Tự tính xác suất lớp, xác suất đặc trưng và điểm dự đoán; không gọi MultinomialNB.
NumPy chỉ dùng mảng, tổng, log và tìm vị trí lớn nhất.
Pandas đọc dữ liệu; scikit-learn chia tập, tạo TF-IDF và đánh giá.

## 2. Bốn phương thức theo ghi chép
```python
model = NaiveBayesSpamClassifier()
model.build_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

| Phương thức | Công việc |
|---|---|
| `__init__(alpha=1)` | Khai báo alpha và biến trạng thái, chưa học |
| `build_model()` | Tạo từ điển rỗng lưu xác suất, đặt lại trạng thái |
| `fit(X_train, y_train)` | Lặp qua lớp để tính prior và word_probability |
| `predict(X_test)` | Tính điểm mỗi lớp cho từng email, trả mảng nhãn |

`self` là đối tượng hiện tại; các phương thức dùng chung các biến self.
Phải build trước fit, fit trước predict. Build lại sẽ xóa trạng thái đã học.

## 3. Công thức trong fit
Gọi N là số email train, N_c là số email lớp c, V là số đặc trưng.

```text
P(c) = N_c / N
S[c,j] = tổng trọng số TF-IDF của đặc trưng j trong các email lớp c
P(j|c) = (S[c,j] + alpha) / (tổng_j S[c,j] + alpha * V)
```

`X_train[y_train == label]` chọn email thuộc lớp đang xét.
`sum(axis=0)` cộng từng cột từ; ravel đổi kết quả thành mảng một chiều.
Alpha=1 cộng một lượng vào mỗi đặc trưng để tránh xác suất bằng 0.
TF-IDF là trọng số không âm, không phải số đếm nguyên hay xác suất.

## 4. Công thức trong predict
```text
score(c) = log P(c) + tổng_j x[j] * log P(j|c)
```

`email.dot(log_probabilities)` thực hiện tổng trong công thức.
Đây là điểm log để so sánh, không phải xác suất hậu nghiệm đã chuẩn hóa.
Không cần tính mẫu số Bayes vì giống nhau cho các lớp của cùng email.
Log đổi phép nhân thành cộng, tránh tích nhiều số nhỏ về 0.
`argmax` trả vị trí điểm lớn nhất; dùng vị trí đó lấy nhãn.
Nếu hòa, chọn lớp đứng trước trong np.unique (ham trước spam).
Email không có từ nào trong từ vựng có vector 0 và được chọn theo prior.

## 5. Ví dụ tính tay
Hai email ham có vector [2,0], [1,0]; hai spam có [0,2], [0,1].
Prior mỗi lớp = 1/2, V=2, alpha=1.
Lớp ham: P(từ 1|ham)=4/5, P(từ 2|ham)=1/5; spam thì ngược lại.
Email [1,0] có điểm ham lớn hơn nên dự đoán ham.
Đây là ví dụ số minh họa, không phải dữ liệu tính điểm test.

## 6. Căn cứ kiến thức
Bài giảng trang 58: Bayes và độc lập có điều kiện; trang 59: Multinomial;
trang 60: vấn đề xác suất bằng 0. Đề Lab 1 nêu TF-IDF.
Công thức làm trơn và log là giải thích triển khai bổ sung, không gán là
công thức đã được trình bày chi tiết trong bài giảng.

## 7. Cách chạy và kiểm tra
Restart Kernel → Run All. Các ví dụ email mới ở phần demo trong notebook.
Đã đối chiếu xác suất và nhãn trên tập test với MultinomialNB, alpha=1: khớp.
Bản tham chiếu chỉ phục vụ kiểm tra ngoài notebook, không dùng khi train/dự đoán chính.
Đã kiểm tra ví dụ tính tay và việc cần fit lại sau build_model.

## Cài đặt Naive Bayes bằng công thức
Class nằm trong Models/KNN.ipynb, không import KNeighborsClassifier.
- Init lưu k=5 và trạng thái chưa build.
- Build kiểm tra K dương, đặt lại dữ liệu lưu.
- Fit kiểm tra K không vượt số mẫu rồi lưu mảng train cùng nhãn.
- Predict lặp từng email: sqrt(sum((train-email)^2)) → sắp xếp → chọn K → đếm phiếu.
NumPy chỉ tính mảng, tổng, căn, sắp xếp và đếm; không dùng model KNN thư viện.
Nếu khoảng cách bằng nhau, stable sort giữ thứ tự train; hòa phiếu chọn nhãn đứng trước.
Fit lưu ví dụ, không học xác suất như Naive Bayes. Sau build lại phải fit lại.
TF-IDF chỉ fit trên train. Demo dùng lại vectorizer và model đã học.
Bản dễ học chuyển sparse thành dense, cần thêm RAM và dự đoán chậm khi nhiều mẫu.
Ví dụ tính tay: train (0,0):ham, (1,0):ham, (4,4):spam, (5,4):spam, (4,5):spam;
query (0,1), K=3 → hai ham và một spam → ham.

## Mã giả chi tiết
Xem [pseudocode.md](pseudocode.md): INIT, BUILD_MODEL, FIT và PREDICT cho cả hai thuật toán.
Mã giả thể hiện các phép tính thành vòng lặp; code NumPy tính gộp để chạy hiệu quả.
