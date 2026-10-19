# Quá trình thực hiện notebook và kết quả

## 1. Đọc dữ liệu và khám phá
Dùng Enron1 CSV: 5.171 email, 3.672 ham và 1.499 spam.
Chọn cột text/label, kiểm tra thiếu, trùng và phân bố nhãn.

## 2. Làm sạch và chia tập
Hàm clean_text viết ngay trong notebook: chữ thường, bỏ HTML/dấu câu,
gộp khoảng trắng; bỏ nội dung rỗng và trùng sau làm sạch.
Còn 4.888 mẫu; chia phân tầng 80/20, seed=42: train 3.910, test 978.

## 3. TF-IDF
Tối đa 1.000 đặc trưng, bỏ stop words tiếng Anh.
Chỉ fit trên train; test và email mới dùng transform.

## 4. Cài đặt thuật toán
Viết class đủ init → build_model → fit → predict.
Fit tự tính prior, tổng trọng số từng đặc trưng theo lớp và làm trơn alpha=1.
Predict tự tính log prior + tổng trọng số nhân log xác suất từ, chọn lớp lớn nhất.
Dùng NumPy cho tính toán; không gọi MultinomialNB trong notebook.

## 5. Đánh giá

| Accuracy | Precision spam | Recall spam | F1 spam |
|---:|---:|---:|---:|
| 94,79% | 89,00% | 94,18% | 91,51% |

Confusion matrix có hàng là nhãn thật, cột là dự đoán:

| Nhãn thật / Dự đoán | ham | spam |
|---|---:|---:|
| ham | 652 | 34 |
| spam | 17 | 275 |

## 6. Demo và kiểm chứng
- Chạy mọi ô từ đầu đến cuối; dự đoán email mới bằng vectorizer đã học.
- Đối chiếu xác suất và nhãn với bản thư viện trên cùng tập test: khớp.
- Kiểm tra ví dụ nhỏ có đáp án tính tay và reset trạng thái: đạt.
- Kết quả CSV lưu trong outputs/; không dùng email demo để tính điểm test.

## 7. Giới hạn
Một lần chia, dữ liệu email lịch sử tiếng Anh; chưa tuning.
TF-IDF là trọng số không âm thay cho số đếm từ nguyên.

## Thực hiện notebook KNN
1. Đọc cùng Enron1 CSV và làm sạch theo notebook Naive Bayes.
2. Chia cùng 3.910 train/978 test, seed=42, phân tầng nhãn.
3. TF-IDF train-only, cùng 1.000 cột và trọng số.
4. Viết class bốn phương thức, Euclidean, K=5, mỗi láng giềng một phiếu.
5. Chạy đánh giá, demo và xuất knn_metrics/predictions/confusion_matrix trong outputs/.
6. Đã kiểm tra tính tay, reset, K không hợp lệ và đối chiếu tham chiếu trên mẫu không hòa.
Đã xác nhận hai notebook dùng cùng chỉ mục train/test và ma trận train TF-IDF.

| Mô hình | Accuracy | Precision spam | Recall spam | F1 spam |
|---|---:|---:|---:|---:|
| KNN tự viết | 95.09% | 93.26% | 90.07% | 91.64% |

Kết quả baseline của một lần chia; chưa kết luận thuật toán nào luôn tốt hơn.

## So sánh và kết luận

Hai mô hình được đánh giá trên cùng 978 email test, với cùng cách tiền xử lý và TF-IDF.

| Chỉ số | KNN | Naive Bayes | Cao hơn |
|---|---:|---:|---|
| Accuracy | 95,09% | 94,79% | KNN |
| Precision spam | 93,26% | 89,00% | KNN |
| Recall spam | 90,07% | 94,18% | Naive Bayes |
| F1 spam | 91,64% | 91,51% | KNN, chênh lệch nhỏ |

KNN đạt Accuracy và Precision cao hơn Naive Bayes: dự đoán đúng tổng thể nhiều hơn
và ít nhận nhầm email thường thành spam hơn trong thực nghiệm này.
Naive Bayes có Recall cao hơn, giúp phát hiện được nhiều email spam hơn.

Nếu ưu tiên hạn chế chặn nhầm email thường, KNN phù hợp hơn trong thực nghiệm này.
Nếu ưu tiên hạn chế bỏ sót spam, Naive Bayes phù hợp hơn.
F1 của hai mô hình gần tương đương nên chưa có cơ sở khẳng định một thuật toán
vượt trội toàn diện.

Kết luận dựa trên một lần chia dữ liệu cố định, chưa chứng minh thuật toán nào
luôn tốt hơn trên các bộ dữ liệu khác.

## Mã giả của notebook
Đã trình bày đầy đủ đầu vào/đầu ra, phép gán, vòng lặp, điều kiện và quy tắc hòa.
Có cả INIT, BUILD_MODEL, FIT, PREDICT; bản tổng hợp ở docs/pseudocode.md.
Chỉ cập nhật phần trình bày, không thay code Python hoặc kết quả thực nghiệm.
