# Dữ liệu email Enron1 dùng cho Lab 1

## 1. Nguồn và phạm vi

[Spam Mails Dataset trên Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).
Theo người đăng, dữ liệu được chuyển từ các email trong thư mục spam và ham của Enron1 sang CSV.
Đây là văn bản email, thường có dòng Subject. Không có đầy đủ metadata/đính kèm.

## 2. Cột dữ liệu

| Cột | Vai trò |
|---|---|
| text | Văn bản email, đầu vào |
| label | ham hoặc spam, nhãn cần dự đoán |
| label_num | 0/1; không dùng vì đã chọn label |
| Unnamed: 0 | Chỉ mục cũ; không dùng làm đặc trưng |

File gốc có 5.171 mẫu: 3.672 ham, 1.499 spam. Đọc bằng pandas, không đếm dòng văn bản
vì nội dung thư có xuống dòng. Không sửa file gốc khi làm sạch.

## 3. Xử lý

Dùng các bước được viết trực tiếp trong notebook: bỏ missing, làm sạch bằng
lower/regex, bỏ nội dung rỗng, kiểm tra xung đột nhãn và bỏ trùng theo văn bản đã làm sạch.
Số mẫu sau làm sạch sẽ khác số mẫu gốc. Chia 80/20 với seed=42 và phân tầng nhãn.
Sau đó chỉ fit TF-IDF trên train; test dùng transform của cùng vectorizer.
`stop_words=english`, `max_features=1000`. Chỉ truyền các đặc trưng từ vào model.

## 4. Tệp xử lý

`EDA.ipynb` minh họa từng bước. Hai notebook Models dùng cùng quy trình.
Không dùng file TF-IDF toàn corpus cũ để đánh giá. CSV cũ đã chuyển vào
`archive/spam_tfidf_features_full_corpus.csv` (giữ local).

Không cần xuất bảng TF-IDF để chạy mô hình. Ma trận X_train/X_test được tạo
ngay trong notebook. Phần cuối notebook Naive Bayes lưu kết quả vào outputs/.

## 5. Giới hạn

Tiếng Anh, email lịch sử; từ ngữ công việc Enron có thể ảnh hưởng kết quả.
Chỉ loại trùng chính xác sau làm sạch, chưa xử lý các thư gần trùng hoặc cùng chuỗi trao đổi.
Kết quả đánh giá xem README và outputs/ của bản hiện tại.

## KNN dùng cùng dữ liệu
Notebook Models/KNN.ipynb đọc cùng CSV, dùng nguyên cách làm sạch, loại trùng,
seed=42, chia phân tầng 80/20 và TF-IDF 1.000 đặc trưng như Naive Bayes.
Cả hai có 3.910 mẫu train, 978 mẫu test; đã kiểm tra cùng chỉ mục và từ vựng.
Kết quả KNN lưu riêng với tiền tố knn_ trong outputs/.
