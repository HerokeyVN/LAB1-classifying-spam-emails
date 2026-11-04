# Dữ liệu email Enron1 dùng cho Lab 1

Chi tiết vai trò EDA, feature engineering và cách đọc biểu đồ nằm ở [DESIGN.md](DESIGN.md). File này mô tả nguồn, cột, số lượng và đường đi của dữ liệu.

## 1. Nguồn và phạm vi

[Spam Mails Dataset trên Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).
Theo người đăng, dữ liệu được chuyển từ các email trong thư mục spam và ham của Enron1 sang CSV.
Đây là văn bản email, thường có dòng Subject. Không có đầy đủ metadata hay đính kèm.
File local: `data/raw/spam_ham_dataset.csv`. Notebook không tải lại và không sửa file này.

## 2. Cột dữ liệu

| Cột | Loại | Vai trò |
|---|---|---|
| text | chuỗi | Nội dung thư, đầu vào sẽ được làm sạch |
| label | phân loại | ham hoặc spam, nhãn cần dự đoán |
| label_num | mã 0/1 của label | Không đưa vào model vì đã có label |
| Unnamed: 0 | chỉ mục cũ | Không phải đặc trưng |

Bước 01 ghi biên bản vào `outputs/reports/collection.json`: 5.171 dòng, bốn cột trên, không ô thiếu. Ham 3.672 (71,01%), spam 1.499 (28,99%). Đọc bằng pandas, không đếm dòng văn bản, vì nội dung thư có xuống dòng.

## 3. Làm sạch rồi mới tách tập

`data/train_data/02_EDA.ipynb` gọi `prepare()`. Hàm lowercase, bỏ HTML, thay ký tự ngoài chữ, số và khoảng trắng, gộp khoảng trắng, bỏ chuỗi rỗng và bỏ trùng đúng `clean_text`. Lần chạy này không có thư rỗng sau clean và không có trùng khác nhãn. Có 283 bản trùng chính xác bị loại. Còn 4.888 email: 3.427 ham (70,11%) và 1.461 spam (29,89%).

`word_count` và `char_count` được tính trên `clean_text` để EDA. Ham median 87 từ / 449 ký tự, spam median 91 từ / 511 ký tự, cả hai đều có thư dài hơn 20.000 ký tự. Hai lớp chồng nhau, nên độ dài không vào model. Bảng đầy đủ: `outputs/reports/numeric_summary.csv`. Bar nhãn: `outputs/figures/data_labels.png`. Histogram và boxplot độ dài: `outputs/figures/data_lengths.png`.

Output của bước này là `data/processed/clean_emails.csv` với `source_row`, `text`, `label`, `clean_text`, `word_count`, `char_count`. Bước 03 đọc đúng file đó.

## 4. Split và TF-IDF

`notebooks/03_feature_engineering.ipynb` chia 80/20, seed 42, stratify theo label, rồi mới fit TF-IDF.

| Phần | Ham | Spam | Tổng |
|---|---:|---:|---:|
| Train | 2742 | 1169 | 3911 |
| Test | 685 | 292 | 977 |

`TfidfVectorizer` tự viết, bỏ từ tiếng Anh ngắn và từ một ký tự, `max_features=1000`. IDF = log((1+N)/(1+df))+1, học trên train. Test chỉ transform. Không còn nội dung `clean_text` nào xuất hiện ở cả hai phía. Ma trận `X_train.npz` là `(3911, 1000)`, `X_test.npz` là `(977, 1000)`, cùng thứ tự dòng với `train.csv` và `test.csv`. `vocabulary.csv` và `vectorizer.json` mô tả từng cột.

KNN và Naive Bayes dùng chung bốn file này. Không fit lại TF-IDF trong notebook model. Không dùng một bảng TF-IDF fit trên toàn corpus để chấm điểm.

## 5. Giới hạn

Tiếng Anh, email lịch sử. Từ ngữ công việc Enron có thể ảnh hưởng kết quả.
Chỉ loại trùng chính xác sau làm sạch, chưa xử lý thư gần trùng hoặc cùng một chuỗi trao đổi.
Kết quả trên 977 email test được ghi trong [DESIGN.md](DESIGN.md) và `outputs/`.
