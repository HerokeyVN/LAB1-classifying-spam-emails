# Mô tả dự án

## Bài toán
Phân loại email tiếng Anh thành `ham` (thư thường) hoặc `spam` (thư rác).
Dữ liệu: Enron1 CSV, 5.171 email trước xử lý; dùng `text` và `label`.
Nguồn: https://www.kaggle.com/datasets/venky73/spam-mails-dataset

## Mục tiêu
- Hiểu quy trình học có giám sát: chuẩn bị → huấn luyện → dự đoán → đánh giá.
- Áp dụng Multinomial Naive Bayes để phân loại văn bản.
- Viết bằng ngôn ngữ Python.
- Có mã giả, notebook giải thích và demo email mới.

## Phạm vi
Chỉ dùng văn bản email; không xử lý đính kèm hay kết nối hộp thư thật.
Tự viết Naive Bayes bằng NumPy; pandas đọc dữ liệu, scikit-learn tạo TF-IDF/chia tập/đánh giá.
Không thêm Pipeline, kế thừa, lưu/nạp model hay tối ưu tham số.

## Căn cứ kiến thức
- Bài giảng `Machine_Learning_Full_Course.pdf`: Là file gom toàn bộ nội dung slide các chương
- Bài giảng `Machine_Learning_Full_Course.pdf`: trang 34–38 về quy trình,
  trang 37–42 về chỉ số, trang 58–60 về Naive Bayes.
- Đề `ML project_ Classifying Spam Emails .pdf`: làm sạch văn bản, TF-IDF,
  huấn luyện, đánh giá và phân loại email mới.
- Kiến thức được yêu cầu ghi chép trên lớp: init → build_model → fit → predict; docs ngắn gọn, tối đa 200 dòng/file, lưu quá trình ở progress/results.

## Sản phẩm
Notebook thực hành, class, script train/demo, dữ liệu gốc, kết quả và tài liệu.
Hướng dẫn chạy: [README](../README.md).

## Bổ sung KNN
Cùng bài toán spam email và dữ liệu với Naive Bayes.
Thêm Models/KNN.ipynb theo bài giảng trang 45–48: Euclidean, K láng giềng, bỏ phiếu.
Tự viết fit/predict bằng NumPy, giữ đủ bốn phương thức; không tuning.
