# Tham số và cách dùng mô hình

## 1. Cấu hình trong config.yaml

- `data.raw_path = data/raw/spam_ham_dataset.csv`: dữ liệu đầu vào.
- `split.test_size = 0.2`: tỷ lệ Test trước làm sạch.
- `split.seed = 42`: tái lập phép chia với cùng dữ liệu và môi trường.
- `tfidf.max_features = 1000`: giới hạn số đặc trưng.
- `knn.k = 5`: số láng giềng; từ 1 đến số mẫu Train.
- `naive_bayes.alpha = 1`: hệ số làm trơn; phải hữu hạn và lớn hơn 0.

## 2. Tiền xử lý

- `clean_text()`: chữ thường, bỏ HTML, thay ký tự ngoài a–z/chữ số, gộp khoảng trắng.
- `_prepare_side()`: làm sạch, bỏ email rỗng/trùng, kiểm tra nội dung trùng khác nhãn.
- `split_data()`: chia rồi làm sạch từng tập, sau đó giữ nội dung trùng giữa hai tập ở Train và bỏ bản trùng khỏi Test.
- Kiểm tra trùng theo `clean_text`; nếu nội dung chung khác nhãn thì báo lỗi. Tỷ lệ Train/Test sau xử lý có thể khác 80/20.
- `vectorize()`: TF-IDF unigram, stop words tiếng Anh, chuẩn hóa L2; chỉ fit trên Train.

## 3. Các phương thức của mỗi class

- `__init__()`: lưu k hoặc alpha.
- `build_model()`: kiểm tra cấu hình, xóa trạng thái đã học, chuẩn bị cho fit.
- `fit(X_train, y_train)`: KNN lưu dữ liệu Train; Naive Bayes học prior và likelihood.
- `predict(X)`: trả nhãn có điểm cao nhất.
- `KNNClassifier.predict_proba()`: tỷ lệ phiếu từng lớp trong k láng giềng.
- `MultinomialNaiveBayes.predict_log_scores()`: log prior + tổng TF-IDF nhân log likelihood.
- KNN dùng khoảng cách Euclid; hòa khoảng cách ưu tiên thứ tự Train. Hòa điểm/phiếu ưu tiên lớp đứng trước trong danh sách nhãn đã sắp xếp.

## 4. Lưu, nạp và dự đoán

- `save_tfidf()`, `save_knn()`, `save_nb()`: lưu trạng thái vào ba file JSON.
- `load_model()`: khôi phục class từ JSON; không học thêm từ Test.
- `transform()`: dùng từ vựng/IDF đã lưu; bỏ qua từ ngoài từ vựng.
- `predict_both()`: trả nhãn của hai mô hình; `return_scores=True` trả thêm điểm ROC từ cùng lần tính.
- Điểm ROC: KNN dùng tỷ lệ phiếu spam; Naive Bayes dùng điểm spam trừ điểm ham, không lấy trị tuyệt đối.
- `train.py` kiểm tra TF-IDF và nhãn dự đoán trước/sau lưu JSON.
- Các chỉ số hiển thị bốn chữ số thập phân; CSV giữ số gốc để tính toán.

## 5. Đổi tham số

- Chạy lại `python src/pipeline.py` để đồng bộ model chính và báo cáo.
- Nếu dò tham số, dùng validation/cross-validation trên Train; không chọn theo Test.
- Xem [báo cáo bài lab](Lab01_descript.md) để đọc công thức và giới hạn kết quả.
