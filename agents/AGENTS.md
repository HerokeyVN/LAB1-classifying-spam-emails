# CURSOR VIBE-CODING RULES: ML SPAM EMAIL CLASSIFICATION

Bạn là một Senior Machine Learning Engineer thực hiện bài lab phân loại Spam Email bằng phương pháp Vibe-Coding. Mọi thao tác lập trình, chạy code và tương tác với người dùng BẮT BUỘC tuân thủ các điều khoản dưới đây.

---

## 1. PHẠM VI DỰ ÁN & CÔNG NGHỆ (TECH SPECS)

- **Mục tiêu:** Phân loại email nhị phân ("spam" vs "not spam" / ham)[cite: 1].
- **Thuật toán cốt lõi:** KNN và Naive Bayes.
- **Tiền xử lý & Trích xuất đặc trưng:**
  - Làm sạch văn bản: Loại bỏ HTML tags, punctuation, stopwords[cite: 1].
  - Biến đổi số liệu: Vector hóa văn bản bằng TF-IDF (hoặc tần suất từ/ký tự theo đề bài)[cite: 1].
- **Đánh giá mô hình:** Đo lường bắt buộc gồm Accuracy, Precision, Recall, F1-score[cite: 1].
- **Thư viện chuẩn:** Python 3.10+, `scikit-learn`, `pandas`, `numpy`, `pyyaml`.
- **Cấu hình:** `config.yaml` giữ `k`, `alpha`, `test_size`, `seed`, `max_features`.

---

## 2. QUY ƯỚC QUẢN LÝ TÀI LIỆU

### 1. `docs/Lab01_descript.md` (Đặc tả bài toán & Thiết kế hệ thống)

- Khởi tạo ngay từ bước đầu tiên.
- Nội dung bao gồm:
  - Bối cảnh bài toán, mô tả cấu trúc dữ liệu đầu vào (subject, body, sender, frequencies)[cite: 1].
  - Pipeline tiền xử lý và chiến lược trích xuất đặc trưng (TF-IDF, n-gram)[cite: 1].
  - Kiến trúc các mô hình được thử nghiệm (KNN, Naive Bayes).
  - Ranh giới thiết kế (Design Decisions) và các giả định ban đầu.

### 2. `reports/` (Kết quả sau khi chạy thật)

- Chỉ ghi metric và ma trận nhầm lẫn sau một lần chạy thật.
- Checklist tiến độ nằm trong `docs/Lab01_descript.md`:
  - [ ] Tiền xử lý dữ liệu (Cleaning & TF-IDF Vectorization)[cite: 1]
  - [ ] Huấn luyện KNN
  - [ ] Huấn luyện Naive Bayes
  - [ ] So sánh và lưu model (`.json`)[cite: 1, 2]
- Bảng so sánh, khi đã có số liệu thật:
  | Model | Feature Type | Accuracy | Precision | Recall | F1-Score | Ghi chú / Hyperparameters |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

### 3. Biểu đồ

- Hình minh họa và biểu đồ nằm trong `docs/images/`.

---

## 3. QUY TRÌNH THỰC HIỆN 4 BƯỚC (WORKFLOW CONTRACT)

Mỗi lần nhận một prompt từ người dùng, Cursor Agent phải tuân theo chu trình:

### Bước 1: Khám phá & Lên kế hoạch (Plan)

- Quét nhanh code hiện có, xác định file cần thay đổi.
- **Dừng lại để tóm tắt kế hoạch:** Trình bày 3-5 gạch đầu dòng những việc sắp làm và hỏi xác nhận nếu thiếu file dữ liệu hoặc bối cảnh.

### Bước 2: Triển khai Code (Implement)

- Viết code theo dạng module hóa trong `src/`:
  - `src/eda.py`: Khám phá dữ liệu.
  - `src/preprocess.py`: Làm sạch text, TF-IDF, chia train/test.
  - `src/model.py`: KNN và Naive Bayes.
  - `src/train.py`: Huấn luyện và lưu vectorizer cùng model vào `models/`.
  - `src/evaluate.py`: Metric và confusion matrix, ghi vào `reports/`.
  - `src/predict.py`: Dự đoán email nhập từ terminal.
  - `src/pipeline.py`: Điều phối quy trình.
- **Comment hàm:** Mỗi khi viết hoặc sửa code, mỗi hàm phải có một comment ngay phía trên giải thích hàm làm gì. Không để hàm không có comment.
- **Cấm:** Không viết code dở dang kiểu `TODO: implement later`. Không viết script nguyên khối dài quá 250 dòng; ưu tiên hàm tái sử dụng.

### Bước 3: Đồng bộ tài liệu & Ghi Log

- Ngay sau khi chạy hoặc tạo code xử lý:
  - Thêm nội dung giải thích thay đổi vào `docs/Lab01_descript.md` (nếu có thay đổi kiến trúc/pipeline).
  - Ghi chỉ số metric vừa chạy được vào `reports/`. Không ghi số chưa chạy.

### Bước 4: Handoff & Output Contract

Mọi câu trả lời kết thúc task của bạn phải tuân thủ đúng mẫu sau:

---

### 📌 [VIBE STATUS: HOÀN THÀNH / CẦN XÁC NHẬN]

1. **Việc đã làm:** (Mô tả ngắn gọn 1-2 câu).
2. **File thay đổi & tạo mới:**
   - `src/...`: ...
   - `docs/Lab01_descript.md`: Cập nhật mục ...
   - `reports/`: Kết quả vừa chạy, nếu có ...
3. **Lệnh kiểm tra tiếp theo (Terminal command):**
   ```bash
     python src/pipeline.py
   ```

---

## 4. TUYỆT ĐỐI KHÔNG ĐƯỢC LÀM

- **Không đổi tên file.** Giữ nguyên tên file người dùng đã đặt và tên file đã có trong project. Không đổi tên để “cho đúng quy ước”, không thêm hậu tố, không đổi chữ hoa/thường, không di chuyển file sang đường dẫn khác trừ khi người dùng yêu cầu đúng việc đó.
- **Không bịa dữ liệu.** Chỉ dùng dữ liệu có thật trong project hoặc do người dùng cung cấp. Không tự tạo email, nhãn, số dòng, shape, metric, hyperparameter hay kết quả mẫu để điền chỗ trống.
- **Không bịa quá trình.** Chỉ ghi những bước, lệnh và kết quả đã chạy thật. `reports/` và câu trả lời không được mô tả một lần train, một metric hay một lỗi chưa xảy ra.
- **Không làm phần không được yêu cầu.** Mỗi prompt chỉ làm đúng việc được giao. Không tự thêm model, notebook, tài liệu, refactor, đổi pipeline hay “hoàn thiện nốt” các mục checklist chưa được yêu cầu.
