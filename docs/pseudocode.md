# Mã giả thuật toán

Quy ước: RỖNG nghĩa là None/chưa có trạng thái, khác bảng rỗng {}.
Chỉ số bắt đầu từ 0; V là số đặc trưng TF-IDF. FIT không trả nhãn dự đoán.
Mã giả trình bày vòng lặp từng phần tử; code NumPy gộp một số vòng lặp thành phép tính mảng.

## KNN

```text
ĐẦU VÀO: X_train (N × V), y_train (N nhãn), X_test (M × V), K
ĐẦU RA: y_pred (M nhãn)

THỦ TỤC INIT(K):
    k ← K
    ready ← SAI
    X_lưu ← RỖNG; y_lưu ← RỖNG

THỦ TỤC BUILD_MODEL():
    NẾU k không phải số nguyên HOẶC k < 1: BÁO LỖI, DỪNG
    X_lưu ← RỖNG; y_lưu ← RỖNG
    ready ← ĐÚNG

THỦ TỤC FIT(X_train, y_train):
    NẾU ready = SAI: BÁO LỖI, DỪNG
    N ← số dòng X_train
    NẾU k > N HOẶC số nhãn y_train ≠ N: BÁO LỖI, DỪNG
    X_lưu ← mảng các vector trong X_train
    y_lưu ← mảng các nhãn trong y_train

HÀM PREDICT(X_test):
    NẾU X_lưu = RỖNG: BÁO LỖI, DỪNG
    y_pred ← danh sách rỗng
    VỚI MỖI vector x TRONG X_test:
        distances ← danh sách rỗng
        VỚI i ← 0 ĐẾN số dòng X_lưu − 1:
            sum_squared ← 0
            VỚI j ← 0 ĐẾN số cột X_lưu − 1:
                sum_squared ← sum_squared + (x[j] − X_lưu[i,j])²
            THÊM (sqrt(sum_squared), i) VÀO distances
        SẮP XẾP distances tăng dần theo khoảng cách, rồi theo i
        neighbors ← k phần tử đầu của distances
        votes ← bảng có giá trị mặc định 0
        VỚI MỖI (distance, i) TRONG neighbors:
            label ← y_lưu[i]
            votes[label] ← votes[label] + 1
        labels ← các nhãn trong votes, sắp xếp tăng dần
        best ← labels[0]
        VỚI MỖI label TRONG labels:
            NẾU votes[label] > votes[best]: best ← label
        THÊM best VÀO y_pred
    TRẢ VỀ y_pred
```

Hòa khoảng cách ưu tiên chỉ số train nhỏ hơn; hòa phiếu chọn nhãn alphabet trước.

## Multinomial Naive Bayes

```text
ĐẦU VÀO: X_train (N × V), y_train (N nhãn), X_test (M × V), alpha > 0
ĐẦU RA: y_pred (M nhãn)

THỦ TỤC INIT(alpha):
    lưu alpha
    classes ← RỖNG; prior ← RỖNG; probability ← RỖNG

THỦ TỤC BUILD_MODEL():
    NẾU alpha ≤ 0: BÁO LỖI, DỪNG
    classes ← RỖNG
    prior ← bảng rỗng
    probability ← bảng rỗng

THỦ TỤC FIT(X_train, y_train):
    NẾU prior = RỖNG: BÁO LỖI, DỪNG
    N ← số dòng X_train
    V ← số cột X_train
    classes ← các nhãn khác nhau trong y_train, sắp xếp tăng dần
    VỚI MỖI lớp c TRONG classes:
        indices ← các i thỏa y_train[i] = c
        prior[c] ← số phần tử indices / N
        totals ← mảng V phần tử bằng 0
        VỚI MỖI i TRONG indices:
            VỚI j ← 0 ĐẾN V − 1:
                totals[j] ← totals[j] + X_train[i,j]
        denominator ← tổng(totals) + alpha × V
        VỚI j ← 0 ĐẾN V − 1:
            probability[c,j] ← (totals[j] + alpha) / denominator

HÀM PREDICT(X_test):
    NẾU classes = RỖNG: BÁO LỖI, DỪNG
    y_pred ← danh sách rỗng
    VỚI MỖI vector x TRONG X_test:
        best_label ← classes[0]
        best_score ← −∞
        VỚI MỖI lớp c TRONG classes:
            score ← log(prior[c])
            VỚI j ← 0 ĐẾN số đặc trưng − 1:
                score ← score + x[j] × log(probability[c,j])
            NẾU score > best_score:
                best_score ← score
                best_label ← c
        THÊM best_label VÀO y_pred
    TRẢ VỀ y_pred
```

X_train chứa trọng số TF-IDF không âm. Totals là tổng trọng số, không phải số đếm từ nguyên.
Log dùng để tránh nhân các số rất nhỏ; điểm không phải hậu nghiệm đã chuẩn hóa.
Hòa điểm giữ lớp đầu trong classes (ham trước spam).
