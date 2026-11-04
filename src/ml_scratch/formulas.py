"""Công thức tách riêng để đọc, tính tay và kiểm tra. Không đọc file và không train."""
import numpy as np

def euclidean_distances(train, email):
    """Khoảng cách Euclid từ một email tới từng dòng train: sqrt(sum_j (x_j - z_j)^2)."""
    # Hiệu từng chiều giữa mỗi email train và email đang xét.
    difference = train - email
    # Cộng bình phương các chiều: sum_j (x_j - z_j)^2.
    squared = np.sum(difference ** 2, axis=1)
    return np.sqrt(squared)

def nearest_indices(distances, k):
    """Chỉ số K khoảng cách nhỏ nhất. Hòa khoảng cách giữ thứ tự train vì argsort stable."""
    # Stable: hai khoảng cách bằng nhau thì dòng train đứng trước vẫn được giữ trước.
    order = np.argsort(distances, kind='stable')
    return order[:k]

def majority_vote(labels):
    """Nhãn có nhiều phiếu nhất. Hòa phiếu thì lớp đứng trước theo thứ tự chữ cái (ham trước spam)."""
    # np.unique xếp nhãn theo chữ cái, nên hòa phiếu thì ham thắng spam.
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]

def class_prior(class_count, total_count):
    """Xác suất lớp P(c) = N_c / N."""
    # N_c là số email train của lớp, N là số email train.
    return class_count / total_count

def laplace_probabilities(totals, alpha):
    """P(từ j | lớp) = (S_j + alpha) / (sum S + alpha * V). Alpha tránh xác suất 0."""
    # S_j + alpha: từ chưa xuất hiện trong lớp vẫn còn xác suất dương.
    numerator = totals + alpha
    # Cộng alpha cho cả V từ để các xác suất trong lớp cộng lại bằng 1.
    denominator = totals.sum() + alpha * len(totals)
    return numerator / denominator

def log_score(email, prior, probabilities):
    """Điểm log của một lớp: log P(c) + sum_j x_j log P(từ j | c)."""
    # log P(c) là điểm nền của lớp, chưa nhìn nội dung thư.
    class_term = np.log(prior)
    # Từ có tfidf bằng 0 không cộng thêm. Làm trên log để khỏi nhân nhiều số rất nhỏ.
    word_term = email.dot(np.log(probabilities)).item()
    return class_term + word_term

def normalized_scores(scores):
    """Chuẩn hóa điểm log thành trọng số dương tổng bằng 1. Trừ max trước exp để tránh tràn số."""
    # Trừ max trước exp để số rất âm không bị về 0 hết và exp không bị tràn.
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    return weights / weights.sum()

def smooth_idf(document_frequency, n_documents):
    """idf(t) = log((1+N)/(1+df(t))) + 1. Cộng 1 để từ xuất hiện ở mọi thư không bị xóa."""
    # Cộng 1 ở tử và mẫu để khỏi chia cho 0 và khỏi lấy log(0). N là số thư train.
    raw = np.log((1 + n_documents) / (1 + document_frequency))
    # Cộng thêm 1: từ có mặt ở mọi thư vẫn còn idf bằng 1, không bị xóa khỏi vector.
    return raw + 1

def accuracy_from_counts(tp, tn, n):
    """Accuracy = (TP + TN) / N."""
    # Đúng trên cả hai lớp, chia cho toàn bộ email test.
    return (tp + tn) / n

def precision_from_counts(tp, fp):
    """Precision = TP / (TP + FP). Mẫu số 0 thì trả 0."""
    # Trong các thư bị gọi là spam, bao nhiêu thư đúng là spam.
    return tp / (tp + fp) if tp + fp else 0.0

def recall_from_counts(tp, fn):
    """Recall = TP / (TP + FN). Mẫu số 0 thì trả 0."""
    # Trong spam thật, bắt được bao nhiêu. Đây cũng là TPR.
    return tp / (tp + fn) if tp + fn else 0.0

def f1_from_rates(precision, recall):
    """F1 = 2PR / (P + R). Mẫu số 0 thì trả 0."""
    # Trung bình điều hòa. Không cộng TN vào công thức này.
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def false_positive_rate(fp, tn):
    """FPR = FP / (FP + TN). Mẫu số 0 thì trả 0."""
    # Trong ham thật, bao nhiêu thư bị gọi nhầm là spam.
    return fp / (fp + tn) if fp + tn else 0.0

def roc_curve(y_positive, scores):
    """FPR, TPR và ngưỡng khi hạ dần score. Điểm bằng nhau lấy kết quả sau khi cộng hết nhóm đó."""
    y = np.asarray(y_positive).astype(int)
    scores = np.asarray(scores, dtype=float)
    # Score cao xếp trước. mergesort giữ thứ tự cũ khi hai score bằng nhau.
    order = np.argsort(-scores, kind='mergesort')
    y = y[order]
    scores = scores[order]
    # Hạ ngưỡng dần: mỗi email được gọi là spam thì cộng một TP hoặc một FP.
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    # Chỉ giữ điểm cuối của mỗi nhóm score bằng nhau.
    last = np.r_[np.where(scores[1:] != scores[:-1])[0], len(scores) - 1]
    tpr = tps[last] / positives if positives else np.zeros(len(last))
    fpr = fps[last] / negatives if negatives else np.zeros(len(last))
    thresholds = scores[last]
    # Thêm gốc (FPR=0, TPR=0), ngưỡng +inf: chưa gọi email nào là spam.
    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[np.inf, thresholds]

def roc_auc(fpr, tpr):
    """Diện tích dưới ROC bằng hình thang: sum (dx * (y1+y2)/2)."""
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    # Mỗi đoạn giữa hai điểm ROC là một hình thang.
    width = fpr[1:] - fpr[:-1]
    height = (tpr[1:] + tpr[:-1]) / 2
    return float(np.sum(width * height))
