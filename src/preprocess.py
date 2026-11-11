"""Chia tập trên text gốc, làm sạch từng phía, rồi fit TF-IDF trên train."""
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


# Đọc config.yaml ở thư mục gốc của project.
def load_config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


# Đưa một thư về chữ thường, bỏ thẻ HTML và ký tự ngoài chữ, số.
def clean_text(text):
    text = re.sub(r"<[^>]*>", " ", str(text).lower()) # bỏ thẻ HTML
    text = re.sub(r"[^a-z0-9\s]", " ", text) # bỏ ký tự ngoài chữ, số
    return " ".join(text.split()) # nối các từ thành một chuỗi


# Làm sạch một phía sau khi đã chia: bỏ thư rỗng và trùng nội dung trong phía đó.
def _prepare_side(frame):
    data = frame.copy()
    data["clean_text"] = data["text"].map(clean_text)
    data = data[data["clean_text"] != ""]
    if data.empty:
        raise ValueError("Một phía không còn thư sau khi làm sạch")
    conflict = data.groupby("clean_text")["label"].nunique()  # nhóm các thư có cùng nội dung và đếm số nhãn khác nhau
    if int(conflict.max()) > 1:
        raise ValueError("Nội dung trùng khác nhãn trong cùng một phía")
    return data.drop_duplicates("clean_text")


# Chia theo nhãn, làm sạch và giữ nội dung chung ở Train.
def split_data(raw, config):
    kwargs = dict(
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["split"]["seed"]),
    )
    train, test = train_test_split(raw, stratify=raw["label"], **kwargs) # stratify=raw["label"] để tạo tập train và test có cùng phân phối nhãn
    train, test = _prepare_side(train), _prepare_side(test)
    overlap = test["clean_text"].isin(train["clean_text"])
    train_labels = train.set_index("clean_text")["label"]
    if (test.loc[overlap, "label"] != test.loc[overlap, "clean_text"].map(train_labels)).any():
        raise ValueError("Nội dung trùng khác nhãn giữa Train và Test")
    # Train đã có bản đại diện: bỏ bản trùng ở Test, không thêm trọng số cho Train.
    test = test.loc[~overlap].copy()
    if test.empty:
        raise ValueError("Test không còn mẫu sau khi xử lý trùng với Train")
    print(f"giu noi dung trung o Train; loai {int(overlap.sum())} ban trung khoi Test")
    return train, test


# Học từ vựng/IDF trên Train và chỉ transform Test bằng cùng vectorizer.
def vectorize(train, test, config):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=int(config["tfidf"]["max_features"]), norm="l2")
    return vectorizer.fit_transform(train["clean_text"]), vectorizer.transform(test["clean_text"]), vectorizer


# Chia train/test trên text gốc, làm sạch từng phía, fit TF-IDF chỉ trên train.
def prepare():
    config = load_config()
    raw = pd.read_csv(ROOT / config["data"]["raw_path"]).dropna(subset=["text", "label"])
    if "Unnamed: 0" in raw.columns:
        raw = raw.rename(columns={"Unnamed: 0": "source_row"})
    train, test = split_data(raw, config)
    columns = [name for name in ("source_row", "text", "clean_text", "label") if name in train.columns] 
    processed = ROOT / "data" / "processed"
    train[columns].to_csv(processed / "train.csv", index=False)
    test[columns].to_csv(processed / "test.csv", index=False)
    x_train, x_test, vectorizer = vectorize(train, test, config)
    print(f"sau lam sach train {len(train)} test {len(test)}")
    return x_train, x_test, train["label"].to_numpy(), test["label"].to_numpy(), vectorizer, config



