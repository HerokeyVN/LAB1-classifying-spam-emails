"""Giao diện dự đoán sử dụng model JSON hiện có."""
import sys
import time
from pathlib import Path
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
from model import predict_both
from preprocess import clean_text

st.set_page_config(page_title="Phân loại email", page_icon="📧")
st.title("Phân loại email spam")
st.caption("Lab 01 — KNN và Multinomial Naive Bayes")
choice = st.selectbox("Mô hình", ["Cả hai", "KNN", "Naive Bayes"])
text = st.text_area("Nội dung email", height=220)
if st.button("Phân loại", type="primary"):
    cleaned = clean_text(text)
    if not cleaned:
        st.warning("Vui lòng nhập nội dung email có chữ hoặc số.")
    else:
        try:
            start = time.perf_counter()
            with st.spinner("Đang phân loại..."):
                predictions = predict_both([cleaned])
            for key, name in (("knn", "KNN"), ("naive_bayes", "Naive Bayes")):
                if choice in ("Cả hai", name):
                    st.write(f"**{name}:** {predictions[key][0]}")
            st.caption(f"Thời gian xử lý: {time.perf_counter() - start:.3f} giây")
        except FileNotFoundError:
            st.error("Chưa có đủ model. Hãy chạy python src/pipeline.py trước.")
        except Exception as error:
            st.error(f"Không thể phân loại: {error}")
