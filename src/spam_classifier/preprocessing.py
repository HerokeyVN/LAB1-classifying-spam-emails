"""Dataset validation and deterministic conversion of email fields to text."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

LABEL_MAP = {"spam": 1, "ham": 0, "not spam": 0, "1": 1, "0": 0}
TEXT_COLUMNS = ("text", "subject", "body", "sender", "recipient")


def clean_text(value: object) -> str:
    """Remove HTML and normalize whitespace; TF-IDF handles tokenization/casing."""
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combine_email_fields(frame: pd.DataFrame) -> pd.Series:
    available = [column for column in TEXT_COLUMNS if column in frame.columns]
    if not available:
        raise ValueError(f"Dataset needs one text column: {', '.join(TEXT_COLUMNS)}")
    return frame[available].apply(
        lambda row: " ".join(clean_text(value) for value in row if clean_text(value)), axis=1
    )


def normalize_labels(labels: Iterable[object]) -> pd.Series:
    normalized = pd.Series(labels).astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - set(LABEL_MAP))
    if invalid:
        raise ValueError(f"Unsupported labels: {invalid}. Use spam/ham or 1/0.")
    return normalized.map(LABEL_MAP).astype(int)


def load_labeled_dataset(path: str) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError("Dataset needs a 'label' column.")
    texts = combine_email_fields(frame)
    labels = normalize_labels(frame["label"])
    if texts.empty or texts.str.len().eq(0).all():
        raise ValueError("Dataset contains no usable email text.")
    return texts, labels
