import pandas as pd
import pytest

from spam_classifier.preprocessing import combine_email_fields, normalize_labels


def test_combines_fields_and_removes_html():
    data = pd.DataFrame({"subject": ["<b>Free</b>"], "body": ["  prize now  "]})
    assert combine_email_fields(data).iloc[0] == "Free prize now"


def test_normalizes_supported_labels():
    assert normalize_labels(["ham", "SPAM", 0, 1]).tolist() == [0, 1, 0, 1]


def test_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unsupported labels"):
        normalize_labels(["maybe"])
