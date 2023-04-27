import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src import data

"""
Fixture : Function that is executed before the test function
Scope :
    function: fixture is destroyed after every test. [default]
    class: fixture is destroyed after the last test in the class.
    module: fixture is destroyed after the last test in the module (script).
    package: fixture is destroyed after the last test in the package.
    session: fixture is destroyed after the last test of the session.
"""


@pytest.fixture(scope="function")
def df():
    data = [
        {"text": "a0", "label": "c0"},
        {"text": "a1", "label": "c1"},
        {"text": "a2", "label": "c1"},
        {"text": "a3", "label": "c2"},
        {"text": "a4", "label": "c2"},
        {"text": "a5", "label": "c2"},
    ]
    df = pd.DataFrame(data * 10)
    return df


def test_get_idx_tag():
    with tempfile.TemporaryDirectory() as dp:
        fp = Path(dp, "tags.txt")
        open(fp, "w").writelines('"LABEL_0": "Tag 0"')

        dict_tags = data.get_idx_tag(fp)

    assert dict_tags == {0: "Tag 0"}


def test_idx_to_tag(df):
    dict_tags = {"c0": 0, "c1": 1, "c2": 2}
    replaced_serie = data.idx_to_tag(serie=df.label, dict_tags=dict_tags)

    assert set(replaced_serie.unique()) == set(dict_tags.values())


@pytest.mark.parametrize(
    "labels, unique_labels",
    [
        ([], ["other"]),
        (["c3"], ["other"]),
        (["c0"], ["c0", "other"]),
        (["c0", "c1", "c2"], ["c0", "c1", "c2"]),
    ],
)
def test_replace_oos_labels(df: pd.DataFrame, labels: List[str], unique_labels: List[str]):
    replaced_df = data.replace_oos_labels(
        df=df.copy(), labels=labels, label_col="label", oos_label="other"
    )
    assert set(replaced_df.label.unique()) == set(unique_labels)


@pytest.mark.parametrize(
    "min_freq, unique_labels",
    [
        (0, ["c0", "c1", "c2"]),
        (10, ["c0", "c1", "c2"]),
        (20, ["other", "c1", "c2"]),
        (30, ["other", "c2"]),
        (40, ["other"]),
    ],
)
def test_replace_minority_labels(df: pd.DataFrame, min_freq: int, unique_labels: List):
    replaced_df = data.replace_minority_labels(
        df=df.copy(), min_freq=min_freq, label_col="label", min_label="other"
    )
    assert set(replaced_df.label.unique()) == set(unique_labels)


@pytest.mark.parametrize(
    "text, lower, stem, stopwords, cleaned_text",
    [
        ("Hello world", False, False, [], "Hello world"),
        ("Hello world", True, False, [], "hello world"),
    ],
)
def test_clean_text(text, lower, stem, stopwords, cleaned_text):
    assert data.clean_text(text, lower, stem, stopwords) == cleaned_text


@pytest.fixture(scope="function")
def label_encoder():
    return data.LabelEncoder()


class TestLabelEncoder:
    def test_empty_init(self, label_encoder):
        assert label_encoder.idx_to_tag == {}
        assert len(label_encoder.tags) == 0

    def test_len(self, label_encoder):
        assert len(label_encoder) == 0

    def test_str(self, label_encoder):
        assert str(label_encoder) == "<LabelEncoder(num_classes=0)>"

    def test_save_and_load(self, label_encoder):
        with tempfile.TemporaryDirectory() as dp:
            fp = Path(dp, "label_encoder.json")
            label_encoder.save(file_path=fp)
            label_encoder = label_encoder.load(file_path=fp)
            assert len(label_encoder.tags) == 0

    # TODO : FIX BUG WITH GLOBAL TEST
    # TEMPORARY FIX : RUN Training tests separately from others
    def test_fit(self, label_encoder):
        label_encoder.fit(["apple", "apple", "banana"])
        assert "apple" in label_encoder.tag_to_idx.keys()
        assert "banana" in label_encoder.tag_to_idx.keys()
        assert len(label_encoder.tags) == 2

    def test_dict_init(self, label_encoder):
        tag_to_idx = {"apple": 0, "banana": 1}
        label_encoder.__init__(tag_to_idx=tag_to_idx)
        assert label_encoder.idx_to_tag == {0: "apple", 1: "banana"}
        assert len(label_encoder.tags) == 2

    def test_encode_decode(self, label_encoder):
        tag_to_idx = {"apple": 0, "banana": 1}
        y_encoded = [0, 0, 1]
        y_decoded = ["apple", "apple", "banana"]
        label_encoder.__init__(tag_to_idx=tag_to_idx)
        label_encoder.fit(y_decoded)
        assert np.array_equal(label_encoder.encode(y_decoded), np.array(y_encoded))
        assert label_encoder.decode(y_encoded) == y_decoded


# TODO : FIX BUG WITH GLOBAL TEST
# TEMPORARY FIX : RUN Training tests separately from others
@pytest.mark.parametrize("train_size", [(0.7), (0.8), (0.65)])
def test_get_data_splits(df: pd.DataFrame, train_size: float):
    df = df.sample(frac=1).reset_index(drop=True)
    df = data.preprocess(df, lower=True, stem=False, min_freq=0)
    label_encoder = data.LabelEncoder().fit(df.label)
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
        X=df.text.to_numpy(), y=label_encoder.encode(df.label), train_size=train_size
    )
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) / float(len(df)) == pytest.approx(train_size, abs=0.05)  # x ± 0.05
    assert len(X_val) / float(len(df)) == pytest.approx((1 - train_size) / 2.0, abs=0.05)
    assert len(X_test) / float(len(df)) == pytest.approx((1 - train_size) / 2.0, abs=0.05)
