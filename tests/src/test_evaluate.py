import numpy as np
import pandas as pd
import pytest
from src import evaluate
from snorkel.slicing import PandasSFApplier


@pytest.fixture(scope="function")
def df():
    data = [
        {
            "text": "Elon Musk is selling dragon flame throwers as a new product market",
            "label": "General News | Opinion",
        },
        {"text": "First short text", "label": "other"},
        {"text": "Second short text.", "label": "other"},
    ]
    df = pd.DataFrame(data)
    return df


def test_get_slice_metrics(df: pd.DataFrame):
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 1])

    slices = PandasSFApplier([evaluate.news_market, evaluate.short_text]).apply(df)

    metrics = evaluate.get_slice_metrics(y_true=y_true, y_pred=y_pred, slices=slices)

    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        print(mask, slice_name)
        print(y_true[mask], y_pred[mask])

    assert metrics["news_market"]["precision"] == 1 / 1
    assert metrics["news_market"]["recall"] == 1 / 1
    assert metrics["short_text"]["precision"] == 1 / 2
    assert metrics["short_text"]["recall"] == 1 / 2

def test_get_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    classes = ["a", "b"]
    performance = evaluate.get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, df=None)
    assert performance["overall"]["precision"] == 2 / 4
    assert performance["overall"]["recall"] == 2 / 4
    assert performance["class"]["a"]["precision"] == 1 / 2
    assert performance["class"]["a"]["recall"] == 1 / 2
    assert performance["class"]["b"]["precision"] == 1 / 2
    assert performance["class"]["b"]["recall"] == 1 / 2
