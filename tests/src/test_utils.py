import tempfile
from pathlib import Path

import numpy as np

from src import utils


def test_set_seed():
    utils.set_seeds()
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)

    utils.set_seeds()
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)

    assert np.array_equal(a, x)
    assert np.array_equal(b, y)


def test_save_load_dict():
    with tempfile.TemporaryDirectory() as dp:
        d = {"key_0": "val_0", "key_1": "val_1"}
        fp = Path(dp, "d.json")
        utils.save_dict(d=d, filepath=fp)
        loaded_d = utils.load_dict(filepath=fp)

        assert loaded_d == d
