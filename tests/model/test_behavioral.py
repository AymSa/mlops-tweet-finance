from pathlib import Path

import pytest

from config import config
from src import main, predict

"""
Behavioral testing is the process of
testing input data and expected outputs
while treating the model as a black box.

invariance: Changes should not affect outputs.

directional: Change should affect outputs.

minimum functionality: Simple combination of inputs and expected outputs.


Each of these types of tests can also include adversarial tests
such as testing with common biased tokens or noisy tokens, etc.
"""


@pytest.fixture(scope="module")
def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Real Estate market is on the decline.",
            "Macro",
        ),
        (
            "Real Estate market is on the uprise.",
            "Macro",
        ),
        (
            "Real Estate market isn't on the uprise.",
            "Macro",
        ),
    ],
)
def test_inv(text, tag, artifacts):
    """invariance via verb injection (changes should not affect outputs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Real Estate market is on the decline.",
            "Macro",
        ),
        (
            "Market is on the uprise.",
            "other",
        ),
        (
            "Electrical vehicle market is on the uprise.",
            "Company | Product News",
        ),
    ],
)
def test_dir(text, tag, artifacts):
    """Directional expectations (changes with known outputs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Macro economy.",
            "Macro",
        ),
        (
            "Emerging markets stocks.",
            "Stock Commentary",
        ),
        (
            "Dandelions is a sweet love song.",
            "other",
        ),
    ],
)
def test_mft(text, tag, artifacts):
    """Minimum Functionality Tests (simple input/output pairs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag
