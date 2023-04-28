from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]


docs_packages = ["mkdocs==1.4.2", "mkdocstrings==0.21.2", "mkdocstrings-python==0.9.0"]

style_packages = ["black==23.3.0", "flake8==6.0.0", "isort==5.12.0"]

test_packages = [
    "pytest==7.3.1",
    "pytest-cov==4.0.0",
    "great-expectations==0.16.8",
    "Send2Trash==1.8.2",
]

notebooks_packages = ["cleanlab==2.3.1", "jupyter==1.0.0", "lime==0.2.0.1", "wordcloud==1.8.2.2"]

setup(
    name="FinBirdAI",
    version=0.1,
    description="Classify financial tweets.",
    author="AymSa",
    python_requires=">=3.10",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extra_requires={
        "dev": docs_packages
        + style_packages
        + test_packages
        + notebooks_packages
        + ["pre-commit==3.2.2"],
        "docs": docs_packages,
        "test": test_packages,
    },
)
