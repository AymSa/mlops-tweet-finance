import logging
import sys
from pathlib import Path

import mlflow
import nltk
from nltk.corpus import stopwords
from rich.logging import RichHandler

nltk.download("stopwords")

# Key directory locations
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
RESULT_DIR = Path(BASE_DIR, "results")
STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")
LOGS_DIR = Path(BASE_DIR, "logs")
BLOB_STORE = Path(STORES_DIR, "blob")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)

# MLflow config
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))


# Assets
TWEETS_PATH = "../datasets/twitter-finance/raw_tweets.csv"
TAGS_PATH = "../datasets/twitter-finance/tags.txt"

# Extra
STOPWORDS = stopwords.words("english")
ACCEPTED_TAGS = [
    "Company | Product News",
    "Stock Commentary",
    "Macro",
    "General News | Opinion",
    "Stock Movement",
    "M&A | Investments",
]

# Logging
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # pretty formatting for console stdout
