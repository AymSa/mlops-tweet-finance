from pathlib import Path
from nltk.corpus import stopwords
import mlflow





# Key directory locations
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
RESULT_DIR = Path(BASE_DIR, "results")
STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

#MLflow config
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
