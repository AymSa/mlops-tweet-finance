from nltk.stem import PorterStemmer
import re
import numpy as np
import json 
from sklearn.model_selection import train_test_split
from collections import Counter

from config import config

def get_idx_tag(path_file):  # TODO : OPTIMIZE
    """Return a dictionnary composed of indices and tags"""
    dict_tags = dict()
    with open(path_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line != "\n":
                key_value = line.split(":")
                key = int(key_value[0].split("_")[-1][:-1])
                value = line.split(":")[1].replace("\n", "")
                dict_tags[key] = value.replace('"', "")[1:]
    return dict_tags


def idx_to_tag(serie, dict_tags):
    return serie.apply(lambda x: dict_tags[x])


def preprocess(df, lower, stem, min_freq):
    """Preprocess the data."""
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="label", oos_label="other"
    )  # replace OOS labels
    df = replace_minority_labels(
        df=df, label_col="label", min_freq=min_freq, min_label="other"
    )  # replace labels below min freq

    return df


def clean_text(text, lower=False, stem=False, stopwords=config.STOPWORDS):
    # Lower
    if lower:
        text = text.lower()

    # Remove links
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", text)

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join(
            [stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")]
        )

    return text


def replace_oos_labels(df, labels, label_col="label", oos_label="other"):
    """Replace out of scope (oos) labels."""
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df


def replace_minority_labels(df, min_freq, label_col="label", min_label="other"):
    """Replace tags below min_freq"""
    tags = Counter(df[label_col].values)
    tags_above_freq = Counter(tag for tag in tags.elements() if (tags[tag] >= min_freq))
    df[label_col] = df[label_col].apply(
        lambda tag: tag if tag in tags_above_freq else min_label
    )

    return df


class LabelEncoder:
    def __init__(self, tag_to_idx = {}) -> None:
        self.tag_to_idx = {} or tag_to_idx
        self.idx_to_tag = {v:k for k,v in self.tag_to_idx.items()}
        self.tags = list(self.tag_to_idx.keys())
        self.indices = list(self.idx_to_tag.keys())


    def __len__(self) -> int:
        return len(self.tags)

    def __str__(self) -> str:
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        tags = np.unique(y)
        for i, tag in enumerate(tags):
            self.tag_to_idx[tag] = i
        self.idx_to_tag = {v:k for k,v in self.tag_to_idx.items()}
        self.tags = list(self.tag_to_idx.keys())
        self.indices = list(self.idx_to_tag.keys())

        return self

    def encode(self, y):
        return [self.tag_to_idx[tag] for tag in y]
    
    def decode(self, y):
        return [self.idx_to_tag[idx] for idx in y] 

    def save(self, file_path):
        with open(file_path, 'w') as f:
            contents = {"tag_to_idx" : self.tag_to_idx}
            json.dump(contents, f, indent=4, sort_keys = False)

    @classmethod 
    def load(cls, file_path):
        with open(file_path, 'r') as f:
            kwargs = json.load(fp=f)
        return cls(**kwargs)
    

def get_data_splits(X, y, train_size = 0.7):
    """Generate balanced data splits."""
    # Stratify ensure that each data split has similar class distributions
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    
    return X_train, X_val, X_test, y_train, y_val, y_test