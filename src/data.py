import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

from config import config


def get_idx_tag(path_file: str) -> Dict:
    """
    Return a dictionnary composed of indices and tags

    Args :
        path_file (str) : location of file containing tags

    Returns:
        None
    """
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


def preprocess(df: pd.DataFrame, lower: bool, stem: bool, min_freq: int) -> pd.DataFrame:
    """
    Preprocess the data.

    Args :
        df (pd.DataFrame)   : Pandas DataFrame with data.
        lower (bool)        : whether to lowercase the text.
        stem (bool)         : whether to stem the text.
        min_freq (int)      : minimum frequence of data points a label must have.

    Returns :
        pd.DataFrame : Preprocessed Data Frame
    """
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="label", oos_label="other"
    )  # replace OOS labels
    df = replace_minority_labels(
        df=df, label_col="label", min_freq=min_freq, min_label="other"
    )  # replace labels below min freq

    return df


def clean_text(
    text: str, lower: bool = False, stem: bool = False, stopwords=config.STOPWORDS
) -> str:
    """
    Clean raw text.

    Args:
        text (str)              : raw text to clean.
        lower (bool, optional)  : whether to lowercase the text. Defaults to False.
        stem (bool, optional)   : whether to stem the text. Defaults to False.
        stopwords (optional)    : specific words to get rid of. Defaults to config.STOPWORDS.

    Returns:
        str: cleaned text.
    """
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
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


def replace_oos_labels(
    df: pd.DataFrame,
    labels: List[str],
    label_col: str = "label",
    oos_label: str = "other",
) -> pd.DataFrame:
    """
    Replace out of scope (oos) labels.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        labels (List): list of accepted labels.
        label_col (str): name of the dataframe column that has the labels. Defaults to "label".
        oos_label (str, optional): name of the new label for OOS labels. Defaults to "other".

    Returns:
        pd.DataFrame: Dataframe with replaced OOS labels.
    """
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df


def replace_minority_labels(
    df: pd.DataFrame, min_freq: int, label_col: str = "label", min_label: str = "other"
) -> pd.DataFrame:
    """
    Replace tags below min_freq

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        min_freq (int): minimum frequence of data points a label must have.
        label_col (str, optional): name of the dataframe column that has the labels. Defaults to "label".
        min_label (str, optional): name of the new label to replace minority labels. Defaults to "other".

    Returns:
        pd.DataFrame: Dataframe with replaced minority labels.
    """
    tags = Counter(df[label_col].values)
    tags_above_freq = Counter(tag for tag in tags.elements() if (tags[tag] >= min_freq))
    df[label_col] = df[label_col].apply(lambda tag: tag if tag in tags_above_freq else min_label)

    return df


class LabelEncoder:
    """
    Encode labels into unique indices.

    ```python
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)
    ```
    """

    def __init__(self, tag_to_idx: Dict = {}) -> None:
        """
        Initialize the label encoder.

        Args:
            tag_to_idx (Dict, optional): mapping between classes and unique indices. Defaults to {}.
        """
        self.tag_to_idx = {} or tag_to_idx
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}
        self.tags = list(self.tag_to_idx.keys())
        self.indices = list(self.idx_to_tag.keys())

    def __len__(self) -> int:
        return len(self.tags)

    def __str__(self) -> str:
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y: List):
        """
        Fit a list of labels to the encoder.

        Args:
            y (List): raw labels.

        Returns:
            Fitted LabelEncoder instance.
        """
        tags = np.unique(y)
        for i, tag in enumerate(tags):
            self.tag_to_idx[tag] = i
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}
        self.tags = list(self.tag_to_idx.keys())
        self.indices = list(self.idx_to_tag.keys())

        return self

    def encode(self, y: List) -> np.ndarray:
        """
        Encode raw labels

        Args:
            y (List): raw labels

        Returns:
            np.ndarray: encoded labels as indices.
        """
        encoded = np.zeros((len(y)), dtype=int)
        for i, tag in enumerate(y):
            encoded[i] = self.tag_to_idx[tag]
        return encoded

    def decode(self, y: List) -> List:
        """
        Decode a list of indices labels

        Args:
            y (List): indices to decode

        Returns:
            List: decoded tags
        """
        decoded = []
        for idx in y:
            decoded.append(self.idx_to_tag[idx])
        return decoded

    def save(self, file_path: str) -> None:
        """
        Save mapping to JSON file.

        Args:
            file_path (str): filepath to save to.
        """
        with open(file_path, "w") as f:
            contents = {"tag_to_idx": self.tag_to_idx}
            json.dump(contents, f, indent=4, sort_keys=False)

    @classmethod
    def load(cls, file_path: str):
        """
        Load instance of LabelEncoder from file.

        Args:
            file_path (str): JSON filepath to load from.

        Returns:
            LabelEncoder instance.
        """
        with open(file_path, "r") as f:
            kwargs = json.load(fp=f)
        return cls(**kwargs)


def get_data_splits(X: pd.Series, y: np.ndarray, train_size: float = 0.7) -> Tuple:
    """
    Generate balanced data splits.

    Args:
        X (pd.Series): input features
        y (np.ndarray): encoded labels
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.

    Returns:
        Tuple: data splits as Numpy arrays.
    """
    # Stratify ensure that each data split has similar class distributions
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)

    return X_train, X_val, X_test, y_train, y_val, y_test
