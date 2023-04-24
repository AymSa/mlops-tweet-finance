import numpy as np
import random
import json
from typing import Dict


def set_seeds(seed: int = 42) -> None:
    """
    Set seeds for deterministic reproducibility.

    Args :
        seed (int, optional): number to be used as the seed. Defaults to 42.

    Returns :
        None
    """
    np.random.seed(seed)
    random.seed(seed)


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """
    Save dictionnary to a json file.

    Args :
        d (Dict): dictionnay to save
        filepath (str): location  where to save the data.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.

    Returns :
        None
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def load_dict(filepath: str) -> Dict:
    """
    Load a dictionnary from a json file.

    Args :
        filepath (str): Destination pat

    Returns :
        Dict: Loaded object
    """
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d
