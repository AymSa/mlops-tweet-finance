import numpy as np 
import random
import json 

def set_seeds(seed = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

def save_dict(d, filepath):
    """Save dict to a json file."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)

def load_dict(filepath):
    """Load a dict from a json file."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d