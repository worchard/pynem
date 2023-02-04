import numpy as np
import pickle
from pathlib import Path

HERE = Path(__file__).parent

def toy5a10e2r() -> dict:
    """
    Toy simulated dataset containing 5 actions, with 2 replicates each, and 10 effects. Dictionary also
    indicates the true alpha and beta values used when generating the data.
    """
    filename = HERE / 'toy5a10e2r.pkl'
    with open(filename, 'rb') as file:
        out = pickle.load(file)
    return out