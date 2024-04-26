import numpy as np


def tensor_distance(t1:np.ndarray, t2:np.ndarray)->float:
    return np.linalg.norm(t1-t2)