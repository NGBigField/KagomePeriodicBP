import numpy as np


def tensor_mean(tensors:list[np.ndarray]) -> np.ndarray:
    dummy = tensors[0]
    shape = dummy.shape
    dtype = dummy.dtype
    n = len(tensors)
    res = np.zeros(shape=shape, dtype=dtype)
    for tensor in tensors:
        res += tensor
    res /= n
    return res