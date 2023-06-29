from tensors import Tensor
import numpy as np


def Z()->Tensor:
    data = np.array([
        [1,  0],
        [0, -1]
    ])
    return Tensor(
        data=data,
        name="Z"
    )

def I()->Tensor:
    data = np.array([
        [1,  0],
        [0,  1]
    ])
    return Tensor(
        data=data,
        name="I"
    )

