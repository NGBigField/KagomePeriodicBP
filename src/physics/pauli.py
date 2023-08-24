import numpy as np
from typing import Generator, overload


x = np.matrix(
    [[0, 1],
     [1, 0]]
)
y = 1j*np.matrix(
    [[0, -1], 
     [1,  0]]
)
z = np.matrix(
    [[1,  0],
     [0, -1]]
)
id = np.matrix(
    [[1, 0],
     [0, 1]]
)


@overload
def all_paulis(with_names:bool=False)->Generator[np.matrix, None, None]:...
@overload
def all_paulis(with_names:bool=True)->Generator[tuple[np.matrix, str], None, None]:...
#
def all_paulis(with_names:bool=False)->Generator[tuple[np.matrix, str]|np.matrix, None, None]:
    for op, name in [(x,'x'), (y,'y'), (z,'z')]:
        if with_names:
            yield op, name
        else:
            yield op
