import numpy as np
from typing import Generator, overload
import functools

x = np.matrix(
    [[0, 1],
     [1, 0]]
)
y = np.matrix(
    [[ 0, -1j], 
     [1j,   0]]
)
z = np.matrix(
    [[1,  0],
     [0, -1]]
)
id = np.matrix(
    [[1, 0],
     [0, 1]]
)


PAULI_OPERATORS : dict[str, np.matrix] = dict(
    x = x,
    y = y,
    z = z
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

@functools.cache
def by_name(name:str)->np.matrix:
    return PAULI_OPERATORS[name]