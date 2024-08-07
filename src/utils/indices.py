# Typing hints:
from typing import (
    Tuple, 
    Iterator,
    Iterable,
    Generator,
    TypeVar
)
_T = TypeVar("_T")

# For smart iterations:
import itertools


# ============================================================================ #
#                               Global Functions                               #
# ============================================================================ #

def all_possible_indices(dimensions: Tuple[int, ...]) -> Iterator :
    """iterate_permutation 

    Loop over all possible values of multiple indices within given dimensions

    Args:
        dimensions (typ.List[int]): The Shape of tensor for which we want to iterate over all possible indices.

    Returns:
        _type_: _description_
    """
    return itertools.product(*(range(dim) for dim in dimensions) )


def indices_gen() -> Generator[int, None, None]:
    i = 0
    while True:
        yield i
        i += 1

def index_of_first_appearance(it:Iterable[_T], item:_T) -> int:
    for i, val in enumerate(it):
        if val==item:
            return i
    raise ValueError("Not found")


def valid_permutation_list(permutation_lits:list[int], must_permute:bool=False) -> bool:
    for prev_i, next_i in enumerate(permutation_lits):
        if permutation_lits.count(next_i) != 1:
            return False
        if must_permute and prev_i == next_i:
            return False
    return True