import itertools

from typing import (
    Any,
    List,
    Sequence,
    Generator,
    Iterator,
    TypeVar,
    Tuple,
    Generic,
)

_T = TypeVar("_T")


def with_alternating_flag(it:Generator[_T, None, None], first_flag_value:bool=True)->Generator[Tuple[bool, _T], None, None]:
    flag = first_flag_value
    for item in it:
        yield flag, item
        flag = not flag

def with_first_indicator(it:Generator[_T, None, None], first_flag_value:bool=True)->Generator[Tuple[bool, _T], None, None]:
    is_first : bool = True
    for item in it:
        yield is_first, item
        is_first = False

def swap_first_elements(it:Generator[_T, None, None]) -> Generator[_T, None, None]:    
    first_val : _T 
    for i, item in enumerate(it):
        if i==0:
            first_val = item
            yield next(it)
        elif i==1:
            yield first_val
        else:
            yield item