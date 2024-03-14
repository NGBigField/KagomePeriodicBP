import itertools
from dataclasses import dataclass, fields

from typing import (
    Generator,
    TypeVar,
    Tuple,
    Iterable,
    Any
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

def iterator_over_multiple_indices(shape:Iterable[int]):
    return itertools.product(*(range(n) for n in shape))


def _meaningful_attribute(attr_name:str, attr_value:Any)->bool:
    if callable(attr_value):
        return False
    if attr_name.startswith("__"):
        return False
    return True
    

def iterate_objects_attributes(obj:Any)->Generator[Any, None, None]:
    for attr_name in dir(obj):
        if hasattr(obj, attr_name):
            attr_value = getattr(obj, attr_name)
            if not _meaningful_attribute(attr_name, attr_value):
                continue
            yield attr_value

def iterate_dataclass_attributes(obj:dataclass)->Generator[Any, None, None]:
    return (getattr(obj, field.name) for field in fields(obj))
