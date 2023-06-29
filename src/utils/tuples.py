import numpy as np

from typing import (
    Tuple,
    TypeVar,
    Any,
    Callable
)

import operator

_NumericType = TypeVar("_NumericType", float, complex, int)

def angle(t1:Tuple[_NumericType,...], t2:Tuple[_NumericType,...])->float:
    assert len(t1)==len(t2)==2
    dx, dy = sub(t2, t1)
    theta = np.angle(dx + 1j*dy) % (2*np.pi)
    return theta.item() # Convert to python native type

def _apply_pairwise(func:Callable[[_NumericType,_NumericType], _NumericType], t1:Tuple[_NumericType,...], t2:Tuple[_NumericType,...])->Tuple[_NumericType,...]:
    list_ = [func(v1, v2) for v1, v2 in zip(t1, t2, strict=True)]
    return tuple(list_)

def sub(t1:Tuple[_NumericType,...], t2:Tuple[_NumericType,...])->Tuple[_NumericType,...]:
    return _apply_pairwise(operator.sub, t1, t2)

def add(t1:Tuple[_NumericType,...], t2:Tuple[_NumericType,...])->Tuple[_NumericType,...]:
    return _apply_pairwise(operator.add, t1, t2)

def copy_with_replaced_val_at_index(t:tuple, i:int, val:Any) -> tuple:
    temp = [x for x in t]
    temp[i] = val
    return tuple(temp)

def equal(t1:Tuple[_NumericType,...], t2:Tuple[_NumericType,...])->bool:
    for v1, v2 in zip(t1, t2, strict=True):
        if v1!=v2:
            return False
    return True

def mean_itemwise(t1:Tuple[_NumericType,...], t2:Tuple[_NumericType,...])->Tuple[_NumericType,...]:
    l = [(v1+v2)/2 for v1, v2 in zip(t1, t2, strict=True)]
    return tuple(l)