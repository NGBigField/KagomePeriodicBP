import numpy as np

from typing import (
    Tuple,
    TypeVar,
    Any,
    Callable
)

import operator
from typing import NamedTuple
from copy import deepcopy

_T1 = TypeVar("_T1")
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


def multiply(t:Tuple[_NumericType,...], scalar_or_t2:_NumericType|tuple[_NumericType,...])->Tuple[_NumericType,...]:
    if isinstance(scalar_or_t2, tuple):
        t2 = scalar_or_t2
    else: 
        t2 = tuple([scalar_or_t2 for _ in t])   # tuple with same length
    return _apply_pairwise(operator.mul, t, t2)


def power(t:Tuple[_NumericType,...], scalar:_NumericType)->Tuple[_NumericType,...]:
    t2 = tuple([scalar for _ in t])
    return _apply_pairwise(operator.pow, t, t2)


def dot_product(t1:Tuple[_NumericType,...], t2:Tuple[_NumericType,...])->_NumericType:
    times_vector = multiply(t1, t2)
    return sum(times_vector)


def copy_with_replaced_val_at_index(t:tuple, i:int, val:Any) -> tuple:
    temp = [x for x in t]
    temp[i] = val
    return tuple(temp)


def copy_with_replaced_val_at_key(t:NamedTuple, key:str, val:Any) -> NamedTuple:
    i = get_index_of_named_tuple_key(t=t, key=key)
    # create native tuple:
    t2 = copy_with_replaced_val_at_index(t, i, val)    
    # use constructor to create instance of the same NamedTuple as t:
    return t.__class__(*t2)  

def get_index_of_named_tuple_key(t:NamedTuple, key:str)->int:
    for i, field in enumerate(t._fields):
        if field == key:
            return i
    raise ValueError(f"Key {key!r} not found in tuple {t}")

def equal(t1:Tuple[_T1,...], t2:Tuple[_T1,...], allow_permutation:bool=False)->bool:
    if len(t1)!=len(t2):
        return False
    
    if allow_permutation:
        return _are_equal_allow_permutation(t1, t2)
    else:
        for v1, v2 in zip(t1, t2, strict=True):
            if v1!=v2:
                return False
        return True


def mean_itemwise(t1:Tuple[_NumericType,...], t2:Tuple[_NumericType,...])->Tuple[_NumericType,...]:
    l = [(v1+v2)/2 for v1, v2 in zip(t1, t2, strict=True)]
    return tuple(l)

def add_element(t:Tuple[_T1,...], element:_T1)->Tuple[_T1,...]:
    lis = list(t)
    lis.append(element)
    return tuple(lis)





def _are_equal_allow_permutation(t1:Tuple[_T1,...], t2:Tuple[_T1,...])->bool:
    l1 : list[_T1] = list(t1)
    l2 : list[_T1] = list(t2)
    while len(l1)>0:
        value = l1[0]
        if value not in l2:
            return False
        l1.remove(value)
        l2.remove(value)
    return True

