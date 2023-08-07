from typing import TypeVar, Any
import numpy as np


_T = TypeVar("_T")
_Iterable = list|np.ndarray|dict
_Numeric = TypeVar("_Numeric", float, int, complex)
_FloatOrComplex = TypeVar("_FloatOrComplex", float, complex)

def subtract(d1:dict[_T, _Numeric], d2:dict[_T, _Numeric])->dict[_T, _Numeric]:
    new_d = dict()
    for key, val1 in d1.items():
        assert key in d2.keys()
        val2 = d2[key]
        new_val = val1-val2
        new_d[key] = new_val
    return new_d