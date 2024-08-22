from typing import TypeVar, Any, Callable, Generator
import numpy as np
from copy import deepcopy

_T = TypeVar("_T")
_T2 = TypeVar("_T2")
_Iterable = list|np.ndarray|dict
_Numeric = TypeVar("_Numeric", float, int, complex)
_FloatOrComplex = TypeVar("_FloatOrComplex", float, complex)

def accumulate_values(d:dict[_T, _Numeric], function:Callable[[_Numeric], _Numeric]=None)->_Numeric:
    total = 0.0
    for _, val in d.items():
        if isinstance(val, dict):
            crnt = accumulate_values(val, function=function)        
        elif val is None:
            continue
        else:
            if function is None:                
                crnt = val
            else:
                crnt = function(val)
        total += crnt
    return total


def subtract(d1:dict[_T, _Numeric], d2:dict[_T, _Numeric])->dict[_T, _Numeric]:
    new_d = dict()
    for key, val1 in d1.items():
        assert key in d2.keys()
        val2 = d2[key]

        if isinstance(val1, dict) and isinstance(val2, dict):
            new_val = subtract(val1, val2)        
        elif val1 is None or val2 is None:
            new_val = None
        else:
            new_val = val1-val2

        new_d[key] = new_val
    return new_d


def same(d1:dict[_T, _T2], d2:dict[_T, _T2])->bool:
    if len(d1) != len(d2):
        return False
    
    d2_copy = deepcopy(d2)
    for key, val1 in d1.items():
        if key not in d2_copy:
            return False
        val2 = d2_copy[key]
        if val1 != val2:
            return False
        d2_copy.pop(key)

    if len(d2_copy)>0:
        return False
    
    return True
        
def statistics_along_key(dictionaries:list[dict[str, Any]], key)->tuple[float, float]:
    values = [dict_[key] for dict_ in dictionaries]
    std = np.std(values)
    mean = np.mean(values)

    return mean, std

def iterate_with_edge_indicators(d:dict[_T, _T2]|np.ndarray) -> Generator[tuple[bool, bool, _T, _T2], None, None]:
    is_first : bool = True
    is_last  : bool = False
    n = len(d)
    for i, (key, item) in enumerate(d.items()):
        if i+1==n:
            is_last = True
        
        yield is_first, is_last, key, item

        is_first = False


def pass_values_from_dict1_to_dict2_on_matching_keys(dict1:dict[_T, _T2], dict2:dict[_T, _T2]) -> None:
    for key, val in dict1.items():
        if key in dict2:
            dict2[key] = val
