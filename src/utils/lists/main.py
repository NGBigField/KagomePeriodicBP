from typing import (
    List,
    TypeVar,
    Any,
    Tuple,
    Type,
    TypeGuard,
    Generator,
    Callable,
)

from copy import deepcopy
import numpy as np

from utils import arguments

_T = TypeVar("_T")
_Iterable = list|np.ndarray|dict
_Numeric = TypeVar("_Numeric", float, int, complex)
_FloatOrComplex = TypeVar("_FloatOrComplex", float, complex)


def _identity(x:_Numeric)->_Numeric:
    return x

def _minimum_required_copies(crnt_len:int, trgt_len:int)->int:
    div, mod = divmod(trgt_len, crnt_len)
    if mod==0:
       return div
    else: 
        return div+1 

def index_by_approx_value(l:list, value:_Numeric, allowed_error:float=1e-6)->int:
    distances = [abs(v-value) if isinstance(v,float|int|complex) else np.inf for v in l]
    within_error = [1 if v<allowed_error else 0 for v in distances]
    num_within_error = sum(within_error)
    if num_within_error>1:
        raise ValueError(f"Too many values in list {l} \nare close to value {value} within an error of {allowed_error} ")
    elif num_within_error<1:
        raise ValueError(f"None of the values in list {l} \nare close to value {value} within an error of {allowed_error} ")
    else:
        return within_error.index(1)


def join_sub_lists(_lists:list[list[_T]])->list[_T]:
    res = []
    for item in _lists:
        if isinstance(item, list):
            res.extend(item)
        else:
            res.append(item)
    return res


def all_same(l:List[_Numeric]|List[np.ndarray]) -> bool:
    dummy = l[0]
    if isinstance(dummy, np.ndarray):
        _same = lambda x, y: np.all(np.equal(x, y))
    else:
        _same = lambda x, y: x==y

    for item in l:
        if not _same(dummy, item):
            return False
            
    return True


def deep_unique(l:list) -> set[_T]:
    seen_values : set[_T] = set()
    
    def _gather_elements(list_:list):        
        for element in list_:
            if isinstance(element, list):
                _gather_elements(element)
            else:
                seen_values.add(element)
    _gather_elements(l)

    return seen_values


def average(l:List[_FloatOrComplex]) -> _FloatOrComplex:
    return sum(l)/len(l)


def sum(list_:List[_Numeric], / , lambda_:Callable[[_Numeric], _Numeric]=_identity ) -> _Numeric:
    dummy = list_[0]
    if isinstance(dummy, int):
        res = int(0)
    elif isinstance(dummy, float):
        res = float(0.0)
    elif isinstance(dummy, complex):
        res = 0.0+1j*0.0
    else:
        raise TypeError(f"Not an expected type of list item {type(dummy)}")
    for item in list_:
        res += lambda_(item)
    return res  


def real(list_:List[_Numeric],/) -> List[int|float]:
    return [ np.real(v) for v in list_ ]  


def product(_list:List[_Numeric]) -> _Numeric:
    prod = 1
    for item in _list:
        prod *= item
    return prod


def all_isinstance( l:list[_T], cls: Type[_T]  ) -> TypeGuard[list[_T]]:
    for item in l:
        if not isinstance(item, cls):
            return False
    return True


def equal(l1:_Iterable, l2:_Iterable) -> bool:

    def lists_comp(l1:list|np.ndarray, l2:list|np.ndarray) -> bool:
        assert all_isinstance([l1,l2], (list, np.ndarray))
        if len(l1) != len(l2):
            return False
        for item1, item2 in zip(l1, l2):
            if equal(item1, item2):
                continue
            else:
                return False
        return True
    
    def dict_comp(d1:dict, d2:dict) -> bool:
        for key, v1 in d1.items():
            if key not in d2:
                return False
            v2 = d2[key]
            if not equal(v1, v2):
                return False
        return True

    
    if all_isinstance([l1, l2], (list, np.ndarray)):
        return lists_comp(l1, l2)
    elif all_isinstance([l1, l2], dict):
        return dict_comp(l1, l2)
    else:
        return l1==l2
    

def share_item(l1:_Iterable, l2:_Iterable) -> bool:
    for v1 in l1:
        for v2 in l2:
            if v1==v2:
                return True
    return False


def copy(l:List[_T]) -> List[_T]:
    duck = l[0]
    if hasattr(duck, "copy") and callable(duck.copy):
        return [item.copy() for item in l] 
    else:
        return [deepcopy(item) for item in l] 


def common_items(l1:list[_T], l2:list[_T]) -> list[_T]:
    s1 = set(l1) 
    s2 = set(l2) 
    return list(s1.intersection(s2))
 

def common_super_class(lis:List[Any]) -> type:
    classes = [type(x).mro() for x in lis]
    for x in classes[0]:
        if all(x in mro for mro in classes):
            return x


def iterate_with_periodic_prev_next_items(l:List[_T], skip_first:bool=False) -> Generator[Tuple[_T, _T, _T], None, None]:
    for i, (is_first, is_last, crnt) in enumerate(iterate_with_edge_indicators(l)):
        prev, next = None, None
        if is_first and skip_first: continue
        if is_first:        prev = l[-1]
        if is_last:         next = l[0]
        if prev is None:    prev = l[i-1]
        if next is None:    next = l[i+1]
        yield prev, crnt, next
        

def iterate_with_edge_indicators(l:List[_T]|np.ndarray) -> Generator[Tuple[bool, bool, _T], None, None]:
    is_first : bool = True
    is_last  : bool = False
    n = len(l)
    for i, item in enumerate(l):
        if i+1==n:
            is_last = True
        
        yield is_first, is_last, item

        is_first = False


def min_max(list_:List[_Numeric])->Tuple[_Numeric, _Numeric]:
    max_ = -np.inf
    min_ = +np.inf
    for item in list_:
        max_ = max(max_, item)        
        min_ = min(min_, item)
    return min_, max_  # type: ignore
    
def swap_items(lis:List[_T], i1:int, i2:int, copy:bool=True) -> List[_T]:
    if copy:
        lis = lis.copy()

    if i1==i2:
        return lis
    
    item1 = lis[i1]
    item2 = lis[i2]
    
    lis[i1] = item2
    lis[i2] = item1
    return lis


def random_item(lis:list[_T])->_T:
    n = len(lis)
    index = np.random.choice(n)
    return lis[index]


def shuffle(lis:list[_T])->list[_T]:
    indices = np.arange(len(lis)) 
    np.random.shuffle(indices)
    new_order = indices.tolist()
    res = rearrange(lis, order=new_order)
    return res

def repeat_list(lis:list[_T], /, num_times:int|None=None, num_items:int|None=None)->list[_T]:
    # Check inputs:
    name, _ = arguments.only_single_input_allowed(function_name="repeat_list", num_times=num_times, num_items=num_items)
    # choose based on inputs:
    match name:
        case "num_times":
            return lis*num_times
        case "num_items":
            d = _minimum_required_copies(len(lis), num_items)
            larger_list = repeat_list(lis, num_times=d)
            return larger_list[:num_items]

def convert_whole_numbers_to_int(lis:List[float|int])->List[float|int]:
    lis_copy : list[int|float] = lis.copy()
    for i, x in enumerate(lis):
        if round(x)==x:
            lis_copy[i] = int(x)
    return lis_copy


def rearrange(l:List[_T], order:List[int]) -> List[_T]:
    # all indices are different and number of indices is correct:
    assert len(set(order))==len(order)==len(l)

    ## rearrange:
    return [ l[i] for i in order ]


def is_sorted(l:list[int]|list[float])->bool:
    return all(a <= b for a, b in zip(l, l[1:]))


def repeated_items(l:list[_T]) -> Generator[tuple[_T, int], None, None]:
    """ repeated_items(l:list[_T]) -> Generator[tuple[_T, int], None, None]
    Replaces a list of repeating items, i.e., [A, A, A, B, C, C, C, C],
    with an iterator over the values and number of repetitions, i.e., [(A, 3), (B, 1), (C, 4)].
    
    Args:
        l (list[_T])

    Yields:
        Generator[tuple[_T, int], None, None]: _description_
    """

    last = l[0] # place holder
    counter = 1
    
    for is_first, is_last, item in iterate_with_edge_indicators(l):
        if is_first:
            last = item
            continue
        
        if item!=last:
            yield last, counter
            counter = 1
            last = item
            continue
                
        counter += 1
    
    yield last, counter
    
    
def next_item_cyclic(lis:list[_T], item:_T)->_T:
    n = len(lis)
    i = lis.index(item)
    if i==n-1:
        return lis[0]
    else:
        return lis[i+1]
    

def prev_item_cyclic(lis:list[_T], item:_T)->_T:
    i = lis.index(item)
    if i==0:
        return lis[-1]
    else:
        return lis[i-1]


def reversed(lis:list[_T])->list[_T]:
    """ Like native `reversed()` but returns list copy instead of an iterator of original list
    """
    res = lis.copy()
    res.reverse()
    return res


def cycle_items(lis:list[_T], k:int, copy:bool=True)->list[_T]:
    """Push items from the end to the beginning of the list in a cyclic manner

    ## Example1:
    >>> l1 = [1, 2, 3, 4, 5]
    >>> l2 = cyclic_items(l1, 2)
    >>> print(l2)   # [3, 4, 5, 1, 2]

    ## Example2:
    >>> l1 = [1, 2, 3, 4, 5]
    >>> l2 = cyclic_items(l1, -1)
    >>> print(l2)   # [2, 3, 4, 5, 1]

    Args:
        lis (list[_T]): list of items
        k (int): num of items to push

    Returns:
        list[_T]: list of items with rotated items.
    """
    if copy:
        lis = lis.copy()

    for _ in range(abs(k)):
        if k>0:
            item = lis.pop()
            lis.insert(0, item)
        else:
            item = lis.pop(0)
            lis.append(item)

    return lis

## Test:
if __name__ == "__main__":
    l = 10*[1] + 20*[2] + 30*[3] + 2*[4] 
    for item, num_repetition in repeated_items():
        print(f"[{item}]*{num_repetition}")
