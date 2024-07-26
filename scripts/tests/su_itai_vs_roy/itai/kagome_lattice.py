#!/usr/bin/python3

from typing import Generator, Literal, Optional, Any, Callable, TypeVar, Final, TypeAlias, TypeGuard, Type, NamedTuple, Generic, Iterable
_T = TypeVar("_T")

import numpy as np
from numpy import ceil, floor, pi

import operator
import functools, itertools
from copy import deepcopy
from dataclasses import dataclass, field, fields
import pickle


DEBUG_MODE = True

# Constants:
EPSILON : Final = 0.000001
NUM_MAIN_DIRECTIONS : Final[int] = 6
MAX_DIRECTIONS_STR_LENGTH : Final[int] = 2

# Types:
PosScalarType                   : TypeAlias = int
EdgeIndicatorType   : TypeAlias = str
EdgesDictType       : TypeAlias = dict[str, tuple[int, int]]

# Define errors:
class LatticeError(Exception): ...
class DirectionError(Exception): ...
class OutsideLatticeError(LatticeError): ...
class TriangularLatticeError(LatticeError): ...



class numerics:
    @staticmethod
    def force_between_0_and_2pi(a:float)->float:
        while a<0:
            a += 2*pi
        while a>=2*pi:
            a -= 2*pi
        return a

    @staticmethod
    def force_zero_for_small_numbers(x:float|int)->float|int:
        if abs(x)<EPSILON:
            return 0
        else:
            return x
        
    @staticmethod
    def force_integers_on_close_to_round(x:float|int)->float|int:
        int_x = int(round(x))
        if abs(int_x-x)<EPSILON:
            return int_x
        return x


    @staticmethod
    def furthest_absolute_integer(x:float|int)->int:
        if x<0:
            return int(floor(x))
        else:
            return int(ceil(x))


def _angle_dist(x:float, y:float)->float:
    x = numerics.force_between_0_and_2pi(x)
    y = numerics.force_between_0_and_2pi(y)
    return abs(x-y)
    

def _unit_vector_from_angle(angle:float)->tuple[int, int]:
    x = numerics.force_integers_on_close_to_round(np.cos(angle))
    y = numerics.force_integers_on_close_to_round(np.sin(angle))
    return (x, y)


_Iterable = list|np.ndarray|dict
_Numeric = TypeVar("_Numeric", float, int, complex)
_FloatOrComplex = TypeVar("_FloatOrComplex", float, complex)


def only_single_input_allowed(function_name:str|None=None, **kwargs:_T)->tuple[str, _T]:
    only_key, only_value = None, None
    for key, value in kwargs.items():
        if value is not None:
            if only_key is not None:
                ## Raise error:
                err_msg = "Only a single input is allowed out of inputs "
                err_msg += f"{list(kwargs.keys())} "
                if function_name is not None:
                    err_msg += f"in function {function_name!r} "
                raise ValueError(err_msg)

            else:
                ## Tag as the only key and value:
                only_key, only_value = key, value

    return only_key, only_value

class lists:
    @staticmethod
    def _identity(x:_Numeric)->_Numeric:
        return x

    @staticmethod
    def _minimum_required_copies(crnt_len:int, trgt_len:int)->int:
        div, mod = divmod(trgt_len, crnt_len)
        if mod==0:
            return div
        else: 
            return div+1 

    @staticmethod
    def index_by_approx_value(l:list[Any], value:_Numeric, allowed_error:float=1e-6)->int:
        distances = [abs(v-value) if isinstance(v,float|int|complex) else np.inf for v in l]
        within_error = [1 if v<allowed_error else 0 for v in distances]
        num_within_error = sum(within_error)
        if num_within_error>1:
            raise ValueError(f"Too many values in list {l} \nare close to value {value} within an error of {allowed_error} ")
        elif num_within_error<1:
            raise ValueError(f"None of the values in list {l} \nare close to value {value} within an error of {allowed_error} ")
        else:
            return within_error.index(1)

    @staticmethod
    def join_sub_lists(_lists:list[list[_T]])->list[_T]:
        res = []
        for item in _lists:
            if isinstance(item, list):
                res.extend(item)
            else:
                res.append(item)
        return res


    def all_same(l:list[_Numeric]|list[np.ndarray]) -> bool:
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


    def average(l:list[_FloatOrComplex]) -> _FloatOrComplex:
        return sum(l)/len(l)


    def sum(list_:list[_Numeric], / , lambda_:Callable[[_Numeric], _Numeric]=_identity ) -> _Numeric:
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


    def real(list_:list[_Numeric],/) -> list[int|float]:
        return [ np.real(v) for v in list_ ]  


    def product(_list:list[_Numeric]) -> _Numeric:
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
            assert lists.all_isinstance([l1,l2], (list, np.ndarray))
            if len(l1) != len(l2):
                return False
            for item1, item2 in zip(l1, l2):
                if lists.equal(item1, item2):
                    continue
                else:
                    return False
            return True
        
        def dict_comp(d1:dict, d2:dict) -> bool:
            for key, v1 in d1.items():
                if key not in d2:
                    return False
                v2 = d2[key]
                if not lists.equal(v1, v2):
                    return False
            return True

        
        if lists.all_isinstance([l1, l2], (list, np.ndarray)):
            return lists_comp(l1, l2)
        elif lists.all_isinstance([l1, l2], dict):
            return dict_comp(l1, l2)
        else:
            return l1==l2
        

    def share_item(l1:_Iterable, l2:_Iterable) -> bool:
        for v1 in l1:
            for v2 in l2:
                if v1==v2:
                    return True
        return False


    def copy(l:list[_T]) -> list[_T]:
        duck = l[0]
        if hasattr(duck, "copy") and callable(duck.copy):
            return [item.copy() for item in l] 
        else:
            return [deepcopy(item) for item in l] 


    def common_super_class(lis:list[Any]) -> type:
        classes = [type(x).mro() for x in lis]
        for x in classes[0]:
            if all(x in mro for mro in classes):
                return x


    def iterate_with_periodic_prev_next_items(l:list[_T], skip_first:bool=False) -> Generator[tuple[_T, _T, _T], None, None]:
        for i, (is_first, is_last, crnt) in enumerate(lists.iterate_with_edge_indicators(l)):
            prev, next = None, None
            if is_first and skip_first: continue
            if is_first:        prev = l[-1]
            if is_last:         next = l[0]
            if prev is None:    prev = l[i-1]
            if next is None:    next = l[i+1]
            yield prev, crnt, next
            

    def iterate_with_edge_indicators(l:list[_T]|np.ndarray) -> Generator[tuple[bool, bool, _T], None, None]:
        is_first : bool = True
        is_last  : bool = False
        n = len(l)
        for i, item in enumerate(l):
            if i+1==n:
                is_last = True
            
            yield is_first, is_last, item

            is_first = False


    def min_max(list_:list[_Numeric])->tuple[_Numeric, _Numeric]:
        max_ = -np.inf
        min_ = +np.inf
        for item in list_:
            max_ = max(max_, item)        
            min_ = min(min_, item)
        return min_, max_  # type: ignore
        
    def swap_items(lis:list[_T], i1:int, i2:int, copy:bool=True) -> list[_T]:
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
        res = lists.rearrange(lis, order=new_order)
        return res

    def repeat_list(lis:list[_T], /, num_times:int|None=None, num_items:int|None=None)->list[_T]:
        # Check inputs:
        name, _ = only_single_input_allowed(function_name="repeat_list", num_times=num_times, num_items=num_items)
        # choose based on inputs:
        match name:
            case "num_times":
                return lis*num_times
            case "num_items":
                d = lists._minimum_required_copies(len(lis), num_items)
                larger_list = lists.repeat_list(lis, num_times=d)
                return larger_list[:num_items]

    def convert_whole_numbers_to_int(lis:list[float|int])->list[float|int]:
        lis_copy : list[int|float] = lis.copy()
        for i, x in enumerate(lis):
            if round(x)==x:
                lis_copy[i] = int(x)
        return lis_copy


    def rearrange(l:list[_T], order:list[int]) -> list[_T]:
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
        
        for is_first, is_last, item in lists.iterate_with_edge_indicators(l):
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


    def reversed(lis:list[_T])->_T:
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


class tuples:
    def angle(t1:tuple[_Numeric,...], t2:tuple[_Numeric,...])->float:
        assert len(t1)==len(t2)==2
        dx, dy = tuples.sub(t2, t1)
        theta = np.angle(dx + 1j*dy) % (2*np.pi)
        return theta.item() # Convert to python native type


    def _apply_pairwise(func:Callable[[_Numeric,_Numeric], _Numeric], t1:tuple[_Numeric,...], t2:tuple[_Numeric,...])->tuple[_Numeric,...]:
        list_ = [func(v1, v2) for v1, v2 in zip(t1, t2, strict=True)]
        return tuple(list_)


    def sub(t1:tuple[_Numeric,...], t2:tuple[_Numeric,...])->tuple[_Numeric,...]:
        return tuples._apply_pairwise(operator.sub, t1, t2)


    def add(t1:tuple[_Numeric,...], t2:tuple[_Numeric,...])->tuple[_Numeric,...]:
        return tuples._apply_pairwise(operator.add, t1, t2)


    def multiply(t:tuple[_Numeric,...], scalar_or_t2:_Numeric|tuple[_Numeric,...])->tuple[_Numeric,...]:
        if isinstance(scalar_or_t2, tuple):
            t2 = scalar_or_t2
        else: 
            t2 = tuple([scalar_or_t2 for _ in t])   # tuple with same length
        return tuples._apply_pairwise(operator.mul, t, t2)


    def power(t:tuple[_Numeric,...], scalar:_Numeric)->tuple[_Numeric,...]:
        t2 = tuple([scalar for _ in t])
        return tuples._apply_pairwise(operator.pow, t, t2)


    def dot_product(t1:tuple[_Numeric,...], t2:tuple[_Numeric,...])->_Numeric:
        times_vector = tuples.multiply(t1, t2)
        return sum(times_vector)


    def copy_with_replaced_val_at_index(t:tuple, i:int, val:Any) -> tuple:
        temp = [x for x in t]
        temp[i] = val
        return tuple(temp)


    def copy_with_replaced_val_at_key(t:NamedTuple, key:str, val:Any) -> NamedTuple:
        i = tuples.get_index_of_named_tuple_key(t=t, key=key)
        # create native tuple:
        t2 = tuples.copy_with_replaced_val_at_index(t, i, val)    
        # use constructor to create instance of the same NamedTuple as t:
        return t.__class__(*t2)  

    def get_index_of_named_tuple_key(t:NamedTuple, key:str)->int:
        for i, field in enumerate(t._fields):
            if field == key:
                return i
        raise ValueError(f"Key {key!r} not found in tuple {t}")

    def equal(t1:tuple[_T,...], t2:tuple[_T,...], allow_permutation:bool=False)->bool:
        if len(t1)!=len(t2):
            return False
        
        if allow_permutation:
            return tuples.angle_are_equal_allow_permutation(t1, t2)
        else:
            for v1, v2 in zip(t1, t2, strict=True):
                if v1!=v2:
                    return False
            return True


    def mean_itemwise(t1:tuple[_Numeric,...], t2:tuple[_Numeric,...])->tuple[_Numeric,...]:
        l = [(v1+v2)/2 for v1, v2 in zip(t1, t2, strict=True)]
        return tuple(l)

    def add_element(t:tuple[_T,...], element:_T)->tuple[_T,...]:
        lis = list(t)
        lis.append(element)
        return tuple(lis)





    def _are_equal_allow_permutation(t1:tuple[_T,...], t2:tuple[_T,...])->bool:
        l1 : list[_T] = list(t1)
        l2 : list[_T] = list(t2)
        while len(l1)>0:
            value = l1[0]
            if value not in l2:
                return False
            l1.remove(value)
            l2.remove(value)
        return True


def draw_now():
    from matplotlib import pyplot as plt
    plt.show(block=False)
    plt.pause(0.01)


def default_value(
    arg:None|_T, 
    default:_T=None, 
    default_factory:Optional[Callable[[], _T ]]=None
) -> _T :
    if arg is not None:
        return arg
    if default is not None:
        return default
    if default_factory is not None:
        return default_factory()
    raise ValueError(f"Must provide either `default` value or function `default_factory` that generates a value")
    

def formatted(
    val:Any, 
    fill:str=' ', 
    alignment:Literal['<','^','>']='>', 
    width:Optional[int]=None, 
    precision:Optional[int]=None,
    signed:bool=False
) -> str:
    
    # Check info:
    try:
        if round(val)==val and precision is None:
            force_int = True
        else:
            force_int = False
    except:
        force_int = False
           
    # Simple formats:
    format = f"{fill}{alignment}"
    if signed:
        format += "+"
    
    # Width:
    width = default_value(width, len(f"{val}"))
    format += f"{width}"            
    
    precision = default_value(precision, 0)
    format += f".{precision}f"    
        
    return f"{val:{format}}"  


def num_out_of_num(num1, num2):
    width = len(str(num2))
    format = lambda num: formatted(num, fill=' ', alignment='>', width=width )
    return format(num1)+"/"+format(num2)


class Direction():
    __slots__ = 'name', 'angle', "unit_vector"

    def __init__(self, name:str, angle:float) -> None:
        self.name = name
        self.angle = numerics.force_between_0_and_2pi(angle)
        self.unit_vector : tuple[int, int] = _unit_vector_from_angle(angle)

    def __str__(self)->str:
        return self.name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.name!r} {self.angle:.5f}"
    
    
    def __hash__(self) -> int:
        angle_with_limited_precision = round(self.angle, 5)
        return hash((self.__class__.__name__, self.name, angle_with_limited_precision))
    
    def __eq__(self, other:object) -> bool:
        if not isinstance(other, Direction):
            return False
        return hash(self)==hash(other)
        # return check.is_equal(self, other)

    ## Get other by relation:
    def opposite(self)->"Direction":
        try:
            res = OPPOSITE_DIRECTIONS[self]
        except KeyError:
            cls = type(self)
            res = cls(name=self.name, angle=self.angle+np.pi)
        return res
    
    def next_clockwise(self)->"Direction":
        return lists.prev_item_cyclic(ORDERED_LISTS[type(self)], self)

    def next_counterclockwise(self)->"Direction":
        return lists.next_item_cyclic(ORDERED_LISTS[type(self)], self)
    
    ## Creation method:
    @classmethod
    def from_angle(cls, angle:float, eps:float=EPSILON)->"Direction":
        ## Where to look
        possible_directions = ORDERED_LISTS[cls]
        # look:
        for dir in possible_directions:
            if _angle_dist(dir.angle, angle)<eps:
                return dir
        raise DirectionError(f"Given angle does not match with any known side")
    
    @classmethod
    def random(cls)->"Direction":
        return lists.random_item(ORDERED_LISTS[cls])
    
    ## iterators over all members:
    @classmethod
    def all_in_counter_clockwise_order(cls)->Generator["Direction", None, None]:
        return iter(ORDERED_LISTS[cls])
    
    @classmethod
    def all_in_clockwise_order(cls)->Generator["Direction", None, None]:
        return reversed(ORDERED_LISTS[cls])
    
    @classmethod
    def all_in_random_order(cls)->Generator["Direction", None, None]:
        return iter(lists.shuffle(ORDERED_LISTS[cls]))
    
    @classmethod
    def iterator_with_str_output(cls, output_func:Callable[[str], Any])->Generator["Direction", None, None]:
        for i, side in enumerate(ORDERED_LISTS[cls]):
            s = " " + num_out_of_num(i+1, NUM_MAIN_DIRECTIONS) + " " + f"{side.name:<{MAX_DIRECTIONS_STR_LENGTH}}"
            output_func(s)
            yield side

    def plot(self)->None:
        ## Some special imports:
        from matplotlib import pyplot as plt
                                
        vector = self.unit_vector
        space = "    "
        x, y = vector[0], vector[1]
        l = 1.1
        plt.figure()
        plt.scatter(0, 0, c="blue", s=100)
        plt.arrow(
            0, 0, x, y, 
            color='black', length_includes_head=True, width=0.01, 
            head_length=0.15, head_width=0.06
        )  
        plt.text(x, y, f"\n\n{space}{self.angle} rad\n{space}{self.unit_vector}", color="blue")
        plt.title(f"Direction {self.name!r}")
        plt.xlim(-l, +l)
        plt.ylim(-l, +l)
        # plt.axis('off')
        plt.grid(color='gray', linestyle=':')

        print(f"Plotted direction {self.name!r}")
    

class LatticeDirection(Direction): 
    R  : "LatticeDirection"
    UR : "LatticeDirection"
    UL : "LatticeDirection"
    L  : "LatticeDirection"
    DL : "LatticeDirection"
    DR : "LatticeDirection" 


class BlockSide(Direction):
    U  : "BlockSide"
    UR : "BlockSide"
    UL : "BlockSide"
    D  : "BlockSide"
    DL : "BlockSide"
    DR : "BlockSide" 

    def orthogonal_counterclockwise_lattice_direction(self)->LatticeDirection:
        return ORTHOGONAL_LATTICE_DIRECTIONS_TO_BLOCK_SIDES[self]
    
    def orthogonal_clockwise_lattice_direction(self)->LatticeDirection:
        return ORTHOGONAL_LATTICE_DIRECTIONS_TO_BLOCK_SIDES[self].opposite()
    
    def matching_lattice_directions(self)->list[LatticeDirection]:
        return MATCHING_LATTICE_DIRECTIONS_TO_BLOCK_SIDES[self]
    
    def opposite_lattice_directions(self)->list[LatticeDirection]:
        return [dir.opposite() for dir in MATCHING_LATTICE_DIRECTIONS_TO_BLOCK_SIDES[self]]

            

# ============================================================================ #
#|                            Main Directions                                 |#
# ============================================================================ #    

# Main direction in the Kagome lattice:
LatticeDirection.R  = LatticeDirection("R" , 0)
LatticeDirection.UR = LatticeDirection("UR", pi/3)
LatticeDirection.UL = LatticeDirection("UL", 2*pi/3)
LatticeDirection.L  = LatticeDirection("L" , pi)
LatticeDirection.DL = LatticeDirection("DL", 4*pi/3)
LatticeDirection.DR = LatticeDirection("DR", 5*pi/3)

# Directions that appear in the hexagonal cell:
BlockSide.U   = BlockSide("U", pi/2)
BlockSide.UR  = BlockSide("UR", pi/2-pi/3)
BlockSide.UL  = BlockSide("UL", pi/2+pi/3)
BlockSide.D   = BlockSide("D", 3*pi/2)
BlockSide.DL  = BlockSide("DL", 3*pi/2-pi/3)
BlockSide.DR  = BlockSide("DR", 3*pi/2+pi/3)




# ============================================================================ #
#|                        Relations between Directions                        |#
# ============================================================================ #    

LATTICE_DIRECTIONS_COUNTER_CLOCKWISE : Final[list[Direction]] = [
    LatticeDirection.DL, LatticeDirection.DR, LatticeDirection.R, LatticeDirection.UR, LatticeDirection.UL, LatticeDirection.L
]

BLOCK_SIDES_COUNTER_CLOCKWISE : Final[list[Direction]] = [
    BlockSide.D, BlockSide.DR, BlockSide.UR, BlockSide.U, BlockSide.UL, BlockSide.DL
]

ORDERED_LISTS = {
    LatticeDirection : LATTICE_DIRECTIONS_COUNTER_CLOCKWISE,
    BlockSide : BLOCK_SIDES_COUNTER_CLOCKWISE
}

OPPOSITE_DIRECTIONS : Final[dict[Direction, Direction]] = {
    # Lattice:
    LatticeDirection.R  : LatticeDirection.L ,
    LatticeDirection.UR : LatticeDirection.DL,
    LatticeDirection.UL : LatticeDirection.DR, 
    LatticeDirection.L  : LatticeDirection.R ,
    LatticeDirection.DL : LatticeDirection.UR,
    LatticeDirection.DR : LatticeDirection.UL,
    # Block:
    BlockSide.U  : BlockSide.D,
    BlockSide.D  : BlockSide.U,
    BlockSide.UR : BlockSide.DL,
    BlockSide.DL : BlockSide.UR,
    BlockSide.UL : BlockSide.DR,
    BlockSide.DR : BlockSide.UL
}

ORTHOGONAL_LATTICE_DIRECTIONS_TO_BLOCK_SIDES : Final[dict[BlockSide, LatticeDirection]] = {
    BlockSide.D  : LatticeDirection.R,
    BlockSide.U  : LatticeDirection.L,
    BlockSide.DR : LatticeDirection.UR,
    BlockSide.DL : LatticeDirection.DR,
    BlockSide.UR : LatticeDirection.UL,
    BlockSide.UL : LatticeDirection.DL
}

MATCHING_LATTICE_DIRECTIONS_TO_BLOCK_SIDES : Final[dict[BlockSide, list[LatticeDirection]]] = {
    BlockSide.D  : [LatticeDirection.DL, LatticeDirection.DR],
    BlockSide.DR : [LatticeDirection.DR, LatticeDirection.R ],
    BlockSide.UR : [LatticeDirection.R,  LatticeDirection.UR],
    BlockSide.U  : [LatticeDirection.UR, LatticeDirection.UL],
    BlockSide.UL : [LatticeDirection.UL, LatticeDirection.L ],
    BlockSide.DL : [LatticeDirection.L , LatticeDirection.DL]
}



@dataclass
class Node():
    index : int
    pos : tuple[PosScalarType, ...]
    edges : list[EdgeIndicatorType]
    directions : list[Direction]
    boundaries : set[BlockSide] = field(default_factory=set)

    def get_edge_in_direction(self, direction:Direction) -> EdgeIndicatorType:
        edge_index = self.directions.index(direction)
        return self.edges[edge_index]
    
    def set_edge_in_direction(self, direction:Direction, value:EdgeIndicatorType) -> None:
        edge_index = self.directions.index(direction)
        self.edges[edge_index] = value

    @property
    def angles(self)->list[float]:
        return [dir.angle for dir in self.directions]


@functools.cache
def total_vertices(N):
	"""
	Returns the total number of vertices in the *bulk* of a hex 
	TN with linear parameter N
	"""
	return 3*N*N - 3*N + 1


@functools.cache
def center_vertex_index(N):
    i = num_rows(N)//2
    j = i
    return get_vertex_index(i, j, N)


@functools.cache
def num_rows(N):
	return 2*N-1


def row_width(i, N):
	"""
	Returns the width of a row i in a hex TN of linear size N. i is the
	row number, and is between 0 -> (2N-2)
	"""
	if i<0 or i>2*N-2:
		return 0
	
	return N+i if i<N else 3*N-i-2

def _get_neighbor_coordinates_in_direction_no_boundary_check(i:int, j:int, direction:LatticeDirection, N:int)->tuple[int, int]:
	## Simple L or R:
	if direction==LatticeDirection.L:  
		return i, j-1
	if direction==LatticeDirection.R:  
		return i, j+1

	## Row dependant:
	middle_row_index = num_rows(N)//2   # above or below middle row

	if direction==LatticeDirection.UR: 
		if i <= middle_row_index: 
			return i-1, j
		else: 
			return i-1, j+1
	
	if direction==LatticeDirection.UL:
		if i <= middle_row_index:
			return i-1, j-1
		else:
			return i-1, j
		
	if direction==LatticeDirection.DL:
		if i < middle_row_index:
			return i+1, j
		else:
			return i+1, j-1
		
	if direction==LatticeDirection.DR:
		if i < middle_row_index:
			return i+1, j+1
		else:
			return i+1, j
		
	TriangularLatticeError(f"Impossible direction {direction!r}")


def check_boundary_vertex(index:int, N)->list[BlockSide]:
	on_boundaries = []

	# Basic Info:
	i, j = get_vertex_coordinates(index, N)
	height = num_rows(N)
	width = row_width(i,N)
	middle_row_index = height//2

	# Boundaries:
	if i==0:
		on_boundaries.append(BlockSide.U)
	if i==height-1:
		on_boundaries.append(BlockSide.D)
	if j==0: 
		if i<=middle_row_index:
			on_boundaries.append(BlockSide.UL)
		if i>=middle_row_index:
			on_boundaries.append(BlockSide.DL)
	if j == width-1:
		if i<=middle_row_index:
			on_boundaries.append(BlockSide.UR)
		if i>=middle_row_index:
			on_boundaries.append(BlockSide.DR)
	
	return on_boundaries



def get_neighbor_coordinates_in_direction(i:int, j:int, direction:LatticeDirection, N:int)->tuple[int, int]:
	i2, j2 = _get_neighbor_coordinates_in_direction_no_boundary_check(i, j, direction, N)

	if i2<0 or i2>=num_rows(N):
		raise OutsideLatticeError()
	
	if j2<0 or j2>=row_width(i2, N):
		raise OutsideLatticeError()
	
	return i2, j2


def get_neighbor(i:int, j:int, direction:LatticeDirection, N:int)->tuple[int, int]:	
	i2, j2 = get_neighbor_coordinates_in_direction(i, j, direction, N)
	return get_vertex_index(i2, j2, N)


def all_neighbors(index:int, N:int)->Generator[tuple[Node, LatticeDirection], None, None]:
	i, j = get_vertex_coordinates(index, N)
	for direction in LatticeDirection.all_in_counter_clockwise_order():
		try: 
			neighbor = get_neighbor(i, j, direction, N)
		except OutsideLatticeError:
			continue
		yield neighbor, direction


def get_vertex_coordinates(index, N)->tuple[int, int]:
	running_index = 0 
	for i in range(num_rows(N)):
		width = row_width(i, N)
		if index < running_index + width:
			j = index - running_index
			return i, j
		running_index += width
	raise TriangularLatticeError("Not found")


def get_vertex_index(i,j,N):
	"""
	Given a location (i,j) of a vertex in the hexagon, return its 
	index number. The vertices are ordered left->right, up->down.
	
	The index number is a number 0->NT-1, where NT=3N^2-3N+1 is the
	total number of vertices in the hexagon.
	
	The index i is the row in the hexagon: i=0 is the upper row.
	
	The index j is the position of the vertex in the row. j=0 is the 
	left-most vertex in the row.
	"""
	
	# Calculate Aw --- the accumulated width of all rows up to row i,
	# but not including row i.
	if i==0:
		Aw = 0
	else:
		Aw = (i*N + i*(i-1)//2 if i<N else 3*N*N -3*N +1 -(2*N-1-i)*(4*N-2-i)//2)
		
	return Aw + j

		
@functools.cache
def get_center_vertex_index(N):
	i = num_rows(N)//2
	j = row_width(i, N)//2
	return get_vertex_index(i, j, N)


def get_node_position(i,j,N):
	w = row_width(i, N)
	x = N - w + 2*j	
	y = N - i
	return x, y


def get_edge_index(i,j,side,N):
	"""
	Get the index of an edge in the triangular PEPS.
	
	Given a vertex (i,j) in the bulk, we have the following rules for 
	labeling its adjacent legs:
	
	       (i-1,j-1)+2*NT+101 (i-1,j)+NT+101
	                         \   /
	                          \ /
	               ij+1 ------ o ------ ij+2
	                          / \
	                         /   \
                   ij+NT+101  ij+2*NT+101
                   
  Above ij is the index of (i,j) and we denote by (i-1,j-1) and (i-1,j)
  the index of these vertices. NT is the total number of vertices in the
  hexagon.
  
  Each of the 6 boundaries has 2N-1 external legs. They are ordered 
  counter-clockwise and labeled as:
  
  d1, d2, ...   --- Lower external legs
  dr1, dr2, ... --- Lower-Right external legs
  ur1, ur2, ... --- Upper-Right external legs
  u1, u2, ...   --- Upper external legs
  ul1, ul2, ... --- Upper-Left external legs
  dl1, dl2, ... --- Lower-Left external legs
  

	Input Parameters:
	------------------
	i,j --- location of the vertex to which the edge belong.
	        i=row, j=column. i=0 ==> upper row, j=0 ==> left-most column.

	side --- The side of the edge. Either 'L', 'R', 'UL', 'UR', 'DL', 'DR'

	N    --- Linear size of the lattice
	
	OUTPUT: the label
	"""

	# The index of the vertex
	ij = get_vertex_index(i,j,N)

	# Calculate the width of the row i (how many vertices are there)
	w = row_width(i, N)
	
	# Total number of vertices in the hexagon
	NT = total_vertices(N)
		
	if side=='L':
		if j>0:
			e = ij
		else:
			if i<N:
				e = f'ul{i*2+1}'
			else:
				e = f'dl{(i-N+1)*2}'
				
	if side=='R':
		if j<w-1:
			e = ij+1
		else:
			if i<N-1:
				e = f'ur{2*(N-1-i)}'
			else:
				e = f'dr{2*(2*N-2-i)+1}'  
				
	if side=='UL':
		if i<N:
			if j>0:
				if i>0:
					e = get_vertex_index(i-1,j-1,N) + 2*NT + 101
				else:
					# i=0
					e = f'u{2*N-1-j*2}'
			else:
				# j=0
				if i==0:
					e = f'u{2*N-1}'
				else:
					e = f'ul{2*i}'
					
		else:
			# so i=N, N+1, ...
			e = get_vertex_index(i-1,j,N) + 2*NT + 101
				
	if side=='UR':
		if i<N:
			if j<w-1:
				if i>0:
					e = get_vertex_index(i-1,j,N) + NT + 101
				else:
					# i=0
					e = f'u{2*N-2-j*2}'
			else:
				# j=w-1
				if i==0:
					e = f'ur{2*N-1}'
				else:
					e = f'ur{2*N-1-2*i}'
		else:
			e = get_vertex_index(i-1,j+1,N) + NT + 101
			
	if side=='DL':
		if i<N-1:
			e = ij + NT + 101
		else:
			# so i=N-1, N, N+1, ...
			if j>0:
				if i<2*N-2:
					e = ij + NT + 101
				else:
					# i=2N-2 --- last row
					e = f'd{2*j}'
			else:
				# j=0
				if i<2*N-2:
					e = f'dl{(i-N+1)*2+1}'
				else:
					e = f'dl{2*N-1}'
					
	if side=='DR':
		if i<N-1:
			e = ij + 2*NT + 101
		else:
			# so i=N-1, N, ...
			if j<w-1:
				if i<2*N-2:
					e = ij + 2*NT + 101
				else:
					# i=2N-2 --- last row
					e = f'd{2*j+1}'
			else:
				# so we're on the last j
				if i<2*N-2:
					e = f'dr{(2*N-2-i)*2}'
				else:
					# at i=2*N-2 (last row)
					e = f'd{2*N-1}'
				
	return e


def create_hex_dicts(N):
	"""
	Creates two dictionaries for a two-way mapping of two different 
	indexing of the bulk vertices in the hexagon TN. These are then used 
	in the rotate_CW function to rotate the hexagon in 60 deg 
	counter-clock-wise.
	
	The two mappings are:
	-----------------------
	
	ij --- The ij index that is defined by row,col=(i,j). This integer
	       is calculated in get_vertex_index()
	       
	(n, side, p). Here the vertices sit on a set of concentrated hexagons.
	              n=1,2, ..., N is the size of the hexagon
	              side = (d, dr, ur, u, ul, dl) the side of the hexagon
	                      on which the vertex sit
	              p=0, ..., n-2 is the position within the side (ordered
	                        in a counter-clock order
	              
	              For example (N,'d',1) is the (i,j)=(2*N-2,1) vertex.
	              
	              
	The function creates two dictionaries:
	
	ij_to_hex[ij] = (n,side, p)
	
	hex_to_ij[(n,side,p)] = ij
	              
	              
	
	Input Parameters:
	-------------------
	
	N --- size of the hexagon.
	
	OUTPUT:
	--------
	
	ij_to_hex, hex_to_ij --- the two dictionaries.
	"""
	
	hex_to_ij = {}
	ij_to_hex = {}
	
	sides = ['d', 'dr', 'ur', 'u', 'ul', 'dl']
	
	i,j = 2*N-1,-1


	# To create the dictionaries, we walk on the hexagon counter-clockwise, 
	# at radius n for n=N ==> n=1. 
	#
	# At each radius, we walk 'd' => 'dr' => 'ur' => 'u' => 'ul' => 'dl'

	for n in range(N, 0, -1):
		i -= 1
		j += 1
		
		#
		# (i,j) hold the position of the first vertex on the radius-n 
		# hexagon
		# 
		
		for side in sides:
			
			for p in range(n-1):
				
				ij = get_vertex_index(i,j,N)
				
				ij_to_hex[ij] = (n,side,p)
				hex_to_ij[(n,side,p)] = ij
				
				if side=='d':
					j +=1
					
				if side=='dr':
					j +=1
					i -=1
					
				if side=='ur':
					j -=1
					i -=1
					
				if side=='u':
					j -= 1
					
				if side=='ul':
					i += 1
					
				if side=='dl':
					i += 1
		

	#
	# Add the central node (its side is '0')
	#
	
	ij = get_vertex_index(i,j,N)
	
	ij_to_hex[ij] = (n,'0',0)
	hex_to_ij[(n,'0',0)] = ij
	
	return ij_to_hex, hex_to_ij


def rotate_CW(N, ijs, ij_to_hex, hex_to_ij):
	
	"""
	
	Takes a list of vertices on the hexagon TN and rotates it 60 degrees
	clockwise.
	
	
	Input Parameters:
	------------------
	
	N --- Hexagon size (radius)
	
	ijs --- The list of vertices
	
	ij_to_hex, hex_to_ij --- two dictionaries used to switch between the
	                         usual (ij) indexing to the internal hexagon
	                         indexing. This is used for the rotation.
	
	OUTPUT:
	--------
	
	new_ijs --- A corresponding list of rotated vertices.
	
	
	Notes: the vertices can be of any type, bulk vertices or external
	       MPS vertices.
	
	
	"""
	
	
	
	NT = total_vertices(N)
	
	new_ijs = []   # Initialize the new vertices list
	
	for ij in ijs:
		
		#
		# Check if ij is a bulk vertex or if it is an external MPS vertex.
		#
		
		if ij>= NT:
			#
			# Rotate an MPS vertex
			# 
			if ij>=NT+10*N-5:
				#
				# We're on the Up-Left edge --- so we rotate to the upper edge
				#
				
				new_ij = ij - 10*N + 5
				
			else:
				#
				# We're other edge --- so rotate clockwise by adding 2N-1
				#
				new_ij = ij + 2*N-1
				
		else:
			#
			# Rotate a bulk vertex using the dictionaries. Once we know
			# the (n,side,p) of a vertex, all we need to do is rotate side
			# to its clockwise neighbor.
			#
			
			(n,side, p) = ij_to_hex[ij]
			
			if side=='u':
				side = 'ur'
			elif side=='ur':
				side = 'dr'
			elif side=='dr':
				side = 'd'
			elif side=='d':
				side = 'dl'
			elif side=='dl':
				side = 'ul'
			elif side=='ul':
				side = 'u'
				
			new_ij = hex_to_ij[(n,side,p)]
			
		new_ijs.append(new_ij)
		
	return new_ijs
			



def rotate_ACW(N, ijs, ij_to_hex, hex_to_ij):
	
	"""
	
	Takes a list of vertices on the hexagon + MPSs TN and rotates it 
	60 degrees anti-clockwise.
	
	
	Input Parameters:
	------------------
	
	N --- Hexagon size (radius)
	
	ijs --- The list of vertices
	
	ij_to_hex, hex_to_ij --- two dictionaries used to switch between the
	                         usual (ij) indexing to the internal hexagon
	                         indexing. This is used for the rotation.
	
	OUTPUT:
	--------
	
	new_ijs --- A corresponding list of rotated vertices.
	
	
	Notes: the vertices can be of any type, bulk vertices or external
	       MPS vertices.
	
	
	"""
	
	
	
	NT = total_vertices(N)
	
	new_ijs = []   # Initialize the new vertices list
	
	for ij in ijs:
		
		#
		# Check if ij is a bulk vertex or if it is an external MPS vertex.
		#
		
		if ij>= NT:
			#
			#     -------------    Rotate an MPS vertex  ------------------
			# 
			if ij>=NT+10*N-5:
				#
				# We're on the Down-Left edge --- so we rotate to the bottom edge
				#
				
				new_ij = ij - 10*N + 5
				
			else:
				#
				# We're other edge --- so rotate anti-clockwise by adding 2N-1
				#
				new_ij = ij + 2*N-1
				
		else:
			#     -------------    Rotate a bulk vertex  ------------------
			#
			# Rotate a bulk vertex using the dictionaries. Once we know
			# the (n,side,p) of a vertex, all we need to do is rotate side
			# to its clockwise neighbor.
			#
			
			(n,side, p) = ij_to_hex[ij]


			if side=='d':
				side = 'dr'
			elif side=='dr':
				side = 'ur'
			elif side=='ur':
				side = 'u'
			elif side=='u':
				side = 'ul'
			elif side=='ul':
				side = 'dl'
			elif side=='dl':
				side = 'd'
				
			new_ij = hex_to_ij[(n,side,p)]
			
		new_ijs.append(new_ij)
		
	return new_ijs
			

    

def create_triangle_lattice(N)->list[Node]:

	"""
	The structure of every node in the list is:
	
	T[d, D_L, D_R, D_UL, D_UR, D_DL, D_DR]
	
	With the following structure
	
                       (3)    (4)
                        UL    UR
                          \   /
                           \ /
                   (1)L  ---o--- R(2)
                           / \
                          /   \
                        DL     DR
                       (5)    (6)

	"""

	NT = total_vertices(N)

	if N<2 and False:
		print("Error in create_random_trianlge_PEPS !!!")
		print(f"N={N} but it must be >= 2 ")
		exit(1)		


	#
	# Create the list of edges. 
	#
	edges_list = []
	for i in range(2*N-1):
		w = row_width(i,N)
		for j in range(w):

			eL  = get_edge_index(i,j,'L' , N)
			eR  = get_edge_index(i,j,'R' , N)
			eUL = get_edge_index(i,j,'UL', N)
			eUR = get_edge_index(i,j,'UR', N)
			eDL = get_edge_index(i,j,'DL', N)
			eDR = get_edge_index(i,j,'DR', N)

			edges_list.append([eL, eR, eUL, eUR, eDL, eDR])
	#
	# Create the list of nodes:
	#
	index = 0
	nodes_list = []
	for i in range(2*N-1):
		w = row_width(i,N)
		for j in range(w):
			n = Node(
				index = index,
				pos = get_node_position(i, j, N),
				edges = edges_list[index],
				directions=[LatticeDirection.L, LatticeDirection.R, LatticeDirection.UL, LatticeDirection.UR, LatticeDirection.DL, LatticeDirection.DR]
			)
			nodes_list.append(n)
			index += 1


	return nodes_list

def all_coordinates(N:int)->Generator[tuple[int, int], None, None]:
	for i in range(num_rows(N)):
		for j in range(row_width(i, N)):
			yield i, j


def unit_vector_corrected_for_sorting_triangular_lattice(direction:Direction)->tuple[float, float]:
	if isinstance(direction, LatticeDirection):
		match direction:
			case LatticeDirection.R :	return (+1,  0)
			case LatticeDirection.L :	return (-1,  0)
			case LatticeDirection.UL:	return (-1, +1)
			case LatticeDirection.UR:	return (+1, +1)
			case LatticeDirection.DL:	return (-1, -1)
			case LatticeDirection.DR:	return (+1, -1)
	elif isinstance(direction, BlockSide):
		match direction:
			case BlockSide.U :	return ( 0, +1)
			case BlockSide.D :	return ( 0, -1)
			case BlockSide.UR:	return (+1, +1)
			case BlockSide.UL:	return (-1, +1)
			case BlockSide.DL:	return (-1, -1)
			case BlockSide.DR:	return (+1, -1)
	else:
		raise TypeError(f"Not a supported type")


def sort_coordinates_by_direction(items:list[tuple[int, int]], direction:Direction, N:int)->list[tuple[int, int]]:
	# unit_vector = direction.unit_vector  # This basic logic break at bigger lattices
	unit_vector = unit_vector_corrected_for_sorting_triangular_lattice(direction)
	def key(ij:tuple[int, int])->float:
		i, j = ij[0], ij[1]
		pos = get_node_position(i, j, N)
		return tuples.dot_product(pos, unit_vector)  # vector dot product
	return sorted(items, key=key)
	

@functools.cache
def vertices_indices_rows_in_direction(N:int, major_direction:BlockSide, minor_direction:LatticeDirection)->list[list[int]]:
	""" arrange nodes by direction:
	"""
	## Arrange indices by position relative to direction, in reverse order
	coordinates_in_reverse = sort_coordinates_by_direction(all_coordinates(N), major_direction.opposite(), N)

	## Bunch vertices by the number of nodes at each possible row (doesn't matter from which direction we look)
	list_of_rows = []
	for i in range(num_rows(N)):

		# collect vertices as much as the row has:
		row = []
		w = row_width(i, N)
		for _ in range(w):
			item = coordinates_in_reverse.pop()
			row.append(item)
		
		# sort row by minor axis:
		sorted_row = sort_coordinates_by_direction(row, minor_direction, N)
		indices = [get_vertex_index(i, j, N) for i,j in sorted_row]
		list_of_rows.append(indices)

	return list_of_rows


def edges_dict_from_edges_list(edges_list:list[list[str]])->EdgesDictType:
    vertices = {}
    for i, i_edges in enumerate(edges_list):
        for e in i_edges:
            if e in vertices:
                (j1,j2) = vertices[e]
                vertices[e] = (i,j1)
            else:
                vertices[e] = (i,i)
    return vertices


def same_dicts(d1:EdgesDictType, d2:EdgesDictType)->bool:

    if len(d1) != len(d2):
        return False
    
    d2_copy = deepcopy(d2)
    for key, tuple1 in d1.items():
        if key not in d2_copy:
            return False
        tuple2 = d2_copy[key]
        if not tuples.equal(tuple1, tuple2, allow_permutation=True):
            return False
        d2_copy.pop(key)

    if len(d2_copy)>0:
        return False
    
    return True
    

def next_clockwise_or_counterclockwise(dir:Direction, clockwise:bool=True)->Direction:
    if clockwise:
        return dir.next_clockwise()
    else:
        return dir.next_counterclockwise()


class create:
    def mean_direction(directions:list[Direction])->Direction:
        angles = [dir.angle for dir in directions]
        angle = sum(angles)/len(angles)
        return Direction(name="mean", angle=angle)
    
    def direction_from_positions(p1:tuple[float, float], p2:tuple[float, float])->Direction:
        angle = tuples.angle(p1, p2)
        return Direction(name="relative", angle=angle)


class check:
    def is_orthogonal(dir1:Direction, dir2:Direction)->bool:
        dir1_ortho_options = [dir1.angle+pi/2, dir1.angle-pi/2]
        for dir1_ortho in dir1_ortho_options:
            if _angle_dist(dir1_ortho, dir2.angle)<EPSILON:
                return True
        return False
    
    def is_opposite(dir1:Direction, dir2:Direction)->bool:
        if isinstance(dir1, BlockSide) and isinstance(dir2, LatticeDirection):
            lattice_options = dir1.opposite_lattice_directions()
            lattice_dir = dir2
            mixed_cased = True
        elif isinstance(dir2, BlockSide) and isinstance(dir1, LatticeDirection) :
            lattice_options = dir2.opposite_lattice_directions()
            lattice_dir = dir1
            mixed_cased = True
        elif check.is_non_specific_direction(dir1) and check.is_non_specific_direction(dir2):  # Not a standard direction
            a1 = dir1.angle
            a2 = numerics.force_between_0_and_2pi(dir2.angle + np.pi)
            return abs(a1-a2)<EPSILON
        else:
            mixed_cased = False

        if mixed_cased:
            return lattice_dir in lattice_options
        else:
            return check.is_equal(dir1.opposite(), dir2)

    def is_equal(dir1:Direction, dir2:Direction) -> bool:
        # Type check:
        assert issubclass(type(dir2), Direction)
        # Fast instance check:
        if dir1 is dir2:
            return True
        # Fast class and name check:
        if (dir1.__class__.__name__==dir2.__class__.__name__ 
            and  dir1.name==dir2.name ):
            return True
        # Slower values check:
        if check.is_non_specific_direction(dir1) or check.is_non_specific_direction(dir2):
            return _angle_dist(dir1.angle, dir2.angle)<EPSILON
        return False
    
    def all_same(l:list[Direction]) -> bool:
        dummy = l[0]
        for item in l:
            if not check.is_equal(dummy, item):
                return False                
        return True

    def is_non_specific_direction(dir:Direction) -> TypeGuard[Direction]:
        if isinstance(dir, LatticeDirection) or isinstance(dir, BlockSide):
            return False
        if isinstance(dir, Direction):
            return True
        return False

class sort:

    def specific_typed_directions_by_clock_order(directions:list[Direction], clockwise:bool=True)->list[Direction]:
        ## Try different first directions:
        for dir_first in directions:
            final_order = [dir_first]
            dir_next = next_clockwise_or_counterclockwise(dir_first, clockwise)
            while dir_next in directions:
                final_order.append(dir_next)
                dir_next = next_clockwise_or_counterclockwise(dir_next, clockwise)
            if len(final_order)==len(directions):
                return final_order
        raise DirectionError("Directions are not related")


    def arbitrary_directions_by_clock_order(first_direction:Direction, directions:list[Direction], clockwise:bool=True)->list[Direction]:
        """Given many arbitrary directions, order them in clockwise order as a continuation of a given starting direction.

        Args:
            first_direction (Direction): The direction from which we draw the relation
            directions (list[Direction]): The options.
            clockwise (bool): The order (defaults to `True`)

        Returns:
            list[Direction]: The directions in clockwise/counter-clockwise order from all options.
        """

        ## Directions hierarchy key:
        if clockwise:
            key = lambda dir: -dir.angle
        else:
            key = lambda dir: dir.angle

        ## Sort:
        sorted_directions = sorted(directions, key=key)

        ## Push first item in list until it is really the first:
        i = sorted_directions.index(first_direction)
        sorted_directions = lists.cycle_items(sorted_directions, -i, copy=False)
        
        return sorted_directions


def plot_lattice(
    lattice:list[Node], edges_dict:dict[str, tuple[int, int]]|None=None, 
    node_color:str="red", node_size=40, edge_style:str="b-", periodic:bool=False
)->None:
    from matplotlib import pyplot as plt

    ## Plot node positions:
    for node in lattice:
        x, y = node.pos
        plt.scatter(x, y, c=node_color, s=node_size)
        plt.text(x,y, s=node.index)

    ## Plot edges:
    edges_list = [node.edges for node in lattice]
    for edges in edges_list:
        for edge in edges:
            nodes = [node for node in lattice if edge in node.edges]

            ## Assert with edges_dict, if given:
            if edges_dict is not None:
                indices = edges_dict[edge]
                assert set(indices) == set((node.index for node in nodes))
            
            if edge.count("-")==2 and not edge[0]=='M':
                assert periodic==True
                assert len(nodes)==2
                for node in nodes:
                    direction = node.directions[ node.edges.index(edge) ]
                    x1, y1 = node.pos
                    x2, y2 = tuples.add((x1, y1), direction.unit_vector)
                    x_text = x2
                    y_text = y2
                    plt.plot([x1, x2], [y1, y2], edge_style)
                    plt.text(x_text, y_text, edge, color="green")
                continue

            elif len(nodes)==2:
                n1, n2 = nodes
                x1, y1 = n1.pos
                x2, y2 = n2.pos
                x_text = (x1+x2)/2
                y_text = (y1+y2)/2

            elif len(nodes)>2:
                raise ValueError(f"len(nodes) = {len(nodes)}")
            
            elif len(nodes)==1:
                node = nodes[0]
                direction = node.directions[ node.edges.index(edge) ] 
                x1, y1 = node.pos
                x2, y2 = tuples.add((x1, y1), direction.unit_vector)
                x_text = x2
                y_text = y2

            plt.plot([x1, x2], [y1, y2], edge_style)
            plt.text(x_text, y_text, edge, color="green")
            

    print("Done")

## constants:

_delta_xs = [0, -1,  1]
_delta_ys = [1, -1, -1]

_CONSTANT_X_SHIFT = 3
_CONSTANT_y_SHIFT = 1

"""
An Upper Triangle:
     Up
      |
      |
      O
     / \
    /   \
Left    Right
"""

## Naming shortcuts:
L  = LatticeDirection.L 
R  = LatticeDirection.R 
UL = LatticeDirection.UL 
UR = LatticeDirection.UR 
DL = LatticeDirection.DL 
DR = LatticeDirection.DR


class KagomeLatticeError(LatticeError):...


@dataclass(frozen=False, slots=True)
class UpperTriangle:
    up    : Node = None
    left  : Node = None
    right : Node = None
    #
    index : int = -1

    def __getitem__(self, key:str)->Node:
        match key:
            case 'up'   : return self.up
            case 'left' : return self.left
            case 'right': return self.right
            case _:
                raise KeyError(f"Not a valid key {key!r}")
            
    def __setitem__(self, key:str, value:Any)->None:
        match key:
            case 'up'   : self.up    = value
            case 'left' : self.left  = value
            case 'right': self.right = value
            case _: 
                raise KeyError("Not an option")

    def all_nodes(self)->Generator[Node, None, None]:
        yield self.up
        yield self.left
        yield self.right
    
    @staticmethod
    def field_names()->list[str]:
        return ['up', 'left', 'right']

class _UnassignedEdgeName():
    def __repr__(self) -> str:
        return "_UnassignedEdgeName"
    


def num_message_connections(N:int)->int:
    return 2*N - 1


def edge_name_from_indices(i1:int, i2:int)->str:
    if   i1<i2:  return f"{i1}-{i2}" 
    elif i1>i2:  return f"{i2}-{i1}" 
    else:
        raise ValueError("Indices must be of different nodes") 


def _derive_node_directions(field:str)->list[LatticeDirection]:
    match field:
        case "up"   : return [LatticeDirection.UL, LatticeDirection.DL, LatticeDirection.DR, LatticeDirection.UR]
        case "left" : return [LatticeDirection.L, LatticeDirection.DL, LatticeDirection.R, LatticeDirection.UR]
        case "right": return [LatticeDirection.UL, LatticeDirection.L, LatticeDirection.DR, LatticeDirection.R]
        case _: raise ValueError(f"Unexpected string {field!r}")


def _tag_boundary_nodes(triangle:UpperTriangle, boundary:BlockSide)->None:
    touching_nodes : list[Node] = []
    if   boundary is BlockSide.U:     touching_nodes = [triangle.up]
    elif boundary is BlockSide.DL:    touching_nodes = [triangle.left]
    elif boundary is BlockSide.DR:    touching_nodes = [triangle.right]
    elif boundary is BlockSide.D:     touching_nodes = [triangle.left, triangle.right]
    elif boundary is BlockSide.UR:    touching_nodes = [triangle.up, triangle.right]
    elif boundary is BlockSide.UL:    touching_nodes = [triangle.up, triangle.left]
    else: 
        raise DirectionError()

    for node in touching_nodes:
        node.boundaries.add(boundary)


def get_upper_triangle_vertices_order(major_direction:BlockSide, minor_direction:LatticeDirection) -> list[list[str]]:
    match major_direction:
        case BlockSide.U:
            if   minor_direction is LatticeDirection.R:    return [['left', 'right'], ['up']]
            elif minor_direction is LatticeDirection.L:    return [['right', 'left'], ['up']]
            else: raise DirectionError("Impossible")
        case BlockSide.UR:
            if   minor_direction is LatticeDirection.DR:    return [['left'], ['up', 'right']]
            elif minor_direction is LatticeDirection.UL:    return [['left'], ['right', 'up']]
            else: raise DirectionError("Impossible")
        case BlockSide.UL:
            if   minor_direction is LatticeDirection.UR:    return [['right'], ['left', 'up']]
            elif minor_direction is LatticeDirection.DL:    return [['right'], ['up', 'left']]
            else: raise DirectionError("Impossible")
        case BlockSide.D:
            return lists.reversed(get_upper_triangle_vertices_order(BlockSide.U, minor_direction))
        case BlockSide.DL:
            return lists.reversed(get_upper_triangle_vertices_order(BlockSide.UR, minor_direction))
        case BlockSide.DR:
            return lists.reversed(get_upper_triangle_vertices_order(BlockSide.UL, minor_direction))

def _create_upper_triangle(triangular_node:Node, indices:list[int])->UpperTriangle:
    upper_triangle = UpperTriangle()
    x0, y0 = triangular_node.pos

    ## Derive Position and Directions:
    for node_index, field, delta_x, delta_y in zip(indices, UpperTriangle.field_names(), _delta_xs, _delta_ys, strict=True):
        x = x0 + delta_x + _CONSTANT_X_SHIFT
        y = y0 + delta_y + _CONSTANT_y_SHIFT
        node = Node(
            index=node_index,
            pos=(x, y),
            edges=[_UnassignedEdgeName(), _UnassignedEdgeName(), _UnassignedEdgeName(), _UnassignedEdgeName()],
            directions=_derive_node_directions(field)
        )
        upper_triangle.__setattr__(field, node)
    return upper_triangle
    
    
def _connect_kagome_nodes_inside_triangle(upper_triangle:UpperTriangle)->None:
        up, left, right = upper_triangle.up, upper_triangle.left, upper_triangle.right 
        # Up-Left:
        edge_name = edge_name_from_indices(up.index, left.index)
        up.edges[up.directions.index(DL)] = edge_name
        left.edges[left.directions.index(UR)] = edge_name
        # Up-Right:
        edge_name = edge_name_from_indices(up.index, right.index)
        up.edges[up.directions.index(DR)] = edge_name
        right.edges[right.directions.index(UL)] = edge_name
        # Left-Right:
        edge_name = edge_name_from_indices(left.index, right.index)
        left.edges[left.directions.index(R)] = edge_name
        right.edges[right.directions.index(L)] = edge_name   


def _name_outer_edges(node:Node, order_ind:int, boundary:BlockSide, kagome_lattice:list[Node], N:int)->None:
    upper_triangle = get_upper_triangle(node.index, kagome_lattice, N)
    _edge_name = lambda ind: f"{boundary}-{ind}"

    if boundary is BlockSide.D:     
        if   node is upper_triangle.left:   node.set_edge_in_direction(DL, _edge_name(order_ind))
        elif node is upper_triangle.right:  node.set_edge_in_direction(DR, _edge_name(order_ind))
        else:
            raise LatticeError()
        
    elif boundary is BlockSide.DR:    
        assert node is upper_triangle.right
        node.set_edge_in_direction(DR, _edge_name(2*order_ind))
        node.set_edge_in_direction(R, _edge_name(2*order_ind+1))

    elif boundary is BlockSide.UR:    
        if   node is upper_triangle.right:  node.set_edge_in_direction(R,  _edge_name(order_ind))
        elif node is upper_triangle.up:     node.set_edge_in_direction(UR, _edge_name(order_ind))
        else: 
            raise LatticeError()

    elif boundary is BlockSide.U:     
        assert node is upper_triangle.up
        node.set_edge_in_direction(UR, _edge_name(2*order_ind))
        node.set_edge_in_direction(UL, _edge_name(2*order_ind+1))

    elif boundary is BlockSide.UL:    
        if   node is upper_triangle.up:     node.set_edge_in_direction(UL, _edge_name(order_ind))
        elif node is upper_triangle.left:   node.set_edge_in_direction(L,  _edge_name(order_ind))
        else: 
            raise LatticeError()

    elif boundary is BlockSide.DL:    
        assert node is upper_triangle.left
        node.set_edge_in_direction(L,  _edge_name(2*order_ind))
        node.set_edge_in_direction(DL, _edge_name(2*order_ind+1))

    else:   
        raise DirectionError("Not a possible hexagonal lattice direction")    


def _connect_kagome_nodes_between_triangles(triangle1:UpperTriangle, triangle2:UpperTriangle, direction1to2:LatticeDirection)->None:
    """ 
    Given two upper triangles `triangle1` and `triangle2`, 
    where the `triangle2` is in direction `direction1to2` relative to `triangle1`,
    find the relevant nodes, and assign common edge between them
    """

    ## Choose the two relevant nodes:
    if   direction1to2 is L:
        n1 = triangle1.left
        n2 = triangle2.right

    elif direction1to2 is DL:
        n1 = triangle1.left
        n2 = triangle2.up

    elif direction1to2 is DR:
        n1 = triangle1.right
        n2 = triangle2.up

    elif direction1to2 is R:
        n1 = triangle1.right
        n2 = triangle2.left

    elif direction1to2 is UR:
        n1 = triangle1.up
        n2 = triangle2.left

    elif direction1to2 is UL:
        n1 = triangle1.up
        n2 = triangle2.right

    else: 
        raise DirectionError(f"Impossible direction {direction1to2!r}")

    ## Assign proper edge name to them:
    edge_name = edge_name_from_indices(n1.index, n2.index)
    leg_index1 = n1.directions.index(direction1to2)
    leg_index2 = n2.directions.index(direction1to2.opposite())
    n1.edges[leg_index1] = edge_name
    n2.edges[leg_index2] = edge_name

def _sorted_boundary_nodes(nodes:list[Node], boundary:BlockSide)->list[Node]:
    # Get relevant nodes:
    boundary_nodes = [node for node in nodes if boundary in node.boundaries]

    # Choose sorting key:
    if   boundary is BlockSide.U:     sorting_key = lambda node: -node.pos[0]
    elif boundary is BlockSide.UR:    sorting_key = lambda node: +node.pos[1] 
    elif boundary is BlockSide.DR:    sorting_key = lambda node: +node.pos[1]
    elif boundary is BlockSide.UL:    sorting_key = lambda node: -node.pos[1]
    elif boundary is BlockSide.DL:    sorting_key = lambda node: -node.pos[1]
    elif boundary is BlockSide.D:     sorting_key = lambda node: +node.pos[0]
    else:
        raise DirectionError(f"Impossible direction {boundary!r}")

    # Sort:
    return sorted(boundary_nodes, key=sorting_key)


def get_upper_triangle(node_index:int, nodes:list[Node], N:int)->UpperTriangle:
    triangle_index = node_index//3
    up_index = triangle_index*3
    left_index, right_index = up_index+1, up_index+2
    return UpperTriangle(
        up    = nodes[up_index],
        left  = nodes[left_index],
        right = nodes[right_index]
    )


def create_kagome_lattice(
    N:int
)->tuple[
    list[Node],
    list[UpperTriangle]
]:

    ## Create the triangular lattice we're based on:
    original_triangular_lattice = create_triangle_lattice(N)
    triangular_lattice_of_upper_triangles : list[UpperTriangle] = []

    ## Position upper-triangles at each node of the kagome lattice:
    kagome_lattice : list[Node] = []
    crnt_kagome_index = 0
    for triangular_node in original_triangular_lattice:
        # Assign crnt indices for the triangle:
        indices = list(range(crnt_kagome_index, crnt_kagome_index+3))
        crnt_kagome_index += 3

        # Scale up the distance between nodes:
        triangular_node.pos = tuples.multiply(triangular_node.pos, (2,4))

        # Create triangle:
        upper_triangle = _create_upper_triangle(triangular_node, indices)
        upper_triangle.index = len(triangular_lattice_of_upper_triangles)
        kagome_lattice.extend(upper_triangle.all_nodes())
        triangular_lattice_of_upper_triangles.append(upper_triangle)
        
    ## Assign Inner edges within the triangle:
    for upper_triangle in triangular_lattice_of_upper_triangles:
        _connect_kagome_nodes_inside_triangle(upper_triangle)         

    ## Assign Edges between triangles:
    for index1, triangle1 in enumerate(triangular_lattice_of_upper_triangles):
        for index2, direction1 in all_neighbors(index1, N):
            triangle2 = triangular_lattice_of_upper_triangles[index2]
            _connect_kagome_nodes_between_triangles(triangle1, triangle2, direction1)

    ## Tag all nodes on boundary:
    for triangle in triangular_lattice_of_upper_triangles:
        on_boundaries = check_boundary_vertex(triangle.index, N)
        for boundary in on_boundaries:
            _tag_boundary_nodes(triangle, boundary)

    ## Use ordered nodes to name Outer Edges
    for boundary in BlockSide.all_in_counter_clockwise_order():
        sorted_nodes = _sorted_boundary_nodes(kagome_lattice, boundary)
        for i, node in enumerate(sorted_nodes):
            _name_outer_edges(node, i, boundary, kagome_lattice, N)
    # The bottom-left node is falsely on its DL leg, fix it:
    bottom_left_corner_node = _sorted_boundary_nodes(kagome_lattice, BlockSide.D)[0]
    bottom_left_corner_node.set_edge_in_direction(DL, f"{BlockSide.D}-0")

    ## Plot test:
    if False:
        plot_lattice(kagome_lattice)
        plot_lattice(original_triangular_lattice, node_color="black", edge_style="y--", node_size=5)
        draw_now()



    return kagome_lattice, triangular_lattice_of_upper_triangles



class MessageDictType(dict[BlockSide, list[Node]]): ...


class KagomeLattice():
    __slots__ =  "N", "nodes", "triangles", "edges", "messages"
    
    def __init__(self, N:int) -> None:
        kagome_lattice, triangular_lattice_of_upper_triangles = create_kagome_lattice(N)
        self.N : int = N
        self.nodes     : list[Node] = kagome_lattice
        self.triangles : list[UpperTriangle]  = triangular_lattice_of_upper_triangles
        self.edges     : dict[str, tuple[int, int]] = edges_dict_from_edges_list(
            [node.edges for node in kagome_lattice]
        )
        self.messages : MessageDictType = {} 


    def plot(self) -> None:
        plot_lattice(self.nodes, edges_dict=self.edges)
        draw_now()


    # ================================================= #
    #|              Basic Derived Properties           |#
    # ================================================= #                    
    @property
    def num_message_connections(self)->int:
        return num_message_connections(self.N)
    
    @property
    def size(self)->int:
        return len(self.nodes)

    # ================================================= #
    #|              Geometry Functions                 |#
    # ================================================= #
    def num_boundary_nodes(self, boundary:BlockSide)->int:
        if boundary in [BlockSide.U, BlockSide.DR, BlockSide.DL]:
            return self.N
        elif boundary in [BlockSide.D, BlockSide.UR, BlockSide.UL]:
            return 2*self.N
        else:
            raise DirectionError("Not a possible boundary direction")

    @functools.cache
    def nodes_indices_rows_in_direction(self, major_direction:BlockSide, minor_direction:LatticeDirection)->list[list[int]]:
        ## Prepare basic data:
        N = self.N
        min_x, max_x, min_y, max_y = self.position_min_max()
        assert check.is_orthogonal(major_direction, minor_direction)
        crnt_vertices_order = get_upper_triangle_vertices_order(major_direction, minor_direction)

        ## Get Upper-Triangles sorted in wanted direction:
        triangle_indices_in_order = vertices_indices_rows_in_direction(N, major_direction, minor_direction)

        ## The results, are each row of upper-triangles, twice, taking the relevant node from the upper-triangle:
        list_of_rows = []
        for row in triangle_indices_in_order:    # Upper-Triangle order:
            for vertices_names in crnt_vertices_order:
                row_indices = self._row_in_direction(row, vertices_names)
                list_of_rows.append(row_indices)
        return list_of_rows


    @functools.cache
    def position_min_max(self)->tuple[int, ...]:
        min_x, max_x = lists.min_max([node.pos[0] for node in self.nodes])
        min_y, max_y = lists.min_max([node.pos[1] for node in self.nodes])
        return min_x, max_x, min_y, max_y

    def get_center_triangle(self)->UpperTriangle:
        index = get_center_vertex_index(self.N)
        return self.triangles[index]

    # ================================================= #
    #|            Retrieve Inner Objects               |#
    # ================================================= #

    def nodes_and_triangles(self)->Generator[tuple[Node, UpperTriangle, ], None, None]:
        triangles_repeated_3_times = itertools.chain.from_iterable(itertools.repeat(triangle, 3) for triangle in self.triangles)
        return zip(self.nodes, triangles_repeated_3_times, strict=True)
    
    
    def get_neighbor(self, node:Node, edge_or_dir:EdgeIndicatorType|LatticeDirection)->Node:
        if isinstance(edge_or_dir, EdgeIndicatorType):
            edge = edge_or_dir
        elif isinstance(edge_or_dir, Direction):
            edge = node.get_edge_in_direction(edge_or_dir)
        else:
            raise TypeError(f"Not an expected type {type(edge_or_dir)!r}")

        i1, i2 = self.edges[edge]
        assert i1!=i2
        if i1 == node.index:
            return self.nodes[i2]
        elif i2 == node.index:
            return self.nodes[i1]
        else:
            raise LatticeError("No neighbor")
        
    @functools.cache
    def sorted_boundary_nodes(self, boundary:BlockSide)->list[Node]:
        return _sorted_boundary_nodes(self.nodes, boundary)
    
    @functools.cache
    def sorted_boundary_edges(self, boundary:BlockSide)->list[EdgeIndicatorType]:
        # Basic info:
        num_boundary_nodes = self.num_boundary_nodes(boundary)
        boundary_nodes = self.sorted_boundary_nodes(boundary)
        assert len(boundary_nodes)==num_boundary_nodes

        # Logic of participating directions:
        participating_directions = boundary.matching_lattice_directions()

        # Logic of participating edges and nodes:
        omit_last_edge = num_boundary_nodes==self.N
        omit_last_node = not omit_last_edge
        
        # Get all edges in order:
        boundary_edges = []
        for _, is_last_node, node in lists.iterate_with_edge_indicators(boundary_nodes):
            if omit_last_node and is_last_node:
                break

            for _, is_last_direction, direction in lists.iterate_with_edge_indicators(participating_directions):
                if omit_last_edge and is_last_node and is_last_direction:
                    break

                if direction in node.directions:
                    boundary_edges.append(node.get_edge_in_direction(direction))

        assert self.num_message_connections == len(boundary_edges)
        return boundary_edges

    def _row_in_direction(self, triangle_indices:list[int], triangle_keys:list[str]) -> list[int]:
        node_indices = []
        for triangle_index in triangle_indices:
            triangle = self.triangles[triangle_index]
            for key in triangle_keys:
                node : Node = triangle.__getattribute__(key)
                node_indices.append(node.index)
        return node_indices
    
    # ================================================= #
    #|                    Visuals                      |#
    # ================================================= #
    def plot_triangles_lattice(self)->None:
        # Visuals import:
        from matplotlib import pyplot as plt
        # basic data:
        N = self.N
        # Plot triangles:
        for upper_triangle in self.triangles:
            ind = upper_triangle.index
            i, j = get_vertex_coordinates(ind, N)
            x, y = get_node_position(i, j, N)
            plt.scatter(x, y, marker="^")
            plt.text(x, y, f" [{ind}]")


    # ================================================= #
    #|                    messages                     |#
    # ================================================= #
    @property
    def has_messages(self)->bool:
        return len(self.messages)==6

    def connect_messages(self) -> None:   
        nodes = self.nodes
        crnt_node_index = self.size
        for side in BlockSide.all_in_counter_clockwise_order():
            if side in BlockSide.all_in_counter_clockwise_order():
                message_nodes = _message_nodes(self, side, crnt_node_index)
                nodes.extend(message_nodes)
                crnt_node_index += len(message_nodes)
        return nodes

    def message_indices(self, direction:BlockSide)->list[int]:
        return _kagome_lattice_derive_message_indices(self.N, direction)
    
    # ================================================= #
    #|              Geometry Functions                 |#
    # ================================================= #
    @property
    def num_message_connections(self)->int:
        N = self.N
        return 2*N - 1
    
    # ================================================= #
    #|                   Neighbors                     |#
    # ================================================= #
    def nodes_connected_to_edge(self, edge:str)->list[Node]:
        i1, i2 = self.edges[edge]
        if i1 == i2:  
            # Only a single node is connected to this edge. i.e., open edge
            return [ self.nodes[i1] ]
        else:
            return [ self.nodes[i] for i in [i1, i2] ]

    def find_neighbor_by_edge(self, node:Node, edge:EdgeIndicatorType)->Node:
        nodes = self.nodes_connected_to_edge(edge)
        assert len(nodes)==2, f"Only tensor '{node.index}' is connected to edge '{edge}'."
        if nodes[0] is node:
            return nodes[1]
        elif nodes[1] is node:
            return nodes[0]
        else:
            raise ValueError(f"Couldn't find a neighbor due to a bug with the tensors connected to the same edge '{edge}'")
        

def index_of_first_appearance(it:Iterable[_T], item:_T) -> int:
    for i, val in enumerate(it):
        if val==item:
            return i
    raise ValueError("Not found")


@functools.cache
def _kagome_lattice_derive_message_indices(N:int, direction:BlockSide)->list[int]:
    # Get info:
    num_lattice_nodes = total_vertices(N)*3
    message_size = 2*N - 1
    direction_index_in_order = index_of_first_appearance( BlockSide.all_in_counter_clockwise_order(), direction )
    # Compute:
    res = np.arange(message_size) + num_lattice_nodes + message_size*direction_index_in_order
    return res.tolist()


def _kagome_disconnect_boundary_edges_to_open_nodes(kagome_lattice:KagomeLattice, edge1:str, edge2:str) -> None:
    ## assert an edge at boundary:
    assert kagome_lattice.edges[edge1][0] == kagome_lattice.edges[edge1][1]
    assert kagome_lattice.edges[edge2][0] == kagome_lattice.edges[edge2][1]

    ## Define node creation function:
    new_node_index = kagome_lattice.size
    def _kagome_disconnect_boundary_edges_to_open_nodes_create_node(edge:str)->Node:
        nonlocal new_node_index

        # Find neighbor and its orientation to the open leg:
        neighbor_index = kagome_lattice.edges[edge][0]
        neighbor = kagome_lattice.nodes[neighbor_index]
        neighbor_leg_index = neighbor.edges.index(edge)
        neighbor_leg_direction = neighbor.directions[neighbor_leg_index]

        # Create node:
        open_node = Node(
            index=new_node_index,
            pos=tuples.add(neighbor.pos, neighbor_leg_direction.unit_vector), 
            edges=[edge], 
            directions=[neighbor_leg_direction.opposite()]
        )
        new_node_index += 1

        return open_node

    ## Create two new tensors:
    n1 = _kagome_disconnect_boundary_edges_to_open_nodes_create_node(edge1)
    n2 = _kagome_disconnect_boundary_edges_to_open_nodes_create_node(edge2)
    
    ## connect new nodes to lattice:
    kagome_lattice.nodes.append(n1)
    kagome_lattice.nodes.append(n2)
    kagome_lattice.edges[edge1]  = tuples.copy_with_replaced_val_at_index(kagome_lattice.edges[edge1], 1, n1.index)
    kagome_lattice.edges[edge2]  = tuples.copy_with_replaced_val_at_index(kagome_lattice.edges[edge2], 1, n2.index)


def _derive_message_node_position(nodes_on_boundary:list[Node], edge:EdgeIndicatorType, boundary_delta:tuple[float,...]):
    node = next(node for node in nodes_on_boundary if edge in node.edges)
    direction = node.directions[ node.edges.index(edge) ]
    delta = tuples.add(boundary_delta, direction.unit_vector)
    # delta = (numerics.furthest_absolute_integer(delta[0]), numerics.furthest_absolute_integer(delta[1]))
    return tuples.add(node.pos, delta)     


def _message_nodes(
    lattice : KagomeLattice,
    boundary_side : BlockSide,
    first_node_index : int
) -> list[Node]:

    # Unpack Inputs:
    msg_dir_to_lattice : BlockSide = boundary_side.opposite()
    mps_order_dir = msg_dir_to_lattice.orthogonal_clockwise_lattice_direction() 

    ## Check data:
    if DEBUG_MODE:
        assert check.is_orthogonal(boundary_side, mps_order_dir), f"MPS must be orthogonal in its own ordering direction to the lattice"

    # Where is the message located compared to the lattice:
    boundary_delta = boundary_side.unit_vector

    # Get all tensors on edge:
    nodes_on_boundary = lattice.sorted_boundary_nodes(boundary_side)
    edges_on_boundary = lattice.sorted_boundary_edges(boundary_side)
    assert ( message_length := len(edges_on_boundary) ) == lattice.num_message_connections

    ## derive shared properties of all message tensors:
    message_edge_names = [ "M-"+str(boundary_side)+f"-{i}" for i in range(message_length-1) ]
    edges_per_message_tensor = [(message_edge_names[0],)] + list( itertools.pairwise(message_edge_names) ) + [(message_edge_names[-1],)]
    message_positions = [_derive_message_node_position(nodes_on_boundary, edge, boundary_delta) for edge in edges_on_boundary]

    ## Prepare output::
    index = 0
    res = []    
    
    ## Connect each message-tensor and its corresponding lattice node:
    for (is_first, is_last, edge_to_lattice), m_edges, m_pos in \
        zip( lists.iterate_with_edge_indicators(edges_on_boundary), edges_per_message_tensor, message_positions, strict=True ):

        ## Change tensor shape according to it's position on the lattice
        if is_first:
            directions = [msg_dir_to_lattice, mps_order_dir]
            edges = [ edge_to_lattice, m_edges[0] ]
        elif is_last:
            directions = [mps_order_dir.opposite(), msg_dir_to_lattice]
            edges = [ m_edges[0], edge_to_lattice ]
        else:
            directions = [mps_order_dir.opposite(), msg_dir_to_lattice,  mps_order_dir]
            assert len(m_edges)==2
            edges = [ m_edges[0], edge_to_lattice , m_edges[1] ]

        ## Add new node to Tensor-Network:
        new_node = Node(
            edges=edges,
            directions=directions, # type: ignore
            pos=m_pos,  # type: ignore
            index=first_node_index+index
        )

        ## Keep edges dict valid:
        for edge in new_node.edges:
            ## fix edges of original lattice:
            if edge in lattice.edges:  # this is an outgoing edge
                boundary_node_index, _same_value = lattice.edges[edge]  # get tensor - should be same value
                assert boundary_node_index==_same_value, "Should be the same tensor, since it is not yet connected!"
                lattice.edges[edge] = (boundary_node_index, new_node.index)

            ## Add new edges:
            else:
                lattice.edges[edge] = (new_node.index, new_node.index)

        res.append(new_node)
        index += 1

    return res




def _new_periodic_edge_name(edge1:str, edge2:str) -> str:
    side1, num1 = edge1.split("-")
    side2, num2 = edge2.split("-")

    # First letter:
    D_or_U_1 = side1[0]
    D_or_U_2 = side2[0]
    assert D_or_U_1 != D_or_U_2
    if D_or_U_1 == "D":
        one_first = True
    elif D_or_U_2 == "D":
        one_first = False
    else:
        raise ValueError("Not an expected case!")
    
    # Order with Down side first
    if one_first:
        return side1+"-"+side2+"-"+num1
    else:
        return side2+"-"+side1+"-"+num2
    

def _kagome_connect_boundary_edges_periodically(kagome_lattice:KagomeLattice, edge1:str, edge2:str) -> None:
    ## assert an edge at boundary:
    assert kagome_lattice.edges[edge1][0] == kagome_lattice.edges[edge1][1]
    assert kagome_lattice.edges[edge2][0] == kagome_lattice.edges[edge2][1]
    
    node1_index = kagome_lattice.edges[edge1][0]
    node2_index = kagome_lattice.edges[edge2][0]         

    ## replace two dict entries with a single entry for two tensors
    new_periodic_edge = _new_periodic_edge_name(edge1, edge2)
    del kagome_lattice.edges[edge1]
    del kagome_lattice.edges[edge2]
    kagome_lattice.edges[new_periodic_edge] = (node1_index, node2_index)

    ## Replace in both nodes, as-well:
    # get nodes:
    node1 = kagome_lattice.nodes[node1_index]
    node2 = kagome_lattice.nodes[node2_index]
    # change edges:
    node1.edges[node1.edges.index(edge1)] = new_periodic_edge
    node2.edges[node2.edges.index(edge2)] = new_periodic_edge


def _connect_boundaries(lattice:KagomeLattice, boundary_condition:str):
    ## connect periodically:
    ordered_half_list = list(BlockSide.all_in_counter_clockwise_order())[0:3]
    for block_side in ordered_half_list:
        boundary_edges          = lattice.sorted_boundary_edges(boundary=block_side)
        opposite_boundary_edges = lattice.sorted_boundary_edges(boundary=block_side.opposite())
        opposite_boundary_edges.reverse()

        for edge1, edge2 in zip(boundary_edges, opposite_boundary_edges, strict=True):
            ## Choose boundary function:
            if boundary_condition=="periodic":
                _kagome_connect_boundary_edges_periodically(lattice, edge1, edge2)
            elif boundary_condition=="open":
                _kagome_disconnect_boundary_edges_to_open_nodes(lattice, edge1, edge2)
            else:
                raise ValueError("Not such option")


def create_kagome_lattice_with_boundary_condition(
    size:int, 
    boundary:str="periodic",  # "periodic"\"open"\"disconnected"
    is_plot:bool=False
) -> tuple[
    dict[str, tuple[int, int]],  # edges dict
    list[list[str]],  # edges list
    list[tuple[int, int]]  # positions list
]:
    # checks:
    assert size>=2, f"size must be an integer bigger than 1"
    assert isinstance(size, int), f"size must be an integer bigger than 1"

    ## Create our classic Kagome exagonal block:
    lattice = KagomeLattice(size)

    ## Fix lattice boundaries:
    if boundary=="disconnected":
        pass
    elif boundary in ["periodic", "open"]:
        _connect_boundaries(lattice, boundary)
    else:
        raise ValueError(f"Not a valid option of input `boundary`={boundary!r}")


    if is_plot:
        plot_lattice(lattice.nodes, periodic=boundary=="periodic")
        draw_now()

    ## Get outputs:
    edges_dict = lattice.edges
    edges_list = [node.edges for node in lattice.nodes]
    pos_list   = [node.pos   for node in lattice.nodes]

    return edges_dict, edges_list, pos_list





## Contraction Order:
from enum import Enum, auto

class ContractionDepth(Enum):
    Full = auto()
    ToMessage = auto()
    ToCore = auto()
    ToEdge = auto()

## Types:
_T = TypeVar("_T")
_EnumType = TypeVar("_EnumType")
_PosFuncType = Callable[[int, int], tuple[int, int] ]
_FULL_CONTRACTION_ORDERS_CACHE_KEY_TYPE : TypeAlias = tuple[int, BlockSide, ContractionDepth]

## Constants:
CACHE_CONTRACTION_ORDER : bool = True
BREAK_MARKER = -100
Full = ContractionDepth.Full
ToCore = ContractionDepth.ToCore
ToMessage = ContractionDepth.ToMessage

## For Caching results:
FULL_CONTRACTION_ORDERS_CACHE : dict[_FULL_CONTRACTION_ORDERS_CACHE_KEY_TYPE, list[int]] = {}


CORE_CONTRACTION_ORDERS : dict[BlockSide, list[int]] = {
    BlockSide.UR : [19, 20, 9, 7, 3, 18, 10, 4, 17, 16, 0, 2, 5, 8, 11, 1, 6, 12, 13, 14, 15],
    BlockSide.DR : [16, 17, 18, 19, 3, 0, 15, 1, 2, 4, 7, 20, 9, 5, 14, 13, 6, 8, 10, 11, 12],
    BlockSide.D  : [16, 15, 14, 1, 0, 17, 13, 2, 18, 19, 3, 4, 5, 6, 12, 7, 8, 11, 10, 9, 20],
}
for side in [BlockSide.UR, BlockSide.DR, BlockSide.D]:
    opposite_list = lists.reversed( CORE_CONTRACTION_ORDERS[side] )
    CORE_CONTRACTION_ORDERS[side.opposite()] = opposite_list



def _validate_core_con_order(contraction_order:list[int])->None:
    given = set(contraction_order)
    needed = set(range(21))
    assert given==needed


class _Bound(Enum):
    first = auto()
    last  = auto()

    def opposite(self)->"_Bound":
        if   self is First: return Last            
        elif self is Last: return First
        else:
            raise ValueError("Not possible")                
    
First = _Bound.first 
Last  = _Bound.last

class _Side(Enum):
    left  = auto() 
    right = auto() 
Left = _Side.left
Right = _Side.right


class _EnumContainer(Generic[_EnumType, _T]):
    def __getitem__(self, key:_EnumType)->_T:
        for field in fields(self):
            if field.name == key.name:
                return getattr(self, field.name)
        raise KeyError(f"No a valid option {key!r}")
    
    def __setitem__(self, key:_EnumType, value:_T)->None:
        for field in fields(self):
            if field.name == key.name:
                return setattr(self, field.name, value)
        raise KeyError(f"No a valid option {key!r}")
            

@dataclass
class _PerBound(_EnumContainer[_Bound, _T]):
    first : _T 
    last  : _T   

    def to_per_side(self, is_reverse_order:bool)->"_PerSide":
        if is_reverse_order:
            return _PerSide(right=self.first, left=self.last)
        else:
            return _PerSide(left=self.first, right=self.last)


@dataclass
class _PerSide(_EnumContainer[_Side, _T]):
    left  : _T 
    right : _T   

    def to_per_order(self, is_reverse_order:bool)->_PerBound[_T]:
        if is_reverse_order:
            return _PerBound(last=self.left, first=self.right)
        else:
            return _PerBound(first=self.left, last=self.right)



class _SideEdges:
    def __init__(self, left_sorted_outer_edges:list[str], right_sorted_outer_edges:list[str]) -> None:
        self.lists : _PerSide[list[str]] = _PerSide(
            left = left_sorted_outer_edges,
            right = right_sorted_outer_edges
        )
        self.crnt_index : _PerSide[int] = _PerSide(
            left = 0,
            right = 0 
        )
        self.exhausted : _PerSide[bool] = _PerSide(
            left = False,
            right = False,
        )

    def crnt(self, side:_Side)->str:
        if self.exhausted[side]:
            return "End"
        index = self.crnt_index[side] 
        item = self.lists[side][index]
        return item

    def next(self, side:_Side)->str:
        self.crnt_index[side] += 1
        try:
            next_item = self.crnt(side)
        except IndexError as e:
            next_item = None
            self.exhausted[side] = True
        return next_item
    
    def __getitem__(self, key:_Side)->_T:
        return self.crnt(key)


class _ReverseOrder:
    __slots__ = ["state", "counter"]
    _num_repeats : Final[int] = 2
    
    def __init__(self) -> None:
        self.counter = 0
        self.state = True
    
    def __bool__(self)->bool:
        return self.state
    
    def __invert__(self)->None:
        self.state = not self.state
        self.counter = 0

    def prev_state(self)->bool:
        if self.counter - 1 < 0:
            return not self.state
        else:
            return self.state

    def check_state_and_update(self)->bool:
        res = self.state
        self.counter += 1
        if self.counter > _ReverseOrder._num_repeats-1:
            ~self   # call invert
        return res

    def set_true(self)->None:
        self.counter = 0
        self.state = True 

    def set_false(self)->None:
        self.counter = 0
        self.state = False 



class _UpToCoreControl:
    """Manages memory of everything we need to know when contraction "Up-To-Core"
    """
    __slots__ = (
        "core_indices", "terminal_indices", "stop_next_iterations", "seen_break"
    )
    
    def __init__(self, tn:KagomeLattice, major_direction:BlockSide) -> None:        
        self.core_indices : set[int] = {node.index for node in tn.get_core_nodes()}
        self.terminal_indices : set[int] = _derive_terminal_indices(tn, major_direction)
        self.stop_next_iterations : bool = False
        self.seen_break : _PerSide[bool] = _PerSide[bool](left=False, right=False)

    def update_seen_break(self, seen_break_now_per_side:_PerSide[bool])->None:
        if seen_break_now_per_side.left:
            self.seen_break.left = True
        if seen_break_now_per_side.right:
            self.seen_break.right = True
        

def _derive_terminal_indices(tn:KagomeLattice, major_direction:BlockSide)->set[int]:
    """derive_terminal_indices
    The convention is that the base of the center upper triangle defines the terminal row for the contraction up to core
    """
    # Minor direction can be random-valid direction not important, since we care only for the indices
    minor_direction = major_direction.orthogonal_clockwise_lattice_direction()
    # Get center triangle:
    center_triangle : UpperTriangle = tn.get_center_triangle()

    ## find the vertex_names of the base, by checking which list has 2 values:
    upper_triangle_order = get_upper_triangle_vertices_order(major_direction, minor_direction)
    # Get the names:
    if len(upper_triangle_order[0])==2:
        vertex_names = upper_triangle_order[0]
    elif len(upper_triangle_order[1])==2:
        vertex_names = upper_triangle_order[1]
    else:
        raise ValueError("Impossible situation. Bug.")
    
    ## Get the indices:
    indices = {center_triangle[name].index for name in vertex_names}

    return indices
        

def _sorted_side_outer_edges(
    lattice:KagomeLattice, direction:BlockSide, with_break:bool=False
)->tuple[
    list[str],
    list[str]
]:
    ## Get all boundary edges. They are sorted to "the right"
    sorted_right_boundary_edges : dict[BlockSide, list[str]] = {
        side: lattice.sorted_boundary_edges(side) for side in BlockSide.all_in_counter_clockwise_order()
    }
    right_last = direction.next_clockwise()
    right_first = right_last.next_clockwise()
    left_last = direction.next_counterclockwise()
    left_first = left_last.next_counterclockwise()

    ## Right edge orders:
    if with_break:
        right_edges = sorted_right_boundary_edges[right_first] + ['Break'] + sorted_right_boundary_edges[right_last] 
        left_edges = sorted_right_boundary_edges[left_last] + ['Break'] + sorted_right_boundary_edges[left_first] 
    else:
        right_edges = sorted_right_boundary_edges[right_first] + sorted_right_boundary_edges[right_last] 
        left_edges = sorted_right_boundary_edges[left_last] + sorted_right_boundary_edges[left_first] 
    left_edges.reverse()

    return left_edges, right_edges

def _decide_by_future_neighbors(
    side_order:_PerBound[_Side], 
    num_neighbors_per_side_before:_PerBound[list[int]], 
    num_neighbors_per_side_after:_PerBound[list[int]]
)->_Side|None:
    ## Decide which side should continue the break:
    for first_last in _Bound:            
        side = side_order[first_last]
        ## How many MPS messages are and were connected:
        num_bfore = num_neighbors_per_side_before[side]
        num_after = num_neighbors_per_side_after[side]
        if num_bfore==0 or num_after==0:
            raise ValueError("Not an expected case")
        elif num_bfore==1: 
            if num_after==2:
                return side
            else:
                continue
        elif num_bfore==2:
            assert num_after==2, "if before we had 2 neighbors, we must still have 2 after."
        else:
            raise ValueError("Not an expected case")
    return None


def _message_break_next_order_logic(
    reverse_order_tracker:_ReverseOrder, seen_break:_PerBound[bool], side_order:_PerBound[_Side],
    num_neighbors_per_side_before:_PerBound[list[int]],
    num_neighbors_per_side_after:_PerBound[list[int]],
)->_ReverseOrder:
    
    ## If only one of the sides seen the break:
    if seen_break.first and not seen_break.last:
        next_side = side_order[Last]
    elif seen_break.last and not seen_break.first:
        next_side = side_order[First]
    elif not seen_break.first and not seen_break.last:
        raise ValueError("Bug. We shouldn't be here.")
    else:
        next_side = _decide_by_future_neighbors(side_order, num_neighbors_per_side_before, num_neighbors_per_side_after)
    
    ## Decide on new order::
    if next_side is Left:
        reverse_order_tracker.set_false()
    elif next_side is Right:
        reverse_order_tracker.set_true()
    elif next_side is None:
        pass   # Keep it as it is
    else:
        raise ValueError("Not an expected option!")
    #
    return reverse_order_tracker


def _find_all_side_neighbors( 
    tn:KagomeLattice, first_last:_Bound,
    nodes:_PerSide[Node], side_order:_PerBound[_Side], 
    side_edges:_SideEdges
)->tuple[
    EdgeIndicatorType,  # last_edge
    list[int]  # neighbor indices
]:
    # prepare results:
    res = []
    # unpack respective values:
    side = side_order[first_last]            
    node = nodes[side]
    edge = side_edges[side]           
    while edge in node.edges:     
        # get neighbor:
        neighbor = tn.find_neighbor_by_edge(node, edge)
        # add neighbor:
        res.append(neighbor.index)  # Add to end 
        # move iterator and get next edge
        edge = side_edges.next(side)  
    return edge, res  # return last edge


def _derive_message_neighbors(
    tn:KagomeLattice, row:list[int], depth:ContractionDepth,
    reverse_order_tracker:_ReverseOrder,
    side_edges:_SideEdges
) -> tuple[
    _PerBound[list[int]],
    list[int],
    _PerBound[bool]
]:
    # Node at each side
    nodes = _PerSide[Node](
        left  = tn.nodes[row[0]],
        right = tn.nodes[row[-1]]
    )

    ## Prepare outputs:
    msg_neighbors_at_bounds = _PerBound[list[int]](first=[], last=[])
    annex_neighbors = []

    # which side is first?       
    if reverse_order_tracker.check_state_and_update():
        side_order = _PerBound[_Side](first=Right, last=Left)
    else:
        side_order = _PerBound[_Side](first=Left, last=Right)

    ## Look for breaks between MPS messages:
    seen_break = _PerBound[bool](first=False, last=False)

    ## Add neighbors if they exist:
    for first_last in _Bound:
        last_edge, neighbors = _find_all_side_neighbors(tn, first_last, nodes, side_order, side_edges )
        msg_neighbors_at_bounds[first_last] += neighbors
            
        ## Special case when MPS messages change from one to another:
        if last_edge == 'Break':
            side = side_order[first_last]
            side_edges.next(side)  # move iterator
            seen_break[first_last] = True

    ## At a break in the side MPS messages:
    if seen_break.first or seen_break.last: 

        # Keep data about neighbors per side:
        num_neighbors_per_side_before = _PerSide[int](left=0, right=0)
        num_neighbors_per_side_before[side_order[First]] = len(msg_neighbors_at_bounds[First])
        num_neighbors_per_side_before[side_order[Last]] = len(msg_neighbors_at_bounds[Last])
        num_neighbors_per_side_after = deepcopy(num_neighbors_per_side_before)

        # Collect lost neighbors
        for first_last in _Bound:
            _, neighbors = _find_all_side_neighbors(tn, first_last, nodes, side_order, side_edges )
            num_neighbors_per_side_after[side_order[first_last]] += len(neighbors)
            annex_neighbors += neighbors  # assign to separate list

        # Next order logic
        reverse_order_tracker = _message_break_next_order_logic(reverse_order_tracker, seen_break, side_order, num_neighbors_per_side_before, num_neighbors_per_side_after)

    ## Seen break per side:
    return msg_neighbors_at_bounds, annex_neighbors, seen_break



def _split_row(row:list[int], row_and_core:set[int])->_PerSide[list[int]]:
    splitted_row = _PerSide(left=[], right=[])
    seen_core = False
    for i in row:
        if i in row_and_core:
            seen_core = True
            continue

        if seen_core:
            splitted_row.right.append(i)
        else:
            splitted_row.left.append(i)

    return splitted_row


def _derive_full_row(row_in:list[int], reverse_order_now:bool, msg_neighbors:_PerBound[list[int]], annex_neighbors:list[int]): 
    if reverse_order_now:
        row_reversed = row_in.copy()
        row_reversed.reverse() 
        return msg_neighbors.first + row_reversed + msg_neighbors.last + annex_neighbors
    else:
        return msg_neighbors.first + row_in + msg_neighbors.last + annex_neighbors


def _derive_row_to_contract(
    control:_UpToCoreControl,
    row:list[int], 
    depth:ContractionDepth, 
    reverse_order_now:bool, 
    msg_neighbors:_PerBound[list[int]], 
    annex_neighbors:list[int],
    seen_break_per_order:_PerBound[bool]  # can be used for debug
)->list[int]:

    ## simple cases:
    if depth is not ToCore: 
        return _derive_full_row(row, reverse_order_now, msg_neighbors, annex_neighbors)
    if control.stop_next_iterations:
        return []  # don't append anything from now on    

    ## Derive intersection between core indices and current row
    core_in_this_row = set(row).intersection(control.core_indices)
    if len(core_in_this_row)==0:
        return _derive_full_row(row, reverse_order_now, msg_neighbors, annex_neighbors)

    ## Prepare values arranged by side:
    # Split row, ignore core indices:
    row_per_side = _split_row(row, core_in_this_row)
    seen_break_now_per_side = seen_break_per_order.to_per_side(reverse_order_now)

    ## If we haven't arrived at a terminal row, all is simple:
    if core_in_this_row.isdisjoint(control.terminal_indices):
        row_per_bound = row_per_side.to_per_order(reverse_order_now)
        # Track if we've seen the break in the side messages:
        control.update_seen_break(seen_break_now_per_side)  
        return msg_neighbors.first + row_per_bound.first + row_per_bound.last + msg_neighbors.last + annex_neighbors

    ## if we are here, this is the last left-side contraction and right shouldn't continue    
    control.stop_next_iterations = True
    msg_neighbors_per_side = msg_neighbors.to_per_side(reverse_order_now)

    ## Add right neighbor, if it is unreachable by the opposite contraction:
    if not control.seen_break.right and seen_break_now_per_side.right:
        assert len(msg_neighbors_per_side.right)>0
        return msg_neighbors_per_side.left + row_per_side.left + msg_neighbors_per_side.right


    ## Return last left row:    
    return msg_neighbors_per_side.left + row_per_side.left


def derive_kagome_tn_contraction_order(
    lattice:KagomeLattice,  
    direction:BlockSide,
    depth:BlockSide
)->list[int]:

    ## Prepare output:
    contraction_order = []    

    # Define Directions:
    major_direction = direction
    minor_right = direction.orthogonal_clockwise_lattice_direction()

    ## Start by fetching the lattice-nodes in order and the messages:
    lattice_rows_ordered_right = lattice.nodes_indices_rows_in_direction(major_direction, minor_right)
    # Side edges:
    left_sorted_outer_edges, right_sorted_outer_edges = _sorted_side_outer_edges(lattice, major_direction, with_break=True)
    side_edges = _SideEdges(left_sorted_outer_edges, right_sorted_outer_edges)
    
    ## Helper objects: 
    # Iterator to switch contraction direction every two rows:
    reverse_order_tracker = _ReverseOrder()
    # Control different sides advance when contracting up to core:
    if depth is ToCore:
        up_to_core_control = _UpToCoreControl(lattice, major_direction)
    else:
        up_to_core_control = None


    ## First message:
    contraction_order += lattice.message_indices(major_direction.opposite()) 

    ## Lattice and its connected nodes:
    for row in lattice_rows_ordered_right:

        reverse_order_now = reverse_order_tracker.state
        msg_neighbors, annex_neighbors, seen_break_per_order = _derive_message_neighbors(lattice, row, depth, reverse_order_tracker, side_edges)
        row_to_contract = _derive_row_to_contract(up_to_core_control, row, depth, reverse_order_now, msg_neighbors, annex_neighbors, seen_break_per_order)

        ## add row to con_order:
        contraction_order.extend( row_to_contract )

    ## Last minor-direction of contraction:
    # make opposite than last direction
    last_order_was_reversed = reverse_order_tracker.prev_state()
    final_order_is_reversed = not last_order_was_reversed
    if final_order_is_reversed:
        last_minor_direction = minor_right.opposite()
    else:
        last_minor_direction = minor_right

    ## Last msg:
    if depth is Full:
        last_msg = lattice.message_indices(major_direction)
        if not final_order_is_reversed:
            last_msg.reverse()
        contraction_order += last_msg

    ## Validate:
    if DEBUG_MODE and depth in [ToMessage, Full]:
        # Both list of messages are in `con_order`:
        assert side_edges.exhausted.left
        assert side_edges.exhausted.right    

    ## we can return `last_minor_direction` if this is important. It is usually NOT.
    return contraction_order


def plot_contraction_order(positions:list[tuple[int,...]], con_order:list[int])->None:
    # for Color transformations 
    import colorsys
    def color_gradient(num_colors:int):
        for i in range(num_colors):
            rgb = colorsys.hsv_to_rgb(i / num_colors, 1.0, 1.0)
            yield rgb
    from matplotlib import pyplot as plt

    not_all_the_way = 0.85
    for (from_, to_), color in zip(itertools.pairwise(con_order), color_gradient(len(positions)) ):
        # if from_<0 or to_<0:
        #     continue  # just a marker for something
        x1, y1 = positions[from_]
        x2, y2 = positions[to_]
        plt.arrow(
            x1, y1, (x2-x1)*not_all_the_way, (y2-y1)*not_all_the_way, 
            width=0.20,
            color=color,
            zorder=0
        )

def get_contraction_order(lattice:KagomeLattice, direction:BlockSide, depth:ContractionDepth, plot_:bool=False)->list[int]:

    global FULL_CONTRACTION_ORDERS_CACHE
    cache_key = (lattice.N, direction, depth)

    if CACHE_CONTRACTION_ORDER and cache_key in FULL_CONTRACTION_ORDERS_CACHE:
        contraction_order = FULL_CONTRACTION_ORDERS_CACHE[cache_key]
    else:
        # Derive:
        contraction_order = derive_kagome_tn_contraction_order(lattice, direction, depth)
        # Save for next time:
        FULL_CONTRACTION_ORDERS_CACHE[cache_key] = contraction_order
    
    ## Plot result:
    if plot_:
        positions = [node.pos for node in lattice.nodes]
        lattice.plot()
        plot_contraction_order(positions, contraction_order)

    ## Return:
    return contraction_order


def _check_share_edge(i1:int, i2:int, edges_list:list[list[str]]) -> str|None:
    edges1 = set(edges_list[i1])
    edges2 = set(edges_list[i2])
    common_edges = edges1.intersection(edges2)
    if len(common_edges)==0:
        return None
    else:
        assert len(common_edges)==1
        return common_edges.pop()
        


def _fix_edges_list_with_fused_corner(
    edges_list : list[list[str]],     
    edges_dict : dict[str,tuple[int, int]],
    contraction_order : list[int]
)->list[list[str]]: 
    
    output_edges : list[list[str]] = deepcopy(edges_list)
    new_fused_edge_name = f"fused_corner"

    ## Find the edge that is not connected:
    for _, crnt, next in lists.iterate_with_periodic_prev_next_items(contraction_order):
        sahred_edge = _check_share_edge(crnt, next, edges_list)
        
        ## if crnt and next share edge
        if isinstance(sahred_edge, str):
            # make sure they share edge
            assert set(edges_dict[sahred_edge])==set((crnt, next))

        ## If they don't shaer edge, create that edge
        elif sahred_edge is None:
            ## Asserr these are corners with only 2 legs:
            assert len(edges_list[crnt]) == len(edges_list[next]) == 2
            ## Create new fused lef
            output_edges[crnt].append(new_fused_edge_name)
            output_edges[next].append(new_fused_edge_name)
            ## This is done only on the first discontinuation:
            break

        else:
            raise TypeError("Not such option")

    return output_edges
        

def main_create_kagome_lattice(
    size:int, periodic:bool
) -> tuple[
    dict[str, tuple[int, int]],  # edges dict
    list[list[str]],  # edges list
    list[tuple[int, int]]  # positions list
]:
    boundary = "periodic" if periodic else "disconnected"
    return create_kagome_lattice_with_boundary_condition(size=size, boundary=boundary)


def main_get_contraction_orders(
    size:int, is_plot:bool=False, is_print:bool=True
) -> tuple[
    dict[   # dict per direction
        str, # Direction key
        tuple[    # value of:
            list[int],              # contraction order
            list[list[str|int]],    # edge list
        ],
    ],
    tuple[
        dict,  # edges dict
        list,  # edges list
        list,  # pos   list
    ]
]:
    ## Prepare output:
    per_direction_dict : dict[str, tuple[list ,list]] = {}


    ## Lattice with correct structure:
    lattice = KagomeLattice(size)
    lattice.connect_messages()

    ## Get outputs for closed Block with messages:
    edges_dict = lattice.edges
    edges_list = [node.edges for node in lattice.nodes]
    pos_list   = [node.pos   for node in lattice.nodes]
    closed_lattice = (edges_dict, edges_list, pos_list)

    ## Get contraction order and corrected edges:
    depth = ContractionDepth.ToMessage
    for direction in BlockSide.all_in_counter_clockwise_order():
        contraction_order = get_contraction_order(lattice=lattice, direction=direction, depth=depth, plot_=is_plot)
        fixed_edges_list = _fix_edges_list_with_fused_corner(edges_list, edges_dict, contraction_order)
        per_direction_dict[str(direction)] = (contraction_order, fixed_edges_list)
        if is_print:
            print(f"In direction {str(direction)!r} is:")
            print(f"    Contraction order: {contraction_order}")
            print(f"    Corrected Edges  : {fixed_edges_list}")
            print("\n")


    return per_direction_dict, closed_lattice 



def main(
    n:int=2, 
    _print:bool=True,
    periodic:bool=False,
    save_at:str=""
) -> dict:
    
    
    def _conditioned_print(*args, **kwargs) -> None:
        if not _print:
            return
        print(*args, **kwargs)
    
    saved_files = dict()

    edges_dict, edges_list, pos_list = main_create_kagome_lattice(size=n, periodic=periodic)
    _conditioned_print(" ====== Periodic Kagome Block ====== \n\n ")
    _conditioned_print("edges_dict:")
    _conditioned_print(edges_dict)
    _conditioned_print("edges_list:")
    _conditioned_print(edges_list)
    _conditioned_print("pos_list:")
    _conditioned_print(pos_list)
    _conditioned_print("")


    N = len(edges_list)
    _conditioned_print("No. of tensors:", N)
    
    es = list(edges_dict.keys())
    _conditioned_print("No of edges: ", len(es))
    
    triplets_no = N//3
    _conditioned_print("No of triplets: ", triplets_no)
    
    _conditioned_print("\n\n")

    blue_edges = []
    red_edges = []
    orange_edges = []
    grey_edges = []
    magneta_edges = []
    green_edges = []
    
    for i in range(triplets_no):
        v=i*3

        blue_edges.append(edges_list[v][0])
        red_edges.append(edges_list[v][1])
        
        orange_edges.append(edges_list[v+1][0])
        grey_edges.append(edges_list[v+1][1])
        
        magneta_edges.append(edges_list[v+2][0])
        green_edges.append(edges_list[v+2][1])
        
    _conditioned_print("blue: ", blue_edges)
    _conditioned_print("red: ", red_edges)
    _conditioned_print("orange: ", orange_edges)
    _conditioned_print("grey: ", grey_edges)
    _conditioned_print("magneta: ", magneta_edges)
    _conditioned_print("green: ", green_edges)
    
    edges_colors_dict = {}
    edges_colors_dict['blue']   = blue_edges
    edges_colors_dict['red']    = red_edges
    edges_colors_dict['orange'] = orange_edges
    edges_colors_dict['grey']   = grey_edges
    edges_colors_dict['magneta']= magneta_edges
    edges_colors_dict['green']  = green_edges

    foutname = save_at+f"Kagome-Lattice-n{n}.pkl"
    fout = open(foutname, 'wb')
    saved_data = (edges_list, edges_colors_dict)
    pickle.dump( saved_data, fout)
    fout.close()

    saved_files["lattice"] = dict(
        path=foutname,
        data=saved_data
    )
        
    _conditioned_print("\n"*6)
    _conditioned_print(" ====== Block with Messages and Fused-Croner Leg ====== \n\n ")

    per_direction_dict, (closed_edges_dict, closed_edges_list, closed_pos_list) = \
        main_get_contraction_orders(size=n, is_print=_print)

    _conditioned_print("\n"*6)
    _conditioned_print(" ====== Closed Block with Messages ====== \n\n ")
    _conditioned_print("edges_dict:")
    _conditioned_print(closed_edges_dict)
    _conditioned_print("edges_list:")
    _conditioned_print(closed_edges_list)
    _conditioned_print("pos_list:")
    _conditioned_print(closed_pos_list)
    _conditioned_print("")

    foutname = save_at+f"Kagome-Lattice-n{n}-with-messages.pkl"
    fout = open(foutname, 'wb')
    saved_data = (closed_edges_list, per_direction_dict)
    pickle.dump( saved_data, fout)
    fout.close()

    saved_files["contraction"] = dict(
        path=foutname,
        data=saved_data
    )

    return saved_files


if __name__ == "__main__":
    main()