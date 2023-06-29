import numpy as np
from enum import Enum
from typing import Generator, Callable, Any, Tuple, TypeVar, Union
from numpy import pi, random
from utils import strings, lists
from functools import cache

EPSILON = 0.000001

def _dist(x:float, y:float)->float:
    return abs(x-y)

def _force_zero_for_small_numbers(x:float|int)->float|int:
    if abs(x)<EPSILON:
        return 0
    else:
        return x
    
def _force_integers_on_close_to_round(x:float|int)->float|int:
    if abs(int(x)-x)<EPSILON:
        return int(x)
    return x
    
def unit_vector_from_angle(angle:float)->Tuple[int, int]:
    x = _force_integers_on_close_to_round(np.cos(angle))
    y = _force_integers_on_close_to_round(np.sin(angle))
    assert isinstance(x, int)
    assert isinstance(y, int)
    return (x, y)
    

_DirOrAngleType = Union["Directions", float]

class DirectionsError(ValueError): ...

    
class Directions(strings.StrEnum):
    Left  = "L"
    Right = "R"
    Up    = "U"
    Down  = "D"

    @property
    def angle(self)->float:
        if   self is Directions.Right: return 0.0
        elif self is Directions.Up   : return pi/2
        elif self is Directions.Left : return pi
        elif self is Directions.Down : return 3*pi/2
        else: 
            raise ValueError(f"Given side is not a valid member")

    def opposite(self)->"Directions":
        if   self is Directions.Right: return Directions.Left
        elif self is Directions.Up   : return Directions.Down
        elif self is Directions.Left : return Directions.Right
        elif self is Directions.Down : return Directions.Up
        else: 
            raise ValueError(f"Given side is not a valid member")
        
    def next_clockwise(self)->"Directions":
        if   self is Directions.Right: return Directions.Down
        elif self is Directions.Down : return Directions.Left
        elif self is Directions.Left : return Directions.Up
        elif self is Directions.Up   : return Directions.Right
        else: 
            raise ValueError(f"Given side is not a valid member")
        
    def next_counterclockwise(self)->"Directions":
        if   self is Directions.Right: return Directions.Up
        elif self is Directions.Up   : return Directions.Left
        elif self is Directions.Left : return Directions.Down
        elif self is Directions.Down : return Directions.Right
        else: 
            raise ValueError(f"Given side is not a valid member")

    @cache
    def unit_vector(self)->Tuple[int, int]:
        return unit_vector_from_angle(self.angle)

    @classmethod
    def from_angle(cls, angle:float)->"Directions":
        for dir in cls:
            if _dist(dir.angle, angle)<EPSILON:
                return dir
        raise DirectionsError(f"Given angle does not match with any known side")

    @classmethod
    def random(cls)->"Directions":
        options = list(cls)
        return lists.random_item(options)

    @classmethod
    def iterator_with_str_output(cls, output_func:Callable[[str], Any])->Generator["Directions", None, None]:
        # Basic Data:
        num_sides = len(cls)
        max_str_len = max([len(str(side.name)) for side in cls])
            
        # Iterate:
        for i, side in enumerate(cls.all_in_counterclockwise_order()):
            s = " " + strings.num_out_of_num(i+1, num_sides) + " " + f"{side.name:<{max_str_len}}"
            output_func(s)
            yield side
            
    @classmethod
    def all_opposite_pairs(cls)->Generator[tuple["Directions", "Directions"], None, None]:
        used_directions : list[cls] = []
        for dir in cls:
            if dir in used_directions:
                continue
            pair = (dir, dir.opposite())
            used_directions.extend(pair)
            yield pair
            
    @classmethod
    def all_in_random_order(cls)->Generator["Directions", None, None]:
        for direction in lists.shuffle(list(cls)):
            yield direction

    @classmethod
    def standard_order(cls)->Generator["Directions", None, None]:
        yield L
        yield R
        yield U
        yield D
        
    @classmethod
    def all_in_counterclockwise_order(cls)->Generator["Directions", None, None]:
        yield D
        yield R
        yield U
        yield L
        

    @classmethod
    def opposite_direction(cls, dir:_DirOrAngleType)->_DirOrAngleType:
        if isinstance(dir, cls):
            return dir.opposite()
        elif isinstance(dir, float):
            return ( dir+pi ) % (2*pi)
        else:
            raise TypeError(f"Not an expected type '{type(dir)}'")    
        
    @classmethod
    def is_equal(cls, dir1:_DirOrAngleType, dir2:_DirOrAngleType)->bool:
        if isinstance(dir1, cls):
            assert isinstance(dir2, cls)
            return dir1 is dir2
        elif isinstance(dir1, float):
            assert isinstance(dir2, float)
            return abs(dir1-dir2) < EPSILON
        else:
            raise TypeError(f"Not an expected type '{type(dir)}'")  

    @classmethod
    def is_orthogonal(cls, dir1:"Directions", dir2:"Directions")->bool:
        if dir1 in [Directions.Left, Directions.Right]:
            return dir2 in [Directions.Up, Directions.Down]
        elif dir1 in [Directions.Up, Directions.Down]:
            return dir2 in [Directions.Left, Directions.Right]
        else:
            TypeError(f"Not a legit option when dir1 is {type(dir1)!r}")


## Short-hand:
L = Directions.Left
R = Directions.Right
U = Directions.Up
D = Directions.Down