import numpy as np
from enum import Enum
from typing import Generator, Callable, Any, Tuple, TypeVar, Union
from numpy import pi, random
from utils import strings, lists, numerics
from functools import cache, cached_property
from abc import ABC, abstractclassmethod

EPSILON = 0.000001

def _dist(x:float, y:float)->float:
    return abs(x-y)

def unit_vector_from_angle(angle:float)->Tuple[int, int]:
    x = numerics.force_integers_on_close_to_round(np.cos(angle))
    y = numerics.force_integers_on_close_to_round(np.sin(angle))
    assert isinstance(x, int)
    assert isinstance(y, int)
    return (x, y)
    

class DirectionsError(ValueError): ...


class Direction():
    __slots__ = 'name', 'angle'

    def __init__(self, name:str, angle:float) -> None:
        self.name = name
        self.angle = angle

    def __str__(self)->str:
        return self.name

    def opposite(self)->"Direction":
        return OppositeDirections[self]

    def next_clockwise(self)->"Direction":
        return lists.next_item_cyclic(DirectionsInClockwiseOrder, self)

    def next_counterclockwise(self)->"Directions":
        return lists.prev_item_cyclic(DirectionsInClockwiseOrder, self)
    
    @cached_property
    def unit_vector(self)->Tuple[int, int]:
        return unit_vector_from_angle(self.angle)
    
    @staticmethod
    def from_angle(angle:float)->"Direction":
        for dir in DirectionsInClockwiseOrder:
            if _dist(dir.angle, angle)<EPSILON:
                return dir
        raise DirectionsError(f"Given angle does not match with any known side")
    
    @staticmethod
    def random(cls)->"Direction":
        return lists.random_item(DirectionsInClockwiseOrder)
    
    


DL = Direction("DL", 4*pi/3)
DR = Direction("DR", 5*pi/3)
R  = Direction("R" , 0)
UR = Direction("UR", pi/3)
UL = Direction("UL", 2*pi/3)
L  = Direction("L" , 2*pi/3)


DirectionsInClockwiseOrder = [DL, DR, R, UR, UL, L]

OppositeDirections : dict[Direction, Direction] = {
    R  : L ,
    UR : DL,
    UL : DR, 
    L  : R ,
    DL : UR,
    DR : UL
}


def iterator_with_str_output(cls, output_func:Callable[[str], Any])->Generator["Direction", None, None]:
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

