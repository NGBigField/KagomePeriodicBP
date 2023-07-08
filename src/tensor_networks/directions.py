# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

import numpy as np
from enum import Enum
from typing import Generator, Callable, Any, Tuple, Final
from numpy import pi, random
from utils import strings, lists, numerics
from functools import cache, cached_property
from abc import ABC, abstractclassmethod

# ============================================================================ #
#|                             Constants                                      |#
# ============================================================================ #

EPSILON : Final = 0.000001
NUM_MAIN_DIRECTIONS : Final = 6

# ============================================================================ #
#|                            Helper Functions                                |#
# ============================================================================ #

def _dist(x:float, y:float)->float:
    return abs(x-y)

def unit_vector_from_angle(angle:float)->Tuple[int, int]:
    x = numerics.force_integers_on_close_to_round(np.cos(angle))
    y = numerics.force_integers_on_close_to_round(np.sin(angle))
    assert isinstance(x, int)
    assert isinstance(y, int)
    return (x, y)
    

# ============================================================================ #
#|                           Class Defimition                                 |#
# ============================================================================ #    

class DirectionsError(ValueError): ...


class Direction():
    __slots__ = 'name', 'angle'

    def __init__(self, name:str, angle:float) -> None:
        self.name = name
        self.angle = angle

    def __str__(self)->str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Direction):
            return False
        if self.name==other.name:
            return True
        if _dist(self.angle, other.angle)<EPSILON:
            return True
        return False
    
    def __hash__(self) -> int:
        return hash((self.name, self.angle))

    def opposite(self)->"Direction":
        return OppositeDirections[self]

    def next_clockwise(self)->"Direction":
        return lists.next_item_cyclic(DirectionsInClockwiseOrder, self)

    def next_counterclockwise(self)->"Direction":
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
    
    
    
# ============================================================================ #
#|                            Main Directions                                 |#
# ============================================================================ #    

DL : Final[Direction] = Direction("Down-Left", 4*pi/3)
DR : Final[Direction] = Direction("Down-Right", 5*pi/3)
R  : Final[Direction] = Direction("Right" , 0)
UR : Final[Direction] = Direction("Up-Right", pi/3)
UL : Final[Direction] = Direction("Up-Left", 2*pi/3)
L  : Final[Direction] = Direction("Left" , 2*pi/3)


# ============================================================================ #
#|                        Relations between Directions                        |#
# ============================================================================ #    

DirectionsInClockwiseOrder : Final[list[Direction]] = [DL, DR, R, UR, UL, L]

OppositeDirections : Final[dict[Direction, Direction]] = {
    R  : L ,
    UR : DL,
    UL : DR, 
    L  : R ,
    DL : UR,
    DR : UL
}


# ============================================================================ #
#|                        Common Data Derived Once                            |#
# ============================================================================ #

MAX_DIRECTIONS_STR_LENGTH = max([len(str(side.name)) for side in DirectionsInClockwiseOrder])

# ============================================================================ #
#|                           Declared Function                                |#
# ============================================================================ #

def random()->Direction:
    return lists.random_item(DirectionsInClockwiseOrder)


def iterator_with_str_output(output_func:Callable[[str], Any])->Generator[Direction, None, None]:
    for i, side in enumerate(DirectionsInClockwiseOrder):
        s = " " + strings.num_out_of_num(i+1, NUM_MAIN_DIRECTIONS) + " " + f"{side.name:<{MAX_DIRECTIONS_STR_LENGTH}}"
        output_func(s)
        yield side


def all_opposite_pairs()->Generator[tuple[Direction, Direction], None, None]:
    yield R  , L 
    yield UR , DL
    yield UL , DR
            

def all_in_random_order()->Generator[Direction, None, None]:
    for direction in lists.shuffle(DirectionsInClockwiseOrder):
        yield direction


def standard_order()->Generator[Direction, None, None]:
    return iter(DirectionsInClockwiseOrder)
    
    
def all_in_counterclockwise_order()->Generator[Direction, None, None]:
    return reversed(DirectionsInClockwiseOrder)
        