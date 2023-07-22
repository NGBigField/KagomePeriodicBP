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

from _error_types import DirectionError

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
    return (x, y)
    

# ============================================================================ #
#|                           Class Defimition                                 |#
# ============================================================================ #    



class Direction():
    __slots__ = 'name', 'angle', "unit_vector"

    def __init__(self, name:str, angle:float) -> None:
        self.name = name
        self.angle = angle
        self.unit_vector = unit_vector_from_angle(angle)

    def __str__(self)->str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Direction {self.name!r} {self.angle}"
    
    def __eq__(self, other: object) -> bool:
        # Type check:
        if not isinstance(other, Direction):
            return False
        # Fast instance check:
        if self is other:
            return True
        # Slower values check:
        if self.name==other.name and _dist(self.angle, other.angle)<EPSILON:
            return True
        return False
    
    def __hash__(self) -> int:
        return hash((self.name, self.angle))

    def opposite(self)->"Direction":
        return OppositeDirections[self]

    def next_clockwise(self)->"Direction":
        return lists.next_item_cyclic(LatticeDirectionsInClockwiseOrder, self)

    def next_counterclockwise(self)->"Direction":
        return lists.prev_item_cyclic(LatticeDirectionsInClockwiseOrder, self)
    
    @staticmethod
    def from_angle(angle:float)->"Direction":
        for dir in LatticeDirectionsInClockwiseOrder:
            if _dist(dir.angle, angle)<EPSILON:
                return dir
        raise DirectionError(f"Given angle does not match with any known side")
    
    
    
# ============================================================================ #
#|                            Main Directions                                 |#
# ============================================================================ #    

# Main direction in the Kagome lattice:
R  : Final[Direction] = Direction("R" , 0)
UR : Final[Direction] = Direction("UR", pi/3)
UL : Final[Direction] = Direction("UL", 2*pi/3)
L  : Final[Direction] = Direction("L" , pi)
DL : Final[Direction] = Direction("DL", 4*pi/3)
DR : Final[Direction] = Direction("DR", 5*pi/3)

# Directions that apear in the hexagonal cell:
U  : Final[Direction] = Direction("U", pi/2)
D  : Final[Direction] = Direction("D", 3*pi/2)

# ============================================================================ #
#|                        Relations between Directions                        |#
# ============================================================================ #    

LatticeDirectionsInClockwiseOrder : Final[list[Direction]] = [DL, DR, R, UR, UL, L]
HexagonalBlockDirections : Final[list[Direction]] = [D, DR, UR, U, UL, DL]

OppositeDirections : Final[dict[Direction, Direction]] = {
    R  : L ,
    UR : DL,
    UL : DR, 
    L  : R ,
    DL : UR,
    DR : UL,
    U  : D,
    D  : U,
}


# ============================================================================ #
#|                        Common Data Derived Once                            |#
# ============================================================================ #

MAX_DIRECTIONS_STR_LENGTH = max([len(str(side.name)) for side in LatticeDirectionsInClockwiseOrder])

# ============================================================================ #
#|                           Declared Function                                |#
# ============================================================================ #

def random()->Direction:
    return lists.random_item(LatticeDirectionsInClockwiseOrder)


def iterator_with_str_output(output_func:Callable[[str], Any])->Generator[Direction, None, None]:
    for i, side in enumerate(LatticeDirectionsInClockwiseOrder):
        s = " " + strings.num_out_of_num(i+1, NUM_MAIN_DIRECTIONS) + " " + f"{side.name:<{MAX_DIRECTIONS_STR_LENGTH}}"
        output_func(s)
        yield side


def all_opposite_pairs()->Generator[tuple[Direction, Direction], None, None]:
    yield R  , L 
    yield UR , DL
    yield UL , DR
            

def lattice_directions_in_random_order()->Generator[Direction, None, None]:
    for direction in lists.shuffle(LatticeDirectionsInClockwiseOrder):
        yield direction


def lattice_directions_in_clockwise_order()->Generator[Direction, None, None]:
    return iter(LatticeDirectionsInClockwiseOrder)


def lattice_directions_in_counterclockwise_order()->Generator[Direction, None, None]:
    return reversed(LatticeDirectionsInClockwiseOrder)


def lattice_directions_in_standard_order()->Generator[Direction, None, None]:
    return lattice_directions_in_clockwise_order()


def hexagonal_block_boundaries()->Generator[Direction, None, None]:
    return iter(HexagonalBlockDirections)