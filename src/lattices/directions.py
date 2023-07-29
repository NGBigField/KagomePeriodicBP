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

def _modulo_pi(a:float)->float:
    while a<0:
        a += 2*pi
    while a>=2*pi:
        a -= 2*pi
    return a

def _angle_dist(x:float, y:float)->float:
    x = _modulo_pi(x)
    y = _modulo_pi(y)
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
        return f"{self.__class__.__name__} {self.name!r} {self.angle}"
    
    def __eq__(self, other: object) -> bool:
        # Type check:
        assert issubclass(type(other), Direction)
        # Fast instance check:
        if self is other:
            return True
        # # Slower values check:
        # if (self.__class__.__name__==other.__class__.__name__ 
        #     and  self.name==other.name 
        #     and _angle_dist(self.angle, other.angle)<EPSILON):
        #     return True
        return False
    
    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.name))

    ## Get other by relation:
    def opposite(self)->"Direction":
        return OPPOSITE_DIRECTIONS[self]
    
    def next_clockwise(self)->"Direction":
        return lists.prev_item_cyclic(ORDERED_LISTS[type(self)], self)

    def next_counterclockwise(self)->"Direction":
        return lists.next_item_cyclic(ORDERED_LISTS[type(self)], self)
    
    ## Creation method:
    @classmethod
    def from_angle(cls, angle:float)->"Direction":
        for dir in ORDERED_LISTS[cls]:
            if _angle_dist(dir.angle, angle)<EPSILON:
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
            s = " " + strings.num_out_of_num(i+1, NUM_MAIN_DIRECTIONS) + " " + f"{side.name:<{MAX_DIRECTIONS_STR_LENGTH}}"
            output_func(s)
            yield side

    

class LatticeDirection(Direction): ...    

class BlockSide(Direction):
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
R  : Final[LatticeDirection] = LatticeDirection("R" , 0)
UR : Final[LatticeDirection] = LatticeDirection("UR", pi/3)
UL : Final[LatticeDirection] = LatticeDirection("UL", 2*pi/3)
L  : Final[LatticeDirection] = LatticeDirection("L" , pi)
DL : Final[LatticeDirection] = LatticeDirection("DL", 4*pi/3)
DR : Final[LatticeDirection] = LatticeDirection("DR", 5*pi/3)

# Directions that apear in the hexagonal cell:
block_U  : Final[BlockSide] = BlockSide("U", pi/2)
block_UR : Final[BlockSide] = BlockSide("UR", pi/2-pi/3)
block_UL : Final[BlockSide] = BlockSide("UL", pi/2+pi/3)
block_D  : Final[BlockSide] = BlockSide("D", 3*pi/2)
block_DL : Final[BlockSide] = BlockSide("DL", 3*pi/2-pi/3)
block_DR : Final[BlockSide] = BlockSide("DR", 3*pi/2+pi/3)


class lattice:
    R   = R
    UR  = UR
    UL  = UL
    L   = L
    DL  = DL
    DR  = DR    
    
class block:
    U   = block_U
    D   = block_D
    UL  = block_UL
    DL  = block_DL
    UR  = block_UR
    DR  = block_DR    

# ============================================================================ #
#|                        Relations between Directions                        |#
# ============================================================================ #    

LATTICE_DIRECTIONS_COUNTER_CLOCKWISE : Final[list[Direction]] = [DL, DR, R, UR, UL, L]
BLOCK_SIDES_COUNTER_CLOCKWISE : Final[list[Direction]] = [block_D, block_DR, block_UR, block_U, block_UL, block_DL]

ORDERED_LISTS = {
    LatticeDirection : LATTICE_DIRECTIONS_COUNTER_CLOCKWISE,
    BlockSide : BLOCK_SIDES_COUNTER_CLOCKWISE
}

OPPOSITE_DIRECTIONS : Final[dict[Direction, Direction]] = {
    # Lattice:
    R  : L ,
    UR : DL,
    UL : DR, 
    L  : R ,
    DL : UR,
    DR : UL,
    # Block:
    block_U  : block_D,
    block_D  : block_U,
    block_UR : block_DL,
    block_DL : block_UR,
    block_UL : block_DR,
    block_DR : block_UL
}

ORTHOGONAL_LATTICE_DIRECTIONS_TO_BLOCK_SIDES : Final[dict[BlockSide, LatticeDirection]] = {
    block_D  : R,
    block_U  : L,
    block_DR : UR,
    block_DL : DR,
    block_UR : UL,
    block_UL : DL
}

MATCHING_LATTICE_DIRECTIONS_TO_BLOCK_SIDES : Final[dict[BlockSide, list[LatticeDirection]]] = {
    block_D  : [DL, DR],
    block_DR : [DR, R ],
    block_UR : [R,  UR],
    block_U  : [UR, UL],
    block_UL : [UL, L ],
    block_DL : [L , DL]
}




# ============================================================================ #
#|                        Common Data Derived Once                            |#
# ============================================================================ #

MAX_DIRECTIONS_STR_LENGTH = 2

# ============================================================================ #
#|                           Declared Function                                |#
# ============================================================================ #

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
        else:
            mixed_cased = False

        if mixed_cased:
            return lattice_dir in lattice_options
        else:
            return dir1.opposite() is dir2

            