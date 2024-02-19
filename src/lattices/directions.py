# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

import numpy as np
from enum import Enum
from typing import Generator, Callable, Any, Final, TypeGuard
from numpy import pi, random
from utils import strings, lists, numerics, tuples
from functools import cache, cached_property
from abc import ABC, abstractclassmethod

from _error_types import DirectionError

# ============================================================================ #
#|                             Constants                                      |#
# ============================================================================ #

EPSILON : Final = 0.000001
NUM_MAIN_DIRECTIONS : Final = 6
MAX_DIRECTIONS_STR_LENGTH = 2

# ============================================================================ #
#|                            Helper Functions                                |#
# ============================================================================ #

def _angle_dist(x:float, y:float)->float:
    x = numerics.force_between_0_and_2pi(x)
    y = numerics.force_between_0_and_2pi(y)
    return abs(x-y)
    

def _unit_vector_from_angle(angle:float)->tuple[int, int]:
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
            s = " " + strings.num_out_of_num(i+1, NUM_MAIN_DIRECTIONS) + " " + f"{side.name:<{MAX_DIRECTIONS_STR_LENGTH}}"
            output_func(s)
            yield side

    def plot(self)->None:
        ## Some special imports:
        from matplotlib import pyplot as plt
        from utils import visuals
                                
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
        visuals.draw_now()
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

# Directions that apear in the hexagonal cell:
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




# ============================================================================ #
#|                            Helper Functions                                |#
# ============================================================================ #


# ============================================================================ #
#|                           Declared Functions                               |#
# ============================================================================ #

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

