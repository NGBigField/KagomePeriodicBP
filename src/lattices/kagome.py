if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

from lattices.triangle import create_triangle_lattice
from lattices._common import NodePlaceHolder
from tensor_networks.directions import Direction
from tensor_networks.directions import DL, DR, R, UR, UL, L
from numpy import pi, cos, sin
from utils import numerics
from dataclasses import dataclass, fields
import itertools



_three_roots_of_unity = [i*2*pi/3 for i in range(3)]
_upper_triangle_angles = [root+pi/2 for root in _three_roots_of_unity]
_x_length_normalization = 1/abs(cos(2*pi/3 + pi/2))
_y_length_normalization = 2
_delta_xs = [numerics.force_integers_on_close_to_round(cos(angle)*_x_length_normalization) for angle in _upper_triangle_angles]
_delta_ys = [numerics.force_integers_on_close_to_round(sin(angle)*_y_length_normalization) for angle in _upper_triangle_angles]
_delta_ys[0] = 1

"""
An Upper Triagle:
     Up
      |
      |
      O
     / \
    /   \
Left    Right
"""

@dataclass(frozen=False, slots=True)
class _UpperTriangle:
    up    : NodePlaceHolder = None
    left  : NodePlaceHolder = None
    right : NodePlaceHolder = None

_upper_triangle_corners = ['up', 'left', 'right']

class _UnassignedEdgeName: ...


def _edge_name_from_indices(i1:int, i2:int)->str:
    if   i1<i2:  return f"{i1}-{i2}" 
    elif i1>i2:  return f"{i2}-{i1}" 
    else:
        raise ValueError("Indices must be of different nodes") 


def _derive_node_directions(field:str)->list[Direction]:
    match field:
        case "up"   : return [UL, DL, DR, UR]
        case "left" : return [L, DL, R, UR]
        case "right": return [UL, L, DR, R]
        case _: raise ValueError(f"Unexpected string {field!r}")


def _get_upper_triangle(triangular_node:NodePlaceHolder)->_UpperTriangle:
    upper_triangle = _UpperTriangle()
    x0, y0 = triangular_node.pos

    ## Derive Position and Directions:
    for field, delta_x, delta_y in zip(_upper_triangle_corners, _delta_xs, _delta_ys, strict=True):
        x = x0 + delta_x
        y = y0 + delta_y
        node = NodePlaceHolder(
            pos = (x, y),
            edges=[_UnassignedEdgeName, _UnassignedEdgeName,_UnassignedEdgeName, _UnassignedEdgeName],
            directions=_derive_node_directions(field)
        )
        upper_triangle.__setattr__(field, node)

    ## Complete Edge of inner nodes:
    for i, j in itertools.combinations(range(3), 2):
        print(i,j)
    return upper_triangle
        

def create_kagome_lattice(N:int)->list[NodePlaceHolder]:

    ## Create the triangular lattice we're based on:
    original_triangular_lattice = create_triangle_lattice(N)
    triangular_lattice_of_upper_triangles : list[_UpperTriangle] = []

    ## Position upper-triangles at each node of the kagome lattice:
    kagome_lattice = []
    for triangular_node in original_triangular_lattice:
        upper_triangle = _get_upper_triangle(triangular_node)
        upper_triangle_nodes = [upper_triangle.up, upper_triangle.left, upper_triangle.right]
        kagome_lattice.extend(upper_triangle_nodes)
        triangular_lattice_of_upper_triangles.append(upper_triangle)
        
    ## Complete the edges
    for upper_triangle in triangular_lattice_of_upper_triangles:
        for corner_name in _upper_triangle_corners:
            node : NodePlaceHolder = upper_triangle.__getattribute__(corner_name) 
            for leg_ind, direction in enumerate(node.directions):
                 pass
                

    return 


def main_test():
    lattice = create_kagome_lattice(2)


if __name__ == "__main__":
    main_test()