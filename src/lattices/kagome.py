if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

from lattices import triangle as triangle_lattice
from lattices._common import NodePlaceHolder
from tensor_networks.directions import Direction
from tensor_networks.directions import DL, DR, R, UR, UL, L
from numpy import pi, cos, sin
from utils import numerics
from dataclasses import dataclass, fields
import itertools
from typing import Generator



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

    def all_nodes(self)->Generator[NodePlaceHolder, None, None]:
        yield self.up
        yield self.left
        yield self.right
    
    @staticmethod
    def field_names()->list[str]:
        return ['up', 'left', 'right']

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


def _create_upper_triangle(triangular_node:NodePlaceHolder, indices:list[int])->_UpperTriangle:
    upper_triangle = _UpperTriangle()
    x0, y0 = triangular_node.pos

    ## Derive Position and Directions:
    for node_index, field, delta_x, delta_y in zip(indices, _UpperTriangle.field_names(), _delta_xs, _delta_ys, strict=True):
        x = x0 + delta_x
        y = y0 + delta_y
        node = NodePlaceHolder(
            index=node_index,
            pos=(x, y),
            edges=[_UnassignedEdgeName, _UnassignedEdgeName,_UnassignedEdgeName, _UnassignedEdgeName],
            directions=_derive_node_directions(field)
        )
        upper_triangle.__setattr__(field, node)
    return upper_triangle
        

def create_kagome_lattice(N:int)->list[NodePlaceHolder]:

    ## Create the triangular lattice we're based on:
    original_triangular_lattice = triangle_lattice.create_triangle_lattice(N)
    triangular_lattice_of_upper_triangles : list[_UpperTriangle] = []

    ## Position upper-triangles at each node of the kagome lattice:
    kagome_lattice = []
    crnt_kagome_index = 0
    for triangular_node in original_triangular_lattice:
        # Assign crnt indices for the triangle:
        indices = list(range(crnt_kagome_index, crnt_kagome_index+3))
        crnt_kagome_index += 3

        # Create triangle:
        upper_triangle = _create_upper_triangle(triangular_node, indices)
        upper_triangle_nodes = [upper_triangle.up, upper_triangle.left, upper_triangle.right]
        kagome_lattice.extend(upper_triangle_nodes)
        triangular_lattice_of_upper_triangles.append(upper_triangle)
        
    ## Assign Edges
    for upper_triangle in triangular_lattice_of_upper_triangles:

        ## Inner edges within the triangle:
        up, left, right = upper_triangle.up, upper_triangle.left, upper_triangle.right 
        # Up-Left:
        edge_name = _edge_name_from_indices(up.index, left.index)
        up.edges[up.directions.index(DL)] = edge_name
        left.edges[left.directions.index(UR)] = edge_name
        # Up-Right:
        edge_name = _edge_name_from_indices(up.index, right.index)
        up.edges[up.directions.index(DR)] = edge_name
        right.edges[right.directions.index(UL)] = edge_name
        # Left-Right:
        edge_name = _edge_name_from_indices(left.index, right.index)
        left.edges[left.directions.index(R)] = edge_name
        right.edges[right.directions.index(L)] = edge_name            

    ## Edges between triangles:
    for index1, triangle1 in enumerate(triangular_lattice_of_upper_triangles):
        print(index1,":")                
        for index2, direction1 in triangle_lattice.all_neighbors(index1, N):
            print(f"    {index2}, {direction1}")                

    return 


def main_test():
    lattice = create_kagome_lattice(4)


if __name__ == "__main__":
    main_test()