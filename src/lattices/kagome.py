if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

from lattices import triangle as triangle_lattice
from lattices import edges
from lattices._common import NodePlaceHolder
from lattices import directions
from lattices.directions import LatticeDirection, BlockSide, lattice, block
from lattices.directions import L, R, UL, UR, DL, DR  # Lattice directions:

from _error_types import LatticeError, DirectionError
from _types import EdgeIndicator

from numpy import pi, cos, sin
from utils import numerics, tuples
from dataclasses import dataclass, fields
from typing import Generator

import itertools


_delta_xs = [0, -1,  1]
_delta_ys = [1, -1, -1]

_CONSTANT_X_SHIFT = 3
_CONSTANT_y_SHIFT = 1

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


class KagomeLatticeError(LatticeError):...


@dataclass(frozen=False, slots=True)
class UpperTriangle:
    up    : NodePlaceHolder = None
    left  : NodePlaceHolder = None
    right : NodePlaceHolder = None
    #
    index : int = -1

    def all_nodes(self)->Generator[NodePlaceHolder, None, None]:
        yield self.up
        yield self.left
        yield self.right
    
    @staticmethod
    def field_names()->list[str]:
        return ['up', 'left', 'right']

class _UnassignedEdgeName():
    def __repr__(self) -> str:
        return "_UnassignedEdgeName"



def edge_name_from_indices(i1:int, i2:int)->str:
    if   i1<i2:  return f"{i1}-{i2}" 
    elif i1>i2:  return f"{i2}-{i1}" 
    else:
        raise ValueError("Indices must be of different nodes") 


def _derive_node_directions(field:str)->list[LatticeDirection]:
    match field:
        case "up"   : return [lattice.UL, lattice.DL, lattice.DR, lattice.UR]
        case "left" : return [lattice.L, lattice.DL, lattice.R, lattice.UR]
        case "right": return [lattice.UL, lattice.L, lattice.DR, lattice.R]
        case _: raise ValueError(f"Unexpected string {field!r}")


def _tag_boundary_nodes(triangle:UpperTriangle, boundary:BlockSide)->None:
    touching_nodes : list[NodePlaceHolder] = []
    if   boundary is block.U:     touching_nodes = [triangle.up]
    elif boundary is block.DL:    touching_nodes = [triangle.left]
    elif boundary is block.DR:    touching_nodes = [triangle.right]
    elif boundary is block.D:     touching_nodes = [triangle.left, triangle.right]
    elif boundary is block.UR:    touching_nodes = [triangle.up, triangle.right]
    elif boundary is block.UL:    touching_nodes = [triangle.up, triangle.left]
    else: 
        raise DirectionError()

    for node in touching_nodes:
        node.boundaries.add(boundary)


def _create_upper_triangle(triangular_node:NodePlaceHolder, indices:list[int])->UpperTriangle:
    upper_triangle = UpperTriangle()
    x0, y0 = triangular_node.pos

    ## Derive Position and Directions:
    for node_index, field, delta_x, delta_y in zip(indices, UpperTriangle.field_names(), _delta_xs, _delta_ys, strict=True):
        x = x0 + delta_x + _CONSTANT_X_SHIFT
        y = y0 + delta_y + _CONSTANT_y_SHIFT
        node = NodePlaceHolder(
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


def _name_outer_edges(node:NodePlaceHolder, order_ind:int, boundary:BlockSide, kagome_lattice:list[NodePlaceHolder], N:int)->None:
    upper_triangle = get_upper_triangle(node.index, kagome_lattice, N)
    _edge_name = lambda ind: f"{boundary}-{ind}"

    if boundary is block.D:     
        if   node is upper_triangle.left:   node.set_edge_in_direction(DL, _edge_name(order_ind))
        elif node is upper_triangle.right:  node.set_edge_in_direction(DR, _edge_name(order_ind))
        else:
            raise LatticeError()
        
    elif boundary is block.DR:    
        assert node is upper_triangle.right
        node.set_edge_in_direction(DR, _edge_name(2*order_ind))
        node.set_edge_in_direction(R, _edge_name(2*order_ind+1))

    elif boundary is block.UR:    
        if   node is upper_triangle.right:  node.set_edge_in_direction(R,  _edge_name(order_ind))
        elif node is upper_triangle.up:     node.set_edge_in_direction(UR, _edge_name(order_ind))
        else: 
            raise LatticeError()

    elif boundary is block.U:     
        assert node is upper_triangle.up
        node.set_edge_in_direction(UR, _edge_name(2*order_ind))
        node.set_edge_in_direction(UL, _edge_name(2*order_ind+1))

    elif boundary is block.UL:    
        if   node is upper_triangle.up:     node.set_edge_in_direction(UL, _edge_name(order_ind))
        elif node is upper_triangle.left:   node.set_edge_in_direction(L,  _edge_name(order_ind))
        else: 
            raise LatticeError()

    elif boundary is block.DL:    
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

def sorted_boundary_nodes(nodes:list[NodePlaceHolder], boundary:BlockSide)->list[NodePlaceHolder]:
    # Get relevant nodes:
    boundary_nodes = [node for node in nodes if boundary in node.boundaries]

    # Choose sorting key:
    if   boundary is block.U:     sorting_key = lambda node: -node.pos[0]
    elif boundary is block.UR:    sorting_key = lambda node: +node.pos[1] 
    elif boundary is block.DR:    sorting_key = lambda node: +node.pos[1]
    elif boundary is block.UL:    sorting_key = lambda node: -node.pos[1]
    elif boundary is block.DL:    sorting_key = lambda node: -node.pos[1]
    elif boundary is block.D:     sorting_key = lambda node: +node.pos[0]
    else:
        raise DirectionError(f"Impossible direction {boundary!r}")

    # Sort:
    return sorted(boundary_nodes, key=sorting_key)


def get_upper_triangle(node_index:int, nodes:list[NodePlaceHolder], N:int)->UpperTriangle:
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
    list[NodePlaceHolder],
    list[UpperTriangle]
]:

    ## Create the triangular lattice we're based on:
    original_triangular_lattice = triangle_lattice.create_triangle_lattice(N)
    triangular_lattice_of_upper_triangles : list[UpperTriangle] = []

    ## Position upper-triangles at each node of the kagome lattice:
    kagome_lattice : list[NodePlaceHolder] = []
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
        for index2, direction1 in triangle_lattice.all_neighbors(index1, N):
            triangle2 = triangular_lattice_of_upper_triangles[index2]
            _connect_kagome_nodes_between_triangles(triangle1, triangle2, direction1)

    ## Tag all nodes on boundary:
    for triangle in triangular_lattice_of_upper_triangles:
        on_boundaries = triangle_lattice.check_boundary_vertex(triangle.index, N)
        for boundary in on_boundaries:
            _tag_boundary_nodes(triangle, boundary)

    ## Use ordered nodes to name Outer Edges
    for boundary in directions.hexagonal_block_boundaries():
        sorted_nodes = sorted_boundary_nodes(kagome_lattice, boundary)
        for i, node in enumerate(sorted_nodes):
            _name_outer_edges(node, i, boundary, kagome_lattice, N)
    # The bottom-left node is falsely on its DL leg, fix it:
    bottom_left_corner_node = sorted_boundary_nodes(kagome_lattice, block.D)[0]
    bottom_left_corner_node.set_edge_in_direction(DL, f"{block.D}-0")

    
    ## Plot test:
    # from lattices._common import plot
    # plot(kagome_lattice)
    # plot(original_triangular_lattice, node_color="black", edge_style="y--", node_size=5)

    return kagome_lattice, triangular_lattice_of_upper_triangles


class KagomeLattice():
    __slots__ =  "N", "nodes", "triangles", "edges"
    
    def __init__(self, N:int) -> None:
        kagome_lattice, triangular_lattice_of_upper_triangles = create_kagome_lattice(N)
        self.N : int = N
        self.nodes     : list[NodePlaceHolder] = kagome_lattice
        self.triangles : list[UpperTriangle]   = triangular_lattice_of_upper_triangles
        self.edges     : dict[str, tuple[int, int]] = edges.edges_dict_from_edges_list(
            [node.edges for node in kagome_lattice]
        )
                    
    def nodes_and_triangles(self)->Generator[tuple[NodePlaceHolder, UpperTriangle, ], None, None]:
        triangles_repeated_3_times = itertools.chain.from_iterable(itertools.repeat(triangle, 3) for triangle in self.triangles)
        return zip(self.nodes, triangles_repeated_3_times, strict=True)
    
    
    def get_neighbor(self, node:NodePlaceHolder, edge_or_dir:EdgeIndicator|LatticeDirection)->NodePlaceHolder:
        if isinstance(edge_or_dir, EdgeIndicator):
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
            
            
        

def main_test():
    lattice = create_kagome_lattice(2)


if __name__ == "__main__":
    main_test()