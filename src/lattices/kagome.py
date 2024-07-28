if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

from lattices import triangle as triangle_lattice
from lattices import edges
from lattices._common import Node, sorted_boundary_nodes
from lattices import directions
from lattices.directions import LatticeDirection, BlockSide, Direction

from _error_types import LatticeError, DirectionError
from _types import EdgeIndicatorType

from numpy import pi, cos, sin
from utils import numerics, tuples, lists
from typing import Generator, Any

# OOP:
from dataclasses import dataclass, fields
from enum import Enum, unique, auto

import itertools
import functools

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


@unique
class BlockBoundaryCondition(Enum):
    OPEN        = auto()  # The block has unique leg names on its boundaries. These edges are not connected
    CLOSED      = auto()  # Same as OPEN, but each edge gets a tensor of dim=1.
    PERIODIC    = auto()  # Edges on the boundary get fused to the edges from the opposite boundary.


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
            
    def __contains__(self, item) -> bool:
        for node in self.all_nodes():
            if item is node:
                return True
        return False

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
    original_triangular_lattice = triangle_lattice.create_triangle_lattice(N)
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
        for index2, direction1 in triangle_lattice.all_neighbors(index1, N):
            triangle2 = triangular_lattice_of_upper_triangles[index2]
            _connect_kagome_nodes_between_triangles(triangle1, triangle2, direction1)

    ## Tag all nodes on boundary:
    for triangle in triangular_lattice_of_upper_triangles:
        on_boundaries = triangle_lattice.check_boundary_vertex(triangle.index, N)
        for boundary in on_boundaries:
            _tag_boundary_nodes(triangle, boundary)

    ## Use ordered nodes to name Outer Edges
    for boundary in BlockSide.all_in_counter_clockwise_order():
        sorted_nodes = sorted_boundary_nodes(kagome_lattice, boundary)
        for i, node in enumerate(sorted_nodes):
            _name_outer_edges(node, i, boundary, kagome_lattice, N)
    # The bottom-left node is falsely on its DL leg, fix it:
    bottom_left_corner_node = sorted_boundary_nodes(kagome_lattice, BlockSide.D)[0]
    bottom_left_corner_node.set_edge_in_direction(DL, f"{BlockSide.D}-0")

    return kagome_lattice, triangular_lattice_of_upper_triangles


class KagomeLattice():
    __slots__ =  "N", "nodes", "triangles", "edges", "_boundary_condition"
    
    def __init__(self, N:int) -> None:
        kagome_lattice, triangular_lattice_of_upper_triangles = create_kagome_lattice(N)
        self.N         : int = N
        self.nodes     : list[Node] = kagome_lattice
        self.triangles : list[UpperTriangle]   = triangular_lattice_of_upper_triangles
        self.edges     : dict[str, tuple[int, int]] = edges.edges_dict_from_edges_list(
            [node.edges for node in kagome_lattice]
        )
        self._boundary_condition : BlockBoundaryCondition = BlockBoundaryCondition.OPEN

                    
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
    def change_boundary_conditions(self, periodic:bool) -> None:
        # Change inner attribute:
        if periodic:
            self._boundary_condition = BlockBoundaryCondition.PERIODIC
        else:
            self._boundary_condition = BlockBoundaryCondition.CLOSED

        ## connect per two opposite faces:
        ordered_half_list = list(BlockSide.all_in_counter_clockwise_order())[0:3]
        for block_side in ordered_half_list:
            boundary_edges          = self.sorted_boundary_edges(boundary=block_side)
            opposite_boundary_edges = self.sorted_boundary_edges(boundary=block_side.opposite())
            opposite_boundary_edges.reverse()

            for edge1, edge2 in zip(boundary_edges, opposite_boundary_edges, strict=True):
                ## Choose boundary function:
                if periodic:
                    _kagome_connect_boundary_edges_periodically(self, edge1, edge2)
                else:
                    _kagome_connect_boundary_edges_to_dummy_nodes(self, edge1, edge2)


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
        assert directions.check.is_orthogonal(major_direction, minor_direction)
        crnt_vertices_order = get_upper_triangle_vertices_order(major_direction, minor_direction)

        ## Get Upper-Triangles sorted in wanted direction:
        triangle_indices_in_order = triangle_lattice.vertices_indices_rows_in_direction(N, major_direction, minor_direction)

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
        index = triangle_lattice.get_center_vertex_index(self.N)
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
        return sorted_boundary_nodes(self.nodes, boundary)
    
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
    def plot(self) -> None:
        plot_lattice(self.nodes, periodic=self._boundary_condition==BlockBoundaryCondition.PERIODIC)        

    def plot_triangles_lattice(self) -> None:
        # Visuals import:
        from utils import visuals
        from matplotlib import pyplot as plt
        # basic data:
        N = self.size
        # Plot triangles:
        for upper_triangle in self.triangles:
            ind = upper_triangle.index
            i, j = triangle_lattice.get_vertex_coordinates(ind, N)
            x, y = triangle_lattice.get_node_position(i, j, N)
            plt.scatter(x, y, marker="^")
            plt.text(x, y, f" [{ind}]")

        

def _kagome_connect_boundary_edges_to_dummy_nodes(kagome_lattice:KagomeLattice, edge1:str, edge2:str) -> None:
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



## Cached results for lattices sizes:
NUM_NODES_TO_LATTICE_SIZE = {}
LATTICE_SIZE_TO_NUM_NODES = {}


def _update_size_caches(num_nodes:int, lattice_size:int) -> None:
    NUM_NODES_TO_LATTICE_SIZE[num_nodes] = lattice_size
    LATTICE_SIZE_TO_NUM_NODES[lattice_size] = num_nodes


def num_nodes_by_lattice_size(N:int) -> int:
    # Look in cache
    if N in LATTICE_SIZE_TO_NUM_NODES:
        num_nodes = LATTICE_SIZE_TO_NUM_NODES[N]
    else:
        # Compute:
        num_nodes = 3*triangle_lattice.total_vertices(N)
        # Update cache:
        _update_size_caches(num_nodes=num_nodes, lattice_size=N)
    return num_nodes


def lattice_size_by_num_nodes(num_nodes:int, _max_attempted_size:int=20) -> int:
    # Look in cache
    if num_nodes in NUM_NODES_TO_LATTICE_SIZE:
        N = NUM_NODES_TO_LATTICE_SIZE[num_nodes]
    else:
        # Compute:
        N = _search_lattice_size_by_num_nodes(num_nodes, _max_attempted_size)
        # Update cache:
        _update_size_caches(num_nodes=num_nodes, lattice_size=N)
    return N


def _search_lattice_size_by_num_nodes(num_nodes:int, _max_attempted_size:int=20) -> int:
    for size in range(2, _max_attempted_size+1):
        crnt_num_nodes = num_nodes_by_lattice_size(size)
        if num_nodes == crnt_num_nodes:
            return size
        
    raise ValueError(f"num_nodes={num_nodes} does not match any Kagome lattice size.")


def plot_lattice(all_nodes:list[Node], node_color:str="red", node_size=40, edge_style:str="b-", edge_name_color:str|None="green", periodic:bool=False)->None:
    from matplotlib import pyplot as plt

    ## Plot node positions:
    for node in all_nodes:
        x, y = node.pos
        plt.scatter(x, y, c=node_color, s=node_size)
        plt.text(x,y, s=f"{node.index}")

    ## Plot edges:
    edges_list = [node.edges for node in all_nodes]
    for edges in edges_list:
        for edge in edges:
            nodes = [node for node in all_nodes if edge in node.edges]
            
            if edge.count("-")==2:
                assert periodic==True
                assert len(nodes)==2
                for node in nodes:
                    direction = node.directions[ node.edges.index(edge) ]
                    x1, y1 = node.pos
                    x2, y2 = tuples.add((x1, y1), direction.unit_vector)
                    x_text = x2
                    y_text = y2
                    plt.plot([x1, x2], [y1, y2], edge_style)
                    if edge_name_color is not None:
                        plt.text(x_text, y_text, edge, color=edge_name_color)
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
            if edge_name_color is not None:
                plt.text(x_text, y_text, edge, color=edge_name_color)
            

    print("Done")


if __name__ == "__main__":
    kagome_lattice = KagomeLattice(N=2)
    kagome_lattice.change_boundary_conditions("periodic")
    kagome_lattice.plot()
