if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)
	sys.path.append(
		pathlib.Path(__file__).parent.parent.parent.__str__()
	)



# Get Global-Config:
from _config_reader import DEBUG_MODE

from typing import Callable
from tensor_networks import KagomeTensorNetwork, TensorNode
from lattices.directions import Direction, BlockSide, LatticeDirection
from enums import ContractionDepth, NodeFunctionality
from containers import ContractionConfig

## For common types:
from _error_types import NetworkConnectionError
from _types import EdgeIndicatorType

# Everyone needs numpy:
import numpy as np

# Use our common utilities:
from utils import lists

# For smart iterations:
import itertools

# For oop:
from enum import Enum, auto
from typing import Generic, TypeVar
from dataclasses import dataclass, fields


# Types:
_T = TypeVar("_T")
_EnumType = TypeVar("_EnumType")
_PosFuncType = Callable[[int, int], tuple[int, int] ]

# Constants:
BREAK_MARKER = -100


class _Bound(Enum):
    first = auto()
    last  = auto()

    def opposite(self)->"_Bound":
        if   self is First: return Last            
        elif self is Last: return First
        else:
            raise ValueError("Not possible")                
    
First = _Bound.first 
Last  = _Bound.last

class _Side(Enum):
    left  = auto() 
    right = auto() 
Left = _Side.left
Right = _Side.right


class _EnumContainer(Generic[_EnumType, _T]):
    def __getitem__(self, key:_EnumType)->_T:
        for field in fields(self):
            if field.name == key.name:
                return getattr(self, field.name)
        raise KeyError(f"No a valid option {key!r}")
    
    def __setitem__(self, key:_EnumType, value:_T)->None:
        for field in fields(self):
            if field.name == key.name:
                return setattr(self, field.name, value)
        raise KeyError(f"No a valid option {key!r}")
            

@dataclass
class _PerBound(_EnumContainer[_Bound, _T]):
    first : _T 
    last  : _T   


@dataclass
class _PerSide(_EnumContainer[_Side, _T]):
    left  : _T 
    right : _T   


class _SideEdges:
    def __init__(self, left_sorted_outer_edges:list[str], right_sorted_outer_edges:list[str]) -> None:
        self.lists : _PerSide[list[str]] = _PerSide(
            left = left_sorted_outer_edges,
            right = right_sorted_outer_edges
        )
        self.crnt_index : _PerSide[int] = _PerSide(
            left = 0,
            right = 0 
        )
        self.exhausted : _PerSide[bool] = _PerSide(
            left = False,
            right = False,
        )

    def crnt(self, side:_Side)->str:
        if self.exhausted[side]:
            return "End"
        index = self.crnt_index[side] 
        item = self.lists[side][index]
        return item

    def next(self, side:_Side)->str:
        self.crnt_index[side] += 1
        try:
            next_item = self.crnt(side)
        except IndexError as e:
            next_item = None
            self.exhausted[side] = True
        return next_item
    
    def __getitem__(self, key:_Side)->_T:
        return self.crnt(key)

class _ReverseOrder:
    __slots__ = ["state", "counter"]
    
    def __init__(self) -> None:
        self.counter = 0
        self.state = True
    
    def __bool__(self)->bool:
        res = self.state
        self.counter += 1
        if self.counter > 1:
            ~self
        return res
    
    def __invert__(self)->None:
        self.counter = 0
        self.state = not self.state

    def set_true(self)->None:
        self.counter = 0
        self.state = True 

    def set_false(self)->None:
        self.counter = 0
        self.state = False 

    


def _determine_direction(axis2:list[int], _pos_func:_PosFuncType) -> Direction:
    ## Prepare locations:
    p1 = 0
    positions = [_pos_func(p1, p2) for p2 in axis2]
    x_pos = [pos[0] for pos in positions]
    y_pos = [pos[1] for pos in positions]
    
    ## Is the main axis X or Y:
    if all([p==0 for p in x_pos]):
        main_positions = y_pos
        sorted_answer   = Direction.Up
        opposite_answer = Direction.Down 
    elif all([p==0 for p in y_pos]):
        main_positions = x_pos
        sorted_answer   = Direction.Right
        opposite_answer = Direction.Left
    else: 
        raise ValueError("This option is not accounted for ): ")

    ## Direction
    if lists.is_sorted(main_positions):
        return sorted_answer
    main_positions.reverse()
    if lists.is_sorted(main_positions):
        return opposite_answer
    
    raise ValueError("The lists are not ordered at all! ): ")


def _sorted_side_outer_edges(
    tn:KagomeTensorNetwork, direction:BlockSide, with_break:bool=False
)->tuple[
    list[str],
    list[str]
]:
    ## Get all boundary edges. They are sorted to "the right"
    sorted_right_boundary_edges : dict[BlockSide, list[str]] = {
        side: tn.lattice.sorted_boundary_edges(side) for side in BlockSide.all_in_counter_clockwise_order()
    }
    right_last = direction.next_clockwise()
    right_first = right_last.next_clockwise()
    left_last = direction.next_counterclockwise()
    left_first = left_last.next_counterclockwise()

    ## Right edge orders:
    if with_break:
        right_edges = sorted_right_boundary_edges[right_first] + ['Break'] + sorted_right_boundary_edges[right_last] 
        left_edges = sorted_right_boundary_edges[left_last] + ['Break'] + sorted_right_boundary_edges[left_first] 
    else:
        right_edges = sorted_right_boundary_edges[right_first] + sorted_right_boundary_edges[right_last] 
        left_edges = sorted_right_boundary_edges[left_last] + sorted_right_boundary_edges[left_first] 
    left_edges.reverse()

    return left_edges, right_edges


def _add_all_side_neighbors( 
    tn:KagomeTensorNetwork, first_last:_Bound,
    nodes:_PerSide[TensorNode], side_order:_PerBound[_Side], 
    side_edges:_SideEdges, msg_neighbors:_PerBound[list[int]],
    force_assigning_at_end=False
)->EdgeIndicatorType:
     # unpack respective values:
    side = side_order[first_last]            
    node = nodes[side]
    edge = side_edges[side]           
    while edge in node.edges:     
        # get neighbor:
        neighbor = tn.find_neighbor(node, edge)
        # Where to assign the neighbor:
        if force_assigning_at_end:  put_at = Last
        else:                       put_at = first_last
        # assign neighbor:
        msg_neighbors[put_at].append(neighbor.index)  # Add to end 
        # move iterator and get next edge
        edge = side_edges.next(side)  
    return edge  # return last edge


def _derive_message_neighbors(
    tn:KagomeTensorNetwork, row:list[int], reverse_order:_ReverseOrder,
    side_edges:_SideEdges
) -> _PerBound[list[int]]:
    # Node at each side
    nodes = _PerSide[TensorNode](
        left  = tn.nodes[row[0]],
        right = tn.nodes[row[-1]]
    )
    msg_neighbors = _PerBound[list[int]](first=[], last=[])

    # which side is first?       
    if reverse_order:
        row.reverse()
        side_order = _PerBound[_Side](first=Right, last=Left)
    else:
        side_order = _PerBound[_Side](first=Left, last=Right)

    ## Add neighbors if they exist:
    for first_last in _Bound:
        last_edge = _add_all_side_neighbors(tn, first_last, nodes, side_order, side_edges, msg_neighbors)
            
        ## Special case when MPS messages change from one to another:
        if last_edge == 'Break':
            side = side_order[first_last]
            side_edges.next(side)  # move iterator

    ## At a break, decide if switch direction or not:
    # and collect the 'lost' neighbor of the first/last node
    if last_edge == 'Break': 

        ## Decide if switch direction or not by checking how has neighbors from both messages:
        if len(msg_neighbors[First])==1:
            next_first_side = side_order[First]
        elif len(msg_neighbors[Last])==1:
            next_first_side = side_order[Last]
        else: 
            raise ValueError("Not an expected option!")
        # Force side order to start from `next_first_side`:
        if next_first_side is Left:
            reverse_order.set_false()
        elif next_first_side is Right:
            reverse_order.set_true()
        else:
            raise ValueError("Not an expected option!")

        # Collect lost neighnors
        for first_last in _Bound:
            _add_all_side_neighbors(tn, first_last, nodes, side_order, side_edges, msg_neighbors, force_assigning_at_end=True)

        ## Assert that now both have 2 neighbours:
        # assert len(msg_neighbors[First])==len(msg_neighbors[Last])==2
        #TODO Check

    return msg_neighbors

def derive_contraction_order(
    tn:KagomeTensorNetwork,  
    direction:BlockSide,
    depth:ContractionDepth|int,
    plot_:bool=False
)->tuple[
    list[int],
    Direction
]:
    
    #TODO For debug:
    from lattices.directions import block
    plot_ = True
    direction = block.U

    ## Prepare output:
    con_order = []    

    # Define Directions:
    major_direction = direction
    minor_right = direction.orthogonal_clockwise_lattice_direction()

    ## Start by fetching the lattice-nodes in order and the messages:
    lattice_rows_ordered_right = tn.lattice.nodes_indices_rows_in_direction(major_direction, minor_right)
    # Side edges:
    left_sorted_outer_edges, right_sorted_outer_edges = _sorted_side_outer_edges(tn, major_direction, with_break=True)
    side_edges = _SideEdges(left_sorted_outer_edges, right_sorted_outer_edges)
    
    ## Helper objects: 
    # # Iterator to switch contraction direction every two rows:
    reverse_order = _ReverseOrder()

    ## First message:
    con_order.extend( tn.message_indices(major_direction.opposite()) )

    ## Lattice and its connected nodes:
    for row in lattice_rows_ordered_right:
        msg_neighbors = _derive_message_neighbors(tn, row, reverse_order, side_edges)
        ## add row to con_order:
        con_order.extend( msg_neighbors.first + row + msg_neighbors.last )

    ## Last msg:
    con_order += tn.message_indices(major_direction)

    ## Plot result:
    if plot_:
        # For plotting contraction order, if needed:
        import matplotlib.pyplot as plt
        from tensor_networks.visualizations import plot_contraction_order
        tn.plot(detailed=False)
        plot_contraction_order(tn.positions, con_order)
        plt.title(f"Contraction in Direction {str(direction)}" )

    ## Check:
    if DEBUG_MODE:
        # Both list of messages are in `con_order`:
        assert side_edges.exhausted.left
        assert side_edges.exhausted.right

    ## Last minor-direction of contraction:
    if reverse_order:
        last_direction = minor_right.opposite()
    else:
        last_direction = minor_right

    return con_order, last_direction



if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import bp_test
    bp_test.main_test()

