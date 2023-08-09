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

# Import our types and calsses:
from tensor_networks import KagomeTensorNetwork, TensorNode
from lattices.directions import Direction, BlockSide, LatticeDirection
from enums import ContractionDepth, NodeFunctionality

## For common types:
from _error_types import NetworkConnectionError
from _types import EdgeIndicatorType

# For lattice logic:
from lattices import kagome

# Everyone needs numpy:
import numpy as np

# Use our common utilities:
from utils import lists

# For oop:
from enum import Enum, auto
from typing import Generic, TypeVar, Final, Callable, TypeAlias
from dataclasses import dataclass, fields
from copy import deepcopy


## Types:
_T = TypeVar("_T")
_EnumType = TypeVar("_EnumType")
_PosFuncType = Callable[[int, int], tuple[int, int] ]
_FULL_CONTRACTION_ORDERS_CACHE_KEY_TYPE : TypeAlias = tuple[int, BlockSide, ContractionDepth]

## Constants:
BREAK_MARKER = -100
Full = ContractionDepth.Full
ToCore = ContractionDepth.ToCore
ToMessage = ContractionDepth.ToMessage

## For Caching results:
FULL_CONTRACTION_ORDERS_CACHE : dict[_FULL_CONTRACTION_ORDERS_CACHE_KEY_TYPE, list[int]] = {}


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

    def to_per_side(self, is_reverse_order:bool)->"_PerSide":
        if is_reverse_order:
            return _PerSide(right=self.first, left=self.last)
        else:
            return _PerSide(left=self.first, right=self.last)


@dataclass
class _PerSide(_EnumContainer[_Side, _T]):
    left  : _T 
    right : _T   

    def to_per_order(self, is_reverse_order:bool)->_PerBound[_T]:
        if is_reverse_order:
            return _PerBound(last=self.left, first=self.right)
        else:
            return _PerBound(first=self.left, last=self.right)



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
    _num_repeats : Final[int] = 2
    
    def __init__(self) -> None:
        self.counter = 0
        self.state = True
    
    def __bool__(self)->bool:
        return self.state
    
    def __invert__(self)->None:
        self.state = not self.state
        self.counter = 0

    def prev_state(self)->bool:
        if self.counter - 1 < 0:
            return not self.state
        else:
            return self.state

    def check_state_and_update(self)->bool:
        res = self.state
        self.counter += 1
        if self.counter > _ReverseOrder._num_repeats-1:
            ~self   # call invert
        return res

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

def _decide_by_future_neighbors(
    side_order:_PerBound[_Side], 
    num_neighbors_per_side_before:_PerBound[list[int]], 
    num_neighbors_per_side_after:_PerBound[list[int]]
)->_Side|None:
    ## Decide which side should continue the break:
    for first_last in _Bound:            
        side = side_order[first_last]
        ## How many MPS messages are and were connected:
        num_bfore = num_neighbors_per_side_before[side]
        num_after = num_neighbors_per_side_after[side]
        if num_bfore==0 or num_after==0:
            raise ValueError("Not an expected case")
        elif num_bfore==1: 
            if num_after==2:
                return side
            else:
                continue
        elif num_bfore==2:
            assert num_after==2, "if before we had 2 neighbors, we must still have 2 after."
        else:
            raise ValueError("Not an expected case")
    return None


def _message_break_next_order_logic(
    reverse_order_tracker:_ReverseOrder, seen_break:_PerBound[bool], side_order:_PerBound[_Side],
    num_neighbors_per_side_before:_PerBound[list[int]],
    num_neighbors_per_side_after:_PerBound[list[int]],
)->_ReverseOrder:
    
    ## If only one of the sides seen the break:
    if seen_break.first and not seen_break.last:
        next_side = side_order[Last]
    elif seen_break.last and not seen_break.first:
        next_side = side_order[First]
    elif not seen_break.first and not seen_break.last:
        raise ValueError("Bug. We shouldn't be here.")
    else:
        next_side = _decide_by_future_neighbors(side_order, num_neighbors_per_side_before, num_neighbors_per_side_after)
    
    ## Decide on new order::
    if next_side is Left:
        reverse_order_tracker.set_false()
    elif next_side is Right:
        reverse_order_tracker.set_true()
    elif next_side is None:
        pass   # Keep it as it is
    else:
        raise ValueError("Not an expected option!")
    #
    return reverse_order_tracker


def _find_all_side_neighbors( 
    tn:KagomeTensorNetwork, first_last:_Bound,
    nodes:_PerSide[TensorNode], side_order:_PerBound[_Side], 
    side_edges:_SideEdges
)->tuple[
    EdgeIndicatorType,  # last_edge
    list[int]  # neighbor indices
]:
    # prepare results:
    res = []
    # unpack respective values:
    side = side_order[first_last]            
    node = nodes[side]
    edge = side_edges[side]           
    while edge in node.edges:     
        # get neighbor:
        neighbor = tn.find_neighbor(node, edge)
        # add neighbor:
        res.append(neighbor.index)  # Add to end 
        # move iterator and get next edge
        edge = side_edges.next(side)  
    return edge, res  # return last edge


def _derive_message_neighbors(
    tn:KagomeTensorNetwork, row:list[int], reverse_order_tracker:_ReverseOrder,
    side_edges:_SideEdges
) -> tuple[
    _PerBound[list[int]],
    _PerBound[bool]
]:
    # Node at each side
    nodes = _PerSide[TensorNode](
        left  = tn.nodes[row[0]],
        right = tn.nodes[row[-1]]
    )
    msg_neighbors_at_bounds = _PerBound[list[int]](first=[], last=[])

    # which side is first?       
    if reverse_order_tracker.check_state_and_update():
        side_order = _PerBound[_Side](first=Right, last=Left)
    else:
        side_order = _PerBound[_Side](first=Left, last=Right)

    ## Look for breaks between MPS messages:
    seen_break = _PerBound[bool](first=False, last=False)

    ## Add neighbors if they exist:
    for first_last in _Bound:
        last_edge, neighbors = _find_all_side_neighbors(tn, first_last, nodes, side_order, side_edges )
        msg_neighbors_at_bounds[first_last] += neighbors
            
        ## Special case when MPS messages change from one to another:
        if last_edge == 'Break':
            side = side_order[first_last]
            side_edges.next(side)  # move iterator
            seen_break[first_last] = True

    ## At a break in the side MPS messages:
    if seen_break.first or seen_break.last: 

        # Keep data about beighbors per side:
        num_neighbors_per_side_before = _PerSide[int](left=0, right=0)
        num_neighbors_per_side_before[side_order[First]] = len(msg_neighbors_at_bounds[First])
        num_neighbors_per_side_before[side_order[Last]] = len(msg_neighbors_at_bounds[Last])
        num_neighbors_per_side_after = deepcopy(num_neighbors_per_side_before)

        # Collect lost neighnors
        for first_last in _Bound:
            _, neighbors = _find_all_side_neighbors(tn, first_last, nodes, side_order, side_edges )
            msg_neighbors_at_bounds[Last] += neighbors # force_assigning_at_end
            num_neighbors_per_side_after[side_order[first_last]] += len(neighbors)

        # Next order logic
        reverse_order_tracker = _message_break_next_order_logic(reverse_order_tracker, seen_break, side_order, num_neighbors_per_side_before, num_neighbors_per_side_after)

    ## Seen break per side:
    return msg_neighbors_at_bounds, seen_break



def _split_row(row:list[int], row_and_core:set[int])->_PerSide[list[int]]:
    splitted_row = _PerSide(left=[], right=[])
    seen_core = False
    for i in row:
        if i in row_and_core:
            seen_core = True
            continue

        if seen_core:
            splitted_row.right.append(i)
        else:
            splitted_row.left.append(i)

    return splitted_row


def  _derive_row_to_contract(
    tn:KagomeTensorNetwork,
    row:list[int], 
    depth:ContractionDepth, 
    reverse_order_now:bool, 
    msg_neighbors:_PerBound[list[int]], 
    seen_break_per_order:_PerBound[bool]
)->list[int]:

    ## Prepare simple `full_row` solution:
    if reverse_order_now:
        row.reverse()
    full_row_solution = msg_neighbors.first + row + msg_neighbors.last

    ## simple case:
    if depth is not ToCore: 
        return full_row_solution

    ## Derive intersection between core indices and current row
    core_indices = {node.index for node in tn.get_core_nodes()}
    row_and_core = set(row).intersection(core_indices)
    if len(row_and_core)==0:
        return full_row_solution

    ## check if we need to stop:
    row_per_side = _split_row(row, row_and_core)
    if reverse_order_now:
        pass

    seen_break_per_side = seen_break_per_order.to_per_side(reverse_order_now)
    if seen_break_per_side.left:
        pass


    return msg_neighbors.first + row + msg_neighbors.last



def derive_full_contraction_order(
    tn:KagomeTensorNetwork,  
    direction:BlockSide,
    depth:BlockSide
)->list[int]:

    ## Prepare output:
    contraction_order = []    

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
    reverse_order_tracker = _ReverseOrder()

    ## First message:
    contraction_order += tn.message_indices(major_direction.opposite()) 

    ## Lattice and its connected nodes:
    for row in lattice_rows_ordered_right:

        reverse_order_now = reverse_order_tracker.state
        msg_neighbors, seen_break_per_order = _derive_message_neighbors(tn, row, reverse_order_tracker, side_edges)
        row_to_contract = _derive_row_to_contract(tn, row, depth, reverse_order_now, msg_neighbors, seen_break_per_order)

        ## add row to con_order:
        contraction_order.extend( row_to_contract )

    ## Last minor-direction of contraction:
    # make opposite than last direction
    last_order_was_reversed = reverse_order_tracker.prev_state()
    final_order_is_reversed = not last_order_was_reversed
    if final_order_is_reversed:
        last_minor_direction = minor_right.opposite()
    else:
        last_minor_direction = minor_right

    ## Last msg:
    if depth is Full:
        last_msg = tn.message_indices(major_direction)
        if not final_order_is_reversed:
            last_msg.reverse()
        contraction_order += last_msg

    ## Validate:
    if DEBUG_MODE :
        # Both list of messages are in `con_order`:
        assert side_edges.exhausted.left
        assert side_edges.exhausted.right    

    ## we can return `last_minor_direction` if this is important. It is usually NOT.
    return contraction_order


def get_contraction_order(tn:KagomeTensorNetwork, direction:BlockSide, depth:ContractionDepth, plot_:bool=False)->list[int]:

    ## Get cached contractions or derive them:
    global FULL_CONTRACTION_ORDERS_CACHE
    cache_key = (tn.lattice.N, direction, depth)
    if cache_key in FULL_CONTRACTION_ORDERS_CACHE:
        contraction_order = FULL_CONTRACTION_ORDERS_CACHE[cache_key]
    else:
        # Derive:
        contraction_order = derive_full_contraction_order(tn, direction, depth)
        # Save for next time:
        FULL_CONTRACTION_ORDERS_CACHE[cache_key] = contraction_order
    

    ## Plot result:
    if plot_:
        _plot_con_order(tn, contraction_order)

    ## Return:
    return contraction_order


def _plot_con_order(tn, con_order)->None:
    # For plotting contraction order, if needed:
    import matplotlib.pyplot as plt
    from tensor_networks.visualizations import plot_contraction_order, draw_now
    tn.plot(detailed=False)
    plot_contraction_order(tn.positions, con_order)
    plt.title(f"Contraction Order")
    draw_now()


if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import bp_test
    bp_test.main_test()

