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
from tensor_networks import KagomeTNRepeatedUnitCell, TensorNode, CoreTN, ModeTN
from lattices.directions import Direction, BlockSide, LatticeDirection, sort
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
from typing import Generic, TypeVar, Final, Callable, TypeAlias, TypedDict, Any
from dataclasses import dataclass, fields
from copy import deepcopy


## Types:
_T = TypeVar("_T")
_EnumType = TypeVar("_EnumType")
_PosFuncType = Callable[[int, int], tuple[int, int] ]
_FULL_CONTRACTION_ORDERS_CACHE_KEY_TYPE : TypeAlias = tuple[int, BlockSide, ContractionDepth]

## Constants:
CACHE_CONTRACTION_ORDER : bool = True
BREAK_MARKER = -100
Full = ContractionDepth.Full
ToCore = ContractionDepth.ToCore
ToMessage = ContractionDepth.ToMessage

## For Caching results:
FULL_CONTRACTION_ORDERS_CACHE : dict[_FULL_CONTRACTION_ORDERS_CACHE_KEY_TYPE, list[int]] = {}


CORE_CONTRATION_ORDERS : dict[BlockSide, list[int]] = {
    BlockSide.UR : [19, 20, 9, 7, 3, 18, 10, 4, 17, 16, 0, 2, 5, 8, 11, 1, 6, 12, 13, 14, 15],
    BlockSide.DR : [16, 17, 18, 19, 3, 0, 15, 1, 2, 4, 7, 20, 9, 5, 14, 13, 6, 8, 10, 11, 12],
    BlockSide.D  : [16, 15, 14, 1, 0, 17, 13, 2, 18, 19, 3, 4, 5, 6, 12, 7, 8, 11, 10, 9, 20],
}
for side in [BlockSide.UR, BlockSide.DR, BlockSide.D]:
    opposite_list = lists.reversed( CORE_CONTRATION_ORDERS[side] )
    CORE_CONTRATION_ORDERS[side.opposite()] = opposite_list



def _validate_core_con_order(contraction_order:list[int])->None:
    given = set(contraction_order)
    needed = set(range(21))
    assert given==needed


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



class _UpToCoreControl:
    """Manages memory of everything we need to know when contraction "Up-To-Core"
    """
    __slots__ = (
        "core_indices", "terminal_indices", "stop_next_iterations", "seen_break"
    )
    
    def __init__(self, tn:KagomeTNRepeatedUnitCell, major_direction:BlockSide) -> None:        
        self.core_indices : set[int] = {node.index for node in tn.get_core_nodes()}
        self.terminal_indices : set[int] = _derive_terminal_indices(tn, major_direction)
        self.stop_next_iterations : bool = False
        self.seen_break : _PerSide[bool] = _PerSide[bool](left=False, right=False)

    def update_seen_break(self, seen_break_now_per_side:_PerSide[bool])->None:
        if seen_break_now_per_side.left:
            self.seen_break.left = True
        if seen_break_now_per_side.right:
            self.seen_break.right = True
        

def _derive_terminal_indices(tn:KagomeTNRepeatedUnitCell, major_direction:BlockSide)->set[int]:
    """derive_terminal_indices
    The convention is that the base of the center upper triangle defines the terminal row for the contraction up to core
    """
    # Minor direction can be random-valid direction, not imporant, since we care only for the indices
    minor_direction = major_direction.orthogonal_clockwise_lattice_direction()
    # Get center triangle:
    center_triangle : kagome.UpperTriangle = tn.get_center_triangle()

    ## find the vertex_names of the base, by checking which list has 2 values:
    upper_triangle_order = kagome.get_upper_triangle_vertices_order(major_direction, minor_direction)
    # Get the names:
    if len(upper_triangle_order[0])==2:
        vertex_names = upper_triangle_order[0]
    elif len(upper_triangle_order[1])==2:
        vertex_names = upper_triangle_order[1]
    else:
        raise ValueError("Impossible situation. Bug.")
    
    ## Get the indices:
    indices = {center_triangle[name].index for name in vertex_names}

    return indices
        

def _sorted_side_outer_edges(
    tn:KagomeTNRepeatedUnitCell, direction:BlockSide, with_break:bool=False
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
    tn:KagomeTNRepeatedUnitCell, first_last:_Bound,
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
    tn:KagomeTNRepeatedUnitCell, row:list[int], depth:ContractionDepth,
    reverse_order_tracker:_ReverseOrder,
    side_edges:_SideEdges
) -> tuple[
    _PerBound[list[int]],
    list[int],
    _PerBound[bool]
]:
    # Node at each side
    nodes = _PerSide[TensorNode](
        left  = tn.nodes[row[0]],
        right = tn.nodes[row[-1]]
    )

    ## Prepare outputs:
    msg_neighbors_at_bounds = _PerBound[list[int]](first=[], last=[])
    annex_neighbors = []

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
            num_neighbors_per_side_after[side_order[first_last]] += len(neighbors)
            annex_neighbors += neighbors  # assign to separate list

        # Next order logic
        reverse_order_tracker = _message_break_next_order_logic(reverse_order_tracker, seen_break, side_order, num_neighbors_per_side_before, num_neighbors_per_side_after)

    ## Seen break per side:
    return msg_neighbors_at_bounds, annex_neighbors, seen_break



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


def _derive_full_row(row_in:list[int], reverse_order_now:bool, msg_neighbors:_PerBound[list[int]], annex_neighbors:list[int]): 
    if reverse_order_now:
        row_reversed = row_in.copy()
        row_reversed.reverse() 
        return msg_neighbors.first + row_reversed + msg_neighbors.last + annex_neighbors
    else:
        return msg_neighbors.first + row_in + msg_neighbors.last + annex_neighbors


def _derive_row_to_contract(
    control:_UpToCoreControl,
    row:list[int], 
    depth:ContractionDepth, 
    reverse_order_now:bool, 
    msg_neighbors:_PerBound[list[int]], 
    annex_neighbors:list[int],
    seen_break_per_order:_PerBound[bool]  # can be used for debug
)->list[int]:

    ## simple cases:
    if depth is not ToCore: 
        return _derive_full_row(row, reverse_order_now, msg_neighbors, annex_neighbors)
    if control.stop_next_iterations:
        return []  # don't append anything from now on    

    ## Derive intersection between core indices and current row
    core_in_this_row = set(row).intersection(control.core_indices)
    if len(core_in_this_row)==0:
        return _derive_full_row(row, reverse_order_now, msg_neighbors, annex_neighbors)

    ## Prepare values arranged by side:
    # Split row, ignore core indices:
    row_per_side = _split_row(row, core_in_this_row)
    seen_break_now_per_side = seen_break_per_order.to_per_side(reverse_order_now)

    ## If we haven't arrived at a terminal row, all is simple:
    if core_in_this_row.isdisjoint(control.terminal_indices):
        row_per_bound = row_per_side.to_per_order(reverse_order_now)
        # Track if we've seen the break in the side messages:
        control.update_seen_break(seen_break_now_per_side)  
        return msg_neighbors.first + row_per_bound.first + row_per_bound.last + msg_neighbors.last + annex_neighbors

    ## if we are here, this is the last left-side contraction and right shouldn't continue    
    control.stop_next_iterations = True
    msg_neighbors_per_side = msg_neighbors.to_per_side(reverse_order_now)

    ## Add right neighbor, if it is unreachable by the opposite contraction:
    if not control.seen_break.right and seen_break_now_per_side.right:
        assert len(msg_neighbors_per_side.right)>0
        return msg_neighbors_per_side.left + row_per_side.left + msg_neighbors_per_side.right


    ## Return last left row:    
    return msg_neighbors_per_side.left + row_per_side.left


def _two_level_deep_sorted_clockwise_search(tn:ModeTN, node_level0:TensorNode, major_direction:BlockSide) -> list[int]:
    ## Prepare data and output:
    indices_level2 : list[int] = []
    indices_level1 = []

    # for all core nodes in clockwise order:
    directions_level1 = major_direction.matching_lattice_directions()
    directions_level1_in_clockwise_order = sort.specific_typed_directions_by_clock_order(directions_level1, clockwise=True )
    for direction_level1 in directions_level1_in_clockwise_order:
        node_level1 = tn.find_neighbor(node_level0, direction_level1)
        indices_level1.append(node_level1.index)

        # for all env nodes in clockwise order:
        directions_level2 = [dir for dir in node_level1.directions if tn.find_neighbor(node_level1, dir).functionality is NodeFunctionality.Environment  ]
        directions_level2_in_clockwise_order = sort.specific_typed_directions_by_clock_order(directions_level2, clockwise=True)
        for direction_level2 in directions_level2_in_clockwise_order:
            node_level2 = tn.find_neighbor(node_level1, direction_level2)
            indices_level2.append(node_level2.index)

    first2, *rest2, last2 = indices_level2
    rest2.reverse()
    return [first2] + indices_level1 + [last2] + rest2


def derive_mode_tn_full_contraction_order(
    tn:ModeTN,  
    direction:BlockSide,
    depth:ContractionDepth
)->list[int]:
    # Check:
    assert direction in tn.major_sides, "Not all direction are possible in the small mode-TN"

    ## Use breadth-first algorithm to get con_order for each side:
    up_to_core = _two_level_deep_sorted_clockwise_search(tn, tn.center_node, direction.opposite())
    from_core = _two_level_deep_sorted_clockwise_search(tn, tn.center_node, direction)

    ## When we start bubblecon, the order is reversed:
    up_to_core.reverse()
    con_order = up_to_core + [tn.center_node.index] + from_core

    ## Trim if we go only up to edge:
    if depth is ContractionDepth.ToEdge:
        con_order = con_order[:-6]
    return con_order


def derive_kagome_tn_contraction_order(
    tn:KagomeTNRepeatedUnitCell,  
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
    # Iterator to switch contraction direction every two rows:
    reverse_order_tracker = _ReverseOrder()
    # Control different sides advance when contracting up to core:
    if depth is ToCore:
        up_to_core_control = _UpToCoreControl(tn, major_direction)
    else:
        up_to_core_control = None


    ## First message:
    contraction_order += tn.message_indices(major_direction.opposite()) 

    ## Lattice and its connected nodes:
    for row in lattice_rows_ordered_right:

        reverse_order_now = reverse_order_tracker.state
        msg_neighbors, annex_neighbors, seen_break_per_order = _derive_message_neighbors(tn, row, depth, reverse_order_tracker, side_edges)
        row_to_contract = _derive_row_to_contract(up_to_core_control, row, depth, reverse_order_now, msg_neighbors, annex_neighbors, seen_break_per_order)

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
    if DEBUG_MODE and depth in [ToMessage, Full]:
        # Both list of messages are in `con_order`:
        assert side_edges.exhausted.left
        assert side_edges.exhausted.right    

    ## we can return `last_minor_direction` if this is important. It is usually NOT.
    return contraction_order



def get_contraction_order(tn:KagomeTNRepeatedUnitCell|CoreTN, direction:BlockSide, depth:ContractionDepth, plot_:bool=False)->list[int]:

    ## In the case where it's a CoreTN (which has an expected canonical structure)
    if isinstance(tn, CoreTN):
        assert depth is ContractionDepth.Full
        contraction_order = CORE_CONTRATION_ORDERS[direction]
        if DEBUG_MODE:
            _validate_core_con_order(contraction_order)
        return contraction_order
    
    ## In the case where it's a Mode (which is small and simple but a bit different each time)
    if isinstance(tn, ModeTN):
        assert depth in [ ContractionDepth.Full, ContractionDepth.ToEdge]
        return derive_mode_tn_full_contraction_order(tn, direction, depth) 


    ## In the case where it's a Full Kagome lattice TN:
    ## Get cached contractions or derive them:
    assert isinstance(tn, KagomeTNRepeatedUnitCell)
    assert depth is not ContractionDepth.ToEdge, f"Contraction to edge method does not exist for a full Kagome TN"
    global FULL_CONTRACTION_ORDERS_CACHE
    cache_key = (tn.lattice.N, direction, depth)

    if CACHE_CONTRACTION_ORDER and cache_key in FULL_CONTRACTION_ORDERS_CACHE:
        contraction_order = FULL_CONTRACTION_ORDERS_CACHE[cache_key]
    else:
        # Derive:
        contraction_order = derive_kagome_tn_contraction_order(tn, direction, depth)
        # Save for next time:
        FULL_CONTRACTION_ORDERS_CACHE[cache_key] = contraction_order
    

    ## Plot result:
    if plot_:
        _plot_con_order(tn, contraction_order, detailed=False, with_arrows=True)

    ## Return:
    return contraction_order


def _plot_con_order(tn, con_order, detailed:bool=False, with_arrows:bool=True)->None:
    # For plotting contraction order, if needed:
    import matplotlib.pyplot as plt
    from tensor_networks.visualizations import plot_contraction_order, plot_contraction_nodes, draw_now
    tn.plot(detailed=detailed)
    if with_arrows:
        plot_contraction_order(tn.positions, con_order)
        plt.title(f"Contraction Order")
    else:
        plot_contraction_nodes(tn.positions, con_order)
        plt.title(f"Contracted Nodes")

    draw_now()


if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import test_contraction
    # bp_test.main_test()
    test_contraction.main_test()

