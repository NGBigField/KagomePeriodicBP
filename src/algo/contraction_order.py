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
from tensor_networks import KagomeTensorNetwork
from lattices.directions import Direction, BlockSide, LatticeDirection
from enums import ContractionDepth, NodeFunctionality
from containers import ContractionConfig

## For common types:
from _error_types import NetworkConnectionError

# Everyone needs numpy:
import numpy as np

# Use our common utilities:
from utils import lists

# For smart iterations:
import itertools


_PosFuncType = Callable[[int, int], tuple[int, int] ]

BREAK_MARKER = -100


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


def derive_contraction_orders(
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
    iterators = dict(
        left  = iter(left_sorted_outer_edges+["End"]), 
        right = iter(right_sorted_outer_edges+["End"])
    )
    edges = {side: next(iterator) for side, iterator in iterators.items()}

    ## First message:
    con_order = tn.message_indices(major_direction.opposite())

    ## Iterator to switch contraction direction every two rows:
    reverse_order = itertools.cycle([True, True, False, False])

    ## Lattice and its connected nodes:
    for row in lattice_rows_ordered_right:

        # Node at each side
        nodes = dict(
            left  = tn.nodes[row[0]],
            right = tn.nodes[row[-1]]
        )
        msg_neighbors = dict(first=[], last=[])

        # which side is first?       
        is_reverse_order = next(reverse_order)
        if is_reverse_order:
            row.reverse()
            side_order = dict(first='right', last='left')
        else:
            side_order = dict(first='left', last='right')

        ## Add neighbors if they exist:
        for order in ['first', 'last']:
            # unpack respective values:
            side = side_order[order]            
            iterator = iterators[side]
            node = nodes[side]
            edge = edges[side]
            # Find msg neigbors:
            while edge in node.edges:                
                neighbor = tn.find_neighbor(node, edge)
                msg_neighbors[order].append(neighbor.index)
                edge = next(iterator)
                edges[side] = edge
            ## Mark places where messages break:
            if edge == 'Break':
                msg_neighbors[order].append(BREAK_MARKER)
                edge = next(iterator)
                edges[side] = edge


        ## add row to con_order:
        con_order.extend( msg_neighbors['first'] + row + msg_neighbors['last'] )


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

    ## Last minor-direction of contraction:
    if is_reverse_order:
        last_direction = minor_right.opposite()
    else:
        last_direction = minor_right

    return con_order, last_direction



if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import bp_test
    bp_test.main_test()

