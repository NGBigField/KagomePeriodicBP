if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)



# Get Global-Config:
from _config_reader import DEBUG_MODE

from typing import Callable
from tensor_networks import KagomeTensorNetwork
from lattices.directions import Direction, BlockSide
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
    direction = block.U

    ## Prepare output:
    con_order = []    

    # Define Directions:
    major_forward = direction
    major_back = major_forward.opposite()
    minor_right = direction.orthogonal_clockwise_lattice_direction()
    minor_left = minor_right.opposite()

    ## Start by fetching the lattice-nodes in order and the messages:
    lattice_rows_ordered_right = tn.lattice.nodes_indices_rows_in_direction(major_forward, minor_right)
    sorted_boundary_edges : dict[BlockSide, list[str]] = {
        side: tn.lattice.sorted_boundary_edges(side) for side in BlockSide.all_in_counter_clockwise_order()
    }

    ## First message:
    if major_back in tn.messages:
        first_message_indices = tn.message_indices(major_back)
        con_order.append(first_message_indices)

    ## with messages?        
    for row in lattice_rows_ordered_right:
        ## Add neighbors if they exist:
        first, last = tn.nodes[row[0]], tn.nodes[row[-1]]
    




        # First:
        # try:
        #     new_first = tn.find_neighbor(first, minor_left)                
        # except NetworkConnectionError:
        #     pass
        # else:
        #     row.insert(0, new_first.index)   
        # # Last:
        # try:
        #     new_last = tn.find_neighbor(last, minor_right)                
        # except NetworkConnectionError:
        #     pass
        # else:
        #     row.append(new_last)


    if plot_:
        # For plotting contraction order, if needed:
        import matplotlib.pyplot as plt
        from tensor_networks.visualizations import plot_contraction_order
        tn.plot(detailed=False)
        plot_contraction_order(tn.positions, con_order)
        plt.title(f"Contraction in Direction {str(direction)}" )

    return con_order, last_direction



if __name__ == "__main__":
    from scripts.core_ite_test import main
    main()

