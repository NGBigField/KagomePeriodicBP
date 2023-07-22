if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)



# Get Global-Config:
from _config_reader import DEBUG_MODE

from typing import Callable
from tensor_networks import KagomeTensorNetwork
from lattices.directions import Direction
from enums import ContractionDepth, NodeFunctionality
from containers import ContractionConfig


# Everyone needs numpy:
import numpy as np

# Use our common utilities:
from utils import lists

# For smart iterations:
import itertools


_PosFuncType = Callable[[int, int], tuple[int, int] ]


def _derive_axes(
    tn:KagomeTensorNetwork,
    direction:Direction,
)->tuple[
    list[int],   # axis1
    tuple[list[int], list[int]],   # axis2_variations
    tuple[Direction, Direction], # axis2_directions
    _PosFuncType,   # pos_func
    bool  # has corners
]:
    ## Derive size of tensor:
    min_x, max_x, min_y, max_y = tn.boundaries()
    xs_l2r = np.linspace(min_x, max_x, tn.original_lattice_dims[1]).tolist()
    xs_r2l = np.linspace(max_x, min_x, tn.original_lattice_dims[1]).tolist()
    ys_d2u = np.linspace(min_y, max_y, tn.original_lattice_dims[0]).tolist()
    ys_u2d = np.linspace(max_y, min_y, tn.original_lattice_dims[0]).tolist()

    ## Direction dictates the axis:
    if direction in [Direction.Left, Direction.Right]:
        axis2_variations = (ys_u2d, ys_d2u)
        axis2_directions = (D, U)
        _pos_func : Callable[ [int, int], tuple[int, int] ] = lambda p1, p2: (p1, p2)  # Define which is the x value, and which is the y value
        if   direction is Direction.Left:   axis1 = xs_r2l
        elif direction is Direction.Right:  axis1 = xs_l2r  
        else: raise ValueError(f"Not an option")
    
    elif direction in [Direction.Up, Direction.Down]:
        axis2_variations = (xs_l2r, xs_r2l)
        axis2_directions = (R, L)
        _pos_func : Callable[ [int, int], tuple[int, int] ] = lambda p1, p2: (p2, p1)  # Define which is the x value, and which is the y value
        if   direction is Direction.Down:   axis1 = ys_u2d            
        elif direction is Direction.Up:     axis1 = ys_d2u
        else: raise ValueError(f"Not an option")

    else:
        raise ValueError(f"Not an option")
    
    ## Check if tn has its corners 
    # (usually corners are missing when a square lattice is connected by messages from each side)
    # try all 4 corners:
    corners = tn.get_corner_nodes()
    # Case where only some corners are missing, is not yet implemented
    if len(corners)==4:
        has_corners = True
    elif len(corners)==0:
        has_corners = False
    else:
        raise NotImplementedError("Case where only some corners are missing, is not yet implemented")
                 
    return axis1, axis2_variations, axis2_directions, _pos_func, has_corners


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
    direction:Direction,
    depth:ContractionDepth|int,
    plot_:bool=False
)->tuple[
    list[int],
    Direction
]:

    ## Derive iterations info:
    axis1, axis2_variations, axis2_directions, _pos_func, has_corners = _derive_axes(tn, direction)
    
    # Prepare output:
    con_order = []    

    ## Various iteration indicators:
    # for snake-like motion
    if ContractionConfig.random_snake_pattern:
        is_reversed_order = lists.random_item([False, True])  # The initial direction is random, from it go like snake as usual
    else:
        is_reversed_order = False  
    # for depth untill core mode:
    has_seen_core = False
    # the second axis - alternating:
    axis2 : list[int] = []
    dir2 : Direction = None  #type: ignore
    
    # Firstly, go over all values of the first axis:
    for i1, (first1, last1, p1) in enumerate(lists.iterate_with_edge_indicators(axis1)):

        # the last row\column is reserved for the outgoing message
        if isinstance(depth, int) and depth==i1:
            break
        if depth is ContractionDepth.ToMessage and last1:  
            break
        if DEBUG_MODE and last1:
            assert depth is ContractionDepth.Full, f"Only full-contraction should arrive at the final row|column. depth is {depth!r}"

        # snake-like motion:
        axis2 = axis2_variations[0] if is_reversed_order else axis2_variations[1]
        dir2 = axis2_directions[0] if is_reversed_order else axis2_directions[1]
        is_reversed_order = not is_reversed_order                        
        
        # Because this is not a perfect square, we need to contract the almost edge tensor before continuing:
        if i1 == 1 and not has_corners:
            axis2 = lists.swap_items(axis2, 0, 1)
        
        # Secondly, go over all values of the second axis:
        this_row_or_column = []
        for first2, last2, p2 in lists.iterate_with_edge_indicators(axis2): 
            # ignore the corners:
            if not has_corners and (first2 or last2) and (first1 or last1):  
                continue

            # Add to contraction order
            node = tn.get_tensor_in_pos(_pos_func(p1, p2))
            this_row_or_column.append(node.index)
            if node.functionality is NodeFunctionality.Core: 
                has_seen_core = True
        
        ## Check if we need to get only up to the core:
        if has_seen_core and depth is ContractionDepth.ToCore:
            break
        
        # Add entire row or column to the contraction list    
        con_order.extend(this_row_or_column)

        ## Determine the direction of the last contracted tensors:

    try:
        last_direction = _determine_direction(axis2, _pos_func)
    except:
        last_direction = dir2


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

