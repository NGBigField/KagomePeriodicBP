if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
    from project_paths import add_src, add_base, add_scripts
    add_src()
    add_base()
    add_scripts()



# Control flags:
from _config_reader import DEBUG_MODE

# Types we need in our module:
from lattices.directions import Direction, LatticeDirection, BlockSide, sort
from tensor_networks import ArbitraryTN, ModeTN, EdgeTN, TensorNode, MPS, CoreTN, get_common_edge, get_common_edge_legs
from tensor_networks import TensorNode
from tensor_networks.node import TensorNode
from enums import ContractionDepth, NodeFunctionality, UpdateMode
from containers import UpdateEdge
from _types import EdgeIndicatorType

# Import utilities that are shared between modules:
from utils import lists, tuples

# For numerics and tensor stuff:
import numpy as np


def _rearrange_tensors_and_legs_into_canonical_order(tn:ArbitraryTN)->None:
    ## Basic info:
    # Get core nodes:
    cores = tn.nodes[:2]
    # Find neighbors, not in order
    neighbors1 = [node for node in tn.all_neighbors(cores[0]) if node is not cores[1]]
    neighbors2 = [node for node in tn.all_neighbors(cores[1]) if node is not cores[0]]
    # Check:
    if DEBUG_MODE:
        assert len(neighbors1) == len(neighbors2) == 3
        assert cores[0].functionality in [NodeFunctionality.CenterCore, NodeFunctionality.AroundCore]
        assert cores[1].functionality in [NodeFunctionality.CenterCore, NodeFunctionality.AroundCore]


    # find middle leg between cores:
    ie1, ie2 = get_common_edge_legs(cores[0], cores[1])
    ## Rearrange the legs of the core nodes and the order of env_tensors connected to them:
    env_index = 2
    for core_node, edge_index in zip([cores[0], cores[1]], [ie1, ie2]):
        ## Correct order of core_node legs:
        direction_to_other_core = core_node.directions[edge_index]
        ordered_directions = sort.arbitrary_directions_by_clock_order(direction_to_other_core, core_node.directions, clockwise=False)
        permutation_order = [core_node.directions.index(dir) for dir in ordered_directions]
        core_node.permute(permutation_order)

        ## Correct the indices of the neighbors in the TN ordering of nodes:
        for is_first, _, direction in lists.iterate_with_edge_indicators(core_node.directions):
            # First directions should now be the other core:
            if is_first:
                if DEBUG_MODE:
                    assert direction is direction_to_other_core
                continue
            
            # Give node a new index:
            neighbor = tn.find_neighbor(core_node, direction)
            crnt_neighbor_index = tn.nodes.index(neighbor)
            tn.swap_nodes(crnt_neighbor_index, env_index)

            env_index += 1

    ## Rearrange legs of env tensors:
    env_tensors = tn.nodes[2:]
    assert len(env_tensors) == 6, "Must have 6 environment tensors"
    for prev, crnt, next in lists.iterate_with_periodic_prev_next_items(env_tensors):
        for i, direction in enumerate(crnt.directions):
            neighbor = tn.find_neighbor(crnt, direction) 
            if   neighbor is prev:   i0 = i
            elif neighbor in cores:  i1 = i
            elif neighbor is next:   i2 = i
            else:   
                raise ValueError("Bug. Should have found a correct neighbor")
        permutation_order = [i0, i1, i2]
        crnt.permute(permutation_order)

    return tn




""" ================================================================================================================ """
""" ================================================================================================================ """
""" ================================================================================================================ """



def _sort_direction_by_angle_keeping_outer_directions_out(center_directions:list[Direction], extrema_directions:list[Direction])->list[Direction]:
    lis = sorted(center_directions+extrema_directions, key=lambda d: d.angle)
    if lis[-2] in extrema_directions and lis[-1] in extrema_directions:
        lis = lists.cycle_items(lis, 1)
    elif lis[0] in extrema_directions and lis[1] in extrema_directions:
        lis = lists.cycle_items(lis, -1)
    return lis


def _split_direction_by_type(directions:list[Direction])->tuple[list[LatticeDirection], list[BlockSide], list[Direction]]:
    lattice, block, arbitrary = [], [], []
    for dir in directions:
        if isinstance(dir, LatticeDirection):
            lattice.append(dir)
        elif isinstance(dir, BlockSide):
            block.append(dir)
        elif isinstance(dir, Direction):
            arbitrary.append(dir)
        else:
            raise TypeError(f"Not an expected type {type(dir)!r}")
    return lattice, block, arbitrary 


def _derive_edge_and_nodes(
    mode_tn:ModeTN, 
    edge_tuple:UpdateEdge
)->tuple[EdgeIndicatorType, TensorNode, TensorNode]:
    """ Find the correct edge nodes """

    is_in_core = edge_tuple.is_in_core()
    is_center_included = mode_tn.mode in edge_tuple

    if is_in_core:
        options = mode_tn.get_nodes_by_functionality(NodeFunctionality.CenterCore)
    else:
        options = mode_tn.get_nodes_by_functionality(NodeFunctionality.AroundCore)
    node1 = next((node for node in options if node.unit_cell_flavor in edge_tuple))

    ## 
    valid_neighbors = [node for node in mode_tn.get_core_nodes() if node is not node1 and mode_tn.are_neighbors(node1, node)]
    if is_in_core:
        options = [node for node in valid_neighbors if node.functionality==NodeFunctionality.CenterCore]
    else:
        options = [node for node in valid_neighbors if node.functionality==NodeFunctionality.AroundCore]
    if is_center_included:
        options.append(mode_tn.center_node)
    node2 = next((node for node in options if node.unit_cell_flavor in edge_tuple))

    edge = get_common_edge(node1, node2)
    return edge, node1, node2     


def _contract_all_nodes_except_neighbors(
    tn:ArbitraryTN,
    node1:TensorNode,
    node2:TensorNode
)->tuple[ArbitraryTN, TensorNode]:

    ## Find immediate neighbors:
    neighbors = []
    common_neighbor : TensorNode = None
    for node in tn.nodes:
        if node in [node1, node2]:
            continue
        is_neighbor1 = tn.are_neighbors(node, node1) 
        is_neighbor2 = tn.are_neighbors(node, node2)
        if is_neighbor1 or is_neighbor2:
            neighbors.append(node)
        if is_neighbor1 and is_neighbor2:
            common_neighbor = node
    assert common_neighbor is not None, "Bug. We must find a common neighbor"

    ## Contract all tensors than we don't need:
    nodes_to_keep_old = neighbors+[node1, node2]
    nodes_to_keep_new = tn.contract_all_nodes_with_exceptions(nodes_to_keep_old)

    ## Update common_neighbor:
    common_neighbor = nodes_to_keep_new[nodes_to_keep_old.index(common_neighbor)]

    ## mark all env nodes:
    for node in tn.nodes:
        if node in [node1, node2]:
            continue
        node.functionality = NodeFunctionality.Environment

    return tn, common_neighbor


def _derive_commons_neighbor_edges_connections_as_a_matrix(
    tn:ArbitraryTN,
    common_neighbor:TensorNode,
)->tuple[list[EdgeIndicatorType], list[EdgeIndicatorType]]:
    inner_directions = [dir for dir in common_neighbor.directions if tn.find_neighbor(common_neighbor, dir).functionality!=NodeFunctionality.Environment]
    outer_directions = [dir for dir in common_neighbor.directions if tn.find_neighbor(common_neighbor, dir).functionality==NodeFunctionality.Environment] 
    directions_sorted = _sort_direction_by_angle_keeping_outer_directions_out(inner_directions, outer_directions)
    assert len(directions_sorted)==4
    edge1 = [common_neighbor.edge_in_dir(directions_sorted[i]) for i in [0,1]]
    edge2 = [common_neighbor.edge_in_dir(directions_sorted[i]) for i in [2,3]]
    return edge1, edge2 


def reduce_mode_to_edge_and_env(
    mode_tn:ModeTN, 
    edge_tuple:UpdateEdge
)->EdgeTN:
    
    ## Get basic data:
    edge, node1, node2 = _derive_edge_and_nodes(mode_tn, edge_tuple)

    ## Contract everything except the mode and its neighbors:
    tn, common_neighbor = _contract_all_nodes_except_neighbors(mode_tn.to_arbitrary_tn(), node1, node2)

    ## Decide which of the common_neighbor's legs fit at which side:
    edge1, edge2 = _derive_commons_neighbor_edges_connections_as_a_matrix(tn, common_neighbor)

    ## Split common neighbor using QE-decomposition:
    _, _ = tn.qr_decomp(common_neighbor, edge1, edge2)

    ## Rearrange legs in a canonical order used in the input of `ite.rho_ij()`
    _rearrange_tensors_and_legs_into_canonical_order(tn)

    edge_tn = EdgeTN.from_arbitrary_tn(tn)
    if DEBUG_MODE:
        edge_tn.validate()

    return tn




if __name__ == "__main__":
    from scripts.contraction_test import main_test
    main_test()