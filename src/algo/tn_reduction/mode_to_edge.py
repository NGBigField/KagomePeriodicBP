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
from lattices.directions import Direction, LatticeDirection, BlockSide
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



def _rearrange_legs(
    c1:TensorNode,
    c2:TensorNode,
    env:list[TensorNode],
)->tuple[
    TensorNode, TensorNode, list[TensorNode]
]:
    ## init Indices:
    env_left_legs_indices = []
    env_mid_legs_indices = []
    env_right_legs_indices = []
    c1_legs_indices = []
    c2_legs_indices = []

    ## core tensors start with legs towards each other:
    i1, i2 = get_common_edge_legs(c1, c2)
    c1_legs_indices.append(i1)
    c2_legs_indices.append(i2)

    ## Go counter-clockwise - relate core to environment:
    # half touching to c1:
    for m in env[0:3]:
        i1, i2 = get_common_edge_legs(c1, m)
        c1_legs_indices.append(i1)
        env_mid_legs_indices.append(i2)
    # half touching c2:
    for m in env[3:]:
        i1, i2 = get_common_edge_legs(c2, m)
        c2_legs_indices.append(i1)
        env_mid_legs_indices.append(i2)

    ## Go counter-clockwise - check leg order of environment:
    for prev, this, next in lists.iterate_with_periodic_prev_next_items(env):
        i1, _ = get_common_edge_legs(this, prev)
        env_left_legs_indices.append(i1)
        i1, _ = get_common_edge_legs(this, next)
        env_right_legs_indices.append(i1)

    ## Permute:
    c1.permute(c1_legs_indices)
    c2.permute(c2_legs_indices)
    for ind, i_left, i_mid, i_right in zip(range(len(env)), env_left_legs_indices, env_mid_legs_indices, env_right_legs_indices ,strict=True):
        env[ind].permute([i_left, i_mid, i_right])

    return c1, c2, env


def _find_node_in_relative_direction(dir:LatticeDirection, n1:TensorNode, n2:TensorNode)->TensorNode:
    vec = dir.unit_vector()
    if tuples.equal( n1.pos, tuples.add(vec, n2.pos) ):
        return n1
    elif tuples.equal( n2.pos, tuples.add(vec, n1.pos) ):
        return n2
    else:
        raise ValueError(f"None of nodes [{n1.name!r}, {n2.name!r}] are in correct relation with direction {dir.name!r}.")


def _rearrange_legs_into_canonical_order(
    tn:ArbitraryTN, side:None # Fix mode\side
)->EdgeTN:
    core_tensors = tn.get_nodes_by_functionality()
    core1 = _find_node_in_relative_direction(side.next_counterclockwise(), *core_tensors)
    core2 = _find_node_in_relative_direction(side.next_clockwise(), *core_tensors)
    environment_nodes = [
        tn.find_neighbor(core1, side),
        tn.find_neighbor(core1, side.next_counterclockwise()),
        tn.find_neighbor(core1, side.opposite()),
        tn.find_neighbor(core2, side.opposite()),
        tn.find_neighbor(core2, side.next_clockwise()),
        tn.find_neighbor(core2, side),
    ]

    ## Rearrange legs in required order:
    core1, core2, environment_nodes = _rearrange_legs(core1, core2, environment_nodes)
    environment_tensors = [physical_tensor_with_split_mid_leg(n) for n in environment_nodes]    # Open environment mps legs:
    return core1, core2, environment_tensors




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
    env_tn = _rearrange_legs_into_canonical_order(0)

    if DEBUG_MODE:
        env_tn.validate()

    return tn




if __name__ == "__main__":
    from scripts.contraction_test import main_test
    main_test()