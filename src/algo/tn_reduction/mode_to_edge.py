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
from tensor_networks import ArbitraryTN, ModeTN, EdgeTN, TensorNode, get_common_edge, get_common_edge_legs
from tensor_networks import TensorNode
from tensor_networks.node import TensorNode, two_nodes_ordered_by_relative_direction
from tensor_networks.mps import mps_index_of_open_leg, MPS
from enums import ContractionDepth, NodeFunctionality
from containers import UpdateEdge, MPSOrientation
from containers.configs import ContractionConfig
from _types import EdgeIndicatorType
from _error_types import TensorNetworkError

# Import utilities that are shared between modules:
from utils import lists, tuples

# Algos we need here:
from algo.contract_tensor_network import contract_tensor_network

# For numerics and tensor stuff:
import numpy as np




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
)->tuple[EdgeIndicatorType, int, int, bool]:
    """ Find the correct edge nodes """

    is_in_core = edge_tuple.is_in_core()
    is_center_included = mode_tn.mode in edge_tuple

    if is_in_core:
        options = mode_tn.get_nodes_by_functionality(NodeFunctionality.CenterCore)
    else:
        options = mode_tn.get_nodes_by_functionality(NodeFunctionality.AroundCore)
    node1 = next((node for node in options if node.cell_flavor in edge_tuple))

    ## 
    valid_neighbors = [node for node in mode_tn.get_core_nodes() if node is not node1 and mode_tn.are_neighbors(node1, node)]
    if is_in_core:
        options = [node for node in valid_neighbors if node.functionality==NodeFunctionality.CenterCore]
    else:
        options = [node for node in valid_neighbors if node.functionality==NodeFunctionality.AroundCore]
    if is_center_included:
        options.append(mode_tn.center_node)
    node2 = next((node for node in options if node.cell_flavor in edge_tuple))

    edge = get_common_edge(node1, node2)
    return edge, node1.index, node2.index, is_center_included     


def _contract_all_nodes_individually_except_neighbors(
    tn:ArbitraryTN,
    i1:int, i2:int
)->tuple[ArbitraryTN, TensorNode]:

    ## Derive the main two nodes:
    node1:TensorNode = tn.nodes[i1]
    node2:TensorNode = tn.nodes[i2]

    ## Find immediate neighbors:
    common_neighbor : TensorNode = None
    other_neighbors = []
    for node in tn.nodes:
        if node is node1 or node is node2:
            continue
        is_neighbor1 = tn.are_neighbors(node, node1) 
        is_neighbor2 = tn.are_neighbors(node, node2)
        if is_neighbor1 and is_neighbor2:
            common_neighbor = node
        elif is_neighbor1 or is_neighbor2:
            other_neighbors.append(node)
    assert common_neighbor is not None, "Bug. We must find a common neighbor"

    ## Contract all tensors than we don't need:
    # this order forces the common neighbor to draw most of the neighbor to be swallowed by it:
    nodes_to_keep_old = [common_neighbor, node1, node2] + other_neighbors   
    # Swallow everything except the neighbors of the 2 core nodes:
    nodes_to_keep_new = tn.contract_all_nodes_into_exceptions(nodes_to_keep_old)

    ## Update common_neighbor:
    common_neighbor = nodes_to_keep_new[0]  # The new list keeps the same order

    ## mark all env nodes:
    for node in tn.nodes:
        if node in [node1, node2]:
            continue
        node.functionality = NodeFunctionality.Environment

    return tn, common_neighbor


def _create_mps_node(
    tn:ArbitraryTN, mps:MPS, mps_orientation:MPSOrientation, mps_tensor:np.ndarray, i:int, opposite_node:TensorNode
)->tuple[TensorNode, int]:
    # Basic info:
    shape = mps_tensor.shape

    # Find the correct legs to connect:
    leg_index, opposite_direction, edge, dim = tn.find_open_leg_of_node(opposite_node)

    # Deriveedges, matching directions and position:
    all_edges_options = [f"env-edge-{i-1}", edge, f"env-edge-{i}"]
    all_directions_options = [mps_orientation.ordered.opposite(), mps_orientation.open_towards, mps_orientation.ordered]
    pos = tuples.add(opposite_node.pos, mps_orientation.open_towards.opposite().unit_vector)
    if i==0:
        edges = all_edges_options[-2:]
        directions = all_directions_options[-2:]
        reshaped_tensor = mps_tensor.reshape(shape[-2:])

    elif i==3:
        edges = all_edges_options[:2]
        directions = all_directions_options[:2]
        reshaped_tensor = mps_tensor.reshape(shape[:2])
    
    else:
        edges = all_edges_options
        directions = all_directions_options
        reshaped_tensor = mps_tensor

    # Create new node from mps:
    mps_node = TensorNode(
        index=tn.size,
        name=f"e-{i}",
        tensor=reshaped_tensor,
        is_ket=False,
        pos=pos,
        edges=edges,
        directions=directions,
        functionality=NodeFunctionality.Environment        
    )

    mps_leg_ind = mps_index_of_open_leg(mps, i)
    assert reshaped_tensor.shape[mps_leg_ind]==dim

    return mps_node, leg_index



def _contract_half_using_bubblecon(
    mode_tn:ModeTN,
    i1:int, i2:int,
    trunc_dim:int,
    copy:bool
)->tuple[ArbitraryTN, list[TensorNode]]:
    
    ## Basic info:
    edge_nodes = [mode_tn.nodes[i] for i in [i1, i2]]
    center_node = mode_tn.center_node
    if DEBUG_MODE:
        assert len(edge_nodes)==2
        assert center_node not in edge_nodes
    
    ## Get the side where the edge is located:
    for side in mode_tn.major_sides:
        side_nodes = mode_tn.get_nodes_on_side(side)
        if all((node in side_nodes for node in edge_nodes)):
            edge_side = side
            break
    else:  # Didn't fint
        raise TensorNetworkError("Couldn't find the correct side to contract mode-tn into edge-tn")

    ## Contract to get an MPS
    mps, contraction_order, orientation = contract_tensor_network(mode_tn, direction=edge_side, depth=ContractionDepth.ToEdge, bubblecon_trunc_dim=trunc_dim, print_progress=False)
    if DEBUG_MODE:
        assert i1 not in contraction_order
        assert i2 not in contraction_order

    ## Create a version of the TN where the contracted nodes are removed:
    tn = mode_tn.to_arbitrary_tn(copy=copy)
    nodes_to_remove = [tn.nodes[i] for i in contraction_order]
    for node_to_remove in nodes_to_remove:
        tn.pop_node(node_to_remove.index)
    edge_nodes = [tn.get_node_in_pos(old_node.pos) for old_node in edge_nodes]

    ## Derive the tensors and legs we need to connect to:
    mps_order_direction = orientation.ordered
    n1, n2 = two_nodes_ordered_by_relative_direction(*edge_nodes, mps_order_direction)
    n3 = tn.find_neighbor(n2, mps_order_direction)
    n0 = tn.find_neighbor(n1, mps_order_direction.opposite())

    ## Glue the MPS to the remaining tensors:
    mps_nodes = []
    for i, (mps_tensor, opposite_node) in enumerate(zip(mps.A, [n0, n1, n2, n3], strict=True)):
        # Create mps node:
        mps_node, leg_index = _create_mps_node(tn, mps, orientation, mps_tensor, i, opposite_node)
        # align direction of opposite node:
        opposite_node.directions[leg_index] = orientation.open_towards.opposite()
        # add to TN:
        tn.add_node(mps_node)
        # keep mps nodes for later:
        mps_nodes.append(mps_node)

    ## Return:
    return tn, mps_nodes


def _derive_commons_neighbors_connections_if_it_was_a_matrix(
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


def reduce_mode_to_edge(
    mode_tn:ModeTN, 
    edge_tuple:UpdateEdge,
    contract_config:ContractionConfig,
    copy:bool=True
)->EdgeTN:
    
    ## Get basic data:
    trunc_dim = contract_config.trunc_dim
    edge, i1, i2, is_center_included = _derive_edge_and_nodes(mode_tn, edge_tuple)

    ## Contract everything except the mode and its neighbors:
    if is_center_included:
        tn = mode_tn.to_arbitrary_tn(copy=copy)
        #TODO: Add truncation dimension using SVD 
        tn, common_neighbor = _contract_all_nodes_individually_except_neighbors(tn, i1, i2)

        # Decide which of the common_neighbor's legs fit at which side:
        edge1, edge2 = _derive_commons_neighbors_connections_if_it_was_a_matrix(tn, common_neighbor)

        # Split common neighbor using QE-decomposition:
        _, _ = tn.qr_decomp(common_neighbor, edge1, edge2)

    else:
        # Replace all nodes up to the edge, with an approximating MPS
        tn, mps_nodes = _contract_half_using_bubblecon(mode_tn, i1, i2, trunc_dim=trunc_dim, copy=copy)            

        # Swallow first and last mps tensors into mps:
        for i_to_contract, i_contract_into in [(0, 1), (3, 2)]:
            node_to_contract = mps_nodes[i_to_contract]
            node_contract_into = mps_nodes[i_contract_into]
            tn.contract(node_to_contract, node_contract_into)

    ## Convert type:
    edge_tn = EdgeTN.from_arbitrary_tn(tn)
    if DEBUG_MODE:
        edge_tn.validate()
        assert tuples.equal(edge_tn.unit_cell_flavors, edge_tuple, allow_permutation=True)

    return edge_tn




if __name__ == "__main__":
    from scripts.test_contraction import main_test
    main_test()