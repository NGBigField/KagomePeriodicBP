# Common types in the code:
from tensor_networks import KagomeTN, MPS

# Everyone needs numpy:
import numpy as np

# Other modules we made and need here   
from libs import bmpslib

# Types we need in our module:
from tensor_networks import KagomeTN, CoreTN, ModeTN, MPS, TensorNode
from tensor_networks.node import NodeFunctionality, UnitCellFlavor
from lattices.directions import LatticeDirection, BlockSide, check
from lattices.edges import edges_dict_from_edges_list
from enums import ContractionDepth
from _types import EdgeIndicatorType
from containers import BubbleConConfig, MPSOrientation, MessageDictType

# Our utilities:
from utils import tuples, numerics

# Our needed algos:
from algo.contraction_order import get_contraction_order

# Other modules we made and need here   
from libs.bubblecon import bubblecon


def _cell_flavor_from_index(i:int) -> UnitCellFlavor:
    m = i%3
    match m:
        case 0: return UnitCellFlavor.A
        case 1: return UnitCellFlavor.B
        case 2: return UnitCellFlavor.C
        case _:
            raise ValueError("Not a matched case")
    

def _plot_tn_with_connected_corner(
    tensors     : list[np.ndarray],  
    edges_list  : list[list[str]],      
    angles_list : list[list[float]],  
    positions   : list[tuple[float, float]],
    tn          : KagomeTN 
)->None:

    from tensor_networks.visualizations import plot_network
    from lattices.directions import LatticeDirection, Direction
    from lattices.triangle import total_vertices

    edges_dict = edges_dict_from_edges_list(edges_list)
    nodes = []


    # Decide num_core_nodes:
    for n in range(2, 20):
        num_core_nodes = total_vertices(n)*3
        if num_core_nodes>len(tensors):
            num_core_nodes = total_vertices(n-1)*3
            break

            

    for i, (tensor, edges, angles, position) in enumerate(zip(tensors, edges_list, angles_list, positions, strict=True)):

        if i<num_core_nodes:
            is_ket = True
            directions = [LatticeDirection.from_angle(a) for a in angles]
            functionality=NodeFunctionality.Undefined
            cell_flavor = _cell_flavor_from_index(i)
        else:
            is_ket = False
            directions = [Direction(name="rand", angle=a) for a in angles]
            functionality=NodeFunctionality.Message
            cell_flavor = UnitCellFlavor.NoneLattice

        node = TensorNode(
            index=i,
            name=f"f{i}",
            tensor=tensor,
            is_ket=is_ket,
            pos=position,
            edges=edges,
            directions=directions,
            functionality=functionality,
            cell_flavor=cell_flavor
        )

        nodes.append(node)

    plot_network(nodes, edges_dict)


def connect_corner_messages(
    tn:KagomeTN, outgoing_dir:BlockSide
)->tuple[
    list[np.ndarray], list[list[EdgeIndicatorType]], list[list[float]]
]:
    
    ## Get the output lists:
    tensors = tn.tensors
    edges_list = tn.edges_list
    angles = tn.angles

    ## Derive basic info:
    first_msg_dir = outgoing_dir.opposite()
    msg_indices1 = tn.message_indices(first_msg_dir)
    msg_indices2 = tn.message_indices(first_msg_dir.next_counterclockwise())
    index = [msg_indices1[-1], msg_indices2[0]]

    ## Expand both tensors with a fake leg of dim 1:
    new_tensor, new_edges, pos, old_angles = [], [], [], []
    for i in [0, 1]:
        grand_index = index[i]
        pos.append( tn.positions[grand_index] )
        tensor = tensors[grand_index]
        new_shape = tuples.add_element(tensor.shape, 1) 
        new_edges.append( edges_list[grand_index] + ['fake_leg']  )        
        new_tensor.append( tensor.copy().reshape(new_shape) )
        old_angles.append( angles[grand_index] )

        
    ## Derive new angles:
    new_a = [0.0, 0.0]
    new_a[0] = tuples.angle(pos[0], pos[1])
    new_a[1] = numerics.force_between_0_and_2pi(new_a[0]+np.pi)
    new_angle = [old_angles[i]+[new_a[i]] for i in [0, 1]]

    ## Assign results to list
    for i in [0, 1]:
        grand_index = index[i]
        tensors[grand_index] = new_tensor[i]
        edges_list[grand_index] = new_edges[i]
        angles[grand_index] = new_angle[i]

    ## Return:
    return tensors, edges_list, angles
    
    

def contract_tensor_network(
    tn:KagomeTN|CoreTN|ModeTN, 
    direction:BlockSide,
    depth:ContractionDepth,
    bubblecon_trunc_dim:int,
    print_progress:bool=True
)->tuple[
    MPS|complex|tuple,
    list[int],
    MPSOrientation,
]:
    """Automatically derive/get the contraction order from a known network structure, and compute the contraction using bubblecon

    Args:
        tn (KagomeTN | CoreTN | ModeTN)
        direction (BlockSide): direction of contraction
        depth (ContractionDepth): how deep to contract
        bubblecon_trunc_dim (int): 
        print_progress (bool, optional): Defaults to True.

    Returns:
        tuple[ MPS|complex|tuple, list[int], MPSOrientation, ]
    """

    ## Derive or load Contraction Order:
    contraction_order = get_contraction_order(tn, direction, depth)

    ## Connect first MPS message to a side tensor, to allow efficient contraction:
    if isinstance(tn, KagomeTN):
        tensors, edges_list, angles = connect_corner_messages(tn, direction)
        # _plot_tn_with_connected_corner(tensors, edges_list, angles, tn.positions, tn)
    elif isinstance(tn, CoreTN|ModeTN):
        tensors, edges_list, angles = tn.tensors, tn.edges_list, tn.angles
    else:
        raise TypeError(f"Not an expected type {type(tn)} of input 'tn'")
    
    # Choose bubblecon compression rule:
    if tn.dimensions.virtual_dim <= 3:
        compression_dict = {'type':'SVD'}
    else:
        compression_dict = {'type':'iter', 'max-iter':10, 'err':1e-6}

    ## Call main function:
    mps = bubblecon(
        tensors, 
        edges_list, 
        angles, 
        bubble_angle=direction.angle,
        swallow_order=contraction_order, 
        D_trunc=bubblecon_trunc_dim,
        opt='high',
        progress_bar=BubbleConConfig.progress_bar and print_progress,
        separate_exp=BubbleConConfig.separate_exp,
        ket_tensors=tn.kets,
        compression=compression_dict
    )

    ## Derive outgoing mps direction
    orientation = MPSOrientation.standard(direction)

    ## Check outputs:
    assert not isinstance(mps, list)  # This is not an expected output

    return mps, contraction_order, orientation
