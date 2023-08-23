
# Control flags:
from _config_reader import DEBUG_MODE

# For a little bit of OOP:
from typing import TypeVar

# Types we need in our module:
from tensor_networks import KagomeTN, ArbitraryTN, ModeTN, TensorNode, MPS, CoreTN, get_common_edge
from tensor_networks import TensorNode
from tensor_networks.node import TensorNode
from enums import ContractionDepth, NodeFunctionality, UpdateMode



def reduce_core_to_mode(
    core_tn:CoreTN, 
    mode:UpdateMode
)->ModeTN:
    
    ## Create a copy which is an arbitrary tn which can be contracted:
    tn = core_tn.to_arbitrary_tn()

    ## Get basic info:
    mode_side = mode.side_in_core  # Decide which side corresponds to the mode:

    ## Also keep a list of nodes that should be contracted:
    new_nodes : list[TensorNode] = []

    ## Contract:
    # For each side not being the major core side
    for side in CoreTN.all_mode_sides:
        if side is mode_side:
            continue
            
        # For each boundary node
        boundary_nodes = [node for node in tn.get_nodes_on_boundary(side)]
        for boundary_node in boundary_nodes:

            # For each neighbor which is on thr environment:
            neighbors = tn.all_neighbors(boundary_node)
            for neighbor in neighbors:
                if neighbor.functionality is NodeFunctionality.Environment:
                    boundary_node = tn.contract(neighbor, boundary_node)  # output is the new boundary tensor
            
            # keep in list:
            new_nodes.append(boundary_node)

    ## Let those new tensors know they are part of the environment:
    for node in new_nodes:
        node.functionality = NodeFunctionality.Environment

    mode_tn = ModeTN.from_arbitrary_tn(tn, mode=mode)
    if DEBUG_MODE:
        mode_tn.validate()

    return mode_tn


