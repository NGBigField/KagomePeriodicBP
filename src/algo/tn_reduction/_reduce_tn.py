"""
This module concludes all the different methods for reduction, in a way that simplifies the code.
Given a (Tensor-Network) TN, and a target structure for reduction, the code decides which function should be executed in order to reach that goal.
"""


# Reduction codes:
from algo.tn_reduction.kagome_to_core import reduce_full_kagome_to_core
from algo.tn_reduction.core_to_mode import reduce_core_to_mode
from algo.tn_reduction.mode_to_edge import reduce_mode_to_edge

# Types in the code:
from tensor_networks import TensorNetwork, KagomeTN, CoreTN, ModeTN, EdgeTN


def _reduce_tn_on_step(tn:TensorNetwork, trunc_dim:int, **kwargs)->TensorNetwork:
    if isinstance(tn, KagomeTN):
        return reduce_full_kagome_to_core(tn, trunc_dim=trunc_dim, **kwargs)
    if isinstance(tn, CoreTN):
        return reduce_core_to_mode(tn, **kwargs)
    if isinstance(tn, ModeTN):
        return reduce_mode_to_edge(tn, trunc_dim=trunc_dim, **kwargs)
    raise TypeError(f"Input TN of type {type(tn)!r} didn't match one of the possible reduction functions")
    

def reduce_tn(tn:TensorNetwork, target_class:type, trunc_dim:int, **kwargs)->TensorNetwork:
    while type(tn) is not target_class:
        tn = _reduce_tn_on_step(tn, trunc_dim=trunc_dim, **kwargs)
    return tn


