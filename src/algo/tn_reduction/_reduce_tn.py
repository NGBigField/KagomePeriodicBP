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

# For type hinting:
from typing import TypeVar, Type, overload, Literal
TensorNetworkOutput = TypeVar("TensorNetworkOutput", CoreTN, ModeTN, EdgeTN)


def _reduce_tn_on_step(tn:TensorNetwork, trunc_dim:int, copy:bool, **kwargs)->TensorNetwork:
    if isinstance(tn, KagomeTN):
        return reduce_full_kagome_to_core(tn, trunc_dim=trunc_dim, **kwargs)
    if isinstance(tn, CoreTN):
        return reduce_core_to_mode(tn, copy=copy, **kwargs)
    if isinstance(tn, ModeTN):
        return reduce_mode_to_edge(tn, trunc_dim=trunc_dim, copy=copy, **kwargs)
    raise TypeError(f"Input TN of type {type(tn)!r} didn't match one of the possible reduction functions")
    
# @overload
# def reduce_tn(tn:TensorNetwork, target_type:type[EdgeTN], trunc_dim:int, copy:bool=True, **kwargs)->EdgeTN:...
# @overload
# def reduce_tn(tn:TensorNetwork, target_type:type[ModeTN], trunc_dim:int, copy:bool=True, **kwargs)->ModeTN:...
# @overload
# def reduce_tn(tn:TensorNetwork, target_type:type[CoreTN], trunc_dim:int, copy:bool=True, **kwargs)->CoreTN:...

def reduce_tn(tn:TensorNetwork, target_type:Type[TensorNetworkOutput], trunc_dim:int, copy:bool=True, **kwargs)->TensorNetworkOutput:
    """A general "fits-all" function that reduces the given TN into a chosen TN.

    # Args:
        tn (TensorNetwork): the given TN
        target_class (type): a smaller TN type that `tn` can be reduces into
        trunc_dim (int): truncation dimension (sometimes called `chi`)

    # Returns:
        TensorNetwork: the reduced `tn` into type `target_class`

    # Examples:
        >>> full_tn : KagomeTN
        >>> core_tn = reduce_tn(full_tn, CoreTN, 16)
    """
    while type(tn) is not target_type:
        tn = _reduce_tn_on_step(tn, trunc_dim=trunc_dim, copy=copy, **kwargs)
    return tn


