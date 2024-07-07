"""
This module concludes all the different methods for reduction, in a way that simplifies the code.
Given a (Tensor-Network) TN, and a target structure for reduction, the code decides which function should be executed in order to reach that goal.
"""

# Reduction codes:
from algo.tn_reduction.kagome_to_core import reduce_full_kagome_to_core
from algo.tn_reduction.core_to_mode import reduce_core_to_mode
from algo.tn_reduction.mode_to_edge import reduce_mode_to_edge

# Types in the code:
from tensor_networks import TensorNetwork, KagomeTNRepeatedUnitCell, CoreTN, ModeTN, EdgeTN

# For type hinting:
from typing import TypeVar, Type, Callable
TensorNetworkOutput = TypeVar("TensorNetworkOutput", CoreTN, ModeTN, EdgeTN)

# For function required kwargs:
from copy import deepcopy
from inspect import getfullargspec


def _remove_unneeded_kwargs(func:callable, kwargs:dict[str, object])->TensorNetwork:
    # Find what is needed by func:
    spec = getfullargspec(func)
    needed_keys = set(spec.args)
    # Find what is not needed from kwargs:
    redundant_keys = []
    for key, _ in kwargs.items():
        if key not in needed_keys:
            redundant_keys.append(key)
    # Remove unneeded:
    for key in redundant_keys:
        kwargs.pop(key)
    return kwargs


def _next_reduction_function(tn:TensorNetwork)->Callable[[TensorNetwork, dict], TensorNetwork]:
    if isinstance(tn, KagomeTNRepeatedUnitCell):
        return reduce_full_kagome_to_core
    if isinstance(tn, CoreTN):
        return reduce_core_to_mode
    if isinstance(tn, ModeTN):
        return reduce_mode_to_edge
    raise TypeError(f"Input TN of type {type(tn)!r} didn't match one of the possible reduction functions")


def _reduce_tn_one_step(tn:TensorNetwork, trunc_dim:int, copy:bool, **kwargs_in)->TensorNetwork:

    ## Choose correct function:
    func = _next_reduction_function(tn)

    ## Keep only needed kwargs for the specific function:
    kwargs_to_use = deepcopy(kwargs_in)
    kwargs_to_use["trunc_dim"]=trunc_dim
    kwargs_to_use["copy"]=copy
    kwargs_to_use = _remove_unneeded_kwargs(func, kwargs_to_use)

    res = func(tn, **kwargs_to_use)

    return res


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
        tn = _reduce_tn_one_step(tn, trunc_dim=trunc_dim, copy=copy, **kwargs)
    return tn


