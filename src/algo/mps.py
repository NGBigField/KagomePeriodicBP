from tensor_networks import MPS
from enums import Directions
from tensor_networks import Node
import numpy as np
from utils import assertions

_MsgDictType = dict[Directions, MPS]


def normalize_messages(mpss:_MsgDictType)->_MsgDictType:
    exponents = [ mps.nr_exp for mps in mpss.values() ]
    mean_exponents = int(sum(exponents)/len(exponents))
    for mps in mpss.values():
        mps.nr_exp += -mean_exponents
    return mpss


def physical_tensor_with_split_mid_leg(node:Node, normalize:bool=False)->np.ndarray:
    assert not node.is_ket
    t = node.tensor
    old_shape = t.shape
    assert len(old_shape)==3
    half_mid_d = assertions.integer( np.sqrt(old_shape[1]) )
    physical_m = t.reshape([old_shape[0], half_mid_d, half_mid_d, old_shape[2]])
    if normalize:
        physical_m /= np.linalg.norm(physical_m)  #TODO check
    return physical_m