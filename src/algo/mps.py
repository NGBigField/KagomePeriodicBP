from libs.bmpslib import mps as MPS
from lattices.directions import Direction
import numpy as np
from utils import assertions
from containers import MessageDictType
from enums import MessageModel

# initial messages creation:
from tensor_networks.mps import init_mps_quantum



def initial_message(
	D : int,  # Physical-Lattice bond dimension
    N : int,  # number of tensors in the edge of the lattice
	message_model:MessageModel|str=MessageModel.RANDOM_QUANTUM
) -> MPS:

    dims : list[int] = [D]*N


    # Convert message model:
    if isinstance(message_model, MessageModel):
       pass
    elif isinstance(message_model, str):
       message_model = MessageModel(message_model)
    else:
       raise TypeError(f"Expected `initial_m_mode` to be of type <str> or <MessageModel> enum. Got {type(message_model)}")
    
    # Initial messages are random |v><v| quantum states
    if message_model==MessageModel.RANDOM_QUANTUM: return init_mps_quantum(dims, random=True)

    # Initial messages are uniform quantum density matrices
    elif message_model==MessageModel.UNIFORM_QUANTUM: return init_mps_quantum(dims, random=False)

    # Initial messages are uniform probability dist'
    elif message_model==MessageModel.UNIFORM_CLASSIC: raise NotADirectoryError("We do not support classical messages in this code")
        
    # Initial messages are random probability dist'
    elif message_model==MessageModel.RANDOM_CLASSIC: raise NotADirectoryError("We do not support classical messages in this code")     

    else:
        raise ValueError("Not a valid option")


def normalize_messages(mpss:MessageDictType)->MessageDictType:
    exponents = [ mps.nr_exp for mps in mpss.values() ]
    mean_exponents = int(sum(exponents)/len(exponents))
    for mps in mpss.values():
        mps.nr_exp += -mean_exponents
    return mpss


def physical_tensor_with_split_mid_leg(node:"TensorNode", normalize:bool=False)->np.ndarray:
    from tensor_networks import TensorNode

    assert not node.is_ket
    t = node.tensor
    old_shape = t.shape
    assert len(old_shape)==3
    half_mid_d = assertions.integer( np.sqrt(old_shape[1]) )
    physical_m = t.reshape([old_shape[0], half_mid_d, half_mid_d, old_shape[2]])
    if normalize:
        physical_m /= np.linalg.norm(physical_m)  #TODO check
    return physical_m