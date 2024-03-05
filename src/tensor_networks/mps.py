# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #
if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)

# Numpy Stuff:
import numpy as np
from numpy import  sqrt, pi, conj
from numpy.linalg import norm, eigvals

# MPS Object:
from libs.bmpslib import mps as MPS
from libs.bmpslib import mps_inner_product

# Control:
from _config_reader import DEBUG_MODE

# some common utils in the code:
from utils import assertions

## Types in the code:
from tensor_networks.node import TensorNode, validated_int_square_root
from containers import MessageDictType
from enums import MessageModel

# ============================================================================ #
#                                 Constants                                    #
# ============================================================================ #

ComplexRandom : bool = True
EPSILON = 0.000001

# ============================================================================ #
#                               Inner Functions                                #
# ============================================================================ #

def _rand_mat(d:int)->np.ndarray:
    return np.random.normal(size=[d,d])

# ============================================================================ #
#                             Declared Functions                               #
# ============================================================================ #

def mps_distance(mps1:MPS, mps2:MPS) -> float:
	"""l2_distance(A, B) -> float:
	Where A is `self` and B is `other`:
	Compute 1 - |<A|B>|

	Args:
		self (MPS)
		other (MPS)
	"""
	# Compute:
	conjB = True
	ip = mps_inner_product(mps1, mps2, conjB)

	# old version:
	distance2 = 2 - 2*ip.real

	# New version:
	distance2 = 1 - abs(ip)

	# Validate:
	error_msg = f"L2 Distance should always be a real & positive value. Instead got {distance2}"
	assert np.imag(distance2)==0, error_msg
	if distance2<0:  # If it's a negative value.. it better be very close to zero
		assert abs(distance2)<EPSILON , error_msg
		return 0.0
	# return:
	return distance2


def init_mps_quantum(D_list, random=False) -> MPS:
	"""

	Creates an initial MPS message.

	The MPS is just a product state, and each state is the vectorization
	of the ID or a random |v><v|. This corresponds, in the double edge 
	model, to a trivial ket-bra contraction or a contraction with a 
	random |v>.

	Input Parameters:
	-------------------

	D_list --- A list "physical" dimensions of the MPS. Because we are
						 in the double-edge model, then each leg is the
						 fusion of two legs, and so its dimension must be of the
						 form D=d^2.

	           Since the MPS describes a product state, the shape of
	           the i'th tensor will be [1,D_i,1]  (right/left bond dimension
	           are 1)
	           
	random --- Whether or not the initial MPS is random (or a trivial
	           identity in double-edge formalism)

	Output:
	--------
	An MPS object which is left-canonical.


	"""

	N = len(D_list)
	mps = MPS(N)

	for i, D in enumerate(D_list):
		D2 = D*D
		D3 = D2*D
  
		if random:
			if ComplexRandom:
				A = _rand_mat(D3)+1j*_rand_mat(D3)
				A /= norm(A)
				ketbra = A @ conj(A.T)
				# ketbra = ketbra/norm(ketbra)
			else:
				A = _rand_mat(D3)
				ketbra = A @ A.T
				ketbra = ketbra/norm(ketbra)
   			
			if DEBUG_MODE:	
				_s = sum(eigvals(ketbra))
				# print(_s)
				assert abs(_s-1)<1e-5, f"Must be a physical state"
      
		else:
			ketbra = np.eye(D2)

		ketbra = ketbra.reshape([D,D,D,D,D,D])
		ketbra = ketbra.transpose([0,3, 1,4, 2,5])
		ketbra = ketbra.reshape([D2, D2, D2])

		if i==0:
			ketbra = ketbra[0, :, :].reshape([1, D2, D2])

		if i==N-1:
			ketbra = ketbra[:, :, 0].reshape([D2, D2, 1])

		mps.set_site(ketbra, i)
  
	#	
	# Make it left-canonical and normalize
	#

	mps.left_canonical_QR()

	mps.set_site(mps.A[N-1]/norm(mps.A[N-1]), N-1)

	return mps



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


def physical_tensor_with_split_mid_leg(node:TensorNode)->np.ndarray:
    assert not node.is_ket
    t = node.tensor
    old_shape = t.shape
    assert len(old_shape)==3
    half_mid_d = validated_int_square_root(old_shape[1])
    physical_m = t.reshape([old_shape[0], half_mid_d, half_mid_d, old_shape[2]])
    return physical_m


def mps_index_of_open_leg(mps:MPS, node_index_in_order:int)->int:
	n = mps.N
	if node_index_in_order==0:
		return 0
	elif node_index_in_order==n-1:
		return 1
	else:
		return 1
