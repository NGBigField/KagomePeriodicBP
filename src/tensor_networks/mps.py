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
from lib.bmpslib import mps as MPS

# ============================================================================ #
#                                   Types                                      #
# ============================================================================ #

# ============================================================================ #
#                                 Constants                                    #
# ============================================================================ #

ComplexRandom : bool = True

# ============================================================================ #
#                               Inner Functions                                #
# ============================================================================ #

def _rand_mat(d:int)->np.ndarray:
    return np.random.normal(size=[d,d])

# ============================================================================ #
#                             Declared Functions                               #
# ============================================================================ #

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
	mp = MPS(N)

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

		mp.set_site(ketbra, i)
  
	#	
	# Make it left-canonical and normalize
	#

	mp.left_canonical_QR()

	mp.set_site(mp.A[N-1]/norm(mp.A[N-1]), N-1)

	return mp




if __name__ == "__main__":
    from scripts.core_ite_test import main
    main()

