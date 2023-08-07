import numpy as np

def fuse_tensor_to_itself(tensor:np.ndarray):
	"""
		Given a PEPS tensor T of the form [d, D1, D2, ...],
		contract it with its conjugate T^* along the physical leg (bond d),
		and fuse all the matching ket-bra pairs of the virtual legs to a
		single double-layer leg.

		The resultant tensor is of the form: [D1**2, D2**2, D3**2, ...]
	"""
	n = len(tensor.shape)
	T2 = np.tensordot(tensor, np.conj(tensor), axes=([0],[0]))
 
	# Permute the legs:    [D1, D2, ..., D1^*, D2^*, ...] ==> [D1, D1^*, D2, D2^*, ...]
	perm = []
	for i in range(n-1):
		perm = perm + [i, i+n-1]

	T2 = T2.transpose(perm)

	# Fuse the ket-bra pairs: [D1, D1^*, D2, D2^*, ...] ==> [D1^2, D2^2, ...]
	dims = [tensor.shape[i]**2 for i in range(1, n)]
	T2 = T2.reshape(dims)
	return T2