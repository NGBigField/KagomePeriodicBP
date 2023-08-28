import numpy as np
from numpy import tensordot, zeros
from physics.pauli import x, y, z, id
from typing import Callable, TypeAlias
from functools import cache


HamiltonianFuncType : TypeAlias = Callable[[int, int, int, int, int], np.ndarray]


def _tensor_product(op1:np.ndarray, op2:np.ndarray)->np.ndarray:
	return tensordot(op1, op2, 0)


def heisenberg_2d()->np.ndarray:
	return _tensor_product(x,x) + _tensor_product(y,y) + _tensor_product(z,z) 


def ising_with_transverse_field(B:float)->np.ndarray:
	z_coef = -1
	x_coef = -B/4
	
	hamiltonian = z_coef*tensordot(z, z, 0) + x_coef*tensordot(x, id, 0) + x_coef*tensordot(id, x, 0)
	return hamiltonian


def zero() -> np.ndarray:
	h = identity()
	return h*0


def identity() -> np.ndarray:
	return _tensor_product(id, id)







## Old index dependant functions ##
# =============================== #


def get_su_d_heisenberg(d=2) -> HamiltonianFuncType:
	return lambda n, i1, j1, i2, j2: SU_Heisenberg(n, i1, j1, i2, j2, d=d)


def SU_Heisenberg(n, i1, j1, i2, j2, d=2) -> np.ndarray:
	"""
	
	The Hamiltonian that we simulate.
	
	Input Parameters:
	----------------
	edge --- the edge of the Hamiltonian term we output.
	
	
	OUTPUT:
	-------
	
	A rank-4 tensor defining the Hamiltonian term on the given edge.
	
	If the edge is (i,j) particles, then its Hamiltonian tensor is given
	by:
	
	   <i-ket|<j-ket| h |i-bra>|j-bra>  =  h[i-ket, i-bra, j-ket, j-bra]
	
	"""

	H = None
	for flavorA in range(d):
		for flavorB in range(d):
			T_AB = zeros([d, d])
			T_AB[flavorA, flavorB] = 1.0

			if H is None:
				H = tensordot(T_AB, T_AB.T, 0)
			else:
				H += tensordot(T_AB, T_AB.T, 0)


	return H

