from collections.abc import Iterable
import numpy as np
from numpy import tensordot, zeros
from physics.pauli import x, y, z, id
from functools import cache

# For type hinting:
from typing import Callable, TypeAlias, Literal, TypeVar, Self
_T = TypeVar("_T")


HamiltonianFuncAndInputs : TypeAlias =	tuple[ Callable[[_T], np.ndarray], _T|tuple[_T]|None ] 



def _tensor_product(op1:np.ndarray, op2:np.ndarray)->np.ndarray:
	return tensordot(op1, op2, 0)


def ferromagnetic_with_transverse_field(direction:Literal['x', 'y', 'z'], strength:float=0.0)->np.ndarray:
	return -1*heisenberg_2d() + transverse_field_in_direction(direction, strength)


def transverse_field_in_direction(direction:Literal['x', 'y', 'z'], strength:float=0.0)->np.ndarray:
	match direction:
		case 'x': op = x
		case 'y': op = y
		case 'z': op = z
		case _:
			raise ValueError(f"Not an option {direction!r}")
	return strength*_tensor_product(op, id) + strength*_tensor_product(id, op)


def heisenberg_2d()->np.ndarray:
	return _tensor_product(x,x) + _tensor_product(y,y) + _tensor_product(z,z) 


def ising_with_transverse_field(B:float)->np.ndarray:
	z_coef = -1
	x_coef = -B/4
	
	hamiltonian = z_coef*_tensor_product(z, z) + x_coef*_tensor_product(x, id) + x_coef*_tensor_product(id, x)
	return hamiltonian


def zero() -> np.ndarray:
	h = identity()
	return h*0


def identity() -> np.ndarray:
	return _tensor_product(id, id)







## Old index dependant functions ##
# =============================== #


def get_su_d_heisenberg(d=2):
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

