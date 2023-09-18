from collections.abc import Iterable
import numpy as np
from numpy import tensordot, zeros
from physics.pauli import x, y, z, id
from functools import cache

# For type hinting:
from typing import Literal



def _tensor_product(op1:np.ndarray, op2:np.ndarray)->np.ndarray:
	return tensordot(op1, op2, 0)


def ferromagnetic_with_transverse_field(direction:Literal['x', 'y', 'z'], strength:float=0.0)->np.ndarray:
	return heisenberg_fm() - transverse_field_in_direction(direction, strength)


def transverse_field_in_direction(direction:Literal['x', 'y', 'z'], strength:float=0.0)->np.ndarray:
	match direction:
		case 'x'|'X': op = x
		case 'y'|'Y': op = y
		case 'z'|'Z': op = z
		case _:
			raise ValueError(f"Not an option {direction!r}")
	return strength*_tensor_product(op, id) + strength*_tensor_product(id, op)


def heisenberg_fm()->np.ndarray:
	return -1*heisenberg_afm()
heisenberg_fm.reference = -0.5


def heisenberg_afm()->np.ndarray:
	return _tensor_product(x,x) + _tensor_product(y,y) + _tensor_product(z,z) 
# heisenberg_afm.reference = -0.438703897456
heisenberg_afm.reference = -0.38620


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
