import numpy as np
from numpy import tensordot, zeros
from physics.pauli import x, y, z, id
from physics.spin_operators import Sx, Sy, Sz

# For type hinting:
from typing import Literal

GLOBAL_FIELD_STRENGTH = 1.0

def _tensor_product(op1:np.ndarray, op2:np.ndarray)->np.ndarray:
	return tensordot(op1, op2, 0)


def ferromagnetic_with_transverse_field(direction:Literal['x', 'y', 'z'], strength:float=0.0)->np.ndarray:
	return heisenberg_fm() - field_in_direction(direction, strength)


def field_in_direction(direction:Literal['x', 'y', 'z'], strength:float=0.0)->np.ndarray:
	match direction:
		case 'x'|'X': op = Sx
		case 'y'|'Y': op = Sy
		case 'z'|'Z': op = Sz
		case _:
			raise ValueError(f"Not an option {direction!r}")
	return strength*_tensor_product(op, id) + strength*_tensor_product(id, op)



def heisenberg_fm()->np.ndarray:
	"""Heisenberg FerroMagnetic model
	"""
	return -1*heisenberg_afm()
heisenberg_fm.reference = -0.5


def heisenberg_fm_with_field(f:float=0.0)->np.ndarray:
	"""Heisenberg FerroMagnetic model with Global field in the x direction
	"""
	return heisenberg_fm() + field_in_direction(direction="x", strength=f)
heisenberg_fm_with_field.reference = heisenberg_fm.reference 


def field(direction='x')->np.ndarray:
	"""Global field in the x\y\z direction
	"""
	return field_in_direction(direction=direction, strength=GLOBAL_FIELD_STRENGTH)
field.reference = - GLOBAL_FIELD_STRENGTH


FULLY_SYMMETRIC : bool = True

def heisenberg_afm()->np.ndarray:
	"""Heisenberg Anti-FerroMagnetic model
	"""
	if FULLY_SYMMETRIC:
		return 1.0*_tensor_product(Sx,Sx) + 1.0*_tensor_product(Sy,Sy) + 1.0*_tensor_product(Sz,Sz) 
	else:
		return 1.0001*_tensor_product(Sx,Sx) + 0.9998*_tensor_product(Sy,Sy) + 0.9999*_tensor_product(Sz,Sz) 
# heisenberg_afm.reference = -0.438703897456
heisenberg_afm.reference = -0.40454  # -0.38620


def heisenberg_afm_with_field(f:float=0.0)->np.ndarray:
	"""Heisenberg Anti-FerroMagnetic model with adjustable strength of global field
	"""
	return heisenberg_afm() + field_in_direction(direction="x", strength=f)
heisenberg_afm_with_field.reference = heisenberg_afm.reference


def ising_with_transverse_field(B:float)->np.ndarray:
	z_coef = -1
	x_coef = -B/4
	
	hamiltonian = z_coef*_tensor_product(Sz, Sz) + x_coef*_tensor_product(Sx, id) + x_coef*_tensor_product(id, Sx)
	return hamiltonian


def zero() -> np.ndarray:
	h = identity()
	return h*0


def identity() -> np.ndarray:
	return _tensor_product(id, id)
