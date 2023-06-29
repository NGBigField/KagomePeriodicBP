# ============================================================================ #
#                                  Imports                                     #
# ============================================================================ #

# for allowing forward referencing of type "Self" in alternative constructors:
from __future__ import annotations

# For matrices and tensors:
import numpy as np

# For type hinting:
from typing import (
    Tuple,
    Optional,
    Type,
    Any,
)

# import from tensors:
from tensor_networks.tensors import (
    Tensor,
    Leg,
    _TensorDateType,
)

# ============================================================================ #
#                               Inner Functions                                #
# ============================================================================ #
def _derive_dims(physical_dim:int, bond_dim:int, num_bonds:int) -> Tuple[int, ...] :
        return [physical_dim] + [bond_dim]*num_bonds

# ============================================================================ #
#                              Declared Functions                              #
# ============================================================================ #

def lattice_tensor(data:_TensorDateType, name:Any='', pos:Optional[Tuple[int, ...]]=None) -> Tensor:
    t = Tensor(data, name, pos)
    # First leg is the physical leg while the other are virtual:
    for i, leg in enumerate(t.legs):
        if i == 0:
            leg.tag = Leg.Tag.Physical
        else:
            leg.tag = Leg.Tag.Virtual    
    return t


def empty(dims:Tuple[int, ...], name='', pos:Optional[Tuple[int, ...]]=None) -> Tensor:  
    data = np.zeros(shape=dims, dtype='complex_')
    return lattice_tensor(data, name, pos)

def empty_given_bonds(bond_dim:int, num_bonds:int, physical_dim:int=2) -> Tensor:
        dims = _derive_dims(physical_dim, bond_dim, num_bonds)
        return cls.empty(dims)

def random(dims:Tuple[int, ...], name='', pos:Optional[Tuple[int, ...]]=None) -> Tensor:  
    data = np.random.normal(size=dims) + 1j*np.random.normal(size=dims)        
    return lattice_tensor(data, name, pos)

def random_given_bonds(bond_dim:int, num_bonds:int, physical_dim:int=2) -> Tensor:
        dims = _derive_dims(physical_dim, bond_dim, num_bonds)
        return cls.random(dims)
