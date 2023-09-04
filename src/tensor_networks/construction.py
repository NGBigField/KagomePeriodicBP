# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

# Global config:
from _config_reader import DEBUG_MODE


# For common numeric functions:
import numpy as np
from numpy import matlib

# For common lattice function and classes:
from tensor_networks.tensor_network import KagomeTN, TensorDims
from tensor_networks.unit_cell import UnitCell

from lattices.kagome import KagomeLattice




# ============================================================================ #
#|                           Declared Function                                |#
# ============================================================================ #
     
def create_kagome_tn(
    d : int,  # Physical dimenstion 
    D : int,  # Virutal\Bond dimenstion  
    N : int,  # Lattice-size - Number of upper-triangles at each edge of the hexagon-block
    unit_cell : UnitCell|None = None
) -> KagomeTN:

    # Get the kagome lattice without tensors:
    lattice = KagomeLattice(N)

    ## Unit cell:
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)
    else:
        assert isinstance(unit_cell, UnitCell)

    tn = KagomeTN(lattice, unit_cell, d=d, D=D)
    
    return tn



