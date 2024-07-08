# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

# Global config:
from _config_reader import DEBUG_MODE

# For common lattice function and classes:
from tensor_networks.tensor_network import KagomeTNRepeatedUnitCell, TensorDims
from unit_cell import UnitCell

from lattices.kagome import KagomeLattice

# Common types in code:
from containers.sizes_and_dimensions import TNDimensions




# ============================================================================ #
#|                           Declared Function                                |#
# ============================================================================ #
     
def create_kagome_tn(
    d : int,  # Physical dimenstion 
    D : int,  # Virutal\Bond dimenstion  
    N : int,  # Lattice-size - Number of upper-triangles at each edge of the hexagon-block
    unit_cell : UnitCell|None = None
) -> KagomeTNRepeatedUnitCell:

    # Get the kagome lattice without tensors:
    lattice = KagomeLattice(N)

    ## Unit cell:
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)
    else:
        assert isinstance(unit_cell, UnitCell)

    tn = KagomeTNRepeatedUnitCell(lattice, unit_cell, d=d, D=D)
    
    return tn


def kagome_tn_from_unit_cell(unit_cell:UnitCell, dims:TNDimensions) -> KagomeTNRepeatedUnitCell:
    return create_kagome_tn(
        d = dims.physical_dim,
        D = dims.virtual_dim,
        N = dims.big_lattice_size,
        unit_cell = unit_cell
    )






