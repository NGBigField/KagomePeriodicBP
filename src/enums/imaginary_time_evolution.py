from enum import Enum, auto
from enums.tensor_networks import UnitCellFlavor
from lattices.directions import BlockSide

class UpdateMode(Enum):
    A = auto()
    B = auto()
    C = auto()

    def is_matches_flavor(self, flavor:UnitCellFlavor)->bool:
        return self.name == flavor.name
    
    @property
    def side_in_core(self)->BlockSide:
        return MODES_SIDES[self]
    
    

MODES_SIDES : dict[UpdateMode, BlockSide] = {
    UpdateMode.A : BlockSide.U,
    UpdateMode.B : BlockSide.DL,
    UpdateMode.C : BlockSide.DR,
}

    

