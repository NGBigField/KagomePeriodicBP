from enum import Enum, auto
from enums.tensor_networks import UnitCellFlavor
from typing import Generator
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
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UpdateMode|UnitCellFlavor):
            return False
        return self.name == other.name
    
    def __hash__(self) -> int:
        return hash((self.name, type(self)))
    
    def __str__(self) -> str:
        return self.name
    
    @staticmethod
    def all_options()->Generator["UpdateMode", None, None]:
        return (mode for mode in UpdateMode)
            
    

MODES_SIDES : dict[UpdateMode, BlockSide] = {
    UpdateMode.A : BlockSide.U,
    UpdateMode.B : BlockSide.DL,
    UpdateMode.C : BlockSide.DR,
}

    

