from dataclasses import dataclass
from typing import Generator
from numpy import ndarray as np_ndarray
from enums import CoreCellType

@dataclass
class UnitCell: 
    A : np_ndarray
    B : np_ndarray
    C : np_ndarray
    
    
    def all(self)->Generator[tuple[np_ndarray, CoreCellType], None, None]:
        yield self.A, CoreCellType.A
        yield self.B, CoreCellType.B
        yield self.C, CoreCellType.C




