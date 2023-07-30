from dataclasses import dataclass
from typing import Generator
from numpy import ndarray as np_ndarray
from enums import UnitCellFlavor
import numpy as np

@dataclass
class UnitCell: 
    A : np_ndarray
    B : np_ndarray
    C : np_ndarray
       
    def all(self)->Generator[tuple[np_ndarray, UnitCellFlavor], None, None]:
        yield self.A, UnitCellFlavor.A
        yield self.B, UnitCellFlavor.B
        yield self.C, UnitCellFlavor.C

    @staticmethod
    def all_keys()->list[str]:
        return ["A", "B", "C"]

    def copy(self)->"UnitCell":
        return UnitCell(
            A=self.A.copy(),
            B=self.B.copy(),
            C=self.C.copy()
        )
    
    @staticmethod
    def random(d:int, D:int)->"UnitCell":
        return UnitCell(
             A = _random_tensor(d, D),
             B = _random_tensor(d, D),
             C = _random_tensor(d, D)
        )





def _random_tensor(d:int, D:int)->np.ndarray:
    rs = np.random.RandomState()
    t = rs.uniform(size=[d]+[D]*4) \
        + 1j*rs.normal(size=[d]+[D]*4)
    t = t/np.linalg.norm(t)
    return t



