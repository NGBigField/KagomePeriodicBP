from dataclasses import dataclass
from typing import Generator
from numpy import ndarray as np_ndarray
from enums import UnitCellFlavor
import numpy as np

from utils import saveload, strings, iterations

UNIT_CELL_SUBFOLDER = "unit_cells"


@dataclass
class UnitCell: 
    A : np_ndarray
    B : np_ndarray
    C : np_ndarray
    _file_name : str|None = None

    def __getitem__(self, key:str)->np_ndarray:
        match key:
            case 'A': return self.A
            case 'B': return self.B
            case 'C': return self.C
            case _: 
                raise KeyError("No an option")
            
    def __setitem__(self, key:str|UnitCellFlavor, value:np_ndarray)->None:
        if isinstance(key, UnitCellFlavor):
            key = key.name
        match key:
            case 'A': self.A = value
            case 'B': self.B = value
            case 'C': self.C = value
            case _: 
                raise KeyError("No an option")

    def items(self)->Generator[tuple[UnitCellFlavor, np_ndarray], None, None]:
        yield UnitCellFlavor.A, self.A
        yield UnitCellFlavor.B, self.B
        yield UnitCellFlavor.C, self.C

    @staticmethod
    def all_keys()->list[str]:
        return ["A", "B", "C"]
    
    
    @staticmethod
    def size()->int:
        return 3

    def copy(self)->"UnitCell":
        return UnitCell(
            A=self.A.copy(),
            B=self.B.copy(),
            C=self.C.copy(),
            _file_name=self._file_name
        )
    
    @staticmethod
    def random(d:int, D:int)->"UnitCell":
        return UnitCell(
             A = _random_tensor(d, D),
             B = _random_tensor(d, D),
             C = _random_tensor(d, D)
        )
    
    @staticmethod
    def random_product_state(d:int, D:int)->"UnitCell":
        tensor = _random_tensor(d, D)
        return UnitCell(
             A = tensor.copy(),
             B = tensor.copy(),
             C = tensor.copy()
        )
    
    @staticmethod
    def zero_product_state(d:int, D:int)->"UnitCell":
        assert d==2, "Zero state makes canonical sense when d==2"
        tensor = _zero_state_tensor(D)
        return UnitCell(
             A = tensor.copy(),
             B = tensor.copy(),
             C = tensor.copy()
        )
    

    def save(self, file_name:str|None=None)->None:
        file_name = self._derive_file_name(file_name)
        return saveload.save(self, name=file_name, sub_folder=UNIT_CELL_SUBFOLDER)

    @staticmethod
    def load(file_name:str, if_exist:bool=False)->"UnitCell":
        return saveload.load(file_name, sub_folder=UNIT_CELL_SUBFOLDER, if_exist=if_exist)
    
    def set_filename(self, filename:str)->None:
        self._file_name = filename

    def _derive_file_name(self, given_name:str|None)->str:
        if given_name is not None:
            assert isinstance(given_name, str)
            return given_name
        
        if self._file_name is not None:
            assert isinstance(self._file_name, str)
            return self._file_name

        return strings.time_stamp()
        


def _zero_state_tensor(D:int)->np.ndarray:
    shape = [2]+[D]*4
    t = np.zeros(shape)
    t[0, 0, 0, 0, 0] = 1
    return t

def _random_tensor(d:int, D:int)->np.ndarray:
    rs = np.random.RandomState()
    t = rs.uniform(size=[d]+[D]*4) \
        + 1j*rs.normal(size=[d]+[D]*4)
    t = t/np.linalg.norm(t)
    return t



