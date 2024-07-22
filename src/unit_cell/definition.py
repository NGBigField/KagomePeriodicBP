from dataclasses import dataclass, field, fields
from typing import Generator, Literal
from numpy import ndarray as np_ndarray
from enums import UnitCellFlavor
import numpy as np
from physics.metrics.distance import tensor_distance

from utils import saveload, strings, iterations, files, assertions

UNIT_CELL_SUBFOLDER_NAME = "unit_cells"
UNIT_CELL_FOLDER_FULLPATH = saveload.DATA_FOLDER + saveload.PATH_SEP + UNIT_CELL_SUBFOLDER_NAME
BEST_UNIT_CELL_FOLDER_FULLPATH = UNIT_CELL_FOLDER_FULLPATH + saveload.PATH_SEP + "best"

@dataclass
class UnitCell: 
    """A 3-sites UnitCell for the Kagome lattice
    >>>        \\  /
    >>>         \\/
    >>>          A
    >>>         / \\ 
    >>>        /   \\
    >>> ------B-----C------
    >>>      /       \\
    >>>     /         \\
        
    3 Tensors are located in 3 sites in an upper-triangle:
        A: Top of the triangle.      Legs are ordered us: [physical-leg, UL, DL, DR, UR]
        B: Left corner of triangle.  Legs are ordered us: [physical-leg, L,  DL, R,  UR]
        C: Right corner of triangle. Legs are ordered us: [physical-leg, UL, L,  DR, R ]
    """

    A : np_ndarray
    B : np_ndarray
    C : np_ndarray
    _file_name : str|None = None
    _rotated : int = 0

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
    def from_upper_triangle(triangle)->"UnitCell":
        return UnitCell(A=triangle.up, B=triangle.left, C=triangle.right)
    
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
            _file_name=self._file_name,
            _rotated=self._rotated
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
    
    def save(self, file_name:str|None=None) -> str:
        file_name = self._derive_file_name(file_name)
        return saveload.save(self, name=file_name, sub_folder=UNIT_CELL_SUBFOLDER_NAME)

    @staticmethod
    def load(file_name:str, if_exist:bool=False)->"UnitCell":
        if file_name=="last":
            file_name = files.get_last_file_in_folder(UNIT_CELL_FOLDER_FULLPATH)
        return saveload.load(file_name, sub_folder=UNIT_CELL_SUBFOLDER_NAME, none_if_not_exist=if_exist)
    
    @staticmethod
    def load_best(D:int, none_if_not_exist:bool=True) -> "UnitCell":
        data = BestUnitCellData.load(D=D, none_if_not_exist=none_if_not_exist)
        if data is None and none_if_not_exist:
            return None
        assert data.D == D
        return data.unit_cell        
    
    @staticmethod
    def are_equal(uc1:"UnitCell", uc2:"UnitCell") -> bool:
        assert isinstance(uc1, UnitCell)
        assert isinstance(uc2, UnitCell)
        if not np.all(uc1.A==uc2.A) : return False
        if not np.all(uc1.B==uc2.B) : return False
        if not np.all(uc1.C==uc2.C) : return False
        if not np.all(uc1._rotated==uc2._rotated==0)  : return False
        return True
    
    @staticmethod
    def distance(uc1:"UnitCell", uc2:"UnitCell") -> float:
        assert isinstance(uc1, UnitCell)
        assert isinstance(uc2, UnitCell)
        distances = [tensor_distance(t1, t2) for (_, t1), (_,t2) in zip(uc1.items(), uc2.items(), strict=True) ]
        return sum(distances)

    def set_filename(self, filename:str)->None:
        self._file_name = filename

    def add_noise(self, noise_fraction:float)->None:

        def _random_tensor_like(t:np.ndarray)->np.ndarray:
            dims = t.shape
            d = dims[0]
            D = dims[1]
            num_virtual_legs = len(dims)-1
            m = _random_quantum_state_tensor(d=d, D=D, num_virtual_legs=num_virtual_legs)
            return m

        def _add_noise(t:np.ndarray)->np.ndarray:
            norm = np.linalg.norm(t)
            scale = norm*noise_fraction
            noise = _random_tensor_like(t)
            noise *= scale
            return t + noise
        
        self.A = _add_noise(self.A)
        self.B = _add_noise(self.B)
        self.C = _add_noise(self.C)

    def distance_from(self, other:"UnitCell") -> float:
        return UnitCell.distance(self, other)

    def _derive_file_name(self, given_name:str|None)->str:
        if given_name is not None:
            assert isinstance(given_name, str)
            return given_name
        
        if self._file_name is not None:
            assert isinstance(self._file_name, str)
            return self._file_name

        return strings.time_stamp()
    
    def derive_dimensions(self) -> tuple[int, int]:
        d = self.A.shape[0]
        D = self.A.shape[-1]
        return d, D
    
    def rotate(self, num_rotations:int|Literal["random"], on_copy:bool=False) -> "UnitCell":
        """Rotate the unit-cell while keeping correct relative leg order.

        Args:
            `num_rotations`: int in [-2, -1, 0, 1, 2] or str=="random".
            
                when positive, rotating clockwise `num_rotations` times.                

                when negative, rotating counter-clockwise `num_rotations` times.
            
                when 0, not rotating (useful when "not-rotating" is part of the random orientations).
            
                when "random", randomizing value.

        Returns:
            copy of UnitCell
            number of rotations (useful when `num_rotations=="random"`)
        """
        if isinstance(num_rotations, int):
            ## Check rotations:
            assert num_rotations in [-2, -1, 0, 1, 2]
            if num_rotations < 0:
                num_rotations += 3                

            # Copy?
            if on_copy:
                res = self.copy()
            else:
                res = self

            # Rotate by small steps:
            for _ in range(num_rotations):
                res._rotate_once_clockwise()
                
        elif isinstance(num_rotations, str):
            assert num_rotations=="random"
            num_rotations = np.random.randint(0, 3)  # rand [0, 1 or 2]
            res = self.rotate(num_rotations, on_copy)

        else:
            raise TypeError("Not an expected type")
        
        return res

    def force_zero_rotation(self) -> None:
        """ Keep unit-cell in its canonical rotation. if the unit_cell is rotated, rotate it back to zero rotation.
        """
        num_rotations = -self._rotated
        self.rotate(num_rotations=num_rotations, on_copy=False)
        assert self._rotated==0

    def _rotate_once_clockwise(self) -> None:
        new = UnitCell(
            A = _permute_physical_tensor(self.B, [2, 3, 4, 1]),
            B = _permute_physical_tensor(self.C, [3, 4, 1, 2]),
            C = _permute_physical_tensor(self.A, [2, 3, 4, 1])
        )
        new._rotated = (self._rotated + 1) % 3
        self.copy_from(new)
    
    def copy_from(self, other:"UnitCell") -> None:
        assert isinstance(other, UnitCell)

        for f in fields(self):
            value = getattr(other, f.name)
            setattr(self, f.name, value)
        

    
def _zero_state_tensor(D:int)->np.ndarray:
    shape = [2]+[D]*4
    t = np.zeros(shape)
    t[0, 0, 0, 0, 0] = 1
    return t

def _random_tensor(d:int, D:int)->np.ndarray:
    rs = np.random.RandomState()
    t = rs.uniform(size=[d]+[D]*4) \
        + 1j*rs.normal(size=[d]+[D]*4)
    t /= np.linalg.norm(t)  # normalize
    return t


BEST_UNIT_CELL_DATA_FOLDER_NAME = UNIT_CELL_SUBFOLDER_NAME + saveload.PATH_SEP + "best"

@dataclass
class BestUnitCellData:
    unit_cell : UnitCell
    mean_energy : float
    D : int

    def is_better_than(self, other:"BestUnitCellData") -> bool:
        return self.mean_energy < other.mean_energy

    @staticmethod
    def load(D:int, none_if_not_exist:bool=True) -> "BestUnitCellData":
        BestUnitCellData.force_folder_exist()
        for file_name in BestUnitCellData._all_files_with_D(D=D):
            return saveload.load(file_name, sub_folder=BEST_UNIT_CELL_DATA_FOLDER_NAME)
        if none_if_not_exist:
            return None  #type: ignore
        else:
            raise FileExistsError(f"No saved file for best unit_cell with D={D}")
        
    def save(self) -> str:
        BestUnitCellData.force_folder_exist()
        BestUnitCellData._remove_all_with_D(D=self.D)
        file_name = self._canonical_file_name()
        return saveload.save(self, file_name, sub_folder=BEST_UNIT_CELL_DATA_FOLDER_NAME)
    
    @staticmethod
    def _folder_fullpath() -> str:
        return BEST_UNIT_CELL_FOLDER_FULLPATH

    @staticmethod
    def _all_files_with_D(D:int) -> Generator[str, None, None]:
        file_names = files.get_all_file_names_in_folder(BEST_UNIT_CELL_FOLDER_FULLPATH)
        for file_name in file_names:
            dim_str, *_ = file_name.split(" ") 
            _, dim_str = dim_str.split("=") 
            if int(dim_str) == D:
                yield file_name

    def _canonical_file_name(self) -> str:
        return f"D={self.D} energy={self.mean_energy}"
    
    @staticmethod
    def _remove_all_with_D(D:int) -> None:
        for file_name in BestUnitCellData._all_files_with_D(D=D):
            saveload.delete(file_name, sub_folder=BEST_UNIT_CELL_DATA_FOLDER_NAME)
    
    @staticmethod
    def force_folder_exist() -> None:
        saveload.force_folder_exists(BEST_UNIT_CELL_FOLDER_FULLPATH)


def _permute_physical_tensor(t:np.ndarray, permutation:list[int]) -> np.ndarray:
    permutation = [0]+permutation  # include physical leg
    return t.copy().transpose(permutation)

def _random_quantum_state_tensor(d:int, D:int, num_virtual_legs) -> np.ndarray:
    ## define needed params and functions:
    dims = [d] + [D]*num_virtual_legs
    def _rand_real_mat() -> np.ndarray:
        return np.random.rand(*dims)  #type: ignore

    ## create a legit tensor by a random tensor and its conjugate:
    t = _rand_real_mat() + 1j*_rand_real_mat()
    # t_conj = t.conjugate()
    t_conj = np.conj(t.transpose([0, 2, 1, 4, 3]))
    r = (t + t_conj)
    r /= np.linalg.norm(r)

    return r