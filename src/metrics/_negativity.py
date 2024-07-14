import numpy as np
import typing as typ
from itertools import combinations
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
# My modules:
from metrics.src.densitymats import DensityMatrix, CommonDensityMatrices
from utils import assertions

# Global Constants:
IS_VALIDATE_CORRECT_METHOD = False
EPSILON = 0.0001


class CheckResultsError(Exception): 
    def __init__(self, values: typ.List[float]) -> None:
        super().__init__()
        self.values = values

    def __str__(self) -> str:
        strings : typ.List[str] = []
        for i, value in enumerate(self.values):
            strings.append(f"val_{i} = {value:.010f}")
        string = "\n".join(strings)
        return string


def _assert_same_results(vals: typ.List[float]) -> None:
    for v1, v2 in combinations(vals, 2):
        if abs(v1-v2)>EPSILON:
            raise CheckResultsError(vals)

def _eigenvalues(rho: DensityMatrix) -> typ.List[complex]:
    eigenvals, _ = np.linalg.eig(rho)
    return eigenvals

def _method1(rho_pt: DensityMatrix) -> float:
    rho_pt_dagger = rho_pt.dagger()
    m = DensityMatrix( rho_pt_dagger @ rho_pt )
    sqrt_of_mat = sqrtm(m)
    _sum = np.trace(sqrt_of_mat)
    res = (_sum - 1)/2
    return res

def _method2(rho_pt: DensityMatrix) -> float:
    eigenvals = _eigenvalues(rho_pt)
    sum = 0.00
    for eigenval in eigenvals:
        if eigenval < 0:
            sum += abs(eigenval)  
    return sum

def negativity(
    rho: DensityMatrix|np.ndarray, 
    num_qubits_on_first_part:typ.Optional[int]=None, 
    part_to_transpose:typ.Literal['first', 'second']='first',
    validate:bool=True
) -> float:
    if isinstance(rho, np.ndarray):
        rho = DensityMatrix(rho)
    assert isinstance(rho, DensityMatrix)
    
    rho_pt = rho.partial_transpose(num_qubits_on_first_part, part_to_transpose, validate) # default to half of qubits are transposed
    res2 = _method2(rho_pt)    
    if IS_VALIDATE_CORRECT_METHOD:        
        res1 = _method1(rho_pt)
        _assert_same_results([res1, res2])
    result = assertions.real(res2)
    return result

@typ.overload
def log_negativity(val: DensityMatrix) -> float: ...
@typ.overload
def log_negativity(val: float) -> float: ...

def log_negativity(val) -> float:
    if isinstance(val, DensityMatrix):
        neg = negativity(val)
    elif isinstance(val, float):
        neg = val
    return np.log2(2*neg+1)

