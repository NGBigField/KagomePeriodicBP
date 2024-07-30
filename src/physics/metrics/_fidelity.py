import numpy as np
from scipy.linalg import sqrtm

def fidelity(x:np.ndarray, y:np.ndarray, force_real:bool=True) -> float:
    """Fidelity between two density matrices is defined as (tr(sqrt( sqrt(x) y sqrt(x) )))^2
    """
    sqrt_x : np.ndarray = sqrtm(x)  #type: ignore
    prod = sqrt_x @ y @ sqrt_x
    sqrt_prod : np.ndarray = sqrtm(prod)  #type: ignore
    trace = np.trace(sqrt_prod)
    result = trace*trace
    if force_real:
        result = result.real
    return result