import numpy as np
from scipy.linalg import sqrtm

def fidelity(x:np.ndarray, y:np.ndarray, force_real:bool=True) -> float:
    """(tr(sqrt( sqrt(x) y sqrt(x) )))^2
    """
    sqrt_x : np.ndarray = sqrtm(x)  #type: ignore
    prod = sqrt_x @ y @ sqrt_x
    trace = np.trace(prod)
    result = trace*trace
    if force_real:
        result = result.real
    return result