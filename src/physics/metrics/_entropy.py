import numpy as np
from scipy.linalg import logm
from .src._common import if_operator_transform_to_density_matrix
from .src.densitymats import DensityMatrix


def von_neumann_entropy(rho:np.ndarray) -> float:
    rho = if_operator_transform_to_density_matrix(rho)

    ln_rho = logm(rho)
    res = -np.trace(rho @ ln_rho)
    res = float(res)
    return res


def entanglement_entropy(rho:np.ndarray, already_reduced_density_matrix:bool=True) -> float:
    rho = if_operator_transform_to_density_matrix(rho)

    if not already_reduced_density_matrix:
        raise NotImplementedError("Need to implement partial Trace")  #TODO
        rho = partial_trace(rho)
    
    return von_neumann_entropy(rho)


