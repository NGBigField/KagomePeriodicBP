import numpy as np
from libs.TenQI import op_to_mat


def if_operator_transform_to_density_matrix(rho:np.ndarray) -> np.ndarray:
    if len(rho.shape)==4:
        rho : np.matrix = op_to_mat(rho)
    return rho