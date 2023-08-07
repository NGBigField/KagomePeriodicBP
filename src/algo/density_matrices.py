import scipy
import numpy as np

from containers.density_matrices import MatrixMetrics

# Common functions:
norm = np.linalg.norm
conj = np.conj


def rho_ij_to_rho(rho_ij_in:np.ndarray)->np.ndarray:
    rho_ij = rho_ij_in.copy()
    assert len(rho_ij.shape)==4
    assert rho_ij.shape[0]==rho_ij.shape[1]==rho_ij.shape[2]==rho_ij.shape[3]
    d = rho_ij.shape[0]
    rho = rho_ij.transpose([0, 2, 1, 3])
    rho = rho.reshape([d*d, d*d])
    return rho


def calc_eigenvalues(rho:np.ndarray)->list[complex]:
    return scipy.linalg.eigvals(rho)

def calc_hermicity(rho:np.ndarray)->float:
    dims = rho.shape
    assert dims[0]==dims[1]
    hermicity = norm(rho-conj(rho.T))/norm(rho)
    return hermicity.item()


def calc_metrics(rho:np.ndarray)->MatrixMetrics:
    # common data:
    eigenvalues = calc_eigenvalues(rho)
    # output:
    return MatrixMetrics(
        eigenvalues = eigenvalues,
        sum_eigenvalues = sum(eigenvalues),
        hermicity = calc_hermicity(rho),
        norm = norm(rho).item(),
        negativity = sum([abs(np.real(val)) for val in eigenvalues if np.real(val)<0]),
        trace=rho.trace()
    )
    

    

    

    


    return output
