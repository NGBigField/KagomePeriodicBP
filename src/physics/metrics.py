import numpy as np
from numpy.linalg import norm
from numpy import conj


def hermicity(rho:np.matrix)->float: 
    numpy_typed = norm(rho - conj(rho.transpose([1,0,3,2])))/norm(rho)
    native_typed = numpy_typed.item()
    return native_typed