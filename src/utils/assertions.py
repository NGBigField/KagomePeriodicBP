# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

from typing import (
    Any,
    Union,
    Optional,
    TypeVar,
    Iterator
)

import numpy as np

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
TOLERANCE = 0.00001  # Used for distance validation
IMAGINARY_TOLERANCE = 0.0001

# ==================================================================================== #
# |                               Inner Functions                                    | #
# ==================================================================================== #

def _assert(condition:bool, reason:Optional[str]=None, default_reason:Optional[str]=None, got:Any=None):
    # if condition passed successfully:
    if condition:
        return
    ## If error is needed:
    # Get error message
    msg = ""
    if reason is not None and isinstance(reason, str):
        msg = reason
    elif default_reason is not None and isinstance(default_reason, str):
        msg = default_reason
    if got is not None:
        msg += f" (got {got})"
    raise AssertionError(msg)


def _is_positive_semidefinite(m:np.matrix) -> bool:
    eigen_vals = np.linalg.eigvals(m)
    if np.any(np.imag(eigen_vals)>TOLERANCE):  # Must be real
        return False
    if np.any(np.real(eigen_vals)<-TOLERANCE):  # Must be positive
        return False
    return True

def _is_hermitian(m:np.matrix) -> bool:
    diff = m.H-m
    shape = m.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if abs(diff[i,j])>TOLERANCE:
                return False
    return True  

# ==================================================================================== #
# |                              Declared Functions                                  | #
# ==================================================================================== #

def real(x:complex|float|int, /, reason:Optional[str]=None) -> float|int:
    if x==0:
        return 0
    imaginary_factor = abs(np.imag(x))/abs(np.real(x))
    _assert( imaginary_factor<IMAGINARY_TOLERANCE, reason=reason, default_reason=f"Must be real", got=x)
    return float(np.real(x))

def integer(x:float|int, /, reason:Optional[str]=None) -> int:
    # _assert( isinstance(x, (int, float)), reason=reason )
    _assert( round(x) == x, reason=reason, got=x)
    return int(x)

def index(x:float|int, /, reason:Optional[str]=None) -> int:
    x = integer(x, reason=reason)
    _assert( x >= 0, reason=reason, got=x)
    return x

def bit(x:float|int, /, reason:Optional[str]=None) -> int:
    x = integer(x, reason=reason)
    _assert( x in [0, 1], reason=reason, got=x)
    return x

def even(x:float|int, /, reason:Optional[str]=None) -> int:
    x = integer(x, reason=reason)
    _assert( float(x)/2 == int(int(x)/2), reason=reason, got=x)
    return x

def odd(x:float|int, /, reason:Optional[str]=None ) -> int:
    x = even(x-1, reason=reason) + 1
    return x

def logical(x:bool|int, /, reason:Optional[str]=None ) -> bool:
    if isinstance(x, bool):
        return x
    else:
        x = bit(x, reason=reason)  # also checks if 0 or 1
        if x == 0:
            return False
        elif x == 1:
            return True
        else:
            raise ValueError("bug: Not an expected option")

def density_matrix(m:np.ndarray, /,*, reason:Optional[str]=None, robust_check:bool=True) -> np.ndarray:
    _assert( isinstance(m, (np.matrix, np.ndarray)), reason=reason, default_reason="Must be a matrix type" )
    if not isinstance(m, np.matrix):
        m = np.matrix(m)
    _assert( len(m.shape)==2, reason=reason, default_reason="Must be a matrix" )
    _assert( m.shape[0]==m.shape[1], reason=reason, default_reason="Must be a square matrix" )
    _assert( abs(np.trace(m)-1)<TOLERANCE, reason=reason, default_reason="Density Matrix must have trace==1")
    if robust_check:
        _assert( _is_hermitian(m), reason=reason, default_reason="Density Matrix must be hermitian")
        _assert( _is_positive_semidefinite(m), reason=reason, default_reason="Density Matrix must be positive semidefinite")
    return m

def depleted_iterator(it:Iterator) -> Iterator:
    try:
        next(it)
    except:
        pass
    else:
        raise AssertionError(f"Iterator is not depleted!")
    return it