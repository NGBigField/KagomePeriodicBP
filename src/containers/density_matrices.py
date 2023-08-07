from dataclasses import dataclass
from containers._meta import container_repr

@dataclass
class MatrixMetrics: 
    eigenvalues : list[complex]
    negativity : float
    sum_eigenvalues : complex
    hermicity : float
    norm : float
    trace : float
    
    def __repr__(self) -> str:
        return container_repr(self)