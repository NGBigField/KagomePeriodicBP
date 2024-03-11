from dataclasses import dataclass, field
from containers._meta import container_repr

@dataclass
class MatrixMetrics: 
    eigenvalues : list[complex]
    negativity : float
    sum_eigenvalues : complex
    hermicity : float
    norm : float
    trace : float
    other : dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return container_repr(self)
    
