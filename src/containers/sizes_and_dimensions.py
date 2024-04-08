from dataclasses import dataclass
from utils import assertions
from containers._meta import _ConfigClass

@dataclass
class TNDimensions(_ConfigClass): 
    physical_dim : int = 2      # sometimes called `d` 
    virtual_dim : int = 3       # sometimes called `D`
    big_lattice_size : int = 3  # sometimes called `N`
                
    def __repr__(self) -> str:
        return super().__repr__()
    