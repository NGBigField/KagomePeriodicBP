from dataclasses import dataclass
from utils import assertions
from containers._meta import container_repr

@dataclass
class TNSizesAndDimensions: 
    virtual_dim : int = 3  
    physical_dim : int = 2
    core_size : int = 2
    big_lattice_size : int = 6
                
    @property
    def core_repeats(self) -> int:
        return assertions.integer(self.big_lattice_size/self.core_size)

    def __repr__(self) -> str:
        return container_repr(self)