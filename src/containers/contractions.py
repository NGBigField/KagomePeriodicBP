from dataclasses import dataclass, field
from containers._meta import _ConfigClass
from lattices.directions import BlockSide, LatticeDirection
from typing import Callable



@dataclass
class BubbleconContractionConfig(_ConfigClass):
    trunc_dim : int = 20
    parallel : bool = False
    progress_bar : bool = True

    def __repr__(self) -> str:
        return super().__repr__()

@dataclass
class BubbleConGlobalConfig(_ConfigClass): 
    # trunc_dim_2=None
    # eps=None  
    override_progress_bar : bool|None = None   # None - not overriding; True - Always On; False - Always off
    separate_exp : bool =True
    iterative_compression_max_ier : int =  200
    iterative_compression_error : float = 1e-8
    d_threshold_for_compression : int = 10

    def bubblecon_compression(self, D:int) -> dict:
        if D <= self.d_threshold_for_compression:
            return {'type':'SVD'}
        else:
            return {
                'type' : 'iter', 
                'max-iter' : self.iterative_compression_max_ier, 
                'err' : self.iterative_compression_error
            }

                
    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class MPSOrientation: 
    open_towards : BlockSide
    ordered : LatticeDirection

    @staticmethod
    def standard(main_direction:BlockSide)->"MPSOrientation":
        return MPSOrientation(
            open_towards = main_direction,
            ordered = main_direction.orthogonal_clockwise_lattice_direction() 
        )
    

