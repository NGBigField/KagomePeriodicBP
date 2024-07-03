from dataclasses import dataclass
from containers._meta import _ConfigClass
from lattices.directions import BlockSide, LatticeDirection


@dataclass
class BubbleConConfig(_ConfigClass): 
    # trunc_dim_2=None
    # eps=None
    progress_bar=True
    separate_exp=True
                
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
    

