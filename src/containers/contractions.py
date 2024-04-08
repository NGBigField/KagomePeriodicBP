from dataclasses import dataclass
from lattices.directions import BlockSide, LatticeDirection

@dataclass
class BubbleConConfig: 
    # trunc_dim_2=None
    # eps=None
    progress_bar=True
    separate_exp=True


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
    

