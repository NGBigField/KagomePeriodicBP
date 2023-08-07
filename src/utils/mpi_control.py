# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

try:
    from mpi4py import MPI
    CAN_USE_MPI : bool = True
except ImportError:
    CAN_USE_MPI : bool = False


from typing import (
    Final,
    Optional,
)


# ============================================================================ #
#|                               Constants                                    |#
# ============================================================================ #

ATLAS = True
QUIT_COMMAND : Final[int] = 100


# ============================================================================ #
#|                                Classes                                     |#
# ============================================================================ #

class MpiControl():
    def __init__(self, active:bool=True) -> None:

        # Properties:
        self.active : bool = False
        self.comm : MPI.Intracomm = None
        self.rank : int = 0
        self.size : int = 0

        # Assignment depanding on real-time-state:
        if active and CAN_USE_MPI:
            self.active = True
            self.comm = MPI.COMM_WORLD  
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

            if self.size<1:
                self.active = False

    @property
    def with_atlas(self) -> bool:
        return ATLAS