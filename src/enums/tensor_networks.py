from enum import Enum, auto

class NodeFunctionality(Enum):
    CenterCore = auto()
    AroundCore = auto()
    Message = auto()
    Padding = auto()
    Environment = auto()
    Undefined = auto()   # used when can't derive the true functionality of a node. Usually an error.


class UnitCellFlavor(Enum):
    A = auto()
    B = auto()
    C = auto()
    NoneLattice = auto()

    def __str__(self) -> str:
        return self.name
