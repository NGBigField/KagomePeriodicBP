from enum import Enum, auto

class NodeFunctionality(Enum):
    Core = auto()
    Message = auto()
    Padding = auto()
    Undefined = auto()   # used when can't derive the true functioanlity of a node. Usually an error.


class CoreCellType(Enum):
    A = auto()
    B = auto()
    C = auto()
    NoCore = auto()
