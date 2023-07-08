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


class InitialTNMode(Enum):
    _SimpleUpdateResult = auto()
    SpecialTensor = auto()
    Random = auto()
    
    @classmethod
    def SimpleUpdateResult(cls, h:float)->"InitialTNMode":
        obj = cls._SimpleUpdateResult
        obj.h = h
        return obj