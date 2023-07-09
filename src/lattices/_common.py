from tensor_networks.node import _EdgeIndicator, _PosScalarType, Direction
from dataclasses import dataclass, field


class LatticeError(ValueError): ...
class OutsideLatticeError(LatticeError): ...

@dataclass
class NodePlaceHolder():
    index : int
    pos : tuple[_PosScalarType, ...]
    edges : list[_EdgeIndicator]
    directions : list[Direction]