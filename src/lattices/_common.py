from tensor_networks.node import _EdgeIndicator, _PosScalarType, Direction
from dataclasses import dataclass, field


@dataclass
class NodePlaceHolder():
    pos : tuple[_PosScalarType, ...]
    edges : list[_EdgeIndicator]
    directions : list[Direction]