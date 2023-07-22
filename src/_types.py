from typing import TypeAlias
from lattices.directions import Direction
from libs.bmpslib import mps as MPS


EdgeIndicator : TypeAlias = str
PosScalarType : TypeAlias = int

MsgDictType = dict[Direction, tuple[MPS, Direction]]