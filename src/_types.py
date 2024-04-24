from typing import TypeAlias


EdgeIndicatorType : TypeAlias = str
EdgesDictType : TypeAlias = dict[str, tuple[int, int]]
EnergyPerEdgeDictType : TypeAlias = dict[str, float]
EnergiesOfEdgesDuringUpdateType : TypeAlias = list[EnergyPerEdgeDictType]
PosScalarType : TypeAlias = int