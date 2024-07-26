from typing import TypeAlias


EdgeIndicatorType               : TypeAlias = str
EdgesDictType                   : TypeAlias = dict[str, tuple[int, int]]
ExpectationsPerDirection        : TypeAlias = dict[str, float]
UnitCellValuePerEdgeDict        : TypeAlias = dict[str, float]
EnergiesOfEdgesDuringUpdateType : TypeAlias = list[UnitCellValuePerEdgeDict]
PosScalarType                   : TypeAlias = int|float
PermutationOrdersType           : TypeAlias = dict[str, list[int]]
UnitCellExpectationValuesDict   : TypeAlias = dict[str, dict[str, float]]
