from dataclasses import dataclass, fields
from _types import UnitCellExpectationValuesDict, UnitCellValuePerEdgeDict, ExpectationsPerDirection
from typing import TypeAlias
from utils import lists, strings

Expectations : TypeAlias = ExpectationsPerDirection


@dataclass
class Measurements():
    energies:       UnitCellValuePerEdgeDict  # energies
    expectations:   UnitCellExpectationValuesDict  # expectations
    entanglement:   UnitCellValuePerEdgeDict  # entanglement

    @property
    def mean_energy(self)->float:
        return sum(self.energies.values())/3

    @property
    def mean_expectation_values(self)->Expectations:
        output = {}
        for xyz in ['x', 'y', 'z']:
            results = [self.expectations[abc][xyz] for abc in ['A', 'B', 'C']]
            output[xyz] = lists.average(results)
        return output


    def __repr__(self) -> str:        
        xyz = self.mean_expectation_values
        x, y, z = xyz['x'], xyz['y'], xyz['z'] 
        _formatted = lambda v: strings.formatted(v, width=10, precision=6, signed=True)
        return f"mean-energy={_formatted(self.mean_energy)} ; xyz=[{_formatted(x), _formatted(y), _formatted(z)}]"