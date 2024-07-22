import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# useful utils:
from utils import visuals, dicts, saveload, csvs, prints

# For TN:
from tensor_networks.construction import kagome_tn_from_unit_cell, UnitCell

# Common types in code:
from containers import TNDimensions
from tensor_networks import CoreTN

# The things we test:
from algo.belief_propagation import _belief_propagation_step, BPConfig, belief_propagation
from algo.tn_reduction import reduce_tn

# For testing and showing performance:
from time import perf_counter
from matplotlib import pyplot as plt

# Useful for math:
import numpy as np


d=2



def main(
    D:int=3
):
    unit_cell = UnitCell.random(d=d, D=D)
    u1 = unit_cell
    print(u1.A[0][0][0][0])
    unit_cell._rotate_once_clockwise()
    unit_cell._rotate_once_clockwise()
    unit_cell._rotate_once_clockwise()
    u2 = unit_cell
    print(u2.A[0][0][0][0])
    assert id(u1) == id(u2), "Must be same unit-cells"
    assert UnitCell.are_equal(u1, u2)

    unit_cell1 = unit_cell.rotate(-1, on_copy=True)
    unit_cell1 = unit_cell1.rotate(+1, on_copy=True)
    assert id(unit_cell) != id(unit_cell1), "Must be different unit-cells"
    assert np.all(unit_cell1.B==unit_cell.B)
    assert unit_cell._rotated==0

    for _ in range(10):
        unit_cell2 = unit_cell.rotate("random", on_copy=True)
        unit_cell2.force_zero_rotation()
        print(id(unit_cell))
        print(id(unit_cell2))
        assert id(unit_cell) != id(unit_cell2), "Must be different unit-cells"
        assert UnitCell.are_equal(unit_cell, unit_cell2)


    print("Done")


if __name__ == "__main__":
    main()