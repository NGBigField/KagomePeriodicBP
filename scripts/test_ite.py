import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from enums import UpdateMode, UnitCellFlavor
from containers import ITEConfig, Config


# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians

# useful utils:
from utils import visuals, dicts, saveload

# For testing and showing performance:
from time import perf_counter
from matplotlib import pyplot as plt

# Useful for math:
import numpy as np



def test_full_ite(
    D = 2,
    N = 3,
    live_plots:bool = True
):
    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.ite.interaction_hamiltonian = (hamiltonians.ising_with_transverse_field, (0) )

    ## Run:
    unit_cell_out, ite_tracker, logger = full_ite(config=config)
    print("Done")






def main_test():
    test_full_ite()


if __name__ == "__main__":
    main_test()