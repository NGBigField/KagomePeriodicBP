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
    D = 4,
    N = 3,
    live_plots:bool = 1,
    parallel:bool = 0
):
    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_2d, None)
    # Parallel:
    config.bp.parallel_msgs = parallel
    config.visuals.progress_bars = 1
    # delta-t's:
    config.ite.time_steps = [0.1]*20 + [0.01]*50 + [0.001]*50 + [0.0001]*50 + [0.00001]*100
    config.ite.bp_every_edge = True

    ## Run:
    unit_cell_out, ite_tracker, logger = full_ite(config=config)
    print("Done")






def main_test():
    test_full_ite()


if __name__ == "__main__":
    main_test()