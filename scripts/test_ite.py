import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from enums import UpdateMode, UnitCellFlavor
from containers import ITEConfig, Config
from tensor_networks import UnitCell


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
    N = 2,
    live_plots:bool = 1,
    parallel:bool = 0
):
    
    unit_cell = UnitCell.load(f"best_heisenberg_D{D}")

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None)
    #
    factor = 4
    config.trunc_dim *= factor
    config.bp.max_swallowing_dim *= factor
    # Parallel:
    config.bp.parallel_msgs = parallel
    config.visuals.progress_bars = 1
    # delta-t's:
    config.ite.time_steps =  [1e-7]*200
    config.ite.bp_every_edge = False
    # BP:
    config.bp.msg_diff_terminate = 1e-7


    ## Run:
    unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    print("Done")






def main_test():
    test_full_ite()


if __name__ == "__main__":
    main_test()