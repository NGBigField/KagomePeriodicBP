import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from tensor_networks import UnitCell


# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians


d = 2


def run_ite(
    D = 2,
    N = 5,
    live_plots:bool = 1,
    parallel:bool = 1,
    afm_or_fm:str = "AFM"  # Anti-Ferro-Magnetic or Ferro-Magnetic
):
    unit_cell_file_name = f"best_heisenberg_{afm_or_fm}_D{D}"
    unit_cell = UnitCell.load(unit_cell_file_name)
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)
    unit_cell._file_name = unit_cell_file_name

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    if afm_or_fm=="AFM":
        config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None)
    elif afm_or_fm=="FM":
        config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm, None)
    else:
        raise ValueError("not matching any option.")
    #
    factor = 16
    config.trunc_dim *= factor
    config.bp.max_swallowing_dim *= factor
    # Parallel:
    config.bp.parallel_msgs = parallel
    config.visuals.progress_bars = 1
    # delta-t's:
    config.ite.time_steps = [0.01]*50 + [0.001]*20 + [0.001]*50 + [0.0001]*50
    config.ite.bp_every_edge = False
    # BP:
    # config.bp.target_msg_diff = 1e-6


    ## Run:
    unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    print("Done")






def main():
    run_ite()


if __name__ == "__main__":
    main()