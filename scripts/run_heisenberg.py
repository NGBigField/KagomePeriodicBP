import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from tensor_networks import UnitCell

from utils import strings
from typing import Iterable


# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians


d = 2


def main(
    D = 2,
    N = 2,
    live_plots:bool|Iterable[bool] = [1, 1, 0],
    parallel:bool = 0,
    chi_factor : int = 1,
    results_filename:str = strings.time_stamp()+"_"+strings.random(4),
    afm_or_fm:str = "FM-T"  # Anti-Ferro-Magnetic or Ferro-Magnetic
)->tuple[float, str]:
    
    unit_cell_file_name = f"crnt_heisenberg_{afm_or_fm}_D{D}_chi{chi_factor}_"+strings.random(3)
    # unit_cell = UnitCell.load(unit_cell_file_name)
    # if unit_cell is None:
    unit_cell = UnitCell.random(d=d, D=D)
    unit_cell.set_filename(results_filename) 

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    if afm_or_fm=="AFM":
        config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None)
    elif afm_or_fm=="FM":
        config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm, None)
    elif afm_or_fm=="FM-T":
        config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm_with_trans_field, None)
    else:
        raise ValueError("not matching any option.")
    #
    config.trunc_dim *= chi_factor
    config.bp.max_swallowing_dim *= chi_factor
    # Parallel:
    config.bp.parallel_msgs = parallel
    config.ite.bp_every_edge = False
    # BP:
    config.bp.target_msg_diff = 1e-6

    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    fullpath = unit_cell_out.save(results_filename+"_final")
    print("Done")

    return energy, fullpath



if __name__ == "__main__":
    main()