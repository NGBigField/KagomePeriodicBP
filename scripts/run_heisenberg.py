import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from tensor_networks import UnitCell

from utils import strings
from typing import Iterable


# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians

import numpy as np

d = 2


def main(
    D = 2,
    N = 2,
    live_plots:bool|Iterable[bool] = [1,1,0],
    results_filename:str = strings.time_stamp()+"_"+strings.random(4),
    parallel:bool = 0,
    chi_factor : int = 1,
    hamiltonian:str = "Field",  # Anti-Ferro-Magnetic or Ferro-Magnetic
    damping:float|None = 0.1
)->tuple[float, str]:
    
    unit_cell = UnitCell.load("2024.03.07_19.57.17- keep")
    # unit_cell = UnitCell.random(d=d, D=D)
    # unit_cell = UnitCell.zero_product_state(d=d, D=D)
    # unit_cell.set_filename(results_filename) 

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.bp.damping = damping
    config.bp.max_swallowing_dim = 4*D**2
    config.bp.parallel_msgs = parallel
    config.trunc_dim *= chi_factor
    config.bp.max_swallowing_dim *= chi_factor
    config.ite.time_steps = [0.01]*50 + [0.001]*50 + [1e-4]*50 + [1e-5]*50

    # Interaction:
    match hamiltonian: 
        case "AFM":   config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None)
        case "FM":    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm, None)
        case "FM-T":  config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm_with_field, None)
        case "Field": config.ite.interaction_hamiltonian = (hamiltonians.field, None)
        case _:
            raise ValueError("Not matching any option.")
    

    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    fullpath = unit_cell_out.save(results_filename+"_final")
    print("Done")

    return energy, fullpath



if __name__ == "__main__":
    main()