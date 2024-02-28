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
    live_plots:bool|Iterable[bool] = [0,0,0],
    parallel:bool = 0,
    chi_factor : int = 1,
    results_filename:str = strings.time_stamp()+"_"+strings.random(4),
    hamiltonian:str = "Field",  # Anti-Ferro-Magnetic or Ferro-Magnetic
    damping:float|None = 0,
    hermitize_messages_between_iterations:bool = True
)->tuple[float, str]:
    
    unit_cell_file_name = f"crnt_heisenberg_{hamiltonian}_D{D}_chi{chi_factor}_"+strings.random(3)
    # unit_cell = UnitCell.load(unit_cell_file_name)
    # if unit_cell is None:
    # unit_cell = UnitCell.random(d=d, D=D)
    unit_cell = UnitCell.zero_product_state(d=d, D=D)
    # unit_cell = UnitCell.random_product_state(d=d, D=D)
    unit_cell.set_filename(results_filename) 

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.bp.damping = damping
    config.bp.hermitize_messages_between_iterations = hermitize_messages_between_iterations
    # config.bp.hermitize_messages_between_iterations = False

    # Interaction:
    match hamiltonian: 
        case "AFM":   config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None)
        case "FM":    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm, None)
        case "FM-T":  config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm_with_field, None)
        case "Field": config.ite.interaction_hamiltonian = (hamiltonians.field, None)
        case _:
            raise ValueError("Not matching any option.")
        
    # chi
    config.trunc_dim *= chi_factor
    config.bp.max_swallowing_dim *= chi_factor
    # Comp
    config.bp.parallel_msgs = parallel

    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    fullpath = unit_cell_out.save(results_filename+"_final")
    print("Done")

    return energy, fullpath



if __name__ == "__main__":
    main()