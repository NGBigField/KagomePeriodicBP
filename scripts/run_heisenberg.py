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


force_values = np.logspace(-14, -20, 500*6)
iter_force_value = iter(force_values)
enter_counter = 0
def decreasing_global_field_func(delta_t:float|None)->float:
    global enter_counter 
    enter_counter += 1
    try:
        next_force_value = next(iter_force_value)
    except StopIteration:
        next_force_value = 0
    return next_force_value


def _config_at_measurement(config:Config)->Config:
    config.dims.big_lattice_size += 0
    config.bp.msg_diff_terminate /= 1
    config.bp.allowed_retries    += 1
    return config


def main(
    D = 2,
    N = 2,
    chi_factor : int = 1.0,
    live_plots:bool|Iterable[bool] = [1, 0, 0],
    results_filename:str = strings.time_stamp()+"_"+strings.random(4),
    parallel:bool = 0,
    hamiltonian:str = "AFM",  # Anti-Ferro-Magnetic or Ferro-Magnetic
    active_bp:bool = True,
    damping:float|None = None
)->tuple[float, str]:
    
    # unit_cell = UnitCell.load("last")
    # unit_cell = UnitCell.load("2024.04.11_09.43.42_CGOP - stable -0.25")
    unit_cell = UnitCell.random(d=d, D=D)
    unit_cell.set_filename(results_filename) 

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.bp.damping = damping
    config.bp.parallel_msgs = parallel
    config.trunc_dim = int(4*D**2+20 * chi_factor)
    config.bp.max_swallowing_dim = int(4*D**2 * chi_factor)
    config.bp.msg_diff_terminate = 1e-14
    config.bp.msg_diff_good_enough = 1e-7
    config.bp.times_to_deem_failure_when_diff_increases = 4
    config.bp.max_iterations = 80
    config.bp.allowed_retries = 2
    config.iterative_process.bp_every_edge = True
    config.iterative_process.num_mode_repetitions_per_segment = 2
    config.iterative_process.num_edge_repetitions_per_mode = 6
    config.iterative_process.start_segment_with_new_bp_message = True
    config.iterative_process.change_config_for_measurements_func = _config_at_measurement
    config.iterative_process.use_bp = active_bp
    config.ite.normalize_tensors_after_update = True
    config.ite.time_steps = [[10**(-exp)]*100 for exp in range(1, 18, 1)]

    # Interaction:
    match hamiltonian: 
        case "FM":    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm, None, None)
        case "FM-T":  config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm_with_field, "delta_t", decreasing_global_field_func)
        case "AFM":   config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None, None)
        case "AFM-T": config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm_with_field, "delta_t", decreasing_global_field_func)
        case "Field": config.ite.interaction_hamiltonian = (hamiltonians.field, None, None)
        case _:
            raise ValueError("Not matching any option.")
    

    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    fullpath = unit_cell_out.save(results_filename+"_final")
    print("Done")

    return energy, fullpath



if __name__ == "__main__":
    main()