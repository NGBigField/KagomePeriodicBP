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


GLOBAL_FIELD = 0.0
def constant_global_field(delta_t:float|None)->float:
    return GLOBAL_FIELD


def _config_at_measurement(config:Config)->Config:
    config.dims.big_lattice_size += 0
    config.bp.msg_diff_terminate /= 10
    config.bp.allowed_retries    += 1
    return config


def main(
    D = 2,
    N = 3,
    chi_factor : int = 1.0,
    live_plots:bool|Iterable[bool] = [0, 0, 0],
    results_filename:str = strings.time_stamp()+"_"+strings.random(4),
    parallel:bool = 0,
    hamiltonian:str = "AFM",  # Anti-Ferro-Magnetic or Ferro-Magnetic
    damping:float|None = 0.1
)->tuple[float, str]:

    ## Choose unit-cell:    
    # unit_cell = UnitCell.load("2024.04.25_20.17.29 ising --- stable")
    # unit_cell = UnitCell.random(d=d, D=D)
    # unit_cell = UnitCell.load("last")

    unit_cell = UnitCell.load_best(D=D)
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)

    unit_cell.set_filename(results_filename) 

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.trunc_dim = int((2*D**2+10) * chi_factor)
    config.bp.max_swallowing_dim = int(2*D**2 * chi_factor)
    config.bp.damping = damping
    config.bp.parallel_msgs = parallel
    config.bp.msg_diff_terminate = 1e-13
    config.bp.msg_diff_good_enough = 1e-4
    config.bp.times_to_deem_failure_when_diff_increases = 4
    config.bp.max_iterations = 50
    config.bp.allowed_retries = 3
    config.iterative_process.num_mode_repetitions_per_segment = 5
    config.iterative_process.num_edge_repetitions_per_mode = 6
    config.iterative_process.change_config_for_measurements_func = _config_at_measurement
    config.iterative_process.start_segment_with_new_bp_message = True
    config.iterative_process.use_bp = True
    config.iterative_process.bp_every_edge = True
    config.ite.random_edge_order = True
    config.ite.normalize_tensors_after_update = True
    config.ite.symmetric_product_formula = True
    config.ite.always_use_lowest_energy_state = True
    config.ite.add_gaussian_noise_fraction = 1e-6
    config.ite.time_steps = [[np.power(10, -float(exp))]*100 for exp in np.arange(3, 7, 1)]
    # config.ite.time_steps = [0.001]*3

    # Interaction:
    match hamiltonian: 
        case "FM":        h = (hamiltonians.heisenberg_fm, None, None)
        case "FM-T":      h = (hamiltonians.heisenberg_fm_with_field, "delta_t", constant_global_field)
        case "AFM":       h = (hamiltonians.heisenberg_afm, None, None)
        case "AFM-T":     h = (hamiltonians.heisenberg_afm_with_field, "delta_t", constant_global_field)
        case "Field":     h = (hamiltonians.field, None, None)
        case "Ising-AFM": h = (hamiltonians.ising_with_transverse_field, "delta_t", constant_global_field)
        case _:
            raise ValueError("Not matching any option.")
        
    config.ite.interaction_hamiltonian = h

    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    fullpath = unit_cell_out.save(results_filename+"_final")
    print("Done")

    return energy, fullpath



if __name__ == "__main__":
    main()