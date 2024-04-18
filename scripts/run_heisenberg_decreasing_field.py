import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from tensor_networks import UnitCell

from utils import strings, prints
from typing import Iterable


# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians

import numpy as np

d = 2


def _config_at_measurement(config:Config)->Config:
    config.dims.big_lattice_size += 0
    config.bp.msg_diff_terminate /= 1
    config.bp.allowed_retries    += 1
    return config



def run_single_ite(
    unit_cell:UnitCell,
    field_strength:float,
    config:Config,
    results_filename:str
):
    ## Set:
    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm_with_field, field_strength, None)
    unit_cell.set_filename(results_filename+f"_{field_strength}") 

    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)

    ## Save
    fullpath = unit_cell_out.save(results_filename+f"_{field_strength}+_final")
    print("Done")

    return unit_cell


def main(
    D = 2,
    N = 2,
    chi_factor : int = 1.0,
    live_plots:bool|Iterable[bool] = [0, 0, 0],
    results_filename:str = strings.time_stamp()+"_"+strings.random(4),
    parallel:bool = 0,
    active_bp:bool = True,
    damping:float|None = None,
    field_strength_values = np.arange(1.5, 0, -0.1)
)->tuple[float, str]:
    
    # unit_cell = UnitCell.load("last")
    # unit_cell = UnitCell.load("2024.04.11_09.43.42_CGOP - stable -0.25")
    unit_cell = UnitCell.random(d=d, D=D)

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.bp.damping = damping
    config.bp.parallel_msgs = parallel
    config.trunc_dim = int(4*D**2+20 * chi_factor)
    config.bp.max_swallowing_dim = int(4*D**2 * chi_factor)
    config.bp.msg_diff_terminate = 1e-6
    config.bp.msg_diff_good_enough = 1e-5
    config.bp.times_to_deem_failure_when_diff_increases = 4
    config.bp.max_iterations = 80
    config.bp.allowed_retries = 2
    config.iterative_process.bp_every_edge = True
    config.iterative_process.num_mode_repetitions_per_segment = 3
    config.iterative_process.num_edge_repetitions_per_mode = 6
    config.iterative_process.start_segment_with_new_bp_message = True
    config.iterative_process.change_config_for_measurements_func = _config_at_measurement
    config.iterative_process.use_bp = active_bp
    config.ite.normalize_tensors_after_update = True
    config.ite.time_steps = [[10**(-exp)]*15 for exp in range(1, 8, 1)]


    prog_bar = prints.ProgressBar(len(field_strength_values), print_prefix="Decreasing field test: ")
    for field_strength in field_strength_values: 
        prog_bar.append_extra_str(f" field={field_strength}")
        prog_bar.next()

        unit_cell = run_single_ite(
            unit_cell=unit_cell,
            field_strength=field_strength, 
            config=config,
            results_filename=results_filename 
        )
        config.ite.time_steps = [[10**(-exp)]*15 for exp in range(3, 8, 1)]

    prog_bar.clear()


if __name__ == "__main__":
    main()