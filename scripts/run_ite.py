import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from unit_cell import UnitCell, get_from

from utils import strings, lists
from typing import Iterable, TypeAlias
_Bool : TypeAlias = bool|int

# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians

import numpy as np

d = 2


crnt_force_value = 1e-2
def decreasing_global_field_func(delta_t:float|None)->float:
    global crnt_force_value
    if delta_t is None:
        return 0
    if delta_t>1e-5:
        r = 0.96
    else:
        r = 0.90
    crnt_force_value *= r
    if crnt_force_value < 1e-16:
        crnt_force_value = 0.0
    return crnt_force_value


GLOBAL_FIELD = 0.0
def constant_global_field(delta_t:float|None)->float:
    return GLOBAL_FIELD


def _config_at_measurement(config:Config)->Config:
    config.dims.big_lattice_size += 0
    config.bp.msg_diff_terminate /= 1
    config.bp.allowed_retries    += 0
    return config


def _get_time_steps(e_start:int, e_end:int, n_per_dt:int)->list[float]:
    time_steps = [[np.power(10, -float(exp))]*n_per_dt for exp in np.arange(e_start, e_end, 1)]
    time_steps = lists.join_sub_lists(time_steps)
    return time_steps


def _get_hamiltonian(hamiltonian:str) -> tuple:
    match hamiltonian: 
        case "FM":        return (hamiltonians.heisenberg_fm, None, None)
        case "FM-T":      return (hamiltonians.heisenberg_fm_with_field, "delta_t", constant_global_field)
        case "AFM":       return (hamiltonians.heisenberg_afm, None, None)
        case "AFM-T":     return (hamiltonians.heisenberg_afm_with_field, "delta_t", decreasing_global_field_func)
        case "Field":     return (hamiltonians.field, None, None)
        case "Ising-AFM": return (hamiltonians.ising_with_transverse_field, "delta_t", constant_global_field)
        case _:
            raise ValueError("Not matching any option.")
        

def _get_unit_cell(D:int, get_from:str) -> tuple[UnitCell, bool]:
    is_random = False

    match get_from:
        case "random":
            unit_cell = UnitCell.random(d=d, D=D)
            is_random = True
            print("Got unit_cell as a random tensors")

        case "last":
            unit_cell = UnitCell.load("last")
            print("Got unit_cell from last result")

        case "best":
            unit_cell = UnitCell.load_best(D=D)
            if unit_cell is None:
                return _get_unit_cell(D=D, get_from="tnsu")
            print("Got unit_cell as the previous best.")

        case "tnsu":
            print("Get unit_cell by simple-update:")
            is_random = True
            unit_cell, energy = get_from.simple_update(D=D)

        case _:
            unit_cell = UnitCell.load(get_from)


    return unit_cell, is_random


def _plot_field_over_time() -> None:
    ## Imports:
    from matplotlib import pyplot as plt

    ## Config:
    n_per_dt = 200
    e_start = 3
    e_end   = 8
    time_steps = _get_time_steps(e_start=e_start, e_end=e_end, n_per_dt=n_per_dt)

    ## Get values:
    values = [ ]
    value = None
    for dt in time_steps:
        value = decreasing_global_field_func(dt)
        values.append(value)

    ## Plot:
    lines = plt.plot(values, linewidth=4)
    ax1 : plt.Axes = lines[0].axes
    ax2 : plt.Axes = ax1.twinx()
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.plot(time_steps, linewidth=2)
    ax1.grid()

    ax1.set_ylabel("force",   color=     "tab:red")
    ax1.tick_params(axis='y', labelcolor="tab:red")
    for line in ax1.lines:
        line.set_color("tab:red")

    ax2.set_ylabel("delta-t",      color="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:blue")
    for line in ax2.lines:
        line.set_color("tab:blue")

    plt.show()
    print("Done.")


def main(
    D = 3,
    N = 2,
    chi_factor : int|float = 1,
    live_plots:_Bool|Iterable[_Bool] = [0, 0, 0],   #type: ignore
    progress_bar:bool=True,
    results_filename:str|None = None,
    parallel:_Bool = 0,
    hamiltonian:str = "AFM-T",  # Anti-Ferro-Magnetic or Ferro-Magnetic
    damping:float|None = 0.1,
    unit_cell_from:str = "best"
)->tuple[float, str]:

    assert N>=2
    assert D>0
    assert chi_factor>0

    if results_filename is None:
        results_filename = strings.time_stamp()+"_"+strings.random(4)+f"_D={D}_N={N}"

    unit_cell, _radom_unit_cell = _get_unit_cell(D=D, get_from=unit_cell_from)
    unit_cell.set_filename(results_filename) 

    ## Config:
    config = Config.derive_from_dimensions(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.visuals.progress_bars = progress_bar
    config.bp.damping = damping
    config.bp.parallel_msgs = parallel
    config.ite.interaction_hamiltonian = _get_hamiltonian(hamiltonian)

    # Chi factor:
    config.trunc_dim = int(config.trunc_dim*chi_factor)
    config.bp.trunc_dim = int(config.bp.trunc_dim*chi_factor)

    config.bp.msg_diff_good_enough = 1e-4
    config.bp.msg_diff_terminate = 1e-10
    config.bp.times_to_deem_failure_when_diff_increases = 4
    config.bp.max_iterations = 40
    config.bp.allowed_retries = 2
    config.iterative_process.num_edge_repetitions_per_mode = 6
    config.iterative_process.change_config_for_measurements_func = _config_at_measurement
    config.iterative_process.start_segment_with_new_bp_message = True
    config.iterative_process.use_bp = True
    config.ite.random_edge_order = False
    config.ite.symmetric_product_formula = True
    config.ite.always_use_lowest_energy_state = False
    config.ite.add_gaussian_noise_fraction = 1e-6
    config.iterative_process.bp_every_edge = True
    config.iterative_process.num_mode_repetitions_per_segment = 3

    ## time steps:
    if D<4:
        n_per_dt = 300
        e_start = 4
        e_end   = 8
    else:
        n_per_dt = 150
        e_start = 3
        e_end   = 7
    # 
    if _radom_unit_cell:
        e_start = 2
    config.ite.time_steps = _get_time_steps(e_start, e_end, n_per_dt)

    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    fullpath = unit_cell_out.save(results_filename+"_final")
    print("Done")

    return energy, fullpath



if __name__ == "__main__":
    # _plot_field_over_time()
    main()

