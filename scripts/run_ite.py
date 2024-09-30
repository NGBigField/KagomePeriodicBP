if __name__ == "__main__":
    import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config, HamiltonianFuncAndInputs
from unit_cell import UnitCell
import unit_cell.get_from as get_unit_cell_from 
import project_paths

from utils import strings, lists, processes
from typing import Iterable, TypeAlias, Literal
_Bool : TypeAlias = bool|Literal[0, 1]

# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians

import numpy as np

d = 2


def _get_condor_io_paths()->tuple[str, str]:
    logs_str = str( project_paths.condor_paths['io_dir'] / "logs" )
    data_str = str( project_paths.condor_paths['io_dir'] / "data" )
    return data_str, logs_str


crnt_force_value = 1e-2
def decreasing_global_field_func(delta_t:float|None)->float:
    global crnt_force_value
    if delta_t is None:
        return 0
    if delta_t>1e-5:
        r = 0.93
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
    config.dims.big_lattice_size += 1
    config.bp.msg_diff_terminate /= 2
    config.bp.allowed_retries    += 1
    config.chi_bp *= 2
    config.chi *= 2
    return config


def _get_time_steps(e_start:int, e_end:int, n_per_dt:int)->list[float]:
    time_steps = [[np.power(10, -float(exp))]*n_per_dt for exp in np.arange(e_start, e_end+1, 1)]
    time_steps = lists.join_sub_lists(time_steps)
    return time_steps


def _get_hamiltonian(hamiltonian:str) -> HamiltonianFuncAndInputs:
    match hamiltonian: 
        case "FM":        out = (hamiltonians.heisenberg_fm, None, None)
        case "FM-T":      out = (hamiltonians.heisenberg_fm_with_field, "delta_t", constant_global_field)
        case "AFM":       out = (hamiltonians.heisenberg_afm, None, None)
        case "AFM-T":     out = (hamiltonians.heisenberg_afm_with_field, "delta_t", decreasing_global_field_func)
        case "Field":     out = (hamiltonians.field, None, None)
        case "Ising-AFM": out = (hamiltonians.ising_with_transverse_field, "delta_t", constant_global_field)
        case _:
            raise ValueError("Not matching any option.")
        
    return out  #type: ignore
        

def _get_unit_cell(D:int, source:Literal["random", "last", "best", "tnsu"], config:Config) -> tuple[UnitCell, bool]:
    is_random = False

    match source:
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
                return _get_unit_cell(D=D, source="random", config=config)
            print("Got unit_cell as the previous best.")

        case "tnsu":
            raise NotImplementedError("Not yet.")
            print("Get unit_cell by simple-update:")
            unit_cell, energy = get_unit_cell_from.simple_update(D=D)

        case _:
            assert isinstance(source, str)
            unit_cell = UnitCell.load(source)


    return unit_cell, is_random


def _plot_field_over_time() -> None:
    ## Imports:
    from matplotlib import pyplot as plt

    ## Config:
    n_per_dt = 500
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
    ax1 : plt.Axes = lines[0].axes  #type: ignore
    ax2 : plt.Axes = ax1.twinx()    #type: ignore
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







## ==================================================== ##
#                         main                           #
## ==================================================== ##

def main(
    D = 2,
    N = 2,
    chi_factor : int|float = 1,
    live_plots:_Bool|Iterable[_Bool] = False, 
    progress_bar:Literal['all_active', 'all_disabled', 'only_main'] = 'all_active',
    results_filename:str|None = None,
    parallel:bool = False,
    hamiltonian:str = "AFM",  # Anti-Ferro-Magnetic or Ferro-Magnetic
    damping:float|None = 0.1,
    unit_cell_from:Literal["random", "last", "best", "tnsu"] = "best",
    monitor_cpu_and_ram:bool = False,
    io : Literal['local', 'condor'] = 'local',
    messages_init : Literal['random', 'uniform'] = 'uniform'
)->tuple[float, str]:

    assert N>=2
    assert D>0
    assert chi_factor>0

    if results_filename is None:
        results_filename = strings.time_stamp()+f"_AFM_D={D}_N={N}_"+strings.random(4)

    ## Config:
    config = Config.derive_from_dimensions(D)
    config.set_parallel(parallel) 
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    config.visuals.progress_bars = progress_bar
    config.bp.damping = damping
    config.ite.interaction_hamiltonian = _get_hamiltonian(hamiltonian)

    # Chi factor:
    config.chi = config.chi*chi_factor
    config.chi_bp = config.chi_bp*chi_factor

    config.bp.msg_diff_good_enough = 1e-6
    config.bp.msg_diff_terminate = 1e-13
    # config.iterative_process.change_config_for_measurements_func = _config_at_measurement
    config.ite.always_use_lowest_energy_state = True
    config.ite.symmetric_second_order_trotterization = True
    config.ite.add_gaussian_noise_fraction = 1e-0
    config.ite.random_edge_order = False
    config.iterative_process.num_mode_repetitions_per_segment = 5

    MessageModel = config.BPConfig.init_msg.__class__
    if messages_init == 'random':
        config.BPConfig.init_msg = MessageModel('RQ')
    elif messages_init == 'uniform':
        config.BPConfig.init_msg = MessageModel('UQ')
    else:
        raise ValueError("Not expected.")

    # Monitor:
    config.monitoring_system.set_all(monitor_cpu_and_ram)

    # IO:
    if io == 'local':
        pass
    elif io == 'condor':
        condor_io_data, condor_io_logs = _get_condor_io_paths()
        config.io.data.fullpath = condor_io_data
        config.io.logs.fullpath = condor_io_logs
    else:
        raise ValueError(f"Not an expected input for io = {io!r}")

    # Change unit-cell folders global pointers:
    if io == 'condor':
        from utils import saveload
        from unit_cell.definition import _set_paths
        saveload.DEFAULT_DATA_FOLDER = config.io.data.fullpath
        _set_paths()

    ## Unit-cell:
    unit_cell, _radom_unit_cell = _get_unit_cell(D=D, source=unit_cell_from, config=config)
    unit_cell.set_filename(results_filename)     

    ## time steps:
    if unit_cell_from == 'best':
        config.ite.time_steps = _get_time_steps(4, 8, 200)
    else:
        config.ite.time_steps = _get_time_steps(2, 8, 100)    

    # if _radom_unit_cell:
    #     append_to_head = []
    #     append_to_head += _get_time_steps(2, 2,  50) 
    #     config.ite.time_steps = append_to_head + config.ite.time_steps

    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config, common_results_name=results_filename)
    fullpath = unit_cell_out.save(results_filename+"_final")
    print("Done")

    return energy, fullpath



if __name__ == "__main__":
    # _plot_field_over_time()
    main()

