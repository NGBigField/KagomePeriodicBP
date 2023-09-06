# For allowing tests and scripts to run while debuging this module
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
    from project_paths import add_src, add_base, add_scripts
    add_src()
    add_base()
    add_scripts()

# Get Global-Config:
from _config_reader import DEBUG_MODE, KEEP_LOGS, ALLOW_VISUALS

# Import containers needed for ite:
from containers.imaginary_time_evolution import ITEConfig, ITEProgressTracker, ITEPerModeStats, ITESegmentStats
from containers import Config

# Import other needed types:
from enums import UpdateMode, NodeFunctionality
from containers import MessageDictType, UpdateEdge
from tensor_networks import KagomeTN, CoreTN, ModeTN, EdgeTN, TensorNode, UnitCell
from tensor_networks.construction import kagome_tn_from_unit_cell
from _error_types import BPNotConvergedError, ITEError
from lattices.directions import Direction

# For numeric stuff:
import numpy as np
from numpy.linalg import norm

# Import our shared utilities
from utils import tuples, lists, assertions, saveload, logs, decorators, errors, prints, visuals

# For copying config:
from copy import deepcopy

# Helper function and types for ITE:
from algo.imaginary_time_evolution._logs_and_prints import print_or_log_bp_message, _log_and_print_finish_message, _log_and_print_starting_message, \
                                                            print_or_log_ite_segment_progress, get_progress_bar
from algo.imaginary_time_evolution._constants import CONVERGENCE_CHECK_LENGTH, DEFAULT_PHYSICAL_DIM
from algo.imaginary_time_evolution._visualization import ITEPlots
from algo.imaginary_time_evolution._tn_update import update_unit_cell

# Import belief propagation code:
from algo.belief_propagation import robust_belief_propagation, belief_propagation, BPStats

# Other algorithms we need:
from algo.measurements import derive_xyz_expectation_values_with_tn, measure_energies_and_observables_together
from algo.tn_reduction import reduce_tn
from libs import ITE as ite


def _initial_full_ite_inputs(config:Config, unit_cell:UnitCell, logger:logs.Logger):
    # Config:
    if config is None:
        config = Config.derive_from_physical_dim(DEFAULT_PHYSICAL_DIM)
    config.post_creation_fix()
    
    # Unit-Cell:
    if unit_cell is None:
        unit_cell = UnitCell.random(d=config.dims.physical_dim, D=config.dims.virtual_dim)
    elif isinstance(unit_cell, UnitCell):
        unit_cell = unit_cell.copy()
    else:
        raise TypeError(f"Not an expected type for input 'initial_core' of type {type(unit_cell)!r}")
    
    # Logger:
    if logger is None:
        logger = logs.get_logger(verbose=config.visuals.verbose, write_to_file=KEEP_LOGS)
    elif not isinstance(logger, logs.Logger):
        raise TypeError(f"Not an expected type for input 'logger' of type {type(logger)!r}")
    
    return config, unit_cell, logger
    


def _fix_config_if_bp_struggled(config:Config, bp_stats:BPStats, logger:logs.Logger):
    if bp_stats.attempts>1: 
        config.bp.max_swallowing_dim = bp_stats.final_config.max_swallowing_dim
        logger.debug(f"        config.bp.max_swallowing_dim updated to {config.bp.max_swallowing_dim}")
        if bp_stats.final_config.max_swallowing_dim>=config.trunc_dim:
            config.trunc_dim = int(bp_stats.final_config.max_swallowing_dim*1.5)
            logger.debug(f"        config.bubblecon_trunc_dim updated to {config.trunc_dim}")
    return config


def _calculate_crnt_observables(
    unit_cell:UnitCell, config:Config, messages:MessageDictType, segment_stats:ITESegmentStats|None
)->tuple[
    dict[tuple[str, str], float],
    dict[str, dict[str, float]],
    float
]:
    ## Unpack inputs:
    if segment_stats is None:
        bp_config = config.bp
    else:
        bp_config = segment_stats.ite_per_mode_stats[-1].bp_stats.final_config
    live_plots = config.visuals.live_plots
    allow_prog_bar = config.visuals.progress_bars

    ## Get a new fresh tn:
    full_tn = kagome_tn_from_unit_cell(unit_cell, config.dims)
    messages, _ = belief_propagation(full_tn, messages, bp_config, live_plots, allow_prog_bar)

    ## Calculate observables:
    return measure_energies_and_observables_together(full_tn, config.ite.interaction_hamiltonian, config.trunc_dim)


def _compute_and_plot_zero_iteration_(unit_cell:UnitCell, config:Config, logger:logs.Logger, ite_tracker:ITEProgressTracker, plots:ITEPlots)->None:
    # Inputs:
    delta_t = 0.0
    messages = None
    segment_stats = ITESegmentStats()

    ## Get the state of the system at iteration 0:
    logger.info("Calculating measurements of initial core...")

    ## Calculate observables:
    energies, expectations, mean_energy = _calculate_crnt_observables(unit_cell, config, messages, None)

    ## Save data, print performance and plot graphs:
    ite_tracker.log_segment(delta_t=delta_t, energy=mean_energy, unit_cell=unit_cell, messages=messages, expectation_values=expectations, stats=segment_stats)
    plots.update(energies, segment_stats, delta_t, expectations, unit_cell, _initial=True)
    logger.info(f"Mean energy at iteration 0: {mean_energy}")


def _check_converged(energies_in:list[complex|None], delta_ts:list[float], crnt_delta_t:float)->bool:

    # Convert to real
    energies = [np.real(e) for e in energies_in if e is not None]

    # Check we have enough items for a full check:
    if len(energies)<CONVERGENCE_CHECK_LENGTH:
        return False

    # Assert lengths
    assert len(energies)==len(delta_ts), f"All lists should have the same length"

    # Get last items:
    last_energies = energies[-CONVERGENCE_CHECK_LENGTH:]
    last_delta_ts = delta_ts[-CONVERGENCE_CHECK_LENGTH:]

    # Check we are checking for the crnt delta_t:
    if not all([delta_t==crnt_delta_t for delta_t in last_delta_ts]):
        return False

    if lists.all_same(last_energies):
        return True
    
    # Check convergence:
    energies_for_this_delta_t = [energy for energy, delta_t in zip(energies, delta_ts, strict=True) if delta_t==crnt_delta_t ]
    min_energy = min(energies_for_this_delta_t)
    if min_energy in last_energies:
        return False
    # max_energy = max(energies_for_this_delta_t)
    # if max_energy in last_energies:
    #     return False

    return True


def _mode_order_without_repetitions(prev_order:list[UpdateMode], ite_config:ITEConfig, num_modes:int)->list[UpdateMode]:
    # Get random variation:
    if ite_config.random_mode_order:
        new_order = list(UpdateMode.all_in_random_order())
    else:
        new_order = list(UpdateMode.all_options())

    # Don't allow two of the same in a row
    _counter = 0
    while prev_order is not None and prev_order[-1] is new_order[0]:
        new_ind = np.random.randint(low=1, high=len(new_order))
        new_order = lists.swap_items(new_order, 0, new_ind)
        _counter += 1
        if _counter>100:
            raise ValueError("Bug.")

    # add or remove items:
    if num_modes==3:
        pass
    elif num_modes<3:
        new_order = new_order[:num_modes]
    elif num_modes>3:
        new_order = new_order*(num_modes//3 + 1)
        new_order = new_order[:num_modes]

    return new_order


def _from_unit_cell_to_stable_mode(
    unit_cell:UnitCell, messages:MessageDictType, config:Config, logger:logs.Logger, mode:UpdateMode
)->tuple[
    ModeTN, MessageDictType, BPStats
]:
    ## Duplicate core into a big tensor-network:
    full_tn = kagome_tn_from_unit_cell(unit_cell, config.dims)

    ## Perform BlockBP:
    messages, bp_stats = robust_belief_propagation(
        full_tn, messages, config.bp, 
        update_plots_between_steps=config.visuals.live_plots, 
        allow_prog_bar=config.visuals.progress_bars
    )
    print_or_log_bp_message(config.bp, config.ite.bp_not_converged_raises_error, bp_stats, logger)

    # If block-bp struggled and increased the virtual dimension, the following iterations must also use a higher dimension:
    config = _fix_config_if_bp_struggled(config, bp_stats, logger)

    ## Contract to mode:
    mode_tn = reduce_tn(full_tn, ModeTN, trunc_dim=config.trunc_dim, mode=mode)
    return mode_tn, messages, bp_stats


def get_imaginary_time_evolution_operator(h:np.ndarray, delta_t:float)->tuple[np.ndarray, np.ndarray]:
    # h = ite_config.interaction_hamiltonian
    g = ite.g_from_exp_h(h, delta_t)
    return h, g



## ==== Main ITE Functions ==== ##

@decorators.add_stats()
def ite_per_mode(
    unit_cell:UnitCell,
    messages:MessageDictType|None,
    delta_t:float,
    logger:logs.Logger,
    config:Config,
    mode:UpdateMode
)->tuple[
    UnitCell,               # unit_cell
    MessageDictType,        # messages
    list[float],            # edge_energies
    ITEPerModeStats         # Stats
]:

    mode_tn, messages, bp_stats = _from_unit_cell_to_stable_mode(unit_cell, messages, config, logger, mode)

    ## for each edge in the mode, update the tensors
    edge_tuples = list(UpdateEdge.all_in_random_order())
    edge_energies = []

    prog_bar = get_progress_bar(config, len(edge_tuples), "Executing ITE per-mode:")
    for is_first, is_last, edge_tuple in lists.iterate_with_edge_indicators(edge_tuples):
        prog_bar.next(extra_str=f"{edge_tuple}")

        if config.visuals.live_plots:
            visuals.refresh()

        if config.ite.bp_every_edge and not is_last and not is_first:
            mode_tn, messages, bp_stats = _from_unit_cell_to_stable_mode(unit_cell, messages, config, logger, mode)

        edge_tn = reduce_tn(mode_tn, EdgeTN, trunc_dim=config.trunc_dim, edge_tuple=edge_tuple)

        # Perform ITE update:
        unit_cell, energy, env_metrics = update_unit_cell(edge_tn, unit_cell, config.ite, delta_t, logger)
        edge_energies.append(energy)

        # Update mode_tn:
        mode_tn.update_unit_cell_tensors(unit_cell)
    prog_bar.clear()

    ## Return results and statistics:
    stats = ITEPerModeStats()
    stats.bp_stats = bp_stats
    stats.env_metrics = env_metrics

    return unit_cell, messages, edge_energies, stats


@decorators.add_stats()
# @decorators.multiple_tries(3)
def ite_segment(
    unit_cell:UnitCell,
    messages:MessageDictType|None,
    delta_t:float,
    logger:logs.Logger,
    config_in:Config,
    prev_stats:ITESegmentStats
)->tuple[
    KagomeTN,           # core
    MessageDictType,    # messages
    ITESegmentStats     # stats
]:
    ## Copy and parse config:
    config = config_in.copy()
    use_prog_bar = config.visuals.progress_bars
    num_modes = config.ite.num_mode_repetitions_per_segment

    ## Init messages or use old ones
    if config.ite.start_segment_with_new_bp_message:
        messages = None  # force bp to start with fresh-new messages

    ## Generate a random mode order without repeating the same mode previous twice in a row
    modes_order = _mode_order_without_repetitions(prev_stats.modes_order, config.ite, num_modes)

    ## Keep track of stats:
    stats = ITESegmentStats()
    stats.modes_order = modes_order
    stats.delta_t = delta_t

    if use_prog_bar:  
        log_method = logger.debug
    else:          
        log_method = logger.info

    if use_prog_bar and len(modes_order)>1:  
        prog_bar = prints.ProgressBar(len(modes_order), print_prefix=f"Executing ITE Segment  ")
    else:
        prog_bar = prints.ProgressBar.inactive()


    for update_mode in modes_order:
        _mode_str = f"update_mode={update_mode.name}"
        prog_bar.next(extra_str=_mode_str)
        log_method(f"    {_mode_str}")

        ## Run:
        try:
            unit_cell, messages, edge_energies, ite_per_mode_stats = ite_per_mode(
                unit_cell, messages, delta_t, logger, config, update_mode
            )
        except BPNotConvergedError as e:
            # prints.print_warning(errors.get_traceback(e))
            raise ITEError(*e.args)
        ## Track results:
        stats.ite_per_mode_stats.append(ite_per_mode_stats)
        logger.debug(f"        Hermicity of environment={ite_per_mode_stats.env_metrics.hermicity!r}")
        logger.debug(f"        Edge-Energies={edge_energies!r}")
        ## inputs for next iteration:
        config.bp = ite_per_mode_stats.bp_stats.final_config

    prog_bar.clear()

    return unit_cell, messages, stats


# @decorators.multiple_tries(3)
def ite_per_delta_t(
    unit_cell:UnitCell, messages:MessageDictType|None, delta_t:float, num_repeats:int, config_in:Config, 
    plots:ITEPlots, logger:logs.Logger, tracker:ITEProgressTracker, segment_stats:ITESegmentStats
) -> tuple[
    KagomeTN, MessageDictType|None, bool, ITESegmentStats
    # core, messages, at_least_one_successful_run, step_stats
]:

    ## derive from input:    
    assert num_repeats>0, f"Got num_repeats={num_repeats}. We can't have ITE without repetitions"
    # Config:
    config = config_in.copy()
    # Progress bar:
    prog_bar = get_progress_bar(config, num_repeats, f"Per delta-t...         ")

    ## Perform ITE for all repetitions of this delta_t: 
    at_least_one_successful_run : bool = False
    for i in range(num_repeats):
        prog_bar.next(extra_str=f"mean-energy={segment_stats.mean_energy}")
        logger_method = print_or_log_ite_segment_progress(config, tracker, logger, delta_t, i, num_repeats, segment_stats)

        ## Preform ITE segment:
        try:
            unit_cell, messages, segment_stats = ite_segment(
                unit_cell, messages, delta_t, logger=logger, config_in=config, prev_stats=segment_stats
            )
        except ITEError as e:
            logger.warn(str(e))
            # if DEBUG_MODE:
            #     raise e
            num_errors = tracker.log_error(e)
            if num_errors>config.ite.num_errors_threshold:
                raise ITEError(f"ITE Algo experienced {num_errors} errors.")
            elif config.ite.segment_error_cause_state_revert:
                try:
                    _, energy, segment_stats, _, unit_cell, messages = tracker.revert_back(1)
                except ITEError:
                    raise e
            continue

        at_least_one_successful_run = True

        ## If bp struggled, we will use the harder config for next times:
        config.bp = segment_stats.ite_per_mode_stats[-1].bp_stats.final_config
        
        ## Calculate observables:
        energies, expectations, mean_energy = _calculate_crnt_observables(unit_cell, config, messages, segment_stats)

        ## Save data, print performance and plot graphs:
        segment_stats.mean_energy = mean_energy
        tracker.log_segment(delta_t=delta_t, energy=mean_energy, unit_cell=unit_cell, messages=messages, expectation_values=expectations, stats=segment_stats)
        plots.update(energies, segment_stats, delta_t, expectations, unit_cell)
        logger_method(f"Mean energy after sequence = {mean_energy}")

        ## Check stopping criteria:
        if config.ite.check_converges and _check_converged(tracker.energies, tracker.delta_ts, delta_t):
            break

    prog_bar.clear()

    return unit_cell, messages, at_least_one_successful_run, segment_stats


def full_ite(
    unit_cell:UnitCell|None=None,
    config:Config|None=None,
    logger:logs.Logger|None=None
)->tuple[
    KagomeTN,          # core
    ITEProgressTracker,     # ITE-Tracker
    logs.Logger             # Logger
]:

    ## Initial Settings:
    config, unit_cell, logger = _initial_full_ite_inputs(config, unit_cell, logger)
    
    ## Initial inputs for first iterations:
    step_stats = ITESegmentStats()  # initial step stats for the first iteration. used for randomized mode order
    messages = None

    ## Prepare tracking lists and plots:
    ite_tracker = ITEProgressTracker(unit_cell=unit_cell, messages=messages, config=config)
    _log_and_print_starting_message(logger, config, ite_tracker)  # Print and log valuable information: 
    plots = ITEPlots(active=config.visuals.live_plots, config=config)

    ## Calculate observables of starting core:
    if config.visuals.live_plots: 
        _compute_and_plot_zero_iteration_(unit_cell, config, logger, ite_tracker, plots)

    ## Repetitively perform ITE algo:
    delta_t_list_with_repetitions = list(lists.repeated_items(config.ite.time_steps))
    # Progress bar:
    prog_bar = get_progress_bar(config, len(delta_t_list_with_repetitions), "Executing ITE Algo...  ")
    # Main loop:
    for delta_t, num_repeats in delta_t_list_with_repetitions:
        prog_bar.next(extra_str=f"delta-t={delta_t}")
        unit_cell, messages, success, step_stats = ite_per_delta_t(unit_cell, messages, delta_t, num_repeats, config, plots, logger, ite_tracker, step_stats)
    
    ## Log finish:
    prog_bar.clear()
    _log_and_print_finish_message(logger, config, ite_tracker, plots)  # Print and log valuable information: 

    return unit_cell, ite_tracker, logger


def robust_full_ite(
    initial_core:KagomeTN|None=None,
    config:Config|None=None,
    logger:logs.Logger|None=None
)->tuple[
    KagomeTN,          # core
    ITEProgressTracker,     # ITE-Tracker
    logs.Logger             # Logger
]:
    # Get copy of inputs:
    if config is not None:
        assert isinstance(config, Config)
        config = deepcopy(config)
    if initial_core is not None:
        assert isinstance(initial_core, KagomeTN)
        initial_core = initial_core.copy()
        initial_core.normalize_tensors()

    # Multiple attempts:
    try:
        return full_ite(config=config, unit_cell=initial_core, logger=logger)
    except Exception as e:
        errors.print_traceback(e)
        # config.strengthen()
        return full_ite(config=config, unit_cell=None, logger=logger)
    


if __name__ == "__main__":
    from scripts.run_heisenberg import main
    main()