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
from _config_reader import DEBUG_MODE

# Import belief propagation code:
from algo.belief_propagation import belief_propagation

# Import containers needed for ite:
from containers.imaginary_time_evolution import ITEConfig, ITEProgressTracker, ITEPerModeStats, ITESegmentStats
from containers.sizes_and_dimensions import TNDimensions
from containers.density_matrices import MatrixMetrics
from containers import Config

# Import other needed types:
from enums import UpdateMode, NodeFunctionality
from containers import MessageDictType
from tensor_networks import KagomeTN, TensorNode, UnitCell
from _error_types import BPNotConvergedError, ITEError
from lattices.directions import Direction

# Other algorithms we need:
from algo.measurements import derive_xyz_expectation_values_with_tn, measure_core_energies
from algo.density_matrices import rho_ij_to_rho, calc_metrics
from libs import ITE as ite


from tensor_networks.construction import repeat_core

# For numeric stuff:
import numpy as np
from numpy.linalg import norm

# the modules we are using, by I.A.

# Import our shared utilities
from utils import tuples, lists, assertions, saveload, logs, decorators, errors, visuals, strings, dicts

# For copying config:
from copy import deepcopy

# Helper function and types for ITE:
from algo.imaginary_time_evolution._logs_and_prints import _print_or_log_bp_message, _log_and_print_finish_message, _log_and_print_starting_message, _print_or_log_ite_segment_msg, _fix_config_if_bp_struggled
from algo.imaginary_time_evolution._constants import CONVERGENCE_CHECK_LENGTH
from algo.imaginary_time_evolution._visualization import ITEPlots
from algo.imaginary_time_evolution._tn_control import kagome_tn_from_unit_cell



def _compute_and_plot_zero_iteration_(unit_cell:UnitCell, config:Config, logger:logs.Logger):
    delta_t = 0.0
    logger.info("Calculating measurements of initial core...")
    tn_open = kagome_tn_from_unit_cell(unit_cell, config.dims)
    tn_stable, messages, bp_stats = belief_propagation(tn_open, messages, deepcopy(config.bp))  # Perform BlockBP:
    tn_stable_around_core = reduce_tn_to_core_and_environment(tn_stable, config.bubblecon_trunc_dim, method=config.reduce2core_method)
    expectation_values = derive_xyz_expectation_values_with_tn(tn_stable_around_core, reduce=False)
    energies_per_site, _ = measure_core_energies(tn_stable_around_core, config.ite.interaction_hamiltonain, config.bubblecon_trunc_dim)
    energy = sum(energies_per_site)/len(energies_per_site) 

    ## Save data, print performance and plot graphs:
    ite_tracker.log_segment(delta_t=delta_t, energy=energy, unit_cell=unit_cell, messages=messages, expectation_values=expectation_values, stats=step_stats)
    plots.update(energies_per_site, step_stats, delta_t, expectation_values, _initial=True)
    logger.info(f"Mean energy at iteration 0: {energy}")

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


def _mode_order_without_repetitions(prev_order:list[UpdateMode], ite_config:ITEConfig)->list[UpdateMode]:
    if not ite_config.random_mode_order:
        return list(UpdateMode)
    new_order = list(UpdateMode.all_in_random_order())
    # Don't allow two of the same in a row
    if prev_order is not None and prev_order[-1] is new_order[0]:
        new_ind = np.random.randint(low=1, high=4)
        new_order = lists.swap_items(new_order, 0, new_ind)
    return new_order


def get_imaginary_time_evolution_operator(h:np.ndarray, delta_t:float)->tuple[np.ndarray, np.ndarray]:
    # h = ite_config.interaction_hamiltonain
    g = ite.g_from_exp_h(h, delta_t)
    return h, g



## ==== Main ITE Functions ==== ##


@decorators.add_stats()
def ite_per_mode(
    core:KagomeTN,
    messages:MessageDictType|None,
    delta_t:float,
    logger:logs.Logger,
    config:Config,
    update_mode:UpdateMode
)->tuple[
    KagomeTN,          # core
    MessageDictType,        # messages
    float,                  # edge_energy
    ITEPerModeStats         # Stats
]:

    ## Duplicate core into a big tensor-network:
    tn_open = _core_to_big_open_tn(core, config.dims)

    ## Perform BlockBP:
    tn_stable, messages, bp_stats = belief_propagation(tn_open, messages, deepcopy(config.bp), live_plots=config.live_plots)
    _print_or_log_bp_message(config.bp, config.ite.bp_not_converged_raises_error, bp_stats, logger)
    # tn_stable, messages, bp_stats = belief_propagation_pashtida(tn_open, messages, config.bp)
    # If block-bp increased the virtual dimension, the other modules must also use a higher dimension:
    config = _fix_config_if_bp_struggled(config, bp_stats, logger)

    ## Reduce to core and its close environment:   
    core1, core2, environment, tn_mode = calc_edge_environment(tn_stable, mode=update_mode, bubblecon_trunc_dim=config.bubblecon_trunc_dim, method=config.reduce2edge_method)
    # core1, core2, environment, tn_mode = get_edge_environment_pashtida(tn_stable, side=update_mode, bubblecon_trunc_dim=config.bubblecon_trunc_dim)
    
    ## Perform the update step:
    core1, core2, edge_energy, env_metrics = update_core_tensors(tn_mode, core1, core2, environment, config.ite, delta_t, logger)

    ## Create core with core tensors:
    new_core = _duplicate_to_core(core1, core2, update_mode, config)

    ## Return results and statistics:
    stats = ITEPerModeStats()
    stats.bp_stats = bp_stats
    stats.env_metrics = env_metrics

    return new_core, messages, edge_energy, stats


@decorators.add_stats()
@decorators.multiple_tries(3)
def ite_segment(
    core:KagomeTN,
    messages:MessageDictType|None,
    delta_t:float,
    logger:logs.Logger,
    config_in:Config,
    prev_stats:ITESegmentStats
)->tuple[
    KagomeTN,          # core
    MessageDictType,        # messages
    KagomeTN,          # tn_stable
    ITESegmentStats         # stats
]:

    config = deepcopy(config_in)
    ## Init messages or use old ones
    if config.ite.start_segment_with_new_bp_message:
        messages = None  # force bp to start with fresh-new messages
    ## Generate a random mode order without repeating the same mode previous twice in a row
    modes_order = _mode_order_without_repetitions(prev_stats.modes_order, config.ite)
    ## Follow stats:
    stats = ITESegmentStats()
    stats.modes_order = modes_order
    stats.delta_t = delta_t

    ## Iterate for each mode:
    num_repeats = config.ite.num_mode_repetitions_per_segment
    for i_rep in range(num_repeats):
        for update_mode in modes_order:
            # Each mode is another edge that needs to be updated:
            logger.info(f"    update_mode={update_mode.name: <4}")
            ## Run:
            core, messages, edge_energy, ite_per_mode_stats = ite_per_mode(
                core=core, messages=messages, delta_t=delta_t, logger=logger, config=config, update_mode=update_mode
            )
            ## Track results:
            stats.ite_per_mode_stats.append(ite_per_mode_stats)
            logger.debug(f"        Hermicity of environment={ite_per_mode_stats.env_metrics.hermicity!r}")
            logger.debug(f"        Edge-Energy={edge_energy!r}")
            ## inputs for next iteration:
            config.bp = deepcopy(ite_per_mode_stats.bp_stats.final_config)

    ## Calc final stable TN:
    logger.debug(f"    Calculating stable reduced TN..")    
    tn_open = _core_to_big_open_tn(core, config.dims)  # Duplicate core into a big tensor-network:
    tn_stable, messages, bp_stats = belief_propagation(tn_open, messages, deepcopy(config.bp))  # Perform BlockBP:
    _print_or_log_bp_message(config.bp, config.ite.bp_not_converged_raises_error, bp_stats, logger)

    return core, messages, tn_stable, stats


# @decorators.multiple_tries(3)
def ite_per_delta_t(
    core:KagomeTN, messages:MessageDictType|None, delta_t:float, num_repeats:int, config:Config, 
    logger:logs.Logger, tracker:ITEProgressTracker, step_stats:ITESegmentStats,
    plots:None="ITEPlots",   #TODO ITE Plots
) -> tuple[
    KagomeTN, MessageDictType|None, bool, ITESegmentStats
    # core, messages, at_least_one_succesful_run, step_stats
]:

    ## derive from input:    
    assert num_repeats>0, f"Got num_repeats={num_repeats}. We can't have ITE without repetitions"

    ## Perform ITE for all repetitions of this delta_t: 
    at_least_one_succesful_run : bool = False
    for i in range(num_repeats):
        _print_or_log_ite_segment_msg(config, tracker, logger, delta_t, i, num_repeats)

        ## Preform ITE segment:
        try:
            core, messages, tn_stable, step_stats = ite_segment(
                core, messages, delta_t, logger=logger, config_in=config, prev_stats=step_stats
            )
        except ITEError as e:
            logger.warn(str(e))
            num_errors = tracker.log_error(e)
            if num_errors>config.ite.num_errors_threshold:
                raise ITEError(f"ITE Algo experienced {num_errors} errors.")
            elif config.ite.segment_error_cause_state_revert:
                try:
                    _, energy, step_stats, _, core, messages = tracker.revert_back(1)
                except ITEError:
                    raise e
            continue

        at_least_one_succesful_run = True
        
        ## Calculate observables:
        tn_stable_around_core = reduce_tn_to_core_and_environment(tn_stable, config.bubblecon_trunc_dim, method=config.reduce2core_method)
        energies_per_site, rdms = measure_core_energies(tn_stable_around_core, config.ite.interaction_hamiltonain, config.bubblecon_trunc_dim)
        energy = sum(energies_per_site)/len(energies_per_site) 
        if config.live_plots:
            expectation_values = measure_xyz_expectation_values_with_rdms(rdms)
        else:
            expectation_values = {}

        ## Save data, print performance and plot graphs:
        tracker.log_segment(delta_t=delta_t, energy=energy, unit_cell=core, messages=messages, expectation_values=expectation_values, stats=step_stats)
        plots.update(energies_per_site, step_stats, delta_t, expectation_values)
        logger.info(f"Mean energy after sequence = {energy}")

        ## Check stopping criteria:
        if config.ite.check_converges and _check_converged(tracker.energies, tracker.delta_ts, delta_t):
            break

    return core, messages, at_least_one_succesful_run, step_stats


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
    # Config:
    if config is None:
        config = Config.derive_from_physical_dim(DEFAULT_D)
    # Unit-Cell:
    if unit_cell is None:
        unit_cell_in = UnitCell.random(d=config.dims.physical_dim, D=config.dims.virtual_dim)
    elif isinstance(unit_cell, UnitCell):
        unit_cell_in = unit_cell.copy()
    else:
        raise TypeError(f"Not an expected type for input 'initial_core' of type {type(unit_cell)!r}")
    # Logger:
    if logger is None:
        logger = logs.get_logger(verbose=config.visuals.verbose)
    elif not isinstance(logger, logs.Logger):
        raise TypeError(f"Not an expected type for input 'logger' of type {type(logger)!r}")
    
    
    ## Initial inputs for first iterations:
    step_stats = ITESegmentStats()  # initial step stats for the first iteration. used for randomized mode order
    messages = None

    ## Prepare tracking lists and plots:
    ite_tracker = ITEProgressTracker(unit_cell=unit_cell_in, messages=messages, config=config)
    _log_and_print_starting_message(logger, config, ite_tracker)  # Print and log valuable information: 
    plots = ITEPlots(active=config.visuals.live_plots, config=config)

    ## Calculate observables of starting core:
    if config.visuals.live_plots:
        _compute_and_plot_zero_iteration_(unit_cell_in, config, logger)

    ## Repetitively perform ITE algo:
    unit_cell_out = unit_cell_in  # for output type check
    for delta_t, num_repeats in lists.repeated_items(config.ite.time_steps):
        unit_cell_out, messages, success, step_stats = ite_per_delta_t(unit_cell_in, messages, delta_t, num_repeats, config, plots, logger, ite_tracker, step_stats)
        if not success:  # One more try
            unit_cell_out, messages, success, step_stats = ite_per_delta_t(unit_cell_in, None, delta_t, num_repeats, config, plots, logger, ite_tracker, step_stats)
            if not success:
                raise ITEError(f"ITE didn't work on delte={delta_t}")
        unit_cell_in = unit_cell_out

    ## Log finish:
    _log_and_print_finish_message(logger, config, ite_tracker, plots)  # Print and log valuable information: 

    return unit_cell_out, ite_tracker, logger


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
    from scripts.test_ite import main_test
    main_test()