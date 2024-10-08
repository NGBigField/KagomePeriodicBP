# Get Global-Config:
from _config_reader import DEBUG_MODE, KEEP_LOGS, ALLOW_VISUALS

# types:
from typing import Callable, NamedTuple
from _types import UnitCellValuePerEdgeDict, EnergiesOfEdgesDuringUpdateType

# Import containers needed for ite:
from containers.imaginary_time_evolution import ITEConfig, ITEPerModeStats, ITESegmentStats
from containers._ite_tracker import ITEProgressTracker
from containers import Config

# Import other needed types:
from enums import UpdateMode
from containers import MessageDictType, UpdateEdge, BestUnitCellData, UpdateEdgesOrder
from tensor_networks import KagomeTNRepeatedUnitCell, CoreTN, ModeTN, EdgeTN, UnitCell
from tensor_networks.construction import kagome_tn_from_unit_cell
from _error_types import BPNotConvergedError, ITEError

# For numeric stuff:
import numpy as np
from numpy.linalg import norm

# Import our shared utilities
from utils import lists, logs, decorators, prints, visuals, strings, processes, saveload

# For copying config:
from copy import deepcopy

# Helper function and types for ITE:
from ._logs_and_prints import print_or_log_bp_message, _log_and_print_finish_message, _log_and_print_starting_message, \
                                                            print_or_log_ite_segment_progress, get_progress_bar, formatted_delta_t_str
from ._constants import CONVERGENCE_CHECK_LENGTH, DEFAULT_PHYSICAL_DIM
from ._tn_update import ite_update_unit_cell
from visualizations.ite import ITEPlots

# Import belief propagation code:
from algo.belief_propagation import robust_belief_propagation, belief_propagation, BPStats

# Other algorithms we need:
from algo.measurements import measure_energies_and_observables_together, mean_expectation_values, MeasurementsOnUnitCell
from algo.tn_reduction import reduce_tn

# For efficient cached search:
from functools import lru_cache


class SegmentResults(NamedTuple):
    unit_cell : UnitCell 
    messages : MessageDictType|None
    energy : float
    stats : ITESegmentStats

    def is_better_than(self, other:"SegmentResults")->bool:
        return self.energy < other.energy 


def _start_process_monitoring_with_log(config:Config, common_results_name:str) -> None:
    ## Derive log path:
    monitor_logs_folder = config.io.logs.subfolder('monitor_process')
    saveload.force_folder_exists(monitor_logs_folder)
    log_fullpath = monitor_logs_folder + saveload.PATH_SEP + common_results_name + ".log"
    ## write config now:
    with open(log_fullpath, 'a') as f:
        print(config, file=f)
        print("\n"*2, file=f)
    ## Write process performance over time using a subprocess:
    processes.monitor_crnt_process(
        track_cpu=config.monitoring_system.monitor_cpu,
        track_ram=config.monitoring_system.monitor_ram,
        print_out=log_fullpath
    )

def _deal_edge_order(config:Config, delta_t:float)->UpdateEdgesOrder:
    
    ## for each edge in the mode, once
    num_edges = config.iterative_process.num_edge_repetitions_per_mode

    ## Random?
    if config.ite.random_edge_order:
        edge_tuples = list(UpdateEdge.all_in_random_order(num_edges=num_edges))
    else:
        assert num_edges==6
        edge_tuples = list(UpdateEdge.all_options())

    ## Define delta_t iterator. 
    # Symmetric? if yes, we got a special case:
    if config.ite.symmetric_second_order_trotterization:
        # Add the revered order for symmetry:
        edge_tuples += lists.reversed(edge_tuples)

        ## the middle edge should appear once with double delta_t:
        middle_index = len(edge_tuples)//2-1
        assert edge_tuples[middle_index]==edge_tuples[middle_index+1], f"Bug. The center edges should have been the same"
        edge_tuples.pop(middle_index+1)
        delta_ts = (delta_t*2 if i==middle_index else delta_t for i in range(len(edge_tuples)))

    else:
        delta_ts = (delta_t for _ in range(len(edge_tuples)))
                
    ## combine 
    edge_order = list(zip(edge_tuples, delta_ts, strict=True))
        
    return edge_order


_crnt_best_energy_cache : dict[int, float] = {}


def _get_crnt_best_energy(D:int) -> float:
    # Use cache:
    if D in _crnt_best_energy_cache:
        return _crnt_best_energy_cache[D]
    # Get from memory
    data = BestUnitCellData.load(D=D, none_if_not_exist=True)
    if data is None:
        mean_energy = np.inf
    else:    
        assert data.D == D
        mean_energy = data.mean_energy
    # Update cache:
    _crnt_best_energy_cache[D] = mean_energy
    # Return:
    return mean_energy


def _set_crnt_best_energy(D:int, mean_energy, unit_cell) -> None:
    crnt_data = BestUnitCellData(unit_cell=unit_cell, mean_energy=mean_energy, D=D)
    crnt_data.save()
    # Update cache:
    _crnt_best_energy_cache[D] = mean_energy

def _change_config_for_measurements_if_applicable(
    config:Config, messages:MessageDictType|None
)->tuple[Config, MessageDictType|None]:
    ## No change cases:
    if config.iterative_process.change_config_for_measurements_func is None:
        return config, messages
    
    ## get changed config:
    func = config.iterative_process.change_config_for_measurements_func
    assert callable(func)
    new_config = func(config.copy())

    ## Decide if we can use the same messages:
    if new_config.dims.big_lattice_size == config.dims.big_lattice_size:
        new_message = messages
    else:
        new_message = None

    return new_config, new_message


def _initial_inputs(
    config:Config|None, unit_cell:UnitCell|None, logger:logs.Logger|None, common_results_name:str|None
)->tuple[Config, UnitCell, logs.Logger, None, ITESegmentStats, ITEProgressTracker, ITEPlots]:
    # Config:
    if config is None:
        config = Config.derive_from_dimensions(DEFAULT_PHYSICAL_DIM)
    config.post_creation_fix()
    
    # Unit-Cell:
    if unit_cell is None:
        unit_cell = UnitCell.random(d=config.dims.physical_dim, D=config.dims.virtual_dim)
    elif isinstance(unit_cell, UnitCell):
        unit_cell = unit_cell.copy()
    else:
        raise TypeError(f"Not an expected type for input 'initial_core' of type {type(unit_cell)!r}")
    
    # Common name?
    if common_results_name is None:
        common_results_name = strings.time_stamp()+"_"+strings.random(5)
    
    # filename:
    if unit_cell._file_name is None:
        unit_cell._file_name = common_results_name

    # Logger:
    if logger is None:
        logger = logs.get_logger(verbose=config.visuals.verbose, write_to_file=KEEP_LOGS, filename=common_results_name, folder=config.io.logs.fullpath)
    elif not isinstance(logger, logs.Logger):
        raise TypeError(f"Not an expected type for input 'logger' of type {type(logger)!r}")
    
    # Initial inputs for first iterations:
    messages = None
    step_stats = ITESegmentStats()  # initial step stats for the first iteration. used for randomized mode order

    ## Init tracking lists and plots:
    ite_tracker = ITEProgressTracker(unit_cell=unit_cell, messages=messages, config=config, filename=common_results_name)
    plots = ITEPlots(active=config.visuals.live_plots, config=config)

    ## Monitor system performance:
    if config.monitoring_system.monitor_cpu or config.monitoring_system.monitor_ram:
        _start_process_monitoring_with_log(config, common_results_name)

    return config, unit_cell, logger, messages, step_stats, ite_tracker, plots



def _harden_bp_config_if_struggled(config:Config, bp_stats:BPStats, logger:logs.Logger) -> Config:
    if not bp_stats.success: 
        config.bp.trunc_dim = bp_stats.final_config.trunc_dim
        logger.debug(f"        config.bp.trunc_dim updated to {config.bp.trunc_dim}")
        if bp_stats.final_config.trunc_dim>=config.chi:
            config.chi = int(bp_stats.final_config.trunc_dim*1.33)
            logger.debug(f"        config.chi updated to {config.chi}")
    return config


def _calculate_unit_cell_measurements(
    unit_cell:UnitCell, config:Config, messages:MessageDictType|None
)->tuple[
    MeasurementsOnUnitCell,
    MessageDictType|None  # messages
]:
    ## Unpack inputs:
    live_plots = config.visuals.live_plots
    allow_prog_bar = config.visuals.progress_bars.is_active_at('blockBP')

    ## if unit_cell is rotated, rotate it back:
    unit_cell.force_zero_rotation()

    ## Sometimes we would like to change the configurations for measurements:
    config, messages = _change_config_for_measurements_if_applicable(config, messages)

    ## Get a new fresh tn:
    full_tn = kagome_tn_from_unit_cell(unit_cell, config.dims)
    
    ## BP:
    if config.iterative_process.use_bp:
        messages, _ = robust_belief_propagation(full_tn, messages, config.bp)
    else:
        full_tn.connect_uniform_messages()

    ## Calculate observables:
    measurements = measure_energies_and_observables_together(full_tn, config.ite.interaction_hamiltonian, config.contraction)
    
    return measurements, messages


def _compute_and_plot_zero_iteration_(unit_cell:UnitCell, config:Config, logger:logs.Logger, ite_tracker:ITEProgressTracker, plots:ITEPlots)->None:
    # Inputs:
    delta_t = 0.0
    messages = None
    segment_stats = ITESegmentStats()

    ## Get the state of the system at iteration 0:
    logger.info("Calculating measurements of initial core...")

    ## Calculate observables:
    measurements, messages = _calculate_unit_cell_measurements(unit_cell, config, messages)
    mean_energy = measurements.mean_energy

    ## Save data, print performance and plot graphs:
    ite_tracker.log_segment(delta_t, unit_cell, messages, measurements, segment_stats)
    # plots.update(energies, [], segment_stats, delta_t, expectations, unit_cell, entanglement, _initial=True)
    plots.update(measurements, energies_at_updates=None, segment_stats=segment_stats, delta_t=delta_t, unit_cell=unit_cell, _initial=True)
    logger.info(f"Mean energy at iteration 0: {mean_energy}")


def _log_per_mode_results(
    logger:logs.Logger, 
    energies_at_update_per_mode:dict[str, float], 
    ite_per_mode_stats:ITEPerModeStats, 
    config:Config
)->None:
    num_decimal = config.visuals.energies_print_decimal_point_length
    hermicity_str = f"{[metric.hermicity for metric in ite_per_mode_stats.env_metrics]!r}"
    energies_str = strings.float_list_to_str([np.real(energy) for energy in energies_at_update_per_mode.values()], num_decimals=num_decimal)
    tensor_distance_str = f"{[metric.other['update_distance'] for metric in ite_per_mode_stats.env_metrics]!r}"
    logger.debug(f"        Hermicity of environment="+hermicity_str)        
    logger.debug(f"        Edge-Energies after each update="+energies_str)
    logger.debug(f"        Tensor update distance="+tensor_distance_str)


def _pre_segment_init_params(
    unit_cell:UnitCell, messages:MessageDictType|None, prev_stats:ITESegmentStats, config_in:Config, 
    logger:logs.Logger, delta_t:float
)->tuple[
    UnitCell,  # unit_cell, 
    MessageDictType|None,  # messages, 
    ITESegmentStats,  # stats, 
    Config,  # config
    EnergiesOfEdgesDuringUpdateType,  # energies_at_updates
    prints.ProgressBar,  # prog_bar, 
    list[UpdateMode],  # modes_order, 
    Callable[[str], None],  # log_method,
    UpdateEdgesOrder  # edge_order
]:
    ## Copy and parse config:
    config = config_in.copy()
    num_modes = config.iterative_process.num_mode_repetitions_per_segment

    ## Init messages or use old ones
    if config.iterative_process.start_segment_with_new_bp_message:
        messages = None  # force bp to start with fresh-new messages

    ## Generate a random mode order without repeating the same mode previous twice in a row
    modes_order = _mode_order_without_repetitions(prev_stats.modes_order, config.ite, num_modes)

    ## Decide the order of the edges:
    edge_order = _deal_edge_order(config, delta_t)

    ## Keep track of stats:
    stats = ITESegmentStats()
    stats.modes_order = modes_order
    stats.delta_t = delta_t
    energies_at_updates : EnergiesOfEdgesDuringUpdateType = []

    prog_bar = get_progress_bar(config, len(modes_order), "Executing ITE Segment  ", 'ITE-per-segment', also_verify=len(modes_order)>1)    
    if config.visuals.progress_bars.is_active_somewhere():
        log_method = logger.debug
    else:          
        log_method = logger.info

    ## Add gaussian noise?
    if config.ite.add_gaussian_noise_fraction is not None:
        # the smallest delta_t, the smallest the noise:  #TODO check if still applicable 
        noise_fraction = config.ite.add_gaussian_noise_fraction * delta_t  
        unit_cell.add_noise(noise_fraction)

    ## Randomly rotate Unit-Cell?
    if config.iterative_process.randomly_rotate_unit_cell_between_segments:
        unit_cell = unit_cell.rotate("random")
    

    return unit_cell, messages, stats, config, energies_at_updates, prog_bar, modes_order, log_method, edge_order


def _log_plot_and_print_segment_results(
    plots:ITEPlots, tracker:ITEProgressTracker, delta_t, unit_cell, messages, logger_method, 
    segment_stats:ITESegmentStats, config,
    energies_at_updates:EnergiesOfEdgesDuringUpdateType, 
    measurements_after_segment:MeasurementsOnUnitCell
)->None:
    ## Unpack:
    mean_energy     = measurements_after_segment.mean_energy
    expectations    = measurements_after_segment.expectations
    entanglement    = measurements_after_segment.entanglement
    energies_at_end = measurements_after_segment.energies
    

    ## Tracker and plot objects:
    tracker.log_segment(delta_t=delta_t, unit_cell=unit_cell, messages=messages, measurements=measurements_after_segment, stats=segment_stats)
    plots.update(
        measurements_at_end=measurements_after_segment, 
        energies_at_updates=energies_at_updates,
        segment_stats=segment_stats,
        delta_t=delta_t,
        unit_cell=unit_cell
    )

    ## Print and log:
    num_decimals = config.visuals.energies_print_decimal_point_length
    xyz_means = mean_expectation_values(expectations)
    energies_str     = strings.float_list_to_str(list(energies_at_end.values()), num_decimals=num_decimals)
    entanglement_str = strings.float_list_to_str(list(entanglement.values())   , num_decimals=num_decimals)
    xyz_str          = strings.float_dict_to_str(xyz_means                     , num_decimals=num_decimals)
    logger_method(f"    Edge-Energies after segment =   "+energies_str)
    logger_method(f"    Edge-Negativities           =   "+entanglement_str)
    logger_method(f"    Expectation-Values          =   "+xyz_str)
    logger_method(f"Mean energy after segment = {mean_energy}")


def _post_segment_measurements_checks_and_visuals(
    config:Config, 
    unit_cell:UnitCell, 
    messages:MessageDictType|None, 
    tracker:ITEProgressTracker,
    plots:ITEPlots, 
    logger_method:Callable[[str], None],
    segment_stats:ITESegmentStats, 
    energies_at_updates:EnergiesOfEdgesDuringUpdateType,
    delta_t:float, 
    best_results:SegmentResults
)->tuple[
    bool,  # should break
    float,  # mean energy after_segment 
    UnitCell, # unit_cell 
    MessageDictType|None, # messages
    SegmentResults  # best results
]:
    should_break = False

    ## Keep unit-cell in its canonical rotation
    # if unit_cell is rotated, rotate it back:
    unit_cell.force_zero_rotation()

    ## Calculate observables:
    measurements_after_segment, messages = _calculate_unit_cell_measurements(unit_cell, config, messages)
    mean_energy = measurements_after_segment.mean_energy

    ## If bp struggled, we will use the harder config for next times:
    if config.iterative_process.keep_harder_bp_config_between_segments:
        raise NotImplementedError()

    ## Save data, print performance and plot graphs:
    segment_stats.mean_energy = mean_energy
    _log_plot_and_print_segment_results(
        plots, tracker, delta_t, unit_cell, messages, logger_method, segment_stats, config,
        energies_at_updates, measurements_after_segment
    )

    ## Check stopping criteria:
    if config.ite.check_converges and _check_converged(tracker.energies, tracker.delta_ts, delta_t):
        should_break = True

    ## Which unit cell has minimal energy in crnt full run:
    crnt_results = SegmentResults(unit_cell=unit_cell, messages=messages, energy=mean_energy, stats=segment_stats)
    if crnt_results.is_better_than(best_results):
        best_results = crnt_results

    ## Which unit cell has minimal energy ever:
    D = config.dims.virtual_dim
    crnt_best_energy = _get_crnt_best_energy(D)
    if mean_energy < crnt_best_energy:
        _set_crnt_best_energy(D, mean_energy, unit_cell)

    return should_break, mean_energy, unit_cell, messages, best_results


def _deal_with_segment_error(
    error:Exception, errors_count:int, logger:logs.Logger, tracker:ITEProgressTracker, config:Config, delta_t:float
)->bool:  # should_break
    
    should_break = False

    logger.warn(str(error)+"\n"*3)
    total_num_errors = tracker.log_error(error)
    if total_num_errors >= config.iterative_process.num_total_errors_threshold:
        raise ITEError(f"ITE Algo experienced {total_num_errors}, and will terminate.")
    
    elif errors_count >= config.iterative_process.num_errors_per_delta_t_threshold:
        logger.warn(f"ITE Algo experienced {errors_count} errors for delta_t={delta_t}, and will continue with the next delta_t.")
        should_break = True

    elif config.iterative_process.segment_error_cause_state_revert:

        raise NotImplementedError("We need to reimplement this option")
        """
        revert_length = 1
        if len(tracker) < revert_length:
            return should_break
        
        try:
            _, mean_energy, segment_stats, _, unit_cell, messages = tracker.revert_back(revert_length)
        except ITEError as tracker_error:
            logger.error(tracker_error)
            raise error
        """

    return should_break


def _check_converged(energies_in:list[float], delta_ts:list[float], crnt_delta_t:float)->bool:

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
    unit_cell:UnitCell, messages:MessageDictType|None, config:Config, logger:logs.Logger, mode:UpdateMode
)->tuple[
    ModeTN, MessageDictType|None, BPStats, Config
]:
    ## Duplicate core into a big tensor-network:
    full_tn = kagome_tn_from_unit_cell(unit_cell, config.dims)
    if DEBUG_MODE: full_tn.validate()

    ## Perform BlockBP:
    if config.iterative_process.use_bp:
        messages, bp_stats = robust_belief_propagation(full_tn, messages, config.bp)

        print_or_log_bp_message(config.bp, config.iterative_process.bp_not_converged_raises_error, bp_stats, logger)

        # If block-bp struggled and increased the virtual dimension, the following iterations must also use a higher dimension:
        config = _harden_bp_config_if_struggled(config, bp_stats, logger)
    else:
        bp_stats = BPStats()
        full_tn.connect_uniform_messages()

    ## Contract to mode:
    mode_tn = reduce_tn(full_tn, ModeTN, contract_config=config.contraction, mode=mode)
    return mode_tn, messages, bp_stats, config



## ==== Main ITE Functions ==== ##

@decorators.add_stats()
def ite_per_mode(
    unit_cell:UnitCell,
    messages:MessageDictType|None,
    logger:logs.Logger,
    config:Config,
    mode:UpdateMode,
    edge_order:UpdateEdgesOrder
)->tuple[
    UnitCell,               # unit_cell
    MessageDictType|None,   # messages
    dict[str, float],       # edge_energies
    ITEPerModeStats         # Stats
]:
    ## prepare statistics and results:
    stats = ITEPerModeStats()
    edge_energies = dict()
    mode_tn : ModeTN

    prog_bar = get_progress_bar(config, len(edge_order), "Executing ITE per-mode:", 'ITE-per-mode')
    for is_first, is_last, (edge_tuple, delta_t) in lists.iterate_with_edge_indicators(list(edge_order)):
        prog_bar.next(extra_str=f"{edge_tuple}")

        if config.visuals.live_plots:
            visuals.refresh()

        if config.iterative_process.bp_every_edge or is_first:
            # Perform BlockBP again, to get converged messages.
            mode_tn, messages, bp_stats, config = _from_unit_cell_to_stable_mode(unit_cell, messages, config, logger, mode)
        else:
            # Just update the tensors 
            mode_tn.update_unit_cell_tensors(unit_cell)

        edge_tn = reduce_tn(mode_tn, EdgeTN, contract_config=config.contraction, edge_tuple=edge_tuple, arange_legs=False)
        permutation_orders = edge_tn.rearrange_tensors_and_legs_into_canonical_order()

        # Perform ITE update:
        old_cell = unit_cell.copy()
        unit_cell, energy, env_metrics = ite_update_unit_cell(edge_tn, unit_cell, permutation_orders, config.ite, delta_t, logger)
        env_metrics.other["update_distance"] = unit_cell.distance_from(old_cell)

        # keep stats:
        edge_energies[str(edge_tuple)] = energy
        stats.env_metrics.append(env_metrics)
        stats.bp_stats.append(bp_stats)

    prog_bar.clear()


    return unit_cell, messages, edge_energies, stats


@decorators.add_stats(memory_usage=True)
def ite_per_segment(
    unit_cell:UnitCell,
    messages:MessageDictType|None,
    delta_t:float,
    logger:logs.Logger,
    config_in:Config,
    prev_stats:ITESegmentStats
)->tuple[
    UnitCell,  # UnitCell
    MessageDictType|None,  # messages
    EnergiesOfEdgesDuringUpdateType,  # edge_energies_at_updates
    ITESegmentStats  # stats
]:

    unit_cell, messages, stats, config, energies_at_updates, \
        prog_bar, modes_order, log_method, edge_order = _pre_segment_init_params(
        unit_cell, messages, prev_stats, config_in, logger, delta_t
    )

    for update_mode in modes_order:
        _mode_str = f"update_mode={update_mode.name}"
        prog_bar.next(extra_str=_mode_str)
        log_method(f"    {_mode_str}")

        ## Run:
        try:
            unit_cell, messages, energies_at_update_per_mode, ite_per_mode_stats = ite_per_mode(
                unit_cell, messages, logger, config, update_mode, edge_order
            )
        except BPNotConvergedError as e:
            prog_bar.clear()
            raise ITEError(*e.args)
        
        ## Track and log results:
        stats.ite_per_mode_stats.append(ite_per_mode_stats)
        energies_at_updates.append(energies_at_update_per_mode)
        _log_per_mode_results(logger, energies_at_update_per_mode, ite_per_mode_stats, config=config)

    prog_bar.clear()

    return unit_cell, messages, energies_at_updates, stats


def ite_per_delta_t(
    unit_cell:UnitCell, messages:MessageDictType|None, delta_t:float, num_repeats:int, config_in:Config, 
    plots:ITEPlots, logger:logs.Logger, tracker:ITEProgressTracker, segment_stats:ITESegmentStats
) -> tuple[
    float,  # Mean energy
    UnitCell,  # Unit Cell
    MessageDictType|None, 
    bool, 
    ITESegmentStats
    # core, messages, at_least_one_successful_run, step_stats
]:

    ## derive from input:    
    assert num_repeats>0, f"Got num_repeats={num_repeats}. We can't have ITE without repetitions"
    # Config:
    config = config_in.copy()
    use_lowest_energy_results = config.ite.always_use_lowest_energy_state
    # Progress bar:
    prog_bar = get_progress_bar(config, num_repeats, f"Per delta-t...         ", level='ITE-per-delta-t')
    # track success:
    at_least_one_successful_run : bool = False
    errors_count : int = 0
    # track "best" unit-cell:
    best_results = SegmentResults(unit_cell=unit_cell, messages=messages, energy=np.inf, stats=segment_stats)

    ## Perform ITE for all repetitions of this delta_t: 
    for i in range(num_repeats):
        prog_bar.next(extra_str=f"mean-energy={segment_stats.mean_energy}")
        logger_method = print_or_log_ite_segment_progress(config, tracker, logger, delta_t, i, num_repeats, segment_stats)

        ## Preform ITE segment:
        try:
            unit_cell, messages, energies_at_updates, segment_stats = ite_per_segment(
                unit_cell, messages, delta_t, logger=logger, config_in=config, prev_stats=segment_stats
            )
        except ITEError as error:
            errors_count += 1
            should_break = _deal_with_segment_error(error=error, errors_count=errors_count, logger=logger, tracker=tracker, config=config, delta_t=delta_t)
            if should_break:
                break
            else:
                continue
        else:
            at_least_one_successful_run = True

        ## Measurements, checks and visuals:
        should_break, mean_energy, unit_cell, messages, best_results = _post_segment_measurements_checks_and_visuals(
            config=config, unit_cell=unit_cell, messages=messages, tracker=tracker, plots=plots, logger_method=logger_method, 
            segment_stats=segment_stats, energies_at_updates=energies_at_updates, delta_t=delta_t, best_results=best_results
        )
        if should_break:
            break

    if use_lowest_energy_results:
        unit_cell, messages, mean_energy, segment_stats = best_results
        

    prog_bar.clear()

    return mean_energy, unit_cell, messages, at_least_one_successful_run, segment_stats


def full_ite(
    unit_cell:UnitCell|None=None,
    config:Config|None=None,
    logger:logs.Logger|None=None,
    common_results_name:str|None=None
)->tuple[
    float,                  # energy
    UnitCell,               # core
    ITEProgressTracker,     # ITE-Tracker
    logs.Logger             # Logger
]:

    ## Initial Settings inputs and visuals:
    config, unit_cell, logger, messages, step_stats, ite_tracker, plots = _initial_inputs(
        config, unit_cell, logger, common_results_name
    )
    _log_and_print_starting_message(logger, config, ite_tracker, unit_cell)  # Print and log valuable information: 

    ## Calculate observables of starting core:
    if config.visuals.live_plots: 
        _compute_and_plot_zero_iteration_(unit_cell, config, logger, ite_tracker, plots)

    ## Repetitively perform ITE algo:
    delta_t_list_with_repetitions = list(lists.repeated_items(config.ite.time_steps))
    # Progress bar:
    prog_bar = get_progress_bar(config, len(delta_t_list_with_repetitions), "Executing ITE Algo...  ", level='ITE-Main')
    # Main loop:
    for delta_t, num_repeats in delta_t_list_with_repetitions:
        prog_bar.next(extra_str="delta-t="+formatted_delta_t_str(delta_t))
        try:
            mean_energy, unit_cell, messages, success, step_stats = ite_per_delta_t(unit_cell, messages, delta_t, num_repeats, config, plots, logger, ite_tracker, step_stats)  
        except Exception as e:
            _log_and_print_finish_message(logger, config, ite_tracker, plots)  # Print and log valuable information: 
            raise e
        
    ## finish:
    prog_bar.clear()
    _log_and_print_finish_message(logger, config, ite_tracker, plots)  # Print and log valuable information: 

    return mean_energy, unit_cell, ite_tracker, logger


if __name__ == "__main__":
    from scripts.run_ite import main
    main()