
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

## Get config:
from utils.config import DEBUG_MODE, ALLOW_VISUALS, VERBOSE_MODE

from lib.ncon import ncon

# Import belief propagation code:
from algo.belief_propagation import belief_propagation, MPS, BPConfig, BPStats

# Import containers of ite:
from containers.imaginary_time_evolution import ITEConfig, ITEProgressTracker, ITEPerModeStats, ITESegmentStats
from containers.sizes_and_dimensions import TNSizesAndDimensions
from containers.density_matrices import MatrixMetrics
from containers import Config

# Import need types:
from enums import UpdateModes, Directions, InitialTNMode
from tensor_networks import KagomeTensorNetwork, NodeFunctionality, TensorNode

# Common errors:
from lib.ITE import ITEError
from _error_types import BPNotConvergedError

# Other needed algorithms:
from algo.tensor_network import get_common_edge_legs, calc_edge_environment, reduce_tn_to_core_and_environment
from algo.core_measurements import measure_xyz_expectation_values_with_tn, measure_core_energies, measure_xyz_expectation_values_with_rdms
from algo.density_matrices import rho_ij_to_rho, calc_metrics

from tensor_networks.construction import create_core, repeat_core

# For numeric stuff:
import numpy as np
from numpy.linalg import norm

# the modules we are using, by I.A.
from lib import ITE as ite

# Import our shared utilities
from utils import tuples, lists, assertions, saveload, logs, decorators, errors, visuals, strings, dicts

# For copying config:
from copy import deepcopy

# For useful tracking plots:
from visualizations.ite import ITEPlots


## ==== Constants ==== ##
ENV_HERMICITY_THRESHOLD = 1e-5
EPSILON = 1e-5
CONVERGENCE_CHECK_LENGTH = 3
DEFAULT_D = 3

## ==== Helper Types ==== ##
_BPMessagesType = dict[Directions, tuple[MPS, Directions]]


## ==== Helper Functions ==== ##
def _common_logger_prints(logger:logs.Logger, config:Config, ite_tracker:ITEProgressTracker)->None:
    logger.info(config)
    logger.info(f"ITE-Tracker saved at {ite_tracker.full_path!r}")
    for handler in logger.handlers:
        if isinstance(handler, logs.logging.FileHandler):
            path = handler.baseFilename
            logger.info(f"Logger saved at      {path!r}")    


def _log_and_print_starting_message(logger:logs.Logger, config:Config, ite_tracker:ITEProgressTracker)->None:
    _common_logger_prints(logger, config, ite_tracker)
    logger.info(" ")


def _log_and_print_finish_message(logger:logs.Logger, config:Config, ite_tracker:ITEProgressTracker, plots:ITEPlots)->None:
    logger.info("\n")
    _common_logger_prints(logger, config, ite_tracker)
    plots.save(logger)
    logger.info("")
    expectation_values = ite_tracker.expectation_values[-1]
    energy = ite_tracker.energies[-1]
    logger.info(f"Last measurements are: \n  energy={energy} \n  expectation_values={expectation_values}")
    logger.info("\n")
    logger.info("=== Finished Full-ITE ===")
    logger.info("\n")


def _print_or_log_ite_segment_msg(
    config:Config, tracker:ITEProgressTracker, logger:logs.Logger,
    delta_t:float, i:int, num_repeats:int
)->None:
    counter = 0
    for delte_t_, num_repeats_ in lists.repeated_items(config.ite.time_steps):
        if abs(delte_t_-delta_t)<1e-10 and num_repeats_==num_repeats:
            counter += i 
            break
        else:
            counter += num_repeats_
    else:  # not found
        raise ValueError(f"Bug: delta_t={delta_t} was not found in config.ite.time_steps.")

    num_segments = len(config.ite.time_steps)
    logger.info(" ")
    logger.info("segment: "+strings.num_out_of_num(counter+1, num_segments)+f" ; delta_t={delta_t}: "+strings.num_out_of_num(i+1, num_repeats))
    logger.info("------------------------------")


def _print_or_log_bp_message(config:BPConfig, not_converged_causes_error:bool, stats:BPStats, logger:logs.Logger):
    space = "        "
    _blue_text = lambda s: strings.add_color(s, strings.PrintColors.BLUE)
    if stats.final_error<config.target_msg_diff:
        if stats.attempts==1:
            _attempt_msg = f"Block-BP Converged at "\
                +_blue_text("first attempt")\
                +f" before its last iteration! Error={stats.final_error:e}"
        else:
            _attempt_msg = f"Block-BP Converged at "\
                +_blue_text(f"attempt #{stats.attempts}")\
                +f". Error={stats.final_error:e}"
        _iter_msg = f", Iteration "+ _blue_text(f"{stats.iterations+1} out of {stats.final_config.max_iterations}")
        logger.debug(space+_attempt_msg+_iter_msg)
    else:
        _msg = f"BlockBP didn't converge to error {config.target_msg_diff}. error is now {stats.final_error}"
        if not_converged_causes_error:
            raise BPNotConvergedError(_msg)
        else:
            logger.warn(space+_msg)


def _core_to_big_open_tn(core:KagomeTensorNetwork, tn_config:TNSizesAndDimensions) -> KagomeTensorNetwork:
    assert tn_config.core_size == core.original_lattice_dims[0] == core.original_lattice_dims[1]
    repeats = assertions.odd(tn_config.big_lattice_size/tn_config.core_size)
    return repeat_core(core, repeats=repeats)


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


def _mode_order_without_repetitions(prev_order:list[UpdateModes], ite_config:ITEConfig)->list[UpdateModes]:
    if not ite_config.random_mode_order:
        return list(UpdateModes)
    new_order = list(UpdateModes.all_in_random_order())
    # Don't allow two of the same in a row
    if prev_order is not None and prev_order[-1] is new_order[0]:
        new_ind = np.random.randint(low=1, high=4)
        new_order = lists.swap_items(new_order, 0, new_ind)
    return new_order


def get_imaginary_time_evolution_operator(h:np.ndarray, delta_t:float)->tuple[np.ndarray, np.ndarray]:
    # h = ite_config.interaction_hamiltonain
    g = ite.g_from_exp_h(h, delta_t)
    return h, g


def _duplicate_to_core(core1:TensorNode, core2:TensorNode, update_mode:UpdateModes, config:Config)->KagomeTensorNetwork:
    """ Arrange 2 cell tensors into a 2x2 core.

    core tensor network is of basic cell
    [ a  b ]
    [ b  a ]
    According to `update_mode`, these tensors are arranged to a tensor-network of type `TensorNetwork`.

    Args:
        core1 (Node)
        core2 (Node)
        update_mode (UpdateMode)

    Returns:
        TensorNetwork
    """
    ## Get basic info:
    i1, i2 = get_common_edge_legs(core1, core2)
    assert core1.dims[i1] == core2.dims[i2]
    common_edge_dim = assertions.integer(np.sqrt(core1.dims[i1]))
    assert core1.physical_tensor is not None
    physical_dim = core1.physical_tensor.shape[0]
    assert common_edge_dim == config.tn.virtual_dim
    assert physical_dim == config.tn.physical_dim

    ## permute legs back to canonical ordering:
    for node in [core1, core2]:
        # node.plot()
        perm = [node.directions.index(dir) for dir in Directions.standard_order()]
        node.permute(perm)
        # node.plot()

    ## arrange tensors in list according to mode:
    p1 = core1.physical_tensor
    p2 = core2.physical_tensor
    assert p1 is not None and p2 is not None
    match update_mode:
        case UpdateModes.Up | UpdateModes.Down:
            a, b = p1, p2
        case UpdateModes.Right | UpdateModes.Left:
            a, b = p2, p1
        case _:
            raise ValueError(f"Not a legit case {update_mode!r}")
    peps_list = [a, b, b, a]

    ## Create 2x2 core tensor network:
    core = create_core(config.tn, creation_mode=peps_list)
    if DEBUG_MODE: core.validate()

    return core



def _calc_environment_equivalent_matrix(environment_tensors:list[np.ndarray]) -> np.ndarray:

    ## Parse inputs:
    num_tensors = len(environment_tensors)
    d_physical = environment_tensors[0].shape[2]
    d_virutal  = environment_tensors[0].shape[0]

    ## Prepare ncon inputs:
    tensors_ids = [i for i, _ in enumerate(environment_tensors)]
    edge_list = []

    for prev, crnt, next in lists.iterate_with_periodic_prev_next_items(tensors_ids):
        tensor_edges = []
        tensor_edges.append(crnt)
        tensor_edges.append(-100-crnt)
        tensor_edges.append(-200-crnt)
        tensor_edges.append(next)

        edge_list.append(tensor_edges)

    ## Get matrix from using ncon:
    env_tensor = ncon(environment_tensors, edge_list)
    assert isinstance(env_tensor, np.ndarray)
    assert len(env_tensor.shape) == num_tensors*2
    assert env_tensor.shape[0] == d_physical
    mat_d = d_physical**num_tensors

    perm_list = []
    half_size = len(env_tensor.shape)//2
    for i in range(half_size):
        perm_list.append(i)
        perm_list.append(i+half_size)

    m = env_tensor.copy()
    # m = m.transpose(perm_list)
    m = m.reshape([mat_d, mat_d])
    # m /= norm(m)

    return m


def _fix_config_if_bp_struggled(config:Config, bp_stats:BPStats, logger:logs.Logger):
    if bp_stats.attempts>1: 
        config.bp.max_swallowing_dim = bp_stats.final_config.max_swallowing_dim
        logger.debug(f"        config.bp.max_swallowing_dim updated to {config.bp.max_swallowing_dim}")
        if bp_stats.final_config.max_swallowing_dim>=config.bubblecon_trunc_dim:
            config.bubblecon_trunc_dim = int(bp_stats.final_config.max_swallowing_dim*1.5)
            logger.debug(f"        config.bubblecon_trunc_dim updated to {config.bubblecon_trunc_dim}")
    return config


def update_core_tensors(
    mode_tn:KagomeTensorNetwork,
    core1:TensorNode,
    core2:TensorNode,
    environment_tensors:list[np.ndarray],
    ite_config:ITEConfig,
    delta_t:float,
    logger:logs.Logger
)->tuple[
    TensorNode,  # core1
    TensorNode,  # core2 
    float, # energy
    MatrixMetrics # env_data
]:

    # environment_mat = _calc_environment_equivalent_matrix(environment_tensors)
    # env_metrics = calc_metrics(environment_mat)

    ## Collect data:
    #  Get Time Evolution Operator
    h, g = get_imaginary_time_evolution_operator(ite_config.interaction_hamiltonain, delta_t)
    # dimensions:
    d_virtual = mode_tn.tensor_dims.virtual
    # Get physical core tensors:
    ti = core1.physical_tensor
    tj = core2.physical_tensor
    assert isinstance(ti, np.ndarray)
    assert isinstance(tj, np.ndarray)

    ## Get metrics
    rdm_before = ite.rho_ij(ti, tj, mps_env=environment_tensors)
    env_metrics = calc_metrics(rho_ij_to_rho(rdm_before))    

    ## Check state:
    if env_metrics.hermicity>ENV_HERMICITY_THRESHOLD:
        raise ITEError(f"env_hermicity={env_metrics.hermicity}")
    sum_eigenvalues = assertions.real(env_metrics.sum_eigenvalues)
    if abs(sum_eigenvalues-1)>ENV_HERMICITY_THRESHOLD:
        raise ITEError(f"env is not psd. sum-eigenvalues={sum_eigenvalues}")
    if env_metrics.negativity>0.1:
        raise ITEError(f"env is not psd. negativity={env_metrics.negativity}")    

    ## Update tensors:
    new_ti, new_tj = ite.apply_2local_gate( g=g, Dmax=d_virtual, Ti=ti, Tj=tj, mps_env=environment_tensors )    

    ## Calc energy and updated env metrics:
    rdm_after = ite.rho_ij(new_ti, new_tj, mps_env=environment_tensors)
    energy_after  = np.dot(rdm_after.flatten(),  h.flatten())
    env_metrics = calc_metrics(rho_ij_to_rho(rdm_after))    

    ## Check change
    if DEBUG_MODE and VERBOSE_MODE:
        #
        rdm_diff = norm(rdm_after-rdm_before)/norm(rdm_before)
        s_ = strings.add_color(f"rdm_diff={rdm_diff}", strings.PrintColors.GREEN)
        print(f"        "+s_)
        #
        ti_diff = norm(new_ti-ti)/norm(ti)
        s_ = strings.add_color(f"ti_diff ={ti_diff}", strings.PrintColors.GREEN)
        print(f"        "+s_)
        #
        tj_diff = norm(new_tj-tj)/norm(tj)
        s_ = strings.add_color(f"tj_diff ={tj_diff}", strings.PrintColors.GREEN)
        print(f"        "+s_)

    ## normalize
    new_ti = new_ti / norm(new_ti)
    new_tj = new_tj / norm(new_tj)

    ## keep new data in nodes:
    core1.tensor = new_ti
    core2.tensor = new_tj
    assert core1.is_ket == True
    assert core2.is_ket == True

    return core1, core2, energy_after, env_metrics


## ==== Main ITE Functions ==== ##


@decorators.add_stats()
def ite_per_mode(
    core:KagomeTensorNetwork,
    messages:_BPMessagesType|None,
    delta_t:float,
    logger:logs.Logger,
    config:Config,
    update_mode:UpdateModes
)->tuple[
    KagomeTensorNetwork,          # core
    _BPMessagesType,        # messages
    float,                  # edge_energy
    ITEPerModeStats         # Stats
]:

    ## Duplicate core into a big tensor-network:
    tn_open = _core_to_big_open_tn(core, config.tn)

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
    core:KagomeTensorNetwork,
    messages:_BPMessagesType|None,
    delta_t:float,
    logger:logs.Logger,
    config_in:Config,
    prev_stats:ITESegmentStats
)->tuple[
    KagomeTensorNetwork,          # core
    _BPMessagesType,        # messages
    KagomeTensorNetwork,          # tn_stable
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
    tn_open = _core_to_big_open_tn(core, config.tn)  # Duplicate core into a big tensor-network:
    tn_stable, messages, bp_stats = belief_propagation(tn_open, messages, deepcopy(config.bp))  # Perform BlockBP:
    _print_or_log_bp_message(config.bp, config.ite.bp_not_converged_raises_error, bp_stats, logger)

    return core, messages, tn_stable, stats

# @decorators.multiple_tries(3)
def ite_per_delta_t(
    core:KagomeTensorNetwork, messages:_BPMessagesType|None, delta_t:float, num_repeats:int, config:Config, 
    plots:ITEPlots, logger:logs.Logger, tracker:ITEProgressTracker, step_stats:ITESegmentStats
) -> tuple[
    KagomeTensorNetwork, _BPMessagesType|None, bool, ITESegmentStats
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
        tracker.log_segment(delta_t=delta_t, energy=energy, core=core, messages=messages, expectation_values=expectation_values, stats=step_stats)
        plots.update(energies_per_site, step_stats, delta_t, expectation_values)
        logger.info(f"Mean energy after sequence = {energy}")

        ## Check stopping criteria:
        if config.ite.check_converges and _check_converged(tracker.energies, tracker.delta_ts, delta_t):
            break

    return core, messages, at_least_one_succesful_run, step_stats


def full_ite(
    initial_core:KagomeTensorNetwork|None=None,
    config:Config|None=None,
    logger:logs.Logger|None=None
)->tuple[
    KagomeTensorNetwork,          # core
    ITEProgressTracker,     # ITE-Tracker
    logs.Logger             # Logger
]:

    ## Initial Settings:
    if config is None:
        config = Config.from_D(DEFAULT_D)
    if initial_core is None:
        core_in = create_core(config.tn, creation_mode=InitialTNMode.Random, _check_core=False)
    elif isinstance(initial_core, KagomeTensorNetwork):
        core_in = initial_core.copy()
    else:
        raise TypeError(f"Not an expected type for input 'initial_core' of type {type(initial_core)!r}")
    if logger is None:
        logger = logs.get_logger()
    elif not isinstance(logger, logs.Logger):
        raise TypeError(f"Not an expected type for input 'logger' of type {type(logger)!r}")
    
    
    ## Initial inputs for first iterations:
    step_stats = ITESegmentStats()  # initial step stats for the first iteration. used for randomized mode order
    messages = None

    ## Prepare tracking lists and plots:
    ite_tracker = ITEProgressTracker(core=core_in, messages=messages, config=config)
    _log_and_print_starting_message(logger, config, ite_tracker)  # Print and log valuable information: 
    plots = ITEPlots(active=config.live_plots, config=config)

    ## Calculate observables of starting core:
    if config.live_plots:
        delta_t = 0.0
        logger.info("Calculating measurements of initial core...")
        tn_open = _core_to_big_open_tn(core_in, config.tn)
        tn_stable, messages, bp_stats = belief_propagation(tn_open, messages, deepcopy(config.bp))  # Perform BlockBP:
        tn_stable_around_core = reduce_tn_to_core_and_environment(tn_stable, config.bubblecon_trunc_dim, method=config.reduce2core_method)
        expectation_values = measure_xyz_expectation_values_with_tn(tn_stable_around_core, reduce=False)
        energies_per_site, _ = measure_core_energies(tn_stable_around_core, config.ite.interaction_hamiltonain, config.bubblecon_trunc_dim)
        energy = sum(energies_per_site)/len(energies_per_site) 

        ## Save data, print performance and plot graphs:
        ite_tracker.log_segment(delta_t=delta_t, energy=energy, core=core_in, messages=messages, expectation_values=expectation_values, stats=step_stats)
        plots.update(energies_per_site, step_stats, delta_t, expectation_values, _initial=True)
        logger.info(f"Mean energy at iteration 0: {energy}")

    ## Repetitively perform ITE algo:
    core_out = core_in  # for output type check
    for delta_t, num_repeats in lists.repeated_items(config.ite.time_steps):
        core_out, messages, success, step_stats = ite_per_delta_t(core_in, messages, delta_t, num_repeats, config, plots, logger, ite_tracker, step_stats)
        if not success:  # One more try
            core_out, messages, success, step_stats = ite_per_delta_t(core_in, None, delta_t, num_repeats, config, plots, logger, ite_tracker, step_stats)
            if not success:
                raise ITEError(f"ITE didn't work on delte={delta_t}")
        core_in = core_out

    ## Log finish:
    _log_and_print_finish_message(logger, config, ite_tracker, plots)  # Print and log valuable information: 

    return core_out, ite_tracker, logger


def robust_full_ite(
    initial_core:KagomeTensorNetwork|None=None,
    config:Config|None=None,
    logger:logs.Logger|None=None
)->tuple[
    KagomeTensorNetwork,          # core
    ITEProgressTracker,     # ITE-Tracker
    logs.Logger             # Logger
]:
    # Get copy of inputs:
    if config is not None:
        assert isinstance(config, Config)
        config = deepcopy(config)
    if initial_core is not None:
        assert isinstance(initial_core, KagomeTensorNetwork)
        initial_core = initial_core.copy()
        initial_core.normalize_tensors()

    # Multiple attempts:
    try:
        return full_ite(config=config, initial_core=initial_core, logger=logger)
    except Exception as e:
        errors.print_traceback(e)
        # config.strengthen()
        return full_ite(config=config, initial_core=None, logger=logger)


if __name__ == "__main__":
    from scripts.core_ite_test import main
    main()

