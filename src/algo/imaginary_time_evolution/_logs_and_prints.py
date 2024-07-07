# For type hints
from typing import Callable

# Import belief propagation code:
from algo.belief_propagation import BPConfig, BPStats

# Import containers of ite:
from containers.imaginary_time_evolution import ITEProgressTracker, ITESegmentStats
from containers import Config
from unit_cell import UnitCell
from algo.imaginary_time_evolution import _visualization 

# Common errors:
from _error_types import BPNotConvergedError

# Import our shared utilities
from utils import lists, logs, strings, prints

# For useful tracking plots:
# from algo.imaginary_time_evolution._visuals import ITEPlots  #TODO


def get_progress_bar(config:Config, num_repeats:int, print_prefix:str)->prints.ProgressBar:
    # Progress bar:
    if config.visuals.progress_bars:
        return prints.ProgressBar(num_repeats, print_prefix=print_prefix)
    else:
        return prints.ProgressBar.inactive()


def _common_logger_prints(logger:logs.Logger, config:Config, ite_tracker:ITEProgressTracker)->None:
    # Choose if debug or info (meaning if to print on screen or only to file):
    logger_method = logger.debug
    # Print basic info
    logger_method(config)
    logger_method(f"ITE-Tracker saved at {ite_tracker.full_path!r}")
    # Print where the logger is saved:
    for handler in logger.handlers:
        if isinstance(handler, logs.logging.FileHandler):
            path = handler.baseFilename
            logger.info(f"Logs saved at      {path!r}")    


def _log_and_print_starting_message(logger:logs.Logger, config:Config, ite_tracker:ITEProgressTracker, unit_cell:UnitCell)->None:
    _common_logger_prints(logger, config, ite_tracker)
    hamiltonian_func = config.ite.interaction_hamiltonian.func
    if hasattr(hamiltonian_func, "reference"):
        reference = getattr(hamiltonian_func, "reference")
        logger.debug(f"Hamiltonian's reference energy is {reference!r}")
    logger.debug(f"unit_cell file: {unit_cell._file_name}")
    logger.debug(" ")


def _log_and_print_finish_message(logger:logs.Logger, config:Config, ite_tracker:ITEProgressTracker, plots:_visualization.ITEPlots)->None: #TODO ITEPlots

    logger.info("\n")
    _common_logger_prints(logger, config, ite_tracker)
    plots.save(logger)
    logger.info("")
    
    if len(ite_tracker)<=0:
        logger.info("\n")
        logger.info("=== Unsuccessful run of Full-ITE ===")

    else:
        expectation_values = ite_tracker.expectation_values[-1]
        energy = ite_tracker.energies[-1]
        logger.info(f"Last measurements are: \n  energy={energy} \n  expectation_values={expectation_values}")
        logger.info("\n")
        logger.info("=== Finished Full-ITE ===")

    logger.info("\n")


def print_or_log_ite_segment_progress(
    config:Config, tracker:ITEProgressTracker, logger:logs.Logger,
    delta_t:float, i:int, num_repeats:int, prev_stats:ITESegmentStats, 
)->Callable[[str], None]:
    counter = 0
    for delta_t_, num_repeats_ in lists.repeated_items(config.ite.time_steps):
        if abs(delta_t_-delta_t)<1e-10 and num_repeats_==num_repeats:
            counter += i 
            break
        else:
            counter += num_repeats_
    else:  # not found
        raise ValueError(f"Bug: delta_t={delta_t} was not found in config.ite.time_steps.")

    num_segments = len(config.ite.time_steps)

    if config.visuals.progress_bars:
        logger_method = logger.debug
    else:
        logger_method = logger.info

    logger_method(" ")
    logger_method("segment: "+strings.num_out_of_num(counter+1, num_segments)+f" ; delta_t={delta_t}: "+strings.num_out_of_num(i+1, num_repeats))
    logger_method("------------------------------")
    
    return logger_method


def print_or_log_bp_message(config:BPConfig, not_converged_causes_error:bool, stats:BPStats, logger:logs.Logger):
    # Define:
    threshold_error = config.msg_diff_good_enough
    space = "        "
    _blue_text = lambda s: prints.add_color(s, prints.PrintColors.BLUE)

    # decision tree:
    if stats.final_error<threshold_error:
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
        _msg = f"BlockBP didn't converge to error {threshold_error} after {stats.attempts} attempts. Error is now {stats.final_error}"
        if not_converged_causes_error:
            raise BPNotConvergedError(_msg)
        else:
            logger.warn(space+_msg+"\n"*4)

