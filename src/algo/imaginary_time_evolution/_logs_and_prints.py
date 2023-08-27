
# Import belief propagation code:
from algo.belief_propagation import BPConfig, BPStats

# Import containers of ite:
from containers.imaginary_time_evolution import ITEProgressTracker
from containers import Config

# Common errors:
from _error_types import BPNotConvergedError

# Import our shared utilities
from utils import lists, logs, strings

# For useful tracking plots:
# from algo.imaginary_time_evolution._visuals import ITEPlots  #TODO



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


def _log_and_print_finish_message(logger:logs.Logger, config:Config, ite_tracker:ITEProgressTracker, plots:None="ITEPlots")->None: #TODO ITEPlots
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



def _fix_config_if_bp_struggled(config:Config, bp_stats:BPStats, logger:logs.Logger):
    if bp_stats.attempts>1: 
        config.bp.max_swallowing_dim = bp_stats.final_config.max_swallowing_dim
        logger.debug(f"        config.bp.max_swallowing_dim updated to {config.bp.max_swallowing_dim}")
        if bp_stats.final_config.max_swallowing_dim>=config.bubblecon_trunc_dim:
            config.bubblecon_trunc_dim = int(bp_stats.final_config.max_swallowing_dim*1.5)
            logger.debug(f"        config.bubblecon_trunc_dim updated to {config.bubblecon_trunc_dim}")
    return config