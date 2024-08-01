# For allowing tests and scripts to run while debugging this module
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )
    from project_paths import add_scripts; add_scripts()


# Everyone needs numpy:
import numpy as np

# other used types in our code:
from tensor_networks import MPS, mps_distance
from tensor_networks.abstract_classes import KagomeTensorNetwork
from lattices.directions import BlockSide
from enums import ContractionDepth
from containers import BPStats, BPConfig, MessageDictType, Message

# needed algo:
from algo.contract_tensor_network import contract_tensor_network

# for ite stuff:
from libs import ITE as ite
from libs import bmpslib

# Common used utilities:
from utils import decorators, lists, parallels, visuals, prints

# OOP:
from copy import deepcopy
from typing import Generator


def _hermitize_messages(messages:MessageDictType) -> MessageDictType:
    hermitized_dict = dict()
    for side, message in messages.items():
        mps = ite.hermitize_a_message(message.mps)
        hermitized_dict[side] = Message(mps=mps, orientation=message.orientation)

    return hermitized_dict


def _compute_error(prev_messages:MessageDictType, out_messages:MessageDictType, msg_diff_squared:bool)->float:
    # The error is the average L_2 distance divided by the total number of coordinates if we stack all messages as one huge vector:
    distances: list[float] = [ 
        mps_distance(prev_messages[direction].mps, out_messages[direction].mps) 
        for direction in BlockSide.all_in_counter_clockwise_order() 
    ]
    if msg_diff_squared:
        return sum(distances)/len(distances)
    else:
        return np.sqrt( sum(distances) )/len(distances)
    

def _single_mps_damping(old_mps:MPS, new_mps:MPS, damping:float, trunc_dim:int):
    inner_prod = bmpslib.mps_inner_product(new_mps, old_mps, conjB=True)

    if inner_prod.real>0:
        sign_inP = 1
    else:
        sign_inP = -1
        
    combined_mps = bmpslib.add_two_MPSs(new_mps, 1-damping, old_mps, sign_inP*damping)
    
    # normalize the combined messages and make them right-canonical
    combined_mps.left_canonical_QR()
    combined_mps.right_canonical(maxD=trunc_dim, nr_bulk=True)
    combined_mps.reset_nr()

    return combined_mps


def _message_damping(prev_messages:MessageDictType, out_messages:MessageDictType, damping:float, trunc_dim:int)->MessageDictType:

    next_messages = {}
    for side, new_message in out_messages.items():
        old_message = prev_messages[side]   
        assert old_message.orientation == new_message.orientation 
        assert old_message.orientation.open_towards == side.opposite()
        next_mps = _single_mps_damping(old_message.mps, new_message.mps, damping, trunc_dim)  # Apply damping per message
        next_messages[side] = Message(next_mps, new_message.orientation)

    return next_messages


def _single_outgoing_message(
    direction:BlockSide, tn:KagomeTensorNetwork, bubblecon_trunc_dim:int, print_progress:bool
) -> Message:
    
    ## use bubble con to compute outgoing message:
    mps, _, mps_direction = contract_tensor_network(
        tn, 
        direction, 
        bubblecon_trunc_dim=bubblecon_trunc_dim,
        depth=ContractionDepth.ToMessage,
        allow_progressbar=print_progress
    )

    assert isinstance(mps, MPS)

    ## Out message:
    return Message(mps, mps_direction)


def _bp_error_str(error:float|None):
    return f"{error}" if error is not None else "NaN"


def _fix_messages(messages:MessageDictType) -> None:
    """ right canonical and normalize mantissa and exponent kept in MPS """
    for key in messages.keys():
        messages[key].mps.right_canonical(nr_bulk=True)
        messages[key].mps.reset_nr() 


def _out_going_messages(
    tn:KagomeTensorNetwork, 
    config:BPConfig,
    prev_error:float|None,
    prog_bar_obj:prints.ProgressBar,
    allow_prog_bar:bool=True
)->MessageDictType:
    ## prepare inputs:
    fixed_arguments = dict(tn=tn, bubblecon_trunc_dim=config.trunc_dim)
    multi_processing = config.parallel_msgs 

    ## Different behavior between parallel or concurrent comp:
    if not multi_processing and allow_prog_bar:
        str_func = lambda s: prog_bar_obj.append_extra_str(s+f" error={_bp_error_str(prev_error)}")
        directions_iter=BlockSide.iterator_with_str_output(str_func)  # append prog-bar msg after each sub-step
        fixed_arguments["print_progress"] = allow_prog_bar
    else:
        prog_bar_obj.append_extra_str(f" error={_bp_error_str(prev_error)}")  # append prog-bar msg now
        directions_iter=BlockSide.all_in_counter_clockwise_order()
        fixed_arguments["print_progress"] = False

    # Compute outgoing messages:
    out_messages : dict[BlockSide, Message] = parallels.concurrent_or_parallel(
        func=_single_outgoing_message, values=directions_iter, value_name="direction", in_parallel=multi_processing, fixed_arguments=fixed_arguments  # type: ignore
    ) 

    ## Next incoming messages are the outgoing messages after applying periodic boundaries:
    out_messages = {side.opposite() : message for side, message in out_messages.items()}

    ## Apply message corrections:
    if config.fix_msg_each_step:
        _fix_messages(out_messages)

    return out_messages


def _belief_propagation_step(
    tn:KagomeTensorNetwork,
    prev_messages:MessageDictType,
    prev_error:float|None,
    config:BPConfig,
    prog_bar_obj:prints.ProgressBar,
    allow_prog_bar:bool=True
)->tuple[
    MessageDictType,   # out_messages
    MessageDictType,   # next_messages
    float              # next_error
]:

    ## Compute out-going message for all possible block-sides:
    out_messages = _out_going_messages(tn, config, prev_error, prog_bar_obj, allow_prog_bar)

    ## Check error between messages:
    next_error = _compute_error(prev_messages, out_messages, config.msg_diff_squared)

    ## Apply damping?
    if config.damping is None or config.damping==0:
        next_messages = out_messages
    else:
        next_messages = _message_damping(prev_messages, out_messages, config.damping, config.trunc_dim)

    return out_messages, next_messages, next_error


@decorators.add_stats()
def belief_propagation(
    tn:KagomeTensorNetwork, 
    messages:MessageDictType|None=None, # initial messages
    config:BPConfig=BPConfig(),
    update_plots_between_steps:bool=False,
    allow_prog_bar:bool=True
) -> tuple[ 
    MessageDictType, # final messages
    BPStats
]:

    ## Unpack Configuration:
    max_iterations = config.max_iterations
    terminating_error = config.msg_diff_terminate
    n_failure_check_len = config.times_to_deem_failure_when_diff_increases

    ## Connect or randomize messages:
    if messages is None:
        tn.connect_random_messages()
    else:
        tn.connect_messages(messages)
    messages = tn.messages
    
    ## Visualizations:
    if allow_prog_bar:
        if max_iterations is None:  steps_iterator = prints.ProgressBar.unlimited( "Performing BlockBP...  ")
        else:                       steps_iterator = prints.ProgressBar(max_iterations, "Performing BlockBP...  ")
    else:
        if max_iterations is None:  steps_iterator = prints.ProgressBar.inactive()
        else:                       steps_iterator = prints.ProgressBar(max_iterations, print_out=False)
        

    ## Initial values (In case no iteration will perform, these are the default values)
    error = None  
    i = 0
    success  : bool = False
    
    ## Track last errors to see if converged
    errors = []
    
    ## Track best result in-case failure with error increasing:
    min_error = np.inf
    min_messages = next_messages = messages
 
    ## Compute outgoing messages until max_iterations or max_error:
    for i in steps_iterator:
               

        ## Preform BP step:
        out_messages, next_messages, error = _belief_propagation_step(tn, next_messages, error, config, steps_iterator, allow_prog_bar)
        # out_messages are what we would return out of the algorithm
        # next_messages is a damped starting-point for the next bp_step
        
        if update_plots_between_steps:
            visuals.refresh()

        # Check success conditions:
        if error<terminating_error:
            success = True
            break

        ## Connect incoming massages to tensor-network for next step:
        tn.connect_messages(next_messages)

        # Check if this message is better than the previous 
        if error<min_error:
            min_error = error
            min_messages = deepcopy(out_messages)
        
        # Check premature-failure conditions:
        errors.append(error)
        if len(errors)>n_failure_check_len and lists.is_sorted(errors[-n_failure_check_len:]):  # Check if all last 3 items are in increasing order
            break
    
    steps_iterator.clear()
    assert isinstance(error, float)
        
    ## Check failure:         
    if not success:
        out_messages = min_messages     
        error = min_error

    ## Hermitize messages:
    if config.hermitize_msgs_when_finished:
        out_messages = _hermitize_messages(out_messages)

    ## Arrange Results
    # tn gets the output messages:
    tn.connect_messages(out_messages)
    stats = BPStats(iterations=i+1, final_error=error, final_config=config, success=success)  

    return out_messages, stats
    

@decorators.add_stats()
def robust_belief_propagation(
    tn:KagomeTensorNetwork, 
    messages:MessageDictType|None=None, # initial messages
    config:BPConfig=BPConfig(),
    update_plots_between_steps:bool=False,
    allow_prog_bar:bool=True
) -> tuple[ 
    MessageDictType, # final messages
    BPStats
]:
    ## Unpack Configuration:
    config = config.copy()  # Don't affect the sender's copy of config
    good_enough_error = config.msg_diff_good_enough
    terminating_error = config.msg_diff_terminate

    ## First attempt inputs:
    messages_in = deepcopy(messages)

    ## Track best and total outputs of individual belief_propagation rungs:
    min_messages = messages_in
    min_error = np.inf
    total_iterations = 0


    ## For each attempt, run and check success:    
    for attempt_ind in range(config.allowed_retries):
        # Run:
        messages, stats = belief_propagation(tn, messages_in, config, update_plots_between_steps, allow_prog_bar)

        # unpack:
        error = stats.final_error
        total_iterations += stats.iterations

        # Check success:
        terminating_condition = stats.final_error < terminating_error
        if terminating_condition:
            messages_out = messages
            error_out = error
            break

        # Track best results:
        if error < min_error:
            min_error = error
            min_messages = deepcopy(messages)

        # Try again with better config:
        config.trunc_dim = int(1.5*config.trunc_dim)
        if isinstance(config.max_iterations, int):
            config.max_iterations += 11
        messages_in = None
        
    else:  # if never had success
        messages_out = min_messages
        error_out = min_error

    ## Did we succeed?
    success = error_out < good_enough_error
    assert isinstance(messages_out, dict)
    tn.connect_messages(messages_out)

    ## Return stats
    overall_stats = BPStats(
        attempts=attempt_ind+1, 
        iterations=total_iterations, 
        final_error=error_out, 
        final_config=stats.final_config,
        success=success
    )  

    return messages_out, overall_stats



if __name__ == "__main__":
    pass