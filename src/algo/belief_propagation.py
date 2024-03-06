# For allowing tests and scripts to run while debugging this module
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )
    from project_paths import add_scripts; add_scripts()


# Get Global-Config:
from _config_reader import DEBUG_MODE

# Everyone needs numpy:
import numpy as np

# other used types in our code:
from tensor_networks import KagomeTN, MPS, mps_distance
from lattices.directions import BlockSide
from enums import ContractionDepth
from containers import BPStats, BPConfig, MessageDictType, Message

# needed algo:
from algo.contract_tensor_network import contract_tensor_network

# for ite stuff:
from libs import ITE as ite
from libs import bmpslib

# Common used utilities:
from utils import decorators, parallel_exec, lists, visuals, prints

# OOP:
from copy import deepcopy


def _hermitize_messages(messages:MessageDictType) -> MessageDictType:
    hermitized_dict = {}
    for side, message in messages.items():
        mps = ite.hermitize_a_message(message.mps)
        hermitized_dict[side] = Message(mps=mps, orientation=message.orientation)

    return hermitized_dict

def _mps_damping(old_mps:Message, new_mps:Message, damping:float, trunc_dim:int):
    inner_prod = bmpslib.mps_inner_product(new_mps, old_mps, conjB=True)

    if inner_prod.real>0:
        sign_inP = 1
    else:
        sign_inP = -1
        
    next_mps = bmpslib.add_two_MPSs(new_mps, 1-damping, old_mps, sign_inP*damping)
    
    # normalize the combined messages and make them right-canonical
    next_mps.left_canonical_QR()
    next_mps.right_canonical(maxD=trunc_dim, nr_bulk=True)

    return next_mps


def _message_damping(prev_messages:MessageDictType, out_messages:MessageDictType, damping:float|None, trunc_dim:int)->MessageDictType:

    next_messages = {}
    for side, new_message in out_messages.items():
        old_message = prev_messages[side]   
        assert old_message.orientation == new_message.orientation 
        assert old_message.orientation.open_towards == side.opposite()
        next_mps = _mps_damping(old_message.mps, new_message.mps, damping, trunc_dim)  # Apply damping per message
        next_messages[side] = Message(next_mps, new_message.orientation)

        #TODO: DEBUG
        mps_distance(old_message.mps, new_message.mps)
        mps_distance(old_message.mps, next_mps)

    return next_messages


def _out_going_message(
    tn:KagomeTN, direction:BlockSide, bubblecon_trunc_dim:int, print_progress:bool
) -> Message:
    

    ## use bubble con to compute outgoing message:
    mps, _, mps_direction = contract_tensor_network(
        tn, 
        direction, 
        bubblecon_trunc_dim=bubblecon_trunc_dim,
        depth=ContractionDepth.ToMessage,
        print_progress=print_progress
    )

    assert isinstance(mps, MPS)

    ## right canonical and normalize mantissa and exponent kept in MPS :
    mps.right_canonical(nr_bulk=True)
    mps.reset_nr()

    ## Out message:
    return Message(mps, mps_direction)


def _bp_error_str(error:float|None):
    return f"{error}" if error is not None else "NaN"


def _belief_propagation_step(
    tn:KagomeTN,
    prev_error:float|None,
    config:BPConfig,
    prog_bar_obj:prints.ProgressBar,
    allow_prog_bar:bool=True
)->tuple[
    MessageDictType,   # next_messages
    float              # next_error
]:

    ## Keep old messages for comparison (error calculation):
    prev_messages : MessageDictType = deepcopy(tn.messages)

    ## Compute out-going message for all possible sizes:
    # prepare inputs:
    fixed_arguments = dict(tn=tn, bubblecon_trunc_dim=config.max_swallowing_dim)
    # The message going-out to the left returns from the right as the new incoming message:
    # multi_processing = config.parallel_msgs and (
    #     tn.dimensions.virtual_dim>2 or config.max_swallowing_dim>=16
    # )  #TODO stricter parallel settings
    multi_processing = config.parallel_msgs 

    if not multi_processing and allow_prog_bar:
        directions=BlockSide.iterator_with_str_output(lambda s: prog_bar_obj.append_extra_str(s+f" error={_bp_error_str(prev_error)}"))
        fixed_arguments["print_progress"] = allow_prog_bar
    else:
        prog_bar_obj.append_extra_str(f" error={_bp_error_str(prev_error)}")
        directions=BlockSide.all_in_counter_clockwise_order()
        fixed_arguments["print_progress"] = False

    out_messages = parallel_exec.concurrent_or_parallel(
        func=_out_going_message, values=directions, value_name="direction", in_parallel=multi_processing, fixed_arguments=fixed_arguments
    ) 

    ## Next incoming messages are the outgoing messages after applying periodic boundaries:
    out_messages = {side.opposite() : message for side, message in out_messages.items() }
    # Apply damping
    if config.damping is None or config.damping==0:
        next_messages = out_messages
    else:
        next_messages = _message_damping(prev_messages, out_messages, config.damping, config.max_swallowing_dim)
    
    ## Connect new incoming massages to tensor-network:
    tn.connect_messages(next_messages)

    ## Check error between messages:
    # The error is the average L_2 distance divided by the total number of coordinates if we stack all messages as one huge vector:
    distances = [ 
        mps_distance(prev_messages[direction].mps, out_messages[direction].mps) 
        for direction in BlockSide.all_in_counter_clockwise_order() 
    ]
    if config.msg_diff_squared:
        next_error = sum(distances)/len(distances)
    else:
        next_error = np.sqrt( sum(distances) )/len(distances)

    return next_messages, next_error


@decorators.add_stats()
def belief_propagation(
    tn:KagomeTN, 
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
    target_error = config.target_msg_diff
    n_failure_check_len = config.times_to_deem_failure_when_diff_increases

    ## Connect or randomize messages:
    if messages is None:
        tn.connect_random_messages()
        messages = tn.messages
    else:
        tn.connect_messages(messages)

    ## Visualizations:
    if allow_prog_bar:
        if max_iterations is None:  steps_iterator = prints.ProgressBar.unlimited( "Performing BlockBP...  ")
        else:                       steps_iterator = prints.ProgressBar(max_iterations, "Performing BlockBP...  ")
    else:
        if max_iterations is None:  steps_iterator = prints.ProgressBar.inactive()
        else:                       steps_iterator = prints.ProgressBar(max_iterations, print_out=None)
        

    ## Initial values (In case no iteration will perform, these are the default values)
    error = None  
    i = 0
    success  : bool = False
    
    ## Track last errors to see if converged
    errors = []
    
    ## Track best result in-case failure with error increasing:
    min_error = np.inf
    min_messages = messages
 
    ## Compute outgoing messages until max_iterations or max_error:
    for i in steps_iterator:
               
        # Preform BP step:
        messages, error = _belief_propagation_step(tn, error, config, steps_iterator, allow_prog_bar)
        
        if update_plots_between_steps:
            visuals.refresh()

        # Check success conditions:
        if error<target_error:
            success = True
            break

        # Check if this message is better than the previois
        if error<min_error:
            min_error = error
            min_messages = deepcopy(messages)
        
        # Check premature-failure conditions:
        errors.append(error)
        if len(errors)>n_failure_check_len and lists.is_sorted(errors[-n_failure_check_len:]):  # Check if all last 3 items are in increasing order
            break
    
    steps_iterator.clear()
    assert isinstance(error, float)

    ## Hermitize messages:
    if config.hermitize_msgs:
        messages = _hermitize_messages(messages)
        
    ## Check failure:         
    if not success:
        messages = min_messages     
        tn.connect_messages(messages)

    stats = BPStats(iterations=i+1, final_error=error, final_config=config, success=success)  
  

    # Check result and finish:
    return messages, stats
    

@decorators.add_stats()
def robust_belief_propagation(
    tn:KagomeTN, 
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
    target_error = config.target_msg_diff

    ## First attempt inputs:
    messages_in = deepcopy(messages)
    min_messages = messages_in
    min_error = np.inf

    ## For each attempt, run and check success:    
    for attempt_ind in range(config.allowed_retries):
        # Run:
        messages_out, stats = belief_propagation(tn, messages_in, config, update_plots_between_steps, allow_prog_bar)

        # Check success:
        success = stats.final_error < target_error
        if success:
            break

        # Track best results:
        error = stats.final_error
        if error < min_error:
            min_error = error
            min_messages = deepcopy(messages_out)

        # Try again with better config:
        config.max_swallowing_dim *= 2
        if isinstance(config.max_iterations, int):
            config.max_iterations += 10
        messages_in = None
        
    else:  # if never had success
        messages_out = min_messages
        tn.connect_messages(min_messages)

    ## Return stats
    overall_stats = BPStats(
        attempts=attempt_ind+1, 
        iterations=stats.iterations, 
        final_error=stats.final_error, 
        final_config=stats.final_config,
        success=success
    )  

    return messages_out, overall_stats



if __name__ == "__main__":
    from scripts.test_bp import main_test
    main_test()