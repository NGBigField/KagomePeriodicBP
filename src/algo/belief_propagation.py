if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)
	sys.path.append(
		pathlib.Path(__file__).parent.parent.parent.__str__()
	)

# Get Global-Config:
from _config_reader import DEBUG_MODE

# Everyone needs numpy:
import numpy as np

# other used types in our code:
from tensor_networks import KagomeTN, MPS
from lattices.directions import BlockSide
from enums import ContractionDepth
from containers import BPStats, BPConfig, MessageDictType, Message

# needed algo:
from algo.contract_tensor_network import contract_tensor_network

# for ite stuff:
from libs import ITE as ite

# Common used utilities:
from utils import decorators, parallel_exec, lists, visuals, prints

# OOP:
from copy import deepcopy


def _out_going_message(
    tn:KagomeTN, direction:BlockSide, bubblecon_trunc_dim:int, print_progress:bool, hermitize:bool
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

    ## Force messages to be hermitian:
    if hermitize:
        mps = ite.hermitize_a_message(mps)

    ## Make left canonical and normalize right-most tensor:
    mps.left_canonical_QR()
    n = mps.N
    t = mps.A[n-1]
    mps.set_site(t/np.linalg.norm(t), n-1)

    ## Out message:
    return Message(mps, mps_direction)


def _bp_error_str(error:float|None):
    return f"{error}" if error is not None else "NaN"


def _verify_copy_messages(tn:KagomeTN, prev_messages:dict[BlockSide, Message])->None:
    _copy_distances = []
    for _side in BlockSide.all_in_counter_clockwise_order():
        _a = prev_messages[_side].mps
        _b = tn.messages[_side].mps
        _copy_distances.append(MPS.l2_distance(_a, _b))
    assert sum(_copy_distances) < 1e-14, "Copying messages has a numeric bug!"


def _belief_propagation_step(
    tn:KagomeTN,
    prev_error:float|None,
    prog_bar:prints.ProgressBar,
    bp_config:BPConfig
)->tuple[
    KagomeTN,  # next_tn_with_messages
    MessageDictType,   # next_messages
    float           # next_eerror
]:

    ## Keep old messages for comparison (error calculation):
    prev_messages = tn.messages

    ## Compute out-going message for all possible sizes:
    # prepare inputs:
    fixed_arguments = dict(tn=tn, bubblecon_trunc_dim=bp_config.max_swallowing_dim, hermitize=bp_config.hermitize_messages_between_iterations)
    # The message going-out to the left returns from the right as the new incoming message:
    multi_processing = bp_config.parallel_computing and (
        tn.tensor_dims.virtual>2 or bp_config.max_swallowing_dim>=16
    )
    if multi_processing:
        prog_bar.append_extra_str(f" error={_bp_error_str(prev_error)}")
        directions=BlockSide.all_in_counter_clockwise_order()
        fixed_arguments["print_progress"]=False
        out_messages = parallel_exec.parallel(func=_out_going_message, values=directions, value_name="direction", fixed_arguments=fixed_arguments) 
    else:
        directions=BlockSide.iterator_with_str_output(lambda s: prog_bar.append_extra_str(s+f" error={_bp_error_str(prev_error)}"))
        fixed_arguments["print_progress"] = True
        out_messages = parallel_exec.concurrent(func=_out_going_message, values=directions, value_name="direction", fixed_arguments=fixed_arguments) 

    ## Next incoming messages are the outgoing messages after applying periodic boundaries:
    next_messages = {direction.opposite() : message for direction, message in out_messages.items() }
    
    ## Connect new incoming massages to tensor-network:
    tn.connect_messages(next_messages)

    ## Check error between messages:
    # The error is the average L_2 distance divided by the total number of coordinates if we stack all messages as one huge vector:
    distances = [ 
        MPS.l2_distance(prev_messages[dir].mps, next_messages[dir].mps) 
        for dir in BlockSide.all_in_counter_clockwise_order() 
    ]
    if bp_config.msg_diff_squared:
        next_error = sum(distances)/len(distances)
    else:
        next_error = np.sqrt( sum(distances) )/len(distances)

    return tn, next_messages, next_error


@decorators.add_stats()
def belief_propagation(
    tn:KagomeTN, 
    messages:MessageDictType|None=None, # initial messages
    bp_config:BPConfig=BPConfig(),
    live_plots:bool=False
) -> tuple[ 
    KagomeTN,
    MessageDictType, # final messages
    BPStats
]:

    ## Unpack Configuration:
    max_iterations = bp_config.max_iterations
    max_error = bp_config.target_msg_diff
    n_failure_check_len = bp_config.times_to_deem_failure_when_diff_increases

    ## Connect or randomize messages:
    if messages is None:
        tn.connect_random_messages()
        messages = tn.messages
    else:
        tn.connect_messages(messages)
    if DEBUG_MODE: tn.validate()

    ## Visualizations:
    if max_iterations is None:  prog_bar = prints.ProgressBar.unlimited( "Performing BlockBP...  ")
    else:                       prog_bar = prints.ProgressBar(max_iterations, "Performing BlockBP...  ")

    ## Initial values (In case no iteration will perform, these are the default values)
    error = None  
    i = 0
    success  : bool = False
    
    ## Track last errors to see if converged
    errors = []
    
    ## Track best result in-case failutre with erorr increasing:
    min_error = 1.0
    min_messages = messages
 
    ## Compute outgoing messages until max_iterations or max_error:
    for i in prog_bar:
               
        # Preform BP step:
        tn, messages, error = _belief_propagation_step(
            tn=tn, prev_error=error, 
            prog_bar=prog_bar, bp_config=bp_config
        )
        
        # Check success conditions:
        if error<max_error:
            success = True
            break
        if error<min_error:
            min_error = error
            min_messages = deepcopy(messages)
        
        if live_plots:
            visuals.refresh()

        # Check premature-failure conditions:
        errors.append(error)
        if len(errors)>n_failure_check_len and lists.is_sorted(errors[-n_failure_check_len:]):  # Check if all last 3 items are in increasing order
            break
    
    prog_bar.clear()
    assert isinstance(error, float)

    ## Finish by check success:                    
    stats = BPStats(iterations=i, final_error=error, final_config=bp_config)  
    if not success :
        if bp_config.allowed_retries > 0 :
            # Retry:
            bp_config.allowed_retries -= 1
            bp_config.max_swallowing_dim *= 2
            if isinstance(bp_config.max_iterations, int):
                bp_config.max_iterations += 10
            messages = None
            tn_with_messages, messages, stats = belief_propagation(tn, messages, bp_config)  # Try again with initial messages
            stats.attempts += 1
        
        else:
            messages = min_messages
            tn.connect_messages(messages)

    # Check result and finish:
    if DEBUG_MODE: tn.validate()
    return tn, messages, stats
    


def _test():

    from utils import saveload, assertions


    def _physical_tensor_with_split_mid_leg(t:np.ndarray)->np.ndarray:
        # Basic data:
        old_shape : tuple[int, ...] = t.shape
        assert len(old_shape)==3        
        half_mid_d = assertions.integer( np.sqrt(old_shape[1]) )
        # new open tensor:
        new_shape = [old_shape[0]] + [half_mid_d]*2 + [old_shape[2]]
        physical_m = t.reshape(*new_shape)
        #
        return physical_m



    d = saveload.load("Block-BP inside step 3")
    open_tn = d["open_tn"]
    prev_messages = d["prev_messages"]
    next_messages = d["next_messages"]

    up_message = prev_messages[Direction.Up][0]
    up_open_tensors = [_physical_tensor_with_split_mid_leg(t) for t in up_message.A]




if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import test_bp
    test_bp.main_test()
