if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)

# Get Global-Config:
from _config_reader import DEBUG_MODE

# Everyone needs numpy:
import numpy as np

# other used types in our code:
from tensor_networks import KagomeTensorNetwork, MPS
from lattices.directions import Direction
from enums import ContractionDepth, MessageModel
from containers import BPStats, BPConfig
from _types import MsgDictType

# needed algo:
from algo.tensor_network import contract_tensor_network, fuse_messages_with_tn

# initial messages creation:
from tensor_networks.mps import init_mps_quantum

# for ite stuff:
from libs import ITE as ite

# Common used utilities:
from utils import decorators, strings, parallel_exec, lists, visuals, prints

# OOP:
from copy import deepcopy


def _out_going_message(
    tn:KagomeTensorNetwork, direction:Direction, bubblecon_trunc_dim:int, print_progress:bool, hermitize:bool
) -> tuple[
    MPS, 
    Direction
]:
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

    return mps, mps_direction


def _bp_error_str(error:float|None):
    return f"{error}" if error is not None else "NaN"


def _belief_propagation_step(
    open_tn:KagomeTensorNetwork,
    prev_error:float|None,
    prev_tn_with_messages:KagomeTensorNetwork,
    prev_messages:MsgDictType,
    prog_bar:prints.ProgressBar,
    bp_config:BPConfig
)->tuple[
    KagomeTensorNetwork,  # next_tn_with_messages
    MsgDictType,   # next_messages
    float           # next_eerror
]:
    
    ## Compute out-going message for all possible sizes:
    # prepare inputs:
    fixed_arguments = dict(tn=prev_tn_with_messages, bubblecon_trunc_dim=bp_config.max_swallowing_dim, hermitize=bp_config.hermitize_messages_between_iterations)
    # The message going-out to the left returns from the right as the new incoming message:
    multi_processing = MULTIPROCESSING and (
        open_tn.tensor_dims.virtual>2 or bp_config.max_swallowing_dim>=16
    )
    if multi_processing:
        prog_bar.append_extra_str(f" error={_bp_error_str(prev_error)}")
        directions=list(Direction)
        fixed_arguments["print_progress"]=False
        out_messages = parallel_exec.parallel(func=_out_going_message, values=directions, value_name="direction", fixed_arguments=fixed_arguments) 
    else:
        directions=Direction.iterator_with_str_output(lambda s: prog_bar.append_extra_str(s+f" error={_bp_error_str(prev_error)}"))
        fixed_arguments["print_progress"] = True
        out_messages = parallel_exec.concurrent(func=_out_going_message, values=directions, value_name="direction", fixed_arguments=fixed_arguments) 

    ## Next incoming messages are the outgoing messages after applying periodic boundaries:
    next_messages = {direction.opposite() : mps_data for direction, mps_data in out_messages.items() }
    
    ## Connect new incoming massages to tensor-network:
    next_tn_with_messages = fuse_messages_with_tn(open_tn, next_messages)

    ## Check error between messages:
    # The error is the average L_2 distance divided by the total number of coordinates if we stack all messages as one huge vector:
    distances = [ MPS.l2_distance(prev_messages[dir][0], next_messages[dir][0]) for dir in Direction ]
    if bp_config.msg_diff_squared:
        next_error = sum(distances)/len(distances)
    else:
        next_error = np.sqrt( sum(distances) )/len(distances)

    return next_tn_with_messages, next_messages, next_error





def initial_message(
	D : int,  # Physical-Lattice bond dimension
    num_edge_tensors : int,  # number of tensors in the edge of the lattice
	message_model:MessageModel|str=MessageModel.RANDOM_QUANTUM
) -> MPS:

    dims : list[int] = [D]*num_edge_tensors


    # Convert message model:
    if isinstance(message_model, MessageModel):
       pass
    elif isinstance(message_model, str):
       message_model = MessageModel(message_model)
    else:
       raise TypeError(f"Expected `initial_m_mode` to be of type <str> or <MessageModel> enum. Got {type(message_model)}")
    
    # Initial messages are uniform probability dist'
    if message_model==MessageModel.UNIFORM_CLASSIC: raise NotADirectoryError("We do not support classical messages in this code")
        
    # Initial messages are random probability dist'
    elif message_model==MessageModel.RANDOM_CLASSIC: raise NotADirectoryError("We do not support classical messages in this code")
    
    # Initial messages are uniform quantum density matrices
    elif message_model==MessageModel.UNIFORM_QUANTUM: return init_mps_quantum(dims, random=False)
        
    # Initial messages are random |v><v| quantum states
    elif message_model==MessageModel.RANDOM_QUANTUM: return init_mps_quantum(dims, random=True)

    else:
        raise ValueError("Not a valid option")



@decorators.add_stats()
def belief_propagation_pashtida(
    open_tn:KagomeTensorNetwork, 
    messages:MsgDictType|None, # initial messages
    bp_config:BPConfig
) -> tuple[ 
    KagomeTensorNetwork,
    MsgDictType, # final messages
    BPStats
]:
    """
    A wrapper for the Pashtida version of BP

    
    Input of inner-function
    	mps_list --- list of the 4 incoming MPS messages, ordered
	             counter-clockwise by: D, R, U, L
    """ 
    ## Check inputs
    assert len(open_tn.original_lattice_dims)==2
    assert open_tn.original_lattice_dims[0] == open_tn.original_lattice_dims[1]

    ## Unpack Inputs:
    # Tensor-Network data:
    T_list = open_tn.tesnors
    e_list = open_tn.edges_list
    a_list = open_tn.angles
    N = open_tn.original_lattice_dims[0]
    # Messages
    if messages is None:
        prev_BP_mps_messages = None
    else:
        prev_BP_mps_messages = [messages[direction][0] for direction in Direction.all_in_counterclockwise_order()]
    # bubblecon cofig:
    D_trunc = bp_config.max_swallowing_dim
    delta = bp_config.target_msg_diff
    max_iter = bp_config.max_iterations

    ## Parallel computing:
    mpi_comm = None

    ## Call main function  
    BP_mps_messages = robust_BP(
        N, T_list, e_list, a_list,
        prev_BP_mps_messages, D_trunc, max_iter, delta, mpi_comm
    )

    ## Pack results
    messages = {}
    for mps, direction in zip(BP_mps_messages, Direction.all_in_counterclockwise_order(), strict=True):
        messages[direction] = (mps, direction.next_counterclockwise())
    tn_with_messages = fuse_messages_with_tn(open_tn, messages) 
    tn_with_messages.validate()
    stats = BPStats( )

    return tn_with_messages, messages, stats
		


@decorators.add_stats()
def belief_propagation(
    open_tn:KagomeTensorNetwork, 
    messages:MsgDictType|None, # initial messages
    bp_config:BPConfig,
    live_plots:bool=False
) -> tuple[ 
    KagomeTensorNetwork,
    MsgDictType, # final messages
    BPStats
]:

    ## Unpack Configuration:
    max_iterations = bp_config.max_iterations
    max_error = bp_config.target_msg_diff
    n_failure_check_len = bp_config.times_to_deem_failure_when_diff_increases

    ## Derive first incoming messages:  
    if messages is None:
        virtual_dim = open_tn.tensor_dims.virtual
        num_edge_tensors = open_tn.original_lattice_dims[0]
        messages = { 
            edge_side: (initial_message(bp_config.init_msg, virtual_dim, num_edge_tensors), edge_side.next_counterclockwise() ) \
            for edge_side in Direction.all_in_counterclockwise_order()  \
        }

    ## Visualizations:
    if max_iterations is None:  prog_bar = strings.ProgressBar.unlimited( "Performing BlockBP...  ")
    else:                       prog_bar = strings.ProgressBar(max_iterations, "Performing BlockBP...  ")

    ## Start with the initial message:
    tn_with_messages = fuse_messages_with_tn(open_tn, messages)
    if DEBUG_MODE: tn_with_messages.validate()

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
        tn_with_messages, messages, error = _belief_propagation_step(
            open_tn=open_tn, prev_error=error, 
            prev_tn_with_messages=tn_with_messages, prev_messages=messages,
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
            tn_with_messages, messages, stats = belief_propagation(open_tn, messages, bp_config)  # Try again with initial messages
            stats.attempts += 1
        
        else:
            messages = min_messages
            tn_with_messages = fuse_messages_with_tn(open_tn, messages)            

    # Check result and finish:
    if DEBUG_MODE: tn_with_messages.validate()
    return tn_with_messages, messages, stats
    


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


def main_test():
    from scripts.core_ite_test import main
    main()

if __name__ == "__main__":
    # _test()
    main_test()