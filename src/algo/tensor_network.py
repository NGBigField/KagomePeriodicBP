if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)


# Control flags:
from utils.config import DEBUG_MODE, VERBOSE_MODE, MULTIPROCESSING

# Everyone needs numpy:
import numpy as np

# For type anotation:
from typing import List, Tuple, Iterable, TypeVar

# Common types in the code:
from tensor_networks import KagomeTensorNetwork, Node, NodeFunctionality, MPS, TensorNetworkError
from enums import Sides, Directions, ContractionDepth, ReduceToEdgeMethod, ReduceToCoreMethod
from containers import BubbleConConfig, ContractionConfig
from physics import pauli

# for comparison to pashtida:
from algo._pashtida import two_way_contraction_to_core, get_smallhex_2env

# Our utilities:
from utils import tuples, lists, assertions, parallel_exec

# Our needed algos:
from tensor_networks.tensor_network import get_common_edge_legs
from tensor_networks.construction import create_kagome_tn, _get_edge_from_tensor_coordinates
from algo.contraction_order import derive_contraction_orders
from algo.mps import physical_tensor_with_split_mid_leg
from lib.bubblecon import bubblecon
from lib import bmpslib
import itertools

# For energy estimation:
from lib.ITE import rho_ij


def _fix_angle(a:float)->float:
    while a < 0:
        a += 2*np.pi
    while a>2*np.pi:
        a -= 2*np.pi
    return a

def _get_corner_tensors(tn:KagomeTensorNetwork) -> list[Node]:
    min_x, max_x, min_y, max_y = tn.boundaries()
    corner_tesnors = [] 
    for x in [min_x, max_x]:
        for y in  [min_y, max_y]:
            t = tn.get_tensor_in_pos((x, y))
            corner_tesnors.append(t)    
    return corner_tesnors

def _sandwich_fused_tensors_with_expectation_values(tn_in:KagomeTensorNetwork, mat:np.matrix, ind:int, plot_:bool=False)->KagomeTensorNetwork:

    ## Get peps tensor and node data
    node = tn_in.nodes[ind]
    assert node.is_ket
    ket = node.physical_tensor
    bra = np.conj(ket)

    D = ket.shape[1]
    D2 = D*D
    
    ket_op = np.tensordot(ket, mat, axes=([0],[0]))
    res = np.tensordot(ket_op, bra, axes=([4],[0]))

    res2 = np.transpose(res, axes=[0, 4, 1, 5, 2, 6, 3, 7])
    fused_data = res2.reshape([D2, D2, D2, D2])

    ## Replace node in original tensor_network:
    tn_out = tn_in.copy()
    tn_out.nodes[ind] = Node(
        is_ket          = False,
        tensor          = fused_data,
        edges           = node.edges,
        directions      = node.directions,
        pos             = node.pos,
        index           = node.index,
        name            = node.name,
        on_boundary     = node.on_boundary,
        functionality   = node.functionality
    )
    if DEBUG_MODE: tn_out.validate()



    return tn_out



def _ordered_nodes_on_edge(tn:KagomeTensorNetwork, edge_side:Sides, mps_order_dir:Directions)->list[Node]:
    nodes_on_edge = tn.nodes_on_boundary(edge_side)
    ## Sort nodes by position along `msg_mps_dir`. 
    # nodes must be ordered from left to right, as seen from the direction of the mps:
    if   mps_order_dir==Directions.Down:    
        def pos_ordering(node:Node)->int: return -node.pos[1]
    elif mps_order_dir==Directions.Up:    
        def pos_ordering(node:Node)->int: return +node.pos[1]
    elif mps_order_dir==Directions.Right:    
        def pos_ordering(node:Node)->int: return +node.pos[0]
    elif mps_order_dir==Directions.Left:    
        def pos_ordering(node:Node)->int: return -node.pos[0]
    else:
        raise ValueError(f"Not a valid option msg_mps_dir={mps_order_dir}")
    # sort:        
    return sorted(nodes_on_edge, key=pos_ordering)


def _fuse_mps_with_tn(
    tn : KagomeTensorNetwork,
    mps : MPS,
    mps_order_dir : Directions,
    edge_side : Sides
) -> KagomeTensorNetwork:

    ## Check data:
    assert Directions.is_orthogonal(edge_side, mps_order_dir), f"MPS must be oethogonal in its own ordering direction to the lattice"
    if not ContractionConfig.last_con_order_determines_mps_order:
        assert mps_order_dir is edge_side.next_counterclockwise()

    # Where is the message located compared to the lattice:
    delta = edge_side.unit_vector()

    ## Derive the common directions defining this incoming message:
    msg_dir_to_lattice = edge_side.opposite()
    
    # Get all tensors on edge:
    nodes_on_edge = _ordered_nodes_on_edge(tn, edge_side, mps_order_dir)
    assert len(mps.A) == len(nodes_on_edge) == mps.N

    ## derive shared properties of all message tensors:
    message_edge_names = [ "M-"+str(edge_side)+f"{i}" for i in range(mps.N-1) ]
    edges_per_message_tensor = [(message_edge_names[0],)] + list( itertools.pairwise(message_edge_names) ) + [(message_edge_names[-1],)]
    message_positions = [tuples.add(node.pos, delta) for node in nodes_on_edge]

    ## Mantissa and exponent:
    # if hasattr(mps, "nr_exp"):          exp = getattr(mps, "nr_exp")
    # else:                               exp = 0
    # if hasattr(mps, "nr_mantissa"):     mantissa = getattr(mps, "nr_mantissa")
    # else:                               mantissa = 1.0
    
    ## Connect each message-tensor and its corresponding lattice node:
    for (is_first, is_last, mps_tensor), lattice_node, m_edges, m_pos in \
        zip( lists.iterate_with_edge_indicators(mps.A), nodes_on_edge, edges_per_message_tensor, message_positions, strict=True ):

        ## Include mantissa+exponent info:
        # mps_tensor *= mantissa*10**exp

        ## Change tensor shape according to it's position on the lattice
        shape = mps_tensor.shape
        assert len(shape)==3, f"All message tensors are 3 dimensional when they are created. This tensor has {len(shape)} dimensions!"
        if is_first:
            new_tensor = mps_tensor.reshape([shape[1], shape[2]])
            directions = [msg_dir_to_lattice, mps_order_dir]
            edges = [ lattice_node.edge_in_dir(edge_side), m_edges[0] ]
        elif is_last:
            new_tensor = mps_tensor.reshape([shape[0], shape[1]])
            directions = [mps_order_dir.opposite(), msg_dir_to_lattice]
            edges = [ m_edges[0], lattice_node.edge_in_dir(edge_side) ]
        else:
            new_tensor = 1.0*mps_tensor
            directions = [mps_order_dir.opposite(), msg_dir_to_lattice,  mps_order_dir]
            assert len(m_edges)==2
            edges = [ m_edges[0], lattice_node.edge_in_dir(edge_side) , m_edges[1] ]


        ## Add new node to Tensor-Network:
        index = tn.size
        new_node = Node(
            is_ket=False,
            tensor=new_tensor,
            edges=edges,
            directions=directions, # type: ignore
            pos=m_pos,  # type: ignore
            index=index,
            name=f"m{index}",
            functionality=NodeFunctionality.Message
        )
        tn.add(new_node)

    ## Expand tensor-network in the correct dimension:
    if edge_side in [Directions.Left, Directions.Right]:
        tn.original_lattice_dims = tuples.add( tn.original_lattice_dims, (0, 1) )
    elif edge_side in [Directions.Up, Directions.Down]:
        tn.original_lattice_dims = tuples.add( tn.original_lattice_dims, (1, 0) )

    return tn


def fuse_messages_with_tn(
    tn : KagomeTensorNetwork,
    mps_dict : dict[Directions, tuple[MPS, Directions]]
) -> KagomeTensorNetwork:   
    # Start with new copy of tn: 
    tn_with_messages = tn.copy()

    # Fuse:
    for edge_side in Directions.all_in_counterclockwise_order():
        mps, mps_direction = mps_dict[edge_side]
        tn_with_messages = _fuse_mps_with_tn(tn_with_messages, mps, mps_direction, edge_side)
    return tn_with_messages


def _calc_and_check_expectation_value(numerator, denominator, force_real:bool) -> float:
    ## Control:
    separate_exp = BubbleConConfig.separate_exp

    ## Check inputs:
    if DEBUG_MODE:
        err_msg = f"Braket results should be scalar values. Got numerator={numerator}, denominator={denominator}"
        for val in [numerator, denominator]:
            if separate_exp:
                assert isinstance(val, tuple), "BubbleCon should return tuple[complex, int]"
                assert len(val)==2, "BubbleCon should return tuple[complex, int]"
                assert isinstance(val[0], complex|float), err_msg  # mantissa
                assert isinstance(val[1], int), f"Second return value of BubbleCon should be the exponent. Instead got {val[1]}"  # exponent
            else:
                assert isinstance(val, complex), err_msg

    ## Assign values in common format man*10**exp
    if separate_exp:
        numerator_mantissa   = numerator[0]
        numerator_exponent   = numerator[1]
        denominator_mantissa = denominator[0]
        denominator_exponent = denominator[1]
    else:
        numerator_mantissa   = numerator
        numerator_exponent   = 0
        denominator_mantissa = denominator
        denominator_exponent = 0

    if numerator_mantissa==denominator_mantissa==0:
        raise FloatingPointError(f"Both numerator and denominator are zero.")
            
    ## Compute Result:
    mantissa = numerator_mantissa/denominator_mantissa
    exponent = numerator_exponent-denominator_exponent

    ## Check result:
    if DEBUG_MODE and force_real:
        mantissa = assertions.real(mantissa, reason = f"Solution should be a real value. Instead got {mantissa}")
    elif force_real:
        mantissa = float(np.real(mantissa))
    else:
        pass

    return mantissa*10**exponent

def _sandwich_with_operator_and_contract_fully(
    node_ind:int,
    tn:KagomeTensorNetwork, 
    operator:np.matrix,
    max_con_dim:int, 
    direction:Directions
) -> complex|tuple:
    # Replace fused-tensor <psi|psi> in `node_ind` with  <psi|Z|psi>:
    tn_with_observable = _sandwich_fused_tensors_with_expectation_values(tn, operator, node_ind)
    ## Calculate Expectation Value:
    numerator, _, _ = contract_tensor_network(
        tn_with_observable, 
        direction=direction, 
        depth=ContractionDepth.Full, 
        bubblecon_trunc_dim=max_con_dim, 
        print_progress=False )
    # complete contraction so must be a number:
    assert isinstance(numerator, complex|tuple)
    return numerator


def _rearrange_legs(
    c1:Node,
    c2:Node,
    env:list[Node],
)->tuple[
    Node, Node, list[Node]
]:
    ## init Indices:
    env_left_legs_indices = []
    env_mid_legs_indices = []
    env_right_legs_indices = []
    c1_legs_indices = []
    c2_legs_indices = []

    ## core tensors start with legs towards each other:
    i1, i2 = get_common_edge_legs(c1, c2)
    c1_legs_indices.append(i1)
    c2_legs_indices.append(i2)

    ## Go counter-clockwise - relate core to environment:
    # half touching to c1:
    for m in env[0:3]:
        i1, i2 = get_common_edge_legs(c1, m)
        c1_legs_indices.append(i1)
        env_mid_legs_indices.append(i2)
    # half touching c2:
    for m in env[3:]:
        i1, i2 = get_common_edge_legs(c2, m)
        c2_legs_indices.append(i1)
        env_mid_legs_indices.append(i2)

    ## Go counter-clockwise - check leg order of environment:
    for prev, this, next in lists.iterate_with_periodic_prev_next_items(env):
        i1, _ = get_common_edge_legs(this, prev)
        env_left_legs_indices.append(i1)
        i1, _ = get_common_edge_legs(this, next)
        env_right_legs_indices.append(i1)

    ## Permute:
    c1.permute(c1_legs_indices)
    c2.permute(c2_legs_indices)
    for ind, i_left, i_mid, i_right in zip(range(len(env)), env_left_legs_indices, env_mid_legs_indices, env_right_legs_indices ,strict=True):
        env[ind].permute([i_left, i_mid, i_right])

    return c1, c2, env


def _find_tensor_in_direction(dir:Directions, n1:Node, n2:Node)->Node:
    vec = dir.unit_vector()
    if tuples.equal( n1.pos, tuples.add(vec, n2.pos) ):
        return n1
    elif tuples.equal( n2.pos, tuples.add(vec, n1.pos) ):
        return n2
    else:
        raise ValueError(f"None of nodes [{n1.name!r}, {n2.name!r}] are in correct relation with direction {dir.name!r}.")


def calc_reduced_tn_around_edge(
    tn_stable:KagomeTensorNetwork, side:Sides, 
    bubblecon_trunc_dim:int, method:ReduceToEdgeMethod, allready_reduced_to_core:bool=False, swallow_corners_:bool=True
)->KagomeTensorNetwork:
    """
        Get reduced tensor_network using bubblecon.
    """

    # Common options:
    parallel = MULTIPROCESSING and ( tn_stable.size>300 or bubblecon_trunc_dim>15 ) 

    # Control:
    reduced_to_core : bool = allready_reduced_to_core
    reduced_to_edge : bool = False

    
    if reduced_to_core:
        tn_core = tn_stable  # just one more simple contraction is needed:
    else:
        match method:
            case ReduceToEdgeMethod.EachDirectionToEdge:
                ## Leave a rectangle of tensors around the edge:
                orthogonal_directions = [side.next_clockwise(), side.next_counterclockwise()]
                half_depth = (tn_stable.original_lattice_dims[0]) // 2
                tn_small = reduce_tn_using_bubblecon(tn_stable, directions=orthogonal_directions, bubblecon_trunc_dim=bubblecon_trunc_dim, depth=ContractionDepth.ToCore)
                tn_small = reduce_tn_using_bubblecon(tn_small, bubblecon_trunc_dim=bubblecon_trunc_dim, directions=[side.opposite()], depth=half_depth-1)
                tn_small = reduce_tn_using_bubblecon(tn_small, bubblecon_trunc_dim=bubblecon_trunc_dim, directions=[side], depth=half_depth)
                tn_edge = swallow_corners(tn_small)
                #
                reduced_to_edge = True
            case ReduceToEdgeMethod.EachDirectionToCore:
                tn_core  = _reduce_tn_to_core_and_environment_EachDirectionToCore(tn_stable, bubblecon_trunc_dim, swallow_corners_, parallel)

            case ReduceToEdgeMethod.DoubleMPSZipping:
                tn_core  = _reduce_tn_to_core_and_environment_DoubleMPSZipping(tn_stable, bubblecon_trunc_dim, swallow_corners_, parallel)


    if not reduced_to_edge:
        tn_edge = reduce_core_and_environment_to_edge_and_environment(tn_core, side, bubblecon_trunc_dim)  #type: ignore


    ## final clean-ups and validation
    if DEBUG_MODE: tn_edge.validate()  #type: ignore
    return tn_edge   #type: ignore



def swallow_corners(tn:KagomeTensorNetwork, _if_no_corners_error:bool=True)->KagomeTensorNetwork:
    # Find corner tensors:
    corner_tensors = tn.get_corner_nodes()
    ## contract corners to a nehigboring tensor in the wide direction
    if _if_no_corners_error and len(corner_tensors)==0: 
        raise TensorNetworkError("No corners to swallow")
    for t in corner_tensors:
        # Find another tensor to swallow this corner-tensor into:
        for direction in Directions.all_in_random_order():
            try:
                neighbor_in_direction = tn.find_neighbor(t, dir=direction)
            except ValueError:
                continue
            else:
                break
        else:
            raise ValueError("Not neighbours were found")
        # Full contraction:
        tn.contract_nodes(t, neighbor_in_direction)    
    return tn


def get_edge_environment_pashtida(
    tn:KagomeTensorNetwork, side:Sides, bubblecon_trunc_dim:int
)->tuple[
    Node, Node,         # core1+2
    list[np.ndarray],   # environment
    KagomeTensorNetwork       # small_tn
]:
    """
    A wrapper for the Pashtida version of BP

    
    """ 
    ## Check inputs
    assert len(tn.original_lattice_dims)==2
    assert tn.original_lattice_dims[0] == tn.original_lattice_dims[1]

    ## Unpack Inputs:
    # Tensor-Network data:
    T_list = [] 
    for node in tn.nodes:
        if node.functionality is NodeFunctionality.Message:
            T_list.append(node.fused_tensor)
        elif node.functionality in [NodeFunctionality.Core, NodeFunctionality.Padding]:
            T_list.append(node.physical_tensor)
        else:
            raise ValueError("Not a valid option")

    e_list = tn.edges_list
    a_list = tn.angles
    N = tn.original_lattice_dims[0]-2
    n = 2


    #TODO delete
    from algo._pashtida import plot_network
    # plot_network(T_list, e_list, a_list, N=N)

    bmps_chi = bubblecon_trunc_dim
    opt_method='high'  # bubblecon optimization level
    # edges and modes in pashtida:
    edge_legs = {'h1':(0,1), 'h2':(2,3), 'v1':(0,2), 'v2':(1,3)}
    match side:
        case Sides.Left:  edge = "v1"
        case Sides.Right: edge = "v2"
        case Sides.Up:    edge = "h1"
        case Sides.Down:  edge = "h2"
        case _:
            raise ValueError(" ")
    updated_edges = [edge]



    ## Call main functions  
    small_T_list, small_e_list, small_angles_list = two_way_contraction_to_core(
        N, n,
        T_list, e_list, a_list,
        bmps_chi, opt_method
    )

    # plot_network(small_T_list, small_e_list, small_angles_list, n)

    local_envs_fused, local_envs_open  = get_smallhex_2env(
        small_T_list, small_e_list, 
        small_angles_list, updated_edges, 
        bmps_chi
    )



    # Go over all the calculated envs and either update the corresponding
    # tensors or calculate the energy (if we're on the last step)
    #
    i,j = edge_legs[edge]

    #
    # Transpose the PEPS tensors T_i, T_j before updating them
    # or calculating their RDMs. This is done using a leg permutation
    # which depends on the mode.
    #
    if edge=='h1' or edge=='h2':
        Ti_perm = [0, 2, 3, 1, 4]
        Tj_perm = [0, 1, 4, 2, 3]

        if edge=='h1':
            A_ind, B_ind = i, j
        else:
            A_ind, B_ind = j, i

    if edge=='v1' or edge=='v2':
        Ti_perm = [0, 4, 2, 3, 1]
        Tj_perm = [0, 3, 1, 4, 2]


        if edge=='v1':
            A_ind, B_ind = i, j
        else:
            A_ind, B_ind = j, i

    Ti = small_T_list[i].transpose(Ti_perm) #type: ignore
    Tj = small_T_list[j].transpose(Tj_perm) #type: ignore

    from tensor_networks.operations import fuse_tensor_to_itself

    match side:
        case Sides.Left:  poses = [(0,0), (0,1)]
        case Sides.Right: poses = [(0,1), (0,0)]
        case Sides.Up:    poses = [(0,0), (1,0)]
        case Sides.Down:  poses = [(1,0), (0,0)]
        case _:
            raise ValueError(" ")
        
    ## Sides:
    dir_towards = side
    dir_to_right = side.next_clockwise()
    dir_to_left = side.next_counterclockwise()
    dir_back = side.opposite()
    # angles:
    angle_offset = dir_to_right.angle
    pi_half = np.pi/2
    UR = _fix_angle(   pi_half/2+angle_offset )
    UL = _fix_angle( 3*pi_half/2+angle_offset )
    DL = _fix_angle( 5*pi_half/2+angle_offset )
    DR = _fix_angle( 7*pi_half/2+angle_offset )

    ## Pack results
    cores = [] 
    for i, t in enumerate([Ti, Tj]): 
        if   i==0:  directions = [dir_to_right, dir_towards, dir_to_left , dir_back    ]
        elif i==1:  directions = [dir_to_left , dir_back,    dir_to_right, dir_towards ]
        else:
            raise ValueError()
        
        i_core = i+1

        if   i==0:  edges = ["cores", f"c{i_core}-1", f"c{i_core}-2", f"c{i_core}-3"]
        elif i==1:  edges = ["cores", f"c{i_core}-4", f"c{i_core}-5", f"c{i_core}-6"]
        else:
            raise ValueError()
        
        

        core = Node(
            tensor=t,
            is_ket=True,
            functionality=NodeFunctionality.Core,
            edges=edges,
            directions=directions,  #type: ignore  #TODO fix
            pos=poses[i],
            index=i, 
            name=f"core{i_core}",
            on_boundary=[]
        )

        cores.append(core)

    core1 = cores[0]
    core2 = cores[1]

    # edges = {
    #     "main": (0,1),
    #     "1-2" : (0,2),
    #     "1-3" : (0,3),
    #     "1-4" : (0,4),
    #     "2-2" : (0,5),
    #     "2-3" : (0,6),
    #     "2-4" : (0,7),
    #     "env1": (2,3),
    #     "env2": (3,4),
    #     "env3": (4,5),
    #     "env4": (5,6),
    #     "env5": (6,7),
    #     "env6": (7,2),
    # }

    edges = {
        "cores": (0,1),
        "c1-1" : (0,0),
        "c1-2" : (0,0),
        "c1-3" : (0,0),
        "c2-4" : (1,1),
        "c2-5" : (1,1),
        "c2-6" : (1,1),
    }

    tn_env = KagomeTensorNetwork(
        nodes=cores,
        edges=edges,
        tensor_dims=tn.tensor_dims,
        _copied=True
    )
    match side:
        case Sides.Left | Sides.Right:  dims = (4, 3)
        case Sides.Up   | Sides.Down:   dims = (3, 4)
        case _:
            raise ValueError(" ")
    tn_env.original_lattice_dims = dims

    # tn_env.plot()

    env_edges = [
        ["env6", "c1-1", "env1"],
        ["env1", "c1-2", "env2"],
        ["env2", "c1-3", "env3"],
        ["env3", "c2-4", "env4"],
        ["env4", "c2-5", "env5"],
        ["env5", "c2-6", "env6"],

    ]

    env_directions = [
        [dir_to_right, dir_back, DL],
        [UR, dir_to_right, DR],
        [UL, dir_towards, dir_to_right],
        [dir_to_left, dir_towards, UR],
        [DL, dir_to_left, UL],
        [DR, dir_back, dir_to_left]
    ]

    env_poses = [
        tuples.add(poses[0], dir_towards.unit_vector()),
        tuples.add(poses[0], dir_to_left.unit_vector()),
        tuples.add(poses[0], dir_back.unit_vector()),
        tuples.add(poses[1], dir_back.unit_vector()),
        tuples.add(poses[1], dir_to_right.unit_vector()),
        tuples.add(poses[1], dir_towards.unit_vector())
    ]


    fused_env = local_envs_fused[edge]
    open_env = local_envs_open[edge]
    assert isinstance(fused_env, list)
    assert isinstance(open_env, list)

    for i, t in enumerate(fused_env):
        node = Node(
            is_ket=False,
            tensor=t,
            functionality=NodeFunctionality.Message,
            edges=env_edges[i],
            directions=env_directions[i],
            pos=env_poses[i],
            index=2+i, 
            name=f"env{i+1}",
            on_boundary=[]
        )

        tn_env.add(node)

    tn_env.validate()
    # tn_env.plot()

    return core1, core2, open_env, tn_env


def rearange_tensors_legs_to_canonical_order(
    tn_env:KagomeTensorNetwork, side:Sides
)->tuple[Node, Node, list[np.ndarray]]:
    core_tensors = tn_env.get_core_nodes()
    core1 = _find_tensor_in_direction(side.next_counterclockwise(), *core_tensors)
    core2 = _find_tensor_in_direction(side.next_clockwise(), *core_tensors)
    environment_nodes = [
        tn_env.find_neighbor(core1, side),
        tn_env.find_neighbor(core1, side.next_counterclockwise()),
        tn_env.find_neighbor(core1, side.opposite()),
        tn_env.find_neighbor(core2, side.opposite()),
        tn_env.find_neighbor(core2, side.next_clockwise()),
        tn_env.find_neighbor(core2, side),
    ]

    ## Rearange legs in required order:
    core1, core2, environment_nodes = _rearrange_legs(core1, core2, environment_nodes)
    environment_tensors = [physical_tensor_with_split_mid_leg(n) for n in environment_nodes]    # Open environment mps legs:
    return core1, core2, environment_tensors


def calc_edge_environment(
    tn:KagomeTensorNetwork, side:Sides, bubblecon_trunc_dim:int, method:ReduceToEdgeMethod=ReduceToEdgeMethod.default(), already_reduced_to_core:bool=False
)->tuple[
    Node, Node,         # core1/2
    list[np.ndarray],         # environment
    KagomeTensorNetwork       # small_tn
]:
    ## Get the smallest Tensor-Network around the mode (edge):
    tn_env = calc_reduced_tn_around_edge(tn, side, bubblecon_trunc_dim, method, already_reduced_to_core)

    ## get all tensors in correct order:
    core1, core2, environment_tensors = rearange_tensors_legs_to_canonical_order(tn_env, side)

    return core1, core2, environment_tensors, tn_env


def calc_interaction_energies_in_core(tn:KagomeTensorNetwork, interaction_hamiltonain:np.ndarray, bubblecon_trunc_dim:int) -> list[float]:
    energies = []
    reduced_tn = reduce_tn_to_core_and_environment(tn, bubblecon_trunc_dim, swallow_corners_=False)
    for side in Sides:
        core1, core2, environment_tensors, tn_env = calc_edge_environment(reduced_tn, side, bubblecon_trunc_dim)
        rdm = rho_ij(core1.physical_tensor, core2.physical_tensor, mps_env=environment_tensors)
        energy  = np.dot(rdm.flatten(),  interaction_hamiltonain.flatten())
        energies.append(energy)
    return energies



def calc_mean_values(
    tn:KagomeTensorNetwork, 
    operators:list[np.matrix], 
    bubblecon_trunc_dim:int, 
    direction:Directions=Directions.random(), 
    force_real:bool=False, 
    reduce:bool=True
) -> list[float]:
    ## Prepare output:
    mean_values = []

    ## Perform all common actions:
    if reduce:
        tn_reduced = reduce_tn_to_core_and_environment(tn, bubblecon_trunc_dim)
    else:
        tn_reduced = tn.copy()
    denominator, _, _ = contract_tensor_network(tn_reduced, direction=direction, depth=ContractionDepth.Full, bubblecon_trunc_dim=bubblecon_trunc_dim)
    assert not isinstance(denominator, MPS), "Full contraction should result in a number, not an MPS"
    center_nodes = tn_reduced.get_core_nodes()
    center_indices = [n.index for n in center_nodes]

    ## Perform calculation per operator:
    for operator in operators:
        mean_value = calc_mean_value(tn_reduced, center_indices, operator, bubblecon_trunc_dim=bubblecon_trunc_dim, direction=direction, denominator=denominator, force_real=force_real)
        mean_values.append(mean_value)
    return mean_values


def calc_mean_value(
    tn:KagomeTensorNetwork, 
    node_indices:List[int], 
    operator:np.matrix,
    bubblecon_trunc_dim:int, 
    direction:Directions = Directions.random(), 
    print_:bool = False,
    force_real:bool = False,
    denominator:complex|tuple|None=None
) -> float:
    ## Common denominator <psi|psi>:
    if isinstance(denominator, complex|tuple):
        pass
    elif denominator is None:
        res, _, _ = contract_tensor_network(tn, direction=direction, depth=ContractionDepth.Full, bubblecon_trunc_dim=bubblecon_trunc_dim)        
        assert not isinstance(res, MPS), "Full contraction should result in a number, not an MPS"
        denominator = res
    else:
        raise TypeError(f"Not an expected type {type(denominator)!r} for denominator")

    ## Get numerators in parallel or one-by-one:    
    fixed_arguments = dict(tn=tn, max_con_dim=bubblecon_trunc_dim, direction=direction, operator=operator)
    in_parallel = MULTIPROCESSING and len(node_indices)>1 and tn.tensor_dims.virtual>2
    numerators = parallel_exec.concurrent_or_parallel(
        func=_sandwich_with_operator_and_contract_fully, 
        values=node_indices, value_name="node_ind", 
        fixed_arguments=fixed_arguments,
        in_parallel=in_parallel
        ) 

    ## Calc numerator/denominator for each value
    expectation_values = [
        _calc_and_check_expectation_value(numerator, denominator, force_real)
        for numerator in numerators.values()
    ]    

    if print_:
        for expectation_value, numerator in zip(expectation_values, numerators.values(), strict=True):
            print(f"\texpectation_value = {expectation_value}  =  {numerator} / {denominator}  ")

    return lists.average(expectation_values)


def contract_tensor_network(
    tn:KagomeTensorNetwork, 
    direction:Directions,
    depth:ContractionDepth|int,
    bubblecon_trunc_dim:int,
    print_progress:bool=True
)->Tuple[
    MPS|complex|tuple,
    List[int],
    Directions,
]:

    ## derive basic data:
    contraction_order, last_direction = derive_contraction_orders(tn, direction, depth=depth)

    ## Call main function:
    mp = bubblecon(
        tn.tesnors, 
        tn.edges_list, 
        tn.angles, 
        bubble_angle=direction.angle,
        swallow_order=contraction_order, 
        D_trunc=bubblecon_trunc_dim,
        opt='high',
        progress_bar=BubbleConConfig.progress_bar and print_progress,
        separate_exp=BubbleConConfig.separate_exp,
        ket_tensors=tn.kets
    )


    ## Derive outgoing mps direction
    if ContractionConfig.last_con_order_determines_mps_order:
        mps_direction = last_direction
    else:
        mps_direction = direction.next_clockwise()

    ## Check outputs:
    assert not isinstance(mp, list)  # This is not an expected output

    return mp, contraction_order, mps_direction


def reduce_tn_using_bubblecon(tn:KagomeTensorNetwork, bubblecon_trunc_dim:int, directions:Iterable[Directions], depth:ContractionDepth|int, parallel:bool=False)->KagomeTensorNetwork:

    # prepare inputs:
    fixed_arguments = dict(tn=tn, bubblecon_trunc_dim=bubblecon_trunc_dim, depth=depth)
    directions = lists.shuffle(list(directions))
    
    # Sandwich Tensor-Network from both sides at once if parallel:
    if parallel:
        fixed_arguments["print_progress"]=False
        con_results = parallel_exec.parallel(func=contract_tensor_network, values=directions, value_name="direction", fixed_arguments=fixed_arguments) 
    else:
        fixed_arguments["print_progress"] = True
        con_results = parallel_exec.concurrent(func=contract_tensor_network, values=directions, value_name="direction", fixed_arguments=fixed_arguments) 
    
    # Rearrange outputs:
    mpss        = {direction:tupl[0] for direction, tupl in con_results.items()}
    con_indices = lists.join_sub_lists([tupl[1] for tupl in con_results.values()])
    mps_directions =                   [tupl[2] for tupl in con_results.values()]


    ## Ignore tensors that are accounted-for by the messages:
    remaining_indices = [node.index for node in tn.nodes if node.index not in con_indices]
    reduced_tn = tn.sub_tn(remaining_indices)
    if DEBUG_MODE: reduced_tn.validate()
    
    ## Connect messages directly to the remaining tensors:
    for (direction, mps), mps_dir in zip(mpss.items(), mps_directions, strict=True):
        assert isinstance(mps, MPS)
        reduced_tn = _fuse_mps_with_tn( reduced_tn, mps, mps_dir, direction.opposite() )
    if DEBUG_MODE: reduced_tn.validate()

    ## Return:
    return reduced_tn


def reduce_core_and_environment_to_edge_and_environment(
    tn_small:KagomeTensorNetwork, side:Sides, bubblecon_trunc_dim:int
)->KagomeTensorNetwork:
    tn_env = reduce_tn_using_bubblecon(tn_small, bubblecon_trunc_dim=bubblecon_trunc_dim, directions=[side], depth=2)
    ## Find and Swallow the two corner tensors:
    remaining_core_tensors = [t for t in tn_env.nodes if t.functionality is NodeFunctionality.Core]
    assert len(remaining_core_tensors)==2
    for t in tn_env.nodes:
        # pass on core nodes:
        if t is remaining_core_tensors[0] or t is remaining_core_tensors[1]:
            continue
        # pass on environment of core nodes:
        if tn_env.are_neigbors(t, remaining_core_tensors[0]) or tn_env.are_neigbors(t, remaining_core_tensors[1]):
            continue
        neighbor_in_direction = tn_env.find_neighbor(t, side)

        tn_env.contract_nodes(t, neighbor_in_direction)
    return tn_env


def _reduce_tn_to_core_and_environment_EachDirectionToCore(small_tn:KagomeTensorNetwork, bubblecon_trunc_dim:int, swallow_corners_:bool, parallel:bool) -> KagomeTensorNetwork:
    for directions in Directions.all_opposite_pairs():
        small_tn = reduce_tn_using_bubblecon(
            tn=small_tn, directions=directions, bubblecon_trunc_dim=bubblecon_trunc_dim, depth=ContractionDepth.ToCore, 
            parallel=parallel
        )    
    ## swallow corners:
    if swallow_corners_:        
        small_tn = swallow_corners(small_tn)

    return small_tn

def _reduce_tn_to_core_and_environment_DoubleMPSZipping(tn:KagomeTensorNetwork, bubblecon_trunc_dim:int, swallow_corners_:bool, parallel:bool) -> KagomeTensorNetwork:
    
    ## Check inputs:
    assert swallow_corners_ is True, f"Zipping methods always swallows the corners"

    ## General data:
    core_indices = [n.index for n in tn.get_core_nodes()]
    mpss : dict[Directions, MPS] = dict()
    N = tn.original_lattice_dims[0]-2
    n = len(core_indices)//2
    d_virtual = tn.tensor_dims.virtual
    d_physical = tn.tensor_dims.physical

    ## Prepare two MPSs:
    for direction in [Directions.Down, Directions.Up]:
        con_order, _ = derive_contraction_orders(tn, direction, depth=ContractionDepth.Full)
        con_order = con_order[0:len(con_order)//2]
        for core_index in core_indices:
            if core_index in con_order:
                con_order.remove(core_index)

        mps = bubblecon(
            tn.tesnors, 
            tn.edges_list, 
            tn.angles, 
            bubble_angle=direction.angle,
            swallow_order=con_order, 
            D_trunc=bubblecon_trunc_dim,
            opt='high',
            progress_bar=BubbleConConfig.progress_bar,
            separate_exp=BubbleConConfig.separate_exp,
            ket_tensors=tn.kets
        )
        assert isinstance(mps, MPS)

        # The contracition going down creates an upper mps
        mpss[direction.opposite()] = mps  

    upper_mps = mpss[Directions.Up]
    lower_mps = mpss[Directions.Down]

	# III. Contract the upper/lower MPS to create an edge tensor from the
	#      left and from the right, creating LTensor, RTensor.
    LTensor : np.ndarray = None  #type: ignore

    jD = (N-n)//2 + 1
    for j in range(0, jD):
        LTensor = bmpslib.updateCLeft(LTensor,
            upper_mps.A[-j-1].transpose([2,1,0]), \
            lower_mps.A[j])

    # LTensor legs are [i_up, i_down]. Absorb it in the first tensor of
    # mps_D_half that going to appear in the small square TN
    lower_mps.A[jD] = np.tensordot(LTensor, lower_mps.A[jD], axes=([1],[0]))

    # Now create RTensor, which is the contraction of the right part
    # of the up/down MPSs
    RTensor : np.ndarray = None  #type: ignore
    jU = (N-n)//2 + 1
    for j in range(0, jU):
        RTensor = bmpslib.updateCRight(RTensor,
            upper_mps.A[j].transpose([2,1,0]), \
            lower_mps.A[-1-j])
    # Absorb the RTensor[i_up, i_down] in the first tensor of mps_up_half
    upper_mps.A[jU] = np.tensordot(RTensor, upper_mps.A[jU], axes=([0],[0]))

    # IV. We now have all tensors we need to define the small TN.
    #     It consists of the original TN in the small square + the MPS
    #     tensors that surround it.

    # first create the small TN
    peps = [n.physical_tensor for n in tn.get_core_nodes()]
    small_tn = create_kagome_tn(core_size=n, D=d_virtual, d=d_physical, padding=0, creation_mode=peps)

    ## Add the surrounding MPS tensors in the edge order D, R, U, L
    env_tensors = lower_mps.A[(jD+n//2):(jD+2*n)] \
        + upper_mps.A[jU:(jU+2*n)] + lower_mps.A[jD:(jD+n//2)]


    env_nodes : list[Node] = []
    num_mps_tensors = len(Sides)*n
    i_mps = 0
    ## Add env:
    for side in Directions.all_in_counterclockwise_order():
        dir_to_lattice = side.opposite()
        dir_mps_order = dir_to_lattice.next_clockwise()

        for node in _ordered_nodes_on_edge(small_tn, side, dir_mps_order):
            
            new_pos = tuples.add(node.pos, side.unit_vector())
            bulk_edge_name = node.edge_in_dir(side)

            i_mps_next = (i_mps + 1) % num_mps_tensors
            edges = [f'env{i_mps}', bulk_edge_name, f'env{i_mps_next}']
            dirs = [dir_to_lattice.next_counterclockwise(),  dir_to_lattice, dir_to_lattice.next_clockwise()]

            new_node = Node(
                is_ket=False,
                tensor=env_tensors[i_mps],
                functionality=NodeFunctionality.Message,
                edges=edges,
                directions=dirs,
                pos=new_pos,
                index=small_tn.size,
                name=f"m{i_mps}",
                on_boundary=[]
            )

            small_tn.add(new_node)
            env_nodes.append(new_node)

            i_mps += 1

    ## Fix corner legs:
    i_mps = 0
    for side in Directions.all_in_counterclockwise_order():
        dir_to_lattice = side.opposite()
        dir_mps_order = dir_to_lattice.next_clockwise()
        for i_per_side, node in enumerate(_ordered_nodes_on_edge(small_tn, side, dir_mps_order)):
            if i_per_side==n-1:
                # Get data
                i_mps_next = (i_mps + 1) % num_mps_tensors
                this_node = env_nodes[i_mps]
                next_node = env_nodes[i_mps_next]
                # get common edge:
                i_this_wrong_leg = this_node.directions.index(dir_mps_order)
                i_next_wrong_leg = next_node.directions.index(dir_to_lattice.opposite())
                # Derive new directions:
                this_angle = tuples.angle(this_node.pos, next_node.pos) 
                try:                        
                    this_dir = Directions.from_angle(this_angle)
                except ValueError:
                    this_dir = this_angle
                next_dir = Directions.opposite_direction(this_dir)
                # Update nodes:
                this_node.directions[i_this_wrong_leg] = this_dir
                next_node.directions[i_next_wrong_leg] = next_dir
            i_mps += 1

    ## Update small_tn size
    small_tn.original_lattice_dims = tuples.add(small_tn.original_lattice_dims, (2, 2))

    return small_tn


def reduce_tn_to_core_and_environment(tn:KagomeTensorNetwork, bubblecon_trunc_dim:int, swallow_corners_:bool=True, method:ReduceToCoreMethod=ReduceToCoreMethod.default()) -> KagomeTensorNetwork:
    tn_copy = tn.copy()
    
    ## Decide if parallel contraction benefit us in this case:   #TODO  needs verification
    parallel = MULTIPROCESSING and ( tn.size>300 or bubblecon_trunc_dim>10 ) and tn.tensor_dims.virtual>3

    ## Perform reduction up to a perfect square arround the core:
    if method is ReduceToCoreMethod.EachDirectionToCore:
        small_tn = _reduce_tn_to_core_and_environment_EachDirectionToCore(tn_copy, bubblecon_trunc_dim, swallow_corners_, parallel)

    elif method is ReduceToCoreMethod.DoubleMPSZipping:
        small_tn = _reduce_tn_to_core_and_environment_DoubleMPSZipping(tn_copy, bubblecon_trunc_dim, swallow_corners_, parallel)

    else:
        raise ValueError(f"No such option {method!r}")


    if DEBUG_MODE: small_tn.validate()
    return small_tn


def full_contraction(tn:KagomeTensorNetwork, /,*, max_dim:int, direction:Directions=Directions.Right):
    # Basic info:    
    min_x, max_x, min_y, max_y = tn.boundaries()
    min_x = int(min_x) 
    max_x = int(max_x) 
    min_y = int(min_y) 
    max_y = int(max_y)
    
    ## Derive order:
    con_order = []
    y_range = list(range(min_y, max_y+1))
    
    for x in range(min_x, max_x+1):
        
        for y in y_range:
            try:
                node = tn.get_tensor_in_pos((x,y))
            except ValueError:
                continue
            con_order.append(node.index)
            
        y_range.reverse()      
    
    ## contract:
    mp = bubblecon(
        tn.fused_tensors, 
        tn.edges_list, 
        tn.angles, 
        bubble_angle=direction.angle,
        swallow_order=con_order, 
        D_trunc=max_dim,
        D_trunc2=BubbleConConfig.trunc_dim_2, 
        eps=BubbleConConfig.eps, 
        opt='high',
        progress_bar=BubbleConConfig.progress_bar,
        separate_exp=BubbleConConfig.separate_exp
    )
    return mp



if __name__ == "__main__":
    from scripts.core_ite_test import main
    main()