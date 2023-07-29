if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)


# Control flags:
from _config_reader import DEBUG_MODE

# Everyone needs numpy:
import numpy as np

# For type anotation:
from typing import List, Tuple, Iterable, TypeVar

# Common types in the code:
from tensor_networks import KagomeTensorNetwork, TensorNode, MPS
from lattices.directions import LatticeDirection, BlockSide
from lattices import directions
from _error_types import TensorNetworkError
from enums import ContractionDepth, ReduceToEdgeMethod, ReduceToCoreMethod, NodeFunctionality
from containers import BubbleConConfig, MessageDictType
from physics import pauli
from _types import EdgeIndicatorType

# Our utilities:
from utils import tuples, lists, assertions, parallel_exec

# Our needed algos:
from tensor_networks.tensor_network import get_common_edge_legs
from tensor_networks.construction import create_kagome_tn, _get_edge_from_tensor_coordinates
from algo.contraction_order import derive_contraction_order, contraction_order_at_depth
from algo.mps import physical_tensor_with_split_mid_leg
from libs.bubblecon import bubblecon
from libs import bmpslib
import itertools

# For a little bit of OOP:
from typing import Any, NamedTuple

# For energy estimation:
from libs.ITE import rho_ij


def _derive_angle(p1:tuple[int,int], p2:tuple[int,int])->float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dx, dy)

def _fix_angle(a:float)->float:
    while a < 0:
        a += 2*np.pi
    while a>2*np.pi:
        a -= 2*np.pi
    return a

def _get_corner_tensors(tn:KagomeTensorNetwork) -> list[TensorNode]:
    min_x, max_x, min_y, max_y = tn.positions_min_max()
    corner_tesnors = [] 
    for x in [min_x, max_x]:
        for y in  [min_y, max_y]:
            t = tn.get_node_in_pos((x, y))
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
    tn_out.nodes[ind] = TensorNode(
        is_ket          = False,
        tensor          = fused_data,
        edges           = node.edges,
        directions      = node.directions,
        pos             = node.pos,
        index           = node.index,
        name            = node.name,
        on_boundary     = node.boundaries,
        functionality   = node.functionality
    )
    if DEBUG_MODE: tn_out.validate()



    return tn_out


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
    direction:BlockSide
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
    c1:TensorNode,
    c2:TensorNode,
    env:list[TensorNode],
)->tuple[
    TensorNode, TensorNode, list[TensorNode]
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


def _find_node_in_relative_direction(dir:LatticeDirection, n1:TensorNode, n2:TensorNode)->TensorNode:
    vec = dir.unit_vector()
    if tuples.equal( n1.pos, tuples.add(vec, n2.pos) ):
        return n1
    elif tuples.equal( n2.pos, tuples.add(vec, n1.pos) ):
        return n2
    else:
        raise ValueError(f"None of nodes [{n1.name!r}, {n2.name!r}] are in correct relation with direction {dir.name!r}.")


def calc_reduced_tn_around_edge(
    tn_stable:KagomeTensorNetwork, mode:None,  #TODO fix mode
    bubblecon_trunc_dim:int, method:ReduceToEdgeMethod, allready_reduced_to_core:bool=False, swallow_corners_:bool=True
)->KagomeTensorNetwork:
    """
        Get reduced tensor_network using bubblecon.
        mode: should be fixed
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
                orthogonal_directions = [mode.next_clockwise(), mode.next_counterclockwise()]
                half_depth = (tn_stable.original_lattice_dims[0]) // 2
                tn_small = reduce_tn_using_bubblecon(tn_stable, directions=orthogonal_directions, bubblecon_trunc_dim=bubblecon_trunc_dim, depth=ContractionDepth.ToCore)
                tn_small = reduce_tn_using_bubblecon(tn_small, bubblecon_trunc_dim=bubblecon_trunc_dim, directions=[mode.opposite()], depth=half_depth-1)
                tn_small = reduce_tn_using_bubblecon(tn_small, bubblecon_trunc_dim=bubblecon_trunc_dim, directions=[mode], depth=half_depth)
                tn_edge = swallow_corners(tn_small)
                #
                reduced_to_edge = True
            case ReduceToEdgeMethod.EachDirectionToCore:
                tn_core  = _reduce_tn_to_core_and_environment_EachDirectionToCore(tn_stable, bubblecon_trunc_dim, swallow_corners_, parallel)

            case ReduceToEdgeMethod.DoubleMPSZipping:
                tn_core  = _reduce_tn_to_core_and_environment_DoubleMPSZipping(tn_stable, bubblecon_trunc_dim, swallow_corners_, parallel)


    if not reduced_to_edge:
        tn_edge = reduce_core_and_environment_to_edge_and_environment(tn_core, mode, bubblecon_trunc_dim)  #type: ignore


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
        for direction in Direction.all_in_random_order():
            try:
                neighbor_in_direction = tn.find_neighbor(t, dir_or_edge=direction)
            except ValueError:
                continue
            else:
                break
        else:
            raise ValueError("Not neighbours were found")
        # Full contraction:
        tn.contract_nodes(t, neighbor_in_direction)    
    return tn


def rearange_tensors_legs_to_canonical_order(
    tn_env:KagomeTensorNetwork, side:None # Fix mode\side
)->tuple[TensorNode, TensorNode, list[np.ndarray]]:
    core_tensors = tn_env.get_core_nodes()
    core1 = _find_node_in_relative_direction(side.next_counterclockwise(), *core_tensors)
    core2 = _find_node_in_relative_direction(side.next_clockwise(), *core_tensors)
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
    tn:KagomeTensorNetwork, mode:None,  #TODO fix mode type
    bubblecon_trunc_dim:int, method:ReduceToEdgeMethod=ReduceToEdgeMethod.default(), already_reduced_to_core:bool=False
)->tuple[
    TensorNode, TensorNode,         # core1/2
    list[np.ndarray],         # environment
    KagomeTensorNetwork       # small_tn
]:
    ## Get the smallest Tensor-Network around the mode (edge):
    tn_env = calc_reduced_tn_around_edge(tn, mode, bubblecon_trunc_dim, method, already_reduced_to_core)

    ## get all tensors in correct order:
    core1, core2, environment_tensors = rearange_tensors_legs_to_canonical_order(tn_env, mode)

    return core1, core2, environment_tensors, tn_env


def calc_interaction_energies_in_core(tn:KagomeTensorNetwork, interaction_hamiltonain:np.ndarray, bubblecon_trunc_dim:int) -> list[float]:
    energies = []
    reduced_tn = reduce_tn_to_core_and_environment(tn, bubblecon_trunc_dim, swallow_corners_=False)
    for side in Direction:
        core1, core2, environment_tensors, tn_env = calc_edge_environment(reduced_tn, side, bubblecon_trunc_dim)
        rdm = rho_ij(core1.physical_tensor, core2.physical_tensor, mps_env=environment_tensors)
        energy  = np.dot(rdm.flatten(),  interaction_hamiltonain.flatten())
        energies.append(energy)
    return energies



def calc_mean_values(
    tn:KagomeTensorNetwork, 
    operators:list[np.matrix], 
    bubblecon_trunc_dim:int, 
    direction:BlockSide=BlockSide.random(), 
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
    direction:BlockSide = BlockSide.random(), 
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


def connect_corner_messages(
    tn:KagomeTensorNetwork, outgoing_dir:BlockSide
)->tuple[
    list[np.ndarray], list[list[EdgeIndicatorType]], list[list[float]]
]:
    
    ## Get the output lists:
    tensors = tn.tensors
    edges_list = tn.edges_list
    angles = tn.angles

    ## Derive basic info:
    first_msg_dir = outgoing_dir.opposite()
    msg_indices1 = tn.message_indices(first_msg_dir)
    msg_indices2 = tn.message_indices(first_msg_dir.next_counterclockwise())
    index = [msg_indices1[-1], msg_indices2[0]]

    ## Expand both tensors with a fake leg of dim 1:
    new_tensor, new_edges, pos, old_angles = [], [], [], []
    for i in [0, 1]:
        grand_index = index[i]
        pos.append( tn.positions[grand_index] )
        tensor = tensors[grand_index]
        new_shape = tuples.add_element(tensor.shape, 1) 
        new_edges.append( edges_list[grand_index] + ['fake_leg']  )        
        new_tensor.append( tensor.copy().reshape(new_shape) )
        old_angles.append( angles[grand_index] )

        
    ## Derive new angles:
    new_a = [0.0, 0.0]
    new_a[0] = _derive_angle(pos[0], pos[1])
    new_a[1] = _fix_angle(new_a[0]+np.pi)
    new_angle = [old_angles[i]+[new_a[i]] for i in [0, 1]]

    ## Assign results to list
    for i in [0, 1]:
        grand_index = index[i]
        tensors[grand_index] = new_tensor[i]
        edges_list[grand_index] = new_edges[i]
        angles[grand_index] = new_angle[i]

    ## Return:
    return tensors, edges_list, angles
    


def contract_tensor_network(
    tn:KagomeTensorNetwork, 
    direction:BlockSide,
    depth:ContractionDepth|int,
    bubblecon_trunc_dim:int,
    print_progress:bool=True
)->Tuple[
    MPS|complex|tuple,
    List[int],
    LatticeDirection,
]:

    ## Derive Contraction Order:
    # get or derive full con-oder:
    full_contraction_order, last_direction = derive_contraction_order(tn, direction)
    # Cut con order according to `depth`:
    contraction_order = contraction_order_at_depth(full_contraction_order, depth, tn.lattice.N)

    ## Connect first MPS message to a side tensor, to allow efficient contraction:
    tensors, edges_list, angles = connect_corner_messages(tn, direction)

    ## Call main function:
    mp = bubblecon(
        tensors, 
        edges_list, 
        angles, 
        bubble_angle=direction.angle,
        swallow_order=contraction_order, 
        D_trunc=bubblecon_trunc_dim,
        opt='high',
        progress_bar=BubbleConConfig.progress_bar and print_progress,
        separate_exp=BubbleConConfig.separate_exp,
        ket_tensors=tn.kets
    )


    ## Derive outgoing mps direction
    mps_direction = direction.orthogonal_clockwise_lattice_direction()

    ## Check outputs:
    assert not isinstance(mp, list)  # This is not an expected output

    return mp, contraction_order, mps_direction


def reduce_tn_using_bubblecon(tn:KagomeTensorNetwork, bubblecon_trunc_dim:int, directions:Iterable[BlockSide], depth:ContractionDepth|int, parallel:bool=False)->KagomeTensorNetwork:

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
    tn_small:KagomeTensorNetwork, side:None, # fix side\mode 
    bubblecon_trunc_dim:int
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
    for directions in Direction.all_opposite_pairs():
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
    mpss : dict[Direction, MPS] = dict()
    N = tn.original_lattice_dims[0]-2
    n = len(core_indices)//2
    d_virtual = tn.tensor_dims.virtual
    d_physical = tn.tensor_dims.physical

    ## Prepare two MPSs:
    for direction in [Direction.Down, Direction.Up]:
        con_order, _ = derive_contraction_order(tn, direction, depth=ContractionDepth.Full)
        con_order = con_order[0:len(con_order)//2]
        for core_index in core_indices:
            if core_index in con_order:
                con_order.remove(core_index)

        mps = bubblecon(
            tn.tensors, 
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

    upper_mps = mpss[Direction.Up]
    lower_mps = mpss[Direction.Down]

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


    env_nodes : list[TensorNode] = []
    num_mps_tensors = len(Direction)*n
    i_mps = 0
    ## Add env:
    for side in Direction.all_in_counterclockwise_order():
        dir_to_lattice = side.opposite()
        dir_mps_order = dir_to_lattice.next_clockwise()

        for node in _ordered_nodes_on_boundary(small_tn, side, dir_mps_order):
            
            new_pos = tuples.add(node.pos, side.unit_vector())
            bulk_edge_name = node.edge_in_dir(side)

            i_mps_next = (i_mps + 1) % num_mps_tensors
            edges = [f'env{i_mps}', bulk_edge_name, f'env{i_mps_next}']
            dirs = [dir_to_lattice.next_counterclockwise(),  dir_to_lattice, dir_to_lattice.next_clockwise()]

            new_node = TensorNode(
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
    for side in Direction.all_in_counterclockwise_order():
        dir_to_lattice = side.opposite()
        dir_mps_order = dir_to_lattice.next_clockwise()
        for i_per_side, node in enumerate(_ordered_nodes_on_boundary(small_tn, side, dir_mps_order)):
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
                    this_dir = Direction.from_angle(this_angle)
                except ValueError:
                    this_dir = this_angle
                next_dir = Direction.opposite_direction(this_dir)
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


def full_contraction(tn:KagomeTensorNetwork, /,*, max_dim:int, direction:BlockSide=BlockSide.random()):
    # Basic info:    
    min_x, max_x, min_y, max_y = tn.positions_min_max()
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
                node = tn.get_node_in_pos((x,y))
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

