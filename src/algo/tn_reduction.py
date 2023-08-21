if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)
	sys.path.append(
		pathlib.Path(__file__).parent.parent.parent.__str__()
	)



# Control flags:
from _config_reader import DEBUG_MODE

# For a little bit of OOP:
from typing import Any, NamedTuple, Generic, TypeVar, Generator, Iterable
from dataclasses import dataclass, field
_T = TypeVar("_T")

# Common types in the code:
from tensor_networks import KagomeTN, ArbitraryTN, ModeTN, TensorNode, MPS, CoreTN, get_common_edge

# Everyone needs numpy:
import numpy as np

# Other modules we made and need here   
from libs import bmpslib

# Other algos we need here:
from algo.contract_tensor_network import contract_tensor_network

# Types we need in our module:
from tensor_networks import KagomeTN, ArbitraryTN, TensorNode, MPS
from tensor_networks.node import TensorNode
from lattices.directions import Direction, LatticeDirection, BlockSide, check
from enums import ContractionDepth, NodeFunctionality, UpdateMode
from containers import MPSOrientation, UpdateEdge

# Our utilities:
from utils import tuples, lists, assertions, prints, parallel_exec


L  = LatticeDirection.L
DR = LatticeDirection.DR
UR = LatticeDirection.UR
R  = LatticeDirection.R
UL = LatticeDirection.UL
DL = LatticeDirection.DL

CORE_CONNECTION_DIRECTIONS = {
    BlockSide.D  : [UL, UR],
    BlockSide.DR : [UL, L ],
    BlockSide.UR : [DL, L ],
    BlockSide.U  : [DL, DR],
    BlockSide.UL : [R , DR],
    BlockSide.DL : [R , UR]
}

CORE_CONNECTION_NODES = {
    BlockSide.D  : [7, 8],
    BlockSide.DR : [8, 6],
    BlockSide.UR : [6, 1],
    BlockSide.U  : [1, 0],
    BlockSide.UL : [0, 3],
    BlockSide.DL : [3, 7]
}

NUM_CONNECTIONS_PER_SIDE = 2  # number of connections per side




@dataclass
class _PerSide(Generic[_T]):
    """_Side(top_down, buttom_up)

    Used for repeating structure where one part comes from the top of the hexagonal block down, meeting first the top of the center triangle,
    and the other part comes from the buttom of the hexagonal block going up, meeting first the base of the center triangle.
    """
    top_down  : _T = field(default=None) 
    buttom_up : _T = field(default=None)

    def items(self)->Generator[tuple[str, _T], None, None]:
        yield "top_down" , self.top_down 
        yield "buttom_up", self.buttom_up 

    @staticmethod
    def side_names()->Generator[str, None, None]:
        yield "top_down" 
        yield "buttom_up"

    def __getitem__(self, key:str)->_T:
        if key == "top_down":  return self.top_down
        elif key == "buttom_up":  return self.buttom_up
        else:
            raise KeyError(f"No a valid option {key!r}")
    
    def __setitem__(self, key:str, value:_T)->None:
        if key == "top_down":  self.top_down = value
        elif key == "buttom_up":  self.buttom_up = value
        else:
            raise KeyError(f"No a valid option {key!r}")



def _zip_mps_at_overlap_outside_core(mpss:_PerSide[MPS], overlap_length:int, num_core_connections:_PerSide[int])->_PerSide[MPS]:
    LTensor : np.ndarray = None  #type: ignore
    for j in range(0, overlap_length):
        LTensor = bmpslib.updateCLeft(LTensor,
            mpss.top_down.A[-j-1].transpose([2,1,0]), \
            mpss.buttom_up.A[j])
    # LTensor legs are [i_up, i_down]. Absorb it in the first tensor of
    # mps 'from buttom up' that going to appear in the small TN
    mpss.buttom_up.A[overlap_length] = np.tensordot(LTensor, mpss.buttom_up.A[overlap_length], axes=([1],[0]))

    # Now create RTensor, which is the contraction of the right part
    # of the up/down MPSs
    RTensor : np.ndarray = None  #type: ignore
    for j in range(0, overlap_length):
        RTensor = bmpslib.updateCRight(RTensor,
            mpss.top_down.A[j].transpose([2,1,0]), \
            mpss.buttom_up.A[-1-j])
    # Absorb the RTensor[i_up, i_down] in the first tensor of mps 'top_down'
    mpss.top_down.A[overlap_length] = np.tensordot(RTensor, mpss.top_down.A[overlap_length], axes=([0],[0]))

    return mpss


def _contract_tn_from_sides_and_create_mpss(
    tn:KagomeTN,
    directions:_PerSide[BlockSide],
    bubblecon_trunc_dim:int,
    parallel:bool
)->tuple[
    _PerSide[MPS],
    _PerSide[list[int]],
    _PerSide[MPSOrientation]
]:
    # Decide control vars:
    print_progress = not parallel
    if parallel:
        prints.print_warning(f"Parallel=True not supported yet")

    # Prepare containers for each side of the zipping contraction:
    mpss : _PerSide[MPS] = _PerSide()
    con_orders : _PerSide[list[int]] = _PerSide()
    orientations : _PerSide[MPSOrientation] = _PerSide()
    # Contract:
    for side, direction in directions.items():
        mps, con_order, orientation = contract_tensor_network(
            tn, direction, depth=ContractionDepth.ToCore, bubblecon_trunc_dim=bubblecon_trunc_dim, print_progress=print_progress
        )
        mpss[side] = mps
        con_orders[side] = con_order
        orientations[side] = orientation

    return mpss, con_orders, orientations

def _reduce_tn_to_core_basic_data(
    tn:KagomeTN, parallel:bool
)->tuple[
    list[TensorNode],
    _PerSide[int],
    int,
    _PerSide[BlockSide]
]:

    # General data:
    N = tn.lattice.N
    core_nodes = tn.get_core_nodes()

    # Caused by the choice to always stop the contraction at the line defined by the base of the triangle:
    num_core_connections = _PerSide[int](
        top_down=7,
        buttom_up=5
    )

    ## Derive data for the zipping algorithm:
    num_side_overlap_connections = 2*N - 3  # length of overlap between sides of the zipping algorithm

    # Choose a random contraction direction that meets the base of the center triangle, first, and goes "up":
    from_buttom_up_direction = lists.random_item([BlockSide.U, BlockSide.DL, BlockSide.DR])  

    directions = _PerSide[BlockSide](
        buttom_up=from_buttom_up_direction,
        top_down=from_buttom_up_direction.opposite()
    )

    return core_nodes, num_core_connections, num_side_overlap_connections, directions


def _reduce_tn_to_core_verifications(con_orders, core_nodes, mpss, orientations, num_core_connections, num_side_overlap_connections, directions)->None:
    ## Check contraction order:
    _s0 = set(con_orders.buttom_up)
    _s1 = set(con_orders.top_down)
    _core = {node.index for node in core_nodes}
    assert _s0.isdisjoint(_s1)
    assert _core.isdisjoint(_s0)
    assert _core.isdisjoint(_s1)
    ## check resulting MPS orientations:
    check.is_opposite(orientations.buttom_up.open_towards, orientations.top_down.open_towards) 
    check.is_opposite(orientations.buttom_up.ordered,      orientations.top_down.ordered) 
    assert orientations.buttom_up.open_towards is directions.buttom_up
    assert orientations.top_down.open_towards is directions.top_down
    ## Check MPS sizes and overlaps:
    assert num_side_overlap_connections > 0
    assertions.integer(num_side_overlap_connections)
    for side in _PerSide.side_names():
        assert mpss[side].N == num_core_connections[side] + 2*num_side_overlap_connections


def _environment_tensors_in_canonical_order(mpss:_PerSide[MPSOrientation], directions:_PerSide[BlockSide], num_side_overlap_connections:int)->list[TensorNode]:
    ## Add the surrounding MPS tensors in the edge order: [D, DR, UR, U, UL, DL] 
    # (as seen from the perspective of the "buttom-up" durection) 
    s = num_side_overlap_connections  # number of indices to ignore from each side
    t = NUM_CONNECTIONS_PER_SIDE
    env_tensors = []
    env_tensors += mpss.buttom_up.A[s+t : s+2*t+1]  # D+DR
    env_tensors += mpss.top_down.A[s : s+1+3*t]  # DR -> UR -> U -> UL
    env_tensors += mpss.buttom_up.A[s : s+t]  # DL
    assert len(env_tensors)==t*6

    ## arrange items:
    # Canonical shape of the small tn hexagonal block is when the first env tensors are the ones connecting to the buttom of the block
    # So check how far are we from this rotation.
    match directions.buttom_up:
        case BlockSide.U:  rotations_distance = 0 
        case BlockSide.DL: rotations_distance = 2
        case BlockSide.DR: rotations_distance = 4
        case _:
            raise ValueError("Not such an option")
    env_tensors = lists.cycle_items(env_tensors, rotations_distance*2)

    return env_tensors


def _add_env_tensors_to_small_tn(small_tn:ArbitraryTN, env_tensors:list[TensorNode])->CoreTN:

    t = NUM_CONNECTIONS_PER_SIDE
    env_nodes : list[TensorNode] = [] 
    i = 0 
    for outside_direction in BlockSide.all_in_counter_clockwise_order():
        inside_direction : BlockSide = outside_direction.opposite()
        dir3 = inside_direction.orthogonal_clockwise_lattice_direction()
        dir1 = inside_direction.orthogonal_counterclockwise_lattice_direction()
        dir2_order = CORE_CONNECTION_DIRECTIONS[outside_direction]
        neighbors = CORE_CONNECTION_NODES[outside_direction]
        for dir2, inside_neighbor_index, j in zip(dir2_order, neighbors, range(2)):

            ## Collect data:
            inside_neighbor = small_tn.nodes[inside_neighbor_index]
            tensor = env_tensors[i]

            ## Derive data:
            dir2_opposite = dir2.opposite()
            new_connection_edge_name = f"{outside_direction.name}-{j}"
            pos = tuples.add(inside_neighbor.pos, tuples.multiply(dir2_opposite.unit_vector, 2))
            prev_edge = f"env{i}"
            if i < t*6-1:
                next_edge = f"env{i+1}"
            elif i == t*6-1:
                next_edge = f"env{0}"
            else:
                raise ValueError("Bug. Shouldn't be here")

            ## Create new node:
            node = TensorNode(
                index = small_tn.size,
                name = f"m{i}",
                tensor = tensor,
                is_ket = False,
                pos = pos,
                edges = [prev_edge, new_connection_edge_name, next_edge],
                directions = [dir1, dir2, dir3],
                functionality=NodeFunctionality.Environment
            )

            ## Update:
            inside_neighbor.boundaries.add(outside_direction)
            inside_neighbor.edges[inside_neighbor.directions.index(dir2_opposite)] = new_connection_edge_name
            small_tn.add_node(node)
            env_nodes.append(node)

            ## for next iteration:
            i += 1

    ## Fix direction of environment:
    for prev, this, next in lists.iterate_with_periodic_prev_next_items(env_nodes):
        if not check.is_opposite(prev.directions[-1], this.directions[0]) and this.directions[-1] is next.directions[-1]:
            prev.directions[-1] = this.directions[-1]

        if not check.is_opposite(this.directions[-1], next.directions[0]) and this.directions[0] is prev.directions[0]:
            next.directions[0] = this.directions[0]

    return CoreTN.from_arbitrary_tn(small_tn)


def reduce_tn_to_core(tn:KagomeTN, bubblecon_trunc_dim:int, parallel:bool=False) -> CoreTN:

    ## I. Parse and derive data
    core_nodes, num_core_connections, num_side_overlap_connections, directions = _reduce_tn_to_core_basic_data(tn, parallel)

    ## II. Prepare two MPSs, contract untill core:
	#      One MPS is "from the buttom-up" and the other is "from the top-down"
    mpss, con_orders, orientations = _contract_tn_from_sides_and_create_mpss(tn, directions, bubblecon_trunc_dim, parallel)
    
    ## Some verifications:
    if DEBUG_MODE:
        _reduce_tn_to_core_verifications(con_orders, core_nodes, mpss, orientations, num_core_connections, num_side_overlap_connections, directions)

	## III. Contract the upper/lower MPSs:
    mpss = _zip_mps_at_overlap_outside_core(mpss, num_side_overlap_connections, num_core_connections)

    # IV. We now have all tensors we need to define the small TN.
    #     It consists of the original TN in the small core + a periodic MPS
    #     tensors that surround it.

    ## first create the small TN and derive the canonical order of the env_tensors:
    core_nodes_in_canonical_order = lists.swap_items(core_nodes, 2, 3, copy=False)
    small_tn = ArbitraryTN(nodes=core_nodes_in_canonical_order)
    env_tensors = _environment_tensors_in_canonical_order(mpss, directions, num_side_overlap_connections, )

    ## add tensors-nodes into the tensor-network with the correct direction 
    core_tn = _add_env_tensors_to_small_tn(small_tn, env_tensors)

    if False:
        core_tn.plot()

    return core_tn 


def reduce_core_to_mode(
    core_tn:CoreTN, 
    mode:UpdateMode
)->ModeTN:
    
    ## Create a copy which is an arbitrary tn which can be contracted:
    tn = core_tn.to_arbitrary_tn()

    ## Get basic info:
    mode_side = mode.side_in_core  # Decide which side corrosponds to the mode:

    ## Also keep a list of nodes that should be contracted:
    new_nodes : list[TensorNode] = []

    ## Contract:
    # For each side not being the major core side
    for side in CoreTN.all_mode_sides:
        if side is mode_side:
            continue
            
        # For each boundry node
        boundary_nodes = [node for node in tn.get_nodes_on_boundary(side)]
        for boundary_node in boundary_nodes:

            # For each beighbor which is on thr environment:
            neigbors = tn.all_neighbors(boundary_node)
            for neigbor in neigbors:
                if neigbor.functionality is NodeFunctionality.Environment:
                    boundary_node = tn.contract_nodes(neigbor, boundary_node)  # output is the new boundary tensor
            
            # keep in list:
            new_nodes.append(boundary_node)

    ## Let those new tensors know they are part of the environemnt:
    for node in new_nodes:
        node.functionality = NodeFunctionality.Environment

    return ModeTN.from_arbitrary_tn(tn, mode=mode)





def reduce_mode_tn_to_edge_and_env(
    mode_tn:ModeTN, 
    edge_tuple:UpdateEdge
)->ArbitraryTN:
    
    edge_tn_not_arranged = _reduce_mode_tn_to_edge_and_env_center_version(mode_tn, edge_tuple)
    edge_tn_not_arranged.plot()
    print("Done")


def _reduce_mode_tn_to_edge_and_env_center_version(
    mode_tn:ModeTN, 
    edge_tuple:UpdateEdge,
)->ArbitraryTN:

    ## The correct edge nodes:
    if mode_tn.mode in edge_tuple:
        node1 = mode_tn.center_node
    else:
        options = mode_tn.get_nodes_by_functionality(NodeFunctionality.Core)
        node1 = next((node for node in options if node.unit_cell_flavor in edge_tuple))

    if edge_tuple.is_in_core():
        options = mode_tn.get_nodes_by_functionality(NodeFunctionality.CenterUnitCell)
    else:
        options = mode_tn.get_nodes_by_functionality(NodeFunctionality.Core)
    node2 = next((node for node in options if node.unit_cell_flavor in edge_tuple and node is not node1))
    edge = get_common_edge(node1, node2)

    ## Create copy than can be contracted:
    tn = mode_tn.to_arbitrary_tn()

    ## Find immediate neighbors:
    neighbors = []
    common_neighbor : TensorNode = None
    for node in tn.nodes:
        if node in [node1, node2]:
            continue
        is_neighbor1 = tn.are_neighbors(node, node1) 
        is_neighbor2 = tn.are_neighbors(node, node2)
        if is_neighbor1 or is_neighbor2:
            neighbors.append(node)
        if is_neighbor1 and is_neighbor2:
            common_neighbor = node
    assert common_neighbor is not None, "Bug. We must find a common neighbor"

    ## contract all other nodes into neighbors:
    nodes_to_keep = neighbors+[node1, node2]
    for old_neighbor in neighbors:
        new_neighbor = old_neighbor
        for node in tn.all_neighbors(new_neighbor):
            if node in nodes_to_keep:
                continue
            new_neighbor = tn.contract_nodes(node, new_neighbor)

        # replace old node in the `to_keep` list:
        nodes_to_keep.remove(old_neighbor)
        nodes_to_keep.append(new_neighbor)
        if old_neighbor is common_neighbor:
            common_neighbor = new_neighbor
        new_neighbor.functionality = NodeFunctionality.Environment

    ## Split common neighbor using QE-decomposition:
    center_directions, _, extrema_directions = _split_direction_by_type(common_neighbor.directions)
    directions_sorted = _sort_direction_by_angle_keeping_extrema(center_directions, extrema_directions)
    assert len(directions_sorted)==4
    edge1 = [common_neighbor.edge_in_dir(directions_sorted[i]) for i in [0,1]]
    edge2 = [common_neighbor.edge_in_dir(directions_sorted[i]) for i in [2,3]]
    # Perform qr-decomp on tn algo:
    _, _ = tn.qr_decomp(common_neighbor, edge1, edge2)

    return tn


def _sort_direction_by_angle_keeping_extrema(center_directions:list[Direction], extrema_directions:list[Direction])->list[Direction]:
    lis = sorted(center_directions+extrema_directions, key=lambda d: d.angle)
    if lis[-2] in extrema_directions and lis[-1] in extrema_directions:
        lis = lists.cycle_items(lis, 1)
    elif lis[0] in extrema_directions and lis[1] in extrema_directions:
        lis = lists.cycle_items(lis, -1)
    return lis


def _split_direction_by_type(directions:list[Direction])->tuple[list[LatticeDirection], list[BlockSide], list[Direction]]:
    lattice, block, arbitrary = [], [], []
    for dir in directions:
        if isinstance(dir, LatticeDirection):
            lattice.append(dir)
        elif isinstance(dir, BlockSide):
            block.append(dir)
        elif isinstance(dir, Direction):
            arbitrary.append(dir)
        else:
            raise TypeError(f"Not an expected type {type(dir)!r}")
    return lattice, block, arbitrary 


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


def _rearange_tensors_legs_to_canonical_order(
    tn_env:KagomeTN, side:None # Fix mode\side
)->tuple[TensorNode, TensorNode, list[np.ndarray]]:
    core_tensors = tn_env.get_nodes_by_functionality()
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

#TODO assert used
def calc_edge_environment(
    tn:KagomeTN, mode:None,  #TODO fix mode type
    bubblecon_trunc_dim:int, already_reduced_to_core:bool=False
)->tuple[
    TensorNode, TensorNode,         # core1/2
    list[np.ndarray],         # environment
    KagomeTN       # small_tn
]:
    ## Get the smallest Tensor-Network around the mode (edge):
    tn_env = calc_reduced_tn_around_edge(tn, mode, bubblecon_trunc_dim, method, already_reduced_to_core)

    ## get all tensors in correct order:
    core1, core2, environment_tensors = _rearange_tensors_legs_to_canonical_order(tn_env, mode)

    return core1, core2, environment_tensors, tn_env




if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import contraction_test
    contraction_test.main_test()

