# For allowing tests and scripts to run while debugging this module
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
    from project_paths import add_scripts; add_scripts()

# Control flags:
from _config_reader import DEBUG_MODE

# For a little bit of OOP:
from typing import Generic, TypeVar, Generator
from dataclasses import dataclass, field
_T = TypeVar("_T")

# Common types in the code:
from tensor_networks import KagomeTNRepeatedUntiCell, ArbitraryTN, ModeTN, TensorNode, MPS, CoreTN, get_common_edge

# Everyone needs numpy:
import numpy as np

# Other modules we made and need here   
from libs import bmpslib

# Other algos we need here:
from algo.contract_tensor_network import contract_tensor_network

# Types we need in our module:
from tensor_networks import KagomeTNRepeatedUntiCell, ArbitraryTN, TensorNode, MPS
from tensor_networks.node import TensorNode, two_nodes_ordered_by_relative_direction
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

VALID_BOTTOM_UP_CONTRACTION_TO_CORE_DIRECTIONS = [BlockSide.U, BlockSide.DL, BlockSide.DR]


@dataclass
class _PerSide(Generic[_T]):
    """_Side(top_down, bottom_up)

    Used for repeating structure where one part comes from the top of the hexagonal block down, meeting first the top of the center triangle,
    and the other part comes from the bottom of the hexagonal block going up, meeting first the base of the center triangle.
    """
    top_down  : _T = field(default=None) 
    bottom_up : _T = field(default=None)

    def items(self)->Generator[tuple[str, _T], None, None]:
        yield "top_down" , self.top_down 
        yield "bottom_up", self.bottom_up 

    @staticmethod
    def side_names()->Generator[str, None, None]:
        yield "top_down" 
        yield "bottom_up"

    def __getitem__(self, key:str)->_T:
        if key == "top_down":  return self.top_down
        elif key == "bottom_up":  return self.bottom_up
        else:
            raise KeyError(f"No a valid option {key!r}")
    
    def __setitem__(self, key:str, value:_T)->None:
        if key == "top_down":  self.top_down = value
        elif key == "bottom_up":  self.bottom_up = value
        else:
            raise KeyError(f"No a valid option {key!r}")


def _contract_half(
    side_key:str, 
    tn:KagomeTNRepeatedUntiCell,
    directions:_PerSide[BlockSide],
    bubblecon_trunc_dim:int,
    print_progress:bool
)->tuple[MPS|complex|tuple, list[int], MPSOrientation]:        
    """Contraction function that contracts half TN:
    """
    direction = directions[side_key]
    mps, con_order, orientation = contract_tensor_network(
        tn, direction, depth=ContractionDepth.ToCore, bubblecon_trunc_dim=bubblecon_trunc_dim, print_progress=print_progress
    )
    return mps, con_order, orientation


def _contract_tn_from_sides_and_create_mpss(
    tn:KagomeTNRepeatedUntiCell,
    directions:_PerSide[BlockSide],
    bubblecon_trunc_dim:int,
    parallel:bool
)->tuple[
    _PerSide[MPS],
    _PerSide[list[int]],
    _PerSide[MPSOrientation]
]:
    
    ## Run in parallel or sequential:
    fixed_arguments = dict(
        tn = tn,
        directions = directions,
        bubblecon_trunc_dim = bubblecon_trunc_dim,
        print_progress = not parallel
    )
    values = list(_PerSide.side_names())
    res = parallel_exec.concurrent_or_parallel(_contract_half, values=values, value_name="side_key", in_parallel=parallel, fixed_arguments=fixed_arguments)

    ## Unpack Results:
    mpss         : _PerSide[MPS]            = _PerSide()
    con_orders   : _PerSide[list[int]]      = _PerSide()
    orientations : _PerSide[MPSOrientation] = _PerSide()
    for key, (mps, con_order, orientation) in res.items():
        mpss[key] = mps
        con_orders[key] = con_order
        orientations[key] = orientation 
    
    return mpss, con_orders, orientations

def _basic_data(
    tn:KagomeTNRepeatedUntiCell, parallel:bool, direction:BlockSide|None
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
        bottom_up=5
    )

    ## Derive data for the zipping algorithm:
    num_side_overlap_connections = 2*N - 3  # length of overlap between sides of the zipping algorithm

    # Choose a random contraction direction that meets the base of the center triangle, first, and goes "up":
    if direction is None:
        from_bottom_up_direction = lists.random_item(VALID_BOTTOM_UP_CONTRACTION_TO_CORE_DIRECTIONS)  
    else:
        assert direction in VALID_BOTTOM_UP_CONTRACTION_TO_CORE_DIRECTIONS, f"Not all directions can be used for contraction to core"
        from_bottom_up_direction = direction

    directions = _PerSide[BlockSide](
        bottom_up=from_bottom_up_direction,
        top_down=from_bottom_up_direction.opposite()
    )

    return core_nodes, num_core_connections, num_side_overlap_connections, directions


def _zip_mps_at_overlap_outside_core(mpss:_PerSide[MPS], overlap_length:int, num_core_connections:_PerSide[int])->_PerSide[MPS]:
    LTensor : np.ndarray = None  #type: ignore
    for j in range(0, overlap_length):
        LTensor = bmpslib.updateCLeft(LTensor,
            mpss.top_down.A[-j-1].transpose([2,1,0]), \
            mpss.bottom_up.A[j])
    # LTensor legs are [i_up, i_down]. Absorb it in the first tensor of
    # mps 'from bottom up' that going to appear in the small TN
    mpss.bottom_up.A[overlap_length] = np.tensordot(LTensor, mpss.bottom_up.A[overlap_length], axes=([1],[0]))

    # Now create RTensor, which is the contraction of the right part
    # of the up/down MPSs
    RTensor : np.ndarray = None  #type: ignore
    for j in range(0, overlap_length):
        RTensor = bmpslib.updateCRight(RTensor,
            mpss.top_down.A[j].transpose([2,1,0]), \
            mpss.bottom_up.A[-1-j])
    # Absorb the RTensor[i_up, i_down] in the first tensor of mps 'top_down'
    mpss.top_down.A[overlap_length] = np.tensordot(RTensor, mpss.top_down.A[overlap_length], axes=([0],[0]))

    return mpss


def _verification(con_orders, core_nodes, mpss, orientations, num_core_connections, num_side_overlap_connections, directions)->None:
    ## Check contraction order:
    _s0 = set(con_orders.bottom_up)
    _s1 = set(con_orders.top_down)
    _core = {node.index for node in core_nodes}
    assert _s0.isdisjoint(_s1)
    assert _core.isdisjoint(_s0)
    assert _core.isdisjoint(_s1)
    ## check resulting MPS orientations:
    check.is_opposite(orientations.bottom_up.open_towards, orientations.top_down.open_towards) 
    check.is_opposite(orientations.bottom_up.ordered,      orientations.top_down.ordered) 
    assert orientations.bottom_up.open_towards == directions.bottom_up
    assert orientations.top_down.open_towards == directions.top_down
    ## Check MPS sizes and overlaps:
    assert num_side_overlap_connections > 0
    assertions.integer(num_side_overlap_connections)
    for side in _PerSide.side_names():
        assert mpss[side].N == num_core_connections[side] + 2*num_side_overlap_connections


def _environment_tensors_in_canonical_order(mpss:_PerSide[MPSOrientation], directions:_PerSide[BlockSide], num_side_overlap_connections:int)->list[TensorNode]:
    ## Add the surrounding MPS tensors in the edge order: [D, DR, UR, U, UL, DL] 
    # (as seen from the perspective of the "bottom-up" direction) 
    s = num_side_overlap_connections  # number of indices to ignore from each side
    t = NUM_CONNECTIONS_PER_SIDE
    env_tensors = []
    env_tensors += mpss.bottom_up.A[s+t : s+2*t+1]  # D+DR
    env_tensors += mpss.top_down.A[s : s+1+3*t]  # DR -> UR -> U -> UL
    env_tensors += mpss.bottom_up.A[s : s+t]  # DL
    assert len(env_tensors)==t*6

    ## arrange items:
    # Canonical shape of the small tn hexagonal block is when the first env tensors are the ones connecting to the bottom of the block
    # So check how far are we from this rotation.
    match directions.bottom_up:
        case BlockSide.U:  rotations_distance = 0 
        case BlockSide.DL: rotations_distance = 2
        case BlockSide.DR: rotations_distance = 4
        case _:
            raise ValueError("Not such an option")
    env_tensors = lists.cycle_items(env_tensors, rotations_distance*2)

    return env_tensors


def _add_env_tensors_to_open_core(small_tn:ArbitraryTN, env_tensors:list[TensorNode])->CoreTN:

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

    return small_tn



def reduce_full_kagome_to_core(tn:KagomeTNRepeatedUntiCell, trunc_dim:int, parallel:bool=False, direction:BlockSide|None=None) -> CoreTN:
    ## 0: Check inputs:
    if DEBUG_MODE:
        tn.validate()
    assert tn.has_messages, "To reduce to core, KagomeTN must have mps messages."

    ## I. Parse and derive data
    core_nodes, num_core_connections, num_side_overlap_connections, directions = _basic_data(tn, parallel, direction)

    ## II. Prepare two MPSs, contract until core:
	#      One MPS is "from the bottom-up" and the other is "from the top-down"
    mpss, con_orders, orientations = _contract_tn_from_sides_and_create_mpss(tn, directions, trunc_dim, parallel)
    
    ## Some verifications:
    if DEBUG_MODE:
        _verification(con_orders, core_nodes, mpss, orientations, num_core_connections, num_side_overlap_connections, directions)

	## III. Contract the upper/lower MPSs:
    mpss = _zip_mps_at_overlap_outside_core(mpss, num_side_overlap_connections, num_core_connections)

    # IV. We now have all tensors we need to define the small TN.
    #     It consists of the original TN in the small core + a periodic MPS
    #     tensors that surround it.

    ## first create the small TN and derive the canonical order of the env_tensors:
    core_nodes_in_canonical_order = lists.swap_items(core_nodes, 2, 3, copy=False)
    open_core_tn = ArbitraryTN(nodes=core_nodes_in_canonical_order)
    env_tensors = _environment_tensors_in_canonical_order(mpss, directions, num_side_overlap_connections, )

    ## add tensors-nodes into the tensor-network with the correct direction 
    small_tn = _add_env_tensors_to_open_core(open_core_tn, env_tensors)

    core_tn = CoreTN.from_arbitrary_tn(small_tn)
    if DEBUG_MODE:
        core_tn.validate()

    return core_tn 




if __name__ == "__main__":
    from scripts.test_parallel import main_test
    main_test()