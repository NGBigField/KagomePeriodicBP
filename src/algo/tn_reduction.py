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
from tensor_networks import KagomeTN, ArbitraryTN, TensorNode, MPS, CoreTN

# Everyone needs numpy:
import numpy as np

# Other modules we made and need here   
from libs import bmpslib

# Other algos we need here:
from algo.contract_tensor_network import contract_kagome_tensor_network

# Types we need in our module:
from tensor_networks import KagomeTN, ArbitraryTN, TensorNode, MPS
from tensor_networks.node import TensorNode
from lattices.directions import LatticeDirection, BlockSide, check
from enums import ContractionDepth, NodeFunctionality, UpdateModes
from containers import MPSOrientation

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
        mps, con_order, orientation = contract_kagome_tensor_network(
            tn, direction, depth=ContractionDepth.ToCore, bubblecon_trunc_dim=bubblecon_trunc_dim, print_progress=print_progress
        )
        mpss[side] = mps
        con_orders[side] = con_order
        orientations[side] = orientation

    return mpss, con_orders, orientations

def _basic_data(
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


def _verifications(con_orders, core_nodes, mpss, orientations, num_core_connections, num_side_overlap_connections, directions)->None:
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
            small_tn.add(node)
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
    core_nodes, num_core_connections, num_side_overlap_connections, directions = _basic_data(tn, parallel)

    ## II. Prepare two MPSs, contract untill core:
	#      One MPS is "from the buttom-up" and the other is "from the top-down"
    mpss, con_orders, orientations = _contract_tn_from_sides_and_create_mpss(tn, directions, bubblecon_trunc_dim, parallel)
    
    ## Some verifications:
    if DEBUG_MODE:
        _verifications(con_orders, core_nodes, mpss, orientations, num_core_connections, num_side_overlap_connections, directions)

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
    bubblecon_trunc_dim:int,
    mode:UpdateModes
)->ArbitraryTN:
    
    ## Create a copy which is an arbitrary tn which can be contracted:
    tn = core_tn.to_arbitrary_tn()

    ## Get basic info:
    mode_side = mode.side_in_core  # Decide which side corrosponds to the mode:

    from utils import visuals
    i = 0
    def plot_():
        nonlocal i
        tn.plot()
        visuals.save_figure(file_name=f"{i}")
        i += 1
    plot_()

    ## Find the nodes that should be contracted:
    for side in CoreTN.all_mode_sides:
        if side is mode_side:
            continue
            
        boundary_nodes = [node for node in tn.get_nodes_on_boundary(side)]
        for boundary_node in boundary_nodes:
            neigbors = tn.all_neighbors(boundary_node)
            for neigbor in neigbors:
                if neigbor.functionality is NodeFunctionality.Environment:
                    boundary_node = tn.contract_nodes(neigbor, boundary_node)
                    plot_()
    
    return tn


def reduce_core_and_environment_to_edge_and_environment(
    tn_small:KagomeTN, side:None, # fix side\mode 
    bubblecon_trunc_dim:int
)->KagomeTN:
    tn_env = reduce_tn_using_bubblecon(tn_small, bubblecon_trunc_dim=bubblecon_trunc_dim, directions=[side], depth=2)
    ## Find and Swallow the two corner tensors:
    remaining_core_tensors = [t for t in tn_env.nodes if t.functionality is NodeFunctionality.CenterUnitCell]
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


#TODO check if needed
def reduce_tn_using_bubblecon(tn:KagomeTN, bubblecon_trunc_dim:int, directions:Iterable[BlockSide], depth:ContractionDepth|int, parallel:bool=False)->KagomeTN:

    # prepare inputs:
    fixed_arguments = dict(tn=tn, bubblecon_trunc_dim=bubblecon_trunc_dim, depth=depth)
    directions = lists.shuffle(list(directions))
    
    # Sandwich Tensor-Network from both sides at once if parallel:
    if parallel:
        fixed_arguments["print_progress"]=False
        con_results = parallel_exec.parallel(func=contract_kagome_tensor_network, values=directions, value_name="direction", fixed_arguments=fixed_arguments) 
    else:
        fixed_arguments["print_progress"] = True
        con_results = parallel_exec.concurrent(func=contract_kagome_tensor_network, values=directions, value_name="direction", fixed_arguments=fixed_arguments) 
    
    # Rearrange outputs:
    mpss        = {direction:tupl[0] for direction, tupl in con_results.items()}
    con_indices = lists.join_sub_lists([tupl[1] for tupl in con_results.values()])
    mps_orientations =                  [tupl[2] for tupl in con_results.values()]


    ## Ignore tensors that are accounted-for by the messages:
    remaining_indices = [node.index for node in tn.nodes if node.index not in con_indices]
    reduced_tn = tn.sub_tn(remaining_indices)
    if DEBUG_MODE: reduced_tn.validate()
    
    ## Connect messages directly to the remaining tensors:
    for (direction, mps), orientation in zip(mpss.items(), mps_orientations, strict=True):
        assert isinstance(mps, MPS)
        reduced_tn = _fuse_mps_with_tn( reduced_tn, mps, orientation, direction.opposite() )
    if DEBUG_MODE: reduced_tn.validate()

    ## Return:
    return reduced_tn



if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import contraction_test
    contraction_test.main_test()

