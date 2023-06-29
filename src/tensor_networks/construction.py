
# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append( pathlib.Path(__file__).parent.parent.__str__() )

from typing import Generator, Tuple, List, Dict

# Get Global-Config:
from utils.config import VERBOSE_MODE, DEBUG_MODE

# For more computationally cheap functions and loops:
import itertools
import functools

# Import our shared utilities
from utils import tuples, lists, assertions, saveload, logs, decorators, errors, visuals, strings, dicts

# For common numeric functions:
import numpy as np
from numpy import pi
from numpy import matlib

# For common lattice function and classes:
from tensor_networks.node import Node, NodeFunctionality
from tensor_networks.tensor_network import TensorNetwork, TensorDims
from tensor_networks.edges import edges_dict_from_edges_list
from enums import Directions, Sides, InitialTNMode
from containers import TNSizesAndDimensions

# For pre-designed tensor-networks from other projects:
from translations import from_special_tensor, from_tnsu

# ============================================================================ #
#|                             Constants                                      |#
# ============================================================================ #

RANDOM_CORE_IS_ABBA_LATTICE = True

# ============================================================================ #
#|                            Inner Functions                                 |#
# ============================================================================ #

def _check_tensor_on_boundary(edges:List[str])->List[Directions]:
    on_edge : List[Directions] = []
    for edge in edges:
        first_letter = edge[0]
        if  first_letter.isalpha() and not first_letter.isdigit():
            on_edge.append(Directions(first_letter))
    return on_edge


def _derive_core_functionality(index:int, core_size:int, padding:int)->NodeFunctionality:
    # Basic Data:
    N = core_size + padding
    p = padding//2  # the pad in a single side
    i, j = _get_coordinates_from_index(index, N)
    # logics:
    if i<p or j<p:
        return NodeFunctionality.Padding
    if i+p>=N or j+p>=N:
        return NodeFunctionality.Padding
    else: 
        return NodeFunctionality.Core
    

def _random_tensor(d:int, D:int)->np.ndarray:
    rs = np.random.RandomState()
    t = rs.uniform(size=[d]+[D]*4) \
        + 1j*rs.normal(size=[d]+[D]*4)
    t = t/np.linalg.norm(t)
    return t



def _all_possible_couples(N:int) -> Generator[Tuple[int, int], None, None]:
	for i, j  in itertools.product(range(N), repeat=2):
		yield i, j

def _all_possible_indices(size:tuple[int,...])->Generator[Tuple[int, ...], None, None]:
    iterators = [range(s) for s in size]
    for indices in itertools.product(*iterators):
          yield indices

def _get_neighbor_from_coordinates(
	i:int,  # row
	j:int,  # column
	side:Sides,  # ['L', 'R', 'U', 'D']
	N:int   # linear size of lattice
)->int:
	"""
	Calculde the idx of the neighboring blocks
	"""
	if side=="L":
		return i*N + (j - 1) % N
	if side=="R":
		return i*N + (j + 1) % N
	if side=="U":
		return ((i - 1) % N)*N + j
	if side=="D":
		return ((i + 1) % N)*N + j
	raise ValueError(f"'{side}' not a supported input")


def _get_index_from_coordinates(
	i:int,  # row
	j:int,  # column
	N:int   # linear size of lattice
) -> int:
	return i*N + j


def _get_position_from_tensor_coordinates(
	i:int,
	j:int,
	N:int
)->tuple[int,...]:
    return (j, N-i-1)

def _get_edge_from_tensor_coordinates(
	i:int,
	j:int,
	side:Sides,  # ['L', 'R', 'U', 'D']
	N:int
)->str:
	"""
	Get the index of an edge in the double-layer PEPS.


	Input Parameters:
	------------------
	i,j --- location of the vertex to which the edge belong.
	        i=row, j=column. i=0 ==> upper row.

	side --- The side of the edge. Either 'L', 'R', 'U', 'D'

	N    --- Linear size of the lattice
	"""

	if side=='L' and j==0:
		return f"L{i}"

	if side=='R' and j==N-1:
		return f"R{i}"

	if side=='D' and i==N-1:
		return f"D{j}"

	if side=='U' and i==0:
		return f"U{j}"

	this = _get_index_from_coordinates(i, j, N)
	neighbor = _get_neighbor_from_coordinates(i, j, side, N)
	return f"{min(this,neighbor)}-{max(this,neighbor)}"


def _get_coordinates_from_index(
	ind:int,  # index of node  
	N:int   # linear size of lattice
) -> Tuple[int, int]:
	
	quotient, remainder = divmod(ind, N)
	return quotient, remainder


# ============================================================================ #
#|                           Declared Function                                |#
# ============================================================================ #

def repeat_core(
    core:TensorNetwork,
    repeats:int
) -> TensorNetwork:
    
    repeats = assertions.odd(repeats, reason="No support yet for odd big tensors")
    
    ## create a matrix of references to core 
    mat_nodes = np.empty( core.original_lattice_dims, dtype=Node)
    mat_references = np.empty( core.original_lattice_dims, dtype=int)
    for ind, ((i, j), node) in enumerate(zip( _all_possible_indices(core.original_lattice_dims), core.nodes)):
        mat_nodes[i, j] = node
        mat_references[i, j] = ind        
    
    ## Easy method to find original core-node by index of expanded tensor-network:
    expanded_references = matlib.repmat(mat_references, repeats, repeats)    

    @functools.cache
    def get_core_node(ind:int)->Node:
        indices = np.where(mat_references==ind)
        indices = [ i[0] for i in indices]
        node = mat_nodes[tuple(indices)]
        assert isinstance(node, Node)
        return node

    ## Check result:
    new_shape = expanded_references.shape
    assert len(new_shape)==2
    assert new_shape[0]==new_shape[1]

    ## Prepare commonly used params:
    directions = list(Directions.standard_order())
    core_size = core.original_lattice_dims[0]
    network_size = new_shape[0]
    padding = network_size-core_size

    ## Assign correct values to all nodes:
    nodes : list[Node] = []
    for ind, (i, j) in enumerate(_all_possible_indices((network_size, network_size))):
        edges = [_get_edge_from_tensor_coordinates(i, j, direction, network_size) for direction in Directions]
        core_node = get_core_node(expanded_references[i,j])
        assert core_node.is_ket == True
        node = Node(
            tensor = core_node.tensor,
            is_ket = True,
            edges = edges,
            pos = _get_position_from_tensor_coordinates(i, j, network_size), 
            on_boundary=_check_tensor_on_boundary(edges),
            directions=directions,  #type: ignore  #TODO fix!
            index=ind,
            name=core_node.name,
            functionality=_derive_core_functionality(ind, core_size=core_size, padding=padding)
        )
        nodes.append(node)

    ## Create TN:
    edges_dict = edges_dict_from_edges_list([node.edges for node in nodes])
    tensor_dims = core.tensor_dims
    tn = TensorNetwork( nodes=nodes, edges=edges_dict, tensor_dims=tensor_dims )
    if DEBUG_MODE: tn.validate()

    return tn
          


def create_physical_peps(
    size:int,
    d_virtual:int,
    d_physical:int,
    creation_mode:InitialTNMode,
    h:float|None=None
)->list[np.ndarray]:
    if creation_mode==InitialTNMode._SimpleUpdateResult:
        assert hasattr(creation_mode, "h"), f"A workaround where the enumeration has also a property"
        h = getattr(creation_mode, "h")
        assert isinstance(h, float)
        return from_tnsu.construct_full_lattice_from_itf_experiment_core(edge_size=size, physical_dim=d_physical, virtual_dim=d_virtual, h=h)
    elif creation_mode==InitialTNMode.SpecialTensor:
        assert d_virtual==3, f"Special-Tensor is a tensor with virtual dimension 3. Got d_virtual={d_virtual}"
        return from_special_tensor.construct_full_lattice_from_special_tensor(size)
    elif creation_mode==InitialTNMode.Random:
        if RANDOM_CORE_IS_ABBA_LATTICE:
            a = _random_tensor(D=d_virtual, d=d_physical)
            b = _random_tensor(D=d_virtual, d=d_physical)
            return [a, b, b, a]
        else:
            return [_random_tensor(D=d_virtual, d=d_physical) for _ in range(size**2) ]
    else:
        raise ValueError(f"Not a valid option. got initial_tn={creation_mode!r}.")    


def create_core(
    config:TNSizesAndDimensions,
    creation_mode:InitialTNMode|list[np.ndarray]=InitialTNMode.Random,
    _check_core:bool=True
) -> TensorNetwork:
    core = create_square_tn(core_size=config.core_size, padding=0, d_physical=config.physical_dim, d_virtual=config.virtual_dim, creation_mode=creation_mode)
    core.nodes[0].name = "A"
    core.nodes[1].name = "B"
    core.nodes[2].name = "B"
    core.nodes[3].name = "A"
    if DEBUG_MODE and _check_core:
         assert core.nodes[0] == core.nodes[3]  # A
         assert core.nodes[1] == core.nodes[2]  # B
    return core
     
     
def create_square_tn(
    core_size   : int, 
    d_virtual   : int, 
    d_physical  : int,
    padding:int=0, 
    creation_mode:InitialTNMode|list[np.ndarray]=InitialTNMode.Random,
) -> TensorNetwork:
    # Derive from inputs:
    tn_size = core_size + padding

    ## Physical Tensors:
    if isinstance(creation_mode, InitialTNMode):
        physical_peps = create_physical_peps(size=tn_size, d_physical=d_physical, d_virtual=d_virtual, creation_mode=creation_mode)
    elif isinstance(creation_mode, list) and isinstance(creation_mode[0], np.ndarray):
        physical_peps = [p.copy() for p in creation_mode]  
    else:
        raise ValueError(f"Not a valid option. got initial_tn={creation_mode!r}.")    
    
    ## Init lists:
    edges_list = []
    pos_list = []
    directions = list(Directions.standard_order())

    ## Edges and Positions:
    for i, j in _all_possible_couples(tn_size):
        edges = [_get_edge_from_tensor_coordinates(i, j, direction, tn_size) for direction in directions]
        edges_list.append(edges)
        pos_list.append( _get_position_from_tensor_coordinates(i, j, tn_size) )

    # Derive edges dict. each key is an edge and each value is the tensors connected to it:
    edges_dict = edges_dict_from_edges_list(edges_list)

    # Validate:
    if DEBUG_MODE: assert len(physical_peps)==len(edges_list)==len(pos_list)

    # Wrap all data:
    nodes = [
        Node(
            tensor=physical,
            is_ket=True,
            edges=edges,
            pos=pos, 
            on_boundary=_check_tensor_on_boundary(edges),
            directions=directions,
            index=index,
            name=f"T{index}",
            functionality=_derive_core_functionality(index, core_size=core_size, padding=padding)
        )
        for index, (physical, edges, pos) 
        in enumerate(zip(physical_peps, edges_list, pos_list))
    ]
    tensor_dims = TensorDims(virtual=d_virtual, physical=d_physical)

    return TensorNetwork( nodes=nodes, edges=edges_dict, tensor_dims=tensor_dims )




# ============================================================================ #
#|                                   Test                                     |#
# ============================================================================ #


if __name__ == "__main__":
     from scripts.core_ite_test import main
     main()