
# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append( pathlib.Path(__file__).parent.parent.__str__() )

# Global config:
from _config_reader import DEBUG_MODE

from typing import Generator, Tuple, List, Dict

# For more computationally cheap functions and loops:
import itertools
import functools
import project_paths

# Import our shared utilities
from utils import tuples, lists, assertions, saveload, logs, decorators, errors, visuals, strings, dicts

# For common numeric functions:
import numpy as np
from numpy import matlib

# For common lattice function and classes:
from tensor_networks.node import TensorNode, NodeFunctionality
from tensor_networks.tensor_network import KagomeTensorNetwork, TensorDims
from lattices.directions import Direction, LatticeDirection
from lattices.edges import edges_dict_from_edges_list
from tensor_networks.unit_cell import UnitCell

from containers import TNSizesAndDimensions
from lattices.kagome import KagomeLattice


# ============================================================================ #
#|                             Constants                                      |#
# ============================================================================ #

RANDOM_CORE_IS_ABBA_LATTICE = True

L  = LatticeDirection.L
R  = LatticeDirection.R
UL = LatticeDirection.UL
UR = LatticeDirection.UR
DL = LatticeDirection.DL
DR = LatticeDirection.DR


# ============================================================================ #
#|                            Inner Functions                                 |#
# ============================================================================ #


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
        return NodeFunctionality.CenterUnitCell
    

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
	direction:Direction,  # ['L', 'R', 'U', 'D']
	N:int   # linear size of lattice
)->int:
	"""
	Calculde the idx of the neighboring blocks
	"""
	if direction==L:
		return i*N + (j - 1) % N
	if direction==R:
		return i*N + (j + 1) % N
	if direction==U:
		return ((i - 1) % N)*N + j
	if direction==D:
		return ((i + 1) % N)*N + j
	raise ValueError(f"'{direction}' not a supported input")


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
	direction:Direction,  # ['L', 'R', 'U', 'D']
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

	if direction=='L' and j==0:
		return f"L{i}"

	if direction=='R' and j==N-1:
		return f"R{i}"

	if direction=='D' and i==N-1:
		return f"D{j}"

	if direction=='U' and i==0:
		return f"U{j}"

	this = _get_index_from_coordinates(i, j, N)
	neighbor = _get_neighbor_from_coordinates(i, j, direction, N)
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
    core:KagomeTensorNetwork,
    repeats:int
) -> KagomeTensorNetwork:
    
    repeats = assertions.odd(repeats, reason="No support yet for odd big tensors")
    
    ## create a matrix of references to core 
    mat_nodes = np.empty( core.original_lattice_dims, dtype=TensorNode)
    mat_references = np.empty( core.original_lattice_dims, dtype=int)
    for ind, ((i, j), node) in enumerate(zip( _all_possible_indices(core.original_lattice_dims), core.nodes)):
        mat_nodes[i, j] = node
        mat_references[i, j] = ind        
    
    ## Easy method to find original core-node by index of expanded tensor-network:
    expanded_references = matlib.repmat(mat_references, repeats, repeats)    

    @functools.cache
    def get_core_node(ind:int)->TensorNode:
        indices = np.where(mat_references==ind)
        indices = [ i[0] for i in indices]
        node = mat_nodes[tuple(indices)]
        assert isinstance(node, TensorNode)
        return node

    ## Check result:
    new_shape = expanded_references.shape
    assert len(new_shape)==2
    assert new_shape[0]==new_shape[1]

    ## Prepare commonly used params:
    directions = list(directions.standard_order())
    core_size = core.original_lattice_dims[0]
    network_size = new_shape[0]
    padding = network_size-core_size

    ## Assign correct values to all nodes:
    nodes : list[TensorNode] = []
    for ind, (i, j) in enumerate(_all_possible_indices((network_size, network_size))):
        edges = [_get_edge_from_tensor_coordinates(i, j, direction, network_size) for direction in Directions]
        core_node = get_core_node(expanded_references[i,j])
        assert core_node.is_ket == True
        node = TensorNode(
            tensor = core_node.tensor,
            is_ket = True,
            edges = edges,
            pos = _get_position_from_tensor_coordinates(i, j, network_size), 
            boundaries=_check_tensor_on_boundary(edges),
            directions=directions,  #type: ignore  #TODO fix!
            index=ind,
            name=core_node.name,
            functionality=_derive_core_functionality(ind, core_size=core_size, padding=padding)
        )
        nodes.append(node)

    ## Create TN:
    edges_dict = edges_dict_from_edges_list([node.edges for node in nodes])
    tensor_dims = core.tensor_dims
    tn = KagomeTensorNetwork( nodes=nodes, edges=edges_dict, tensor_dims=tensor_dims )
    if DEBUG_MODE: tn.validate()

    return tn
         
     
     
def create_kagome_tn(
    d : int,  # Physical dimenstion 
    D : int,  # Virutal\Bond dimenstion  
    N : int,  # Lattice-size - Number of upper-triangles at each edge of the hexagon-block
    unit_cell : UnitCell|None = None
) -> KagomeTensorNetwork:

    # Get the kagome lattice without tensors:
    lattice = KagomeLattice(N)

    ## Unit cell:
    if unit_cell is None:
        unit_cell = UnitCell.random()
    else:
        assert isinstance(unit_cell, UnitCell)

    tn = KagomeTensorNetwork(lattice, unit_cell, d=d, D=D)
    
    return tn




# ============================================================================ #
#|                                   Test                                     |#
# ============================================================================ #


if __name__ == "__main__":
    project_paths.add_scripts()
    project_paths.add_base()
    from scripts.build_tn import draw_lattice
    draw_lattice()
    