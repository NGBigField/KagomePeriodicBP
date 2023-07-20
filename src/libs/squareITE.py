#----------------------------------------------------------
#
#               sqaureITE.py
#
#  Use blockbp to calculate the ground state of a 2-local
#  Hamiltoninan on a square grid with periodic boundary
#  conditions.
#
#
#----------------------------------------------------------
#

# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)

# Import required packages:
from lib import bmpslib
from lib import bubblecon
from lib import ITE
import lib.blockbp as blockbp

from lib.ITE import fuse_tensor 



# For type hinting:
from typing import (
	Any,
	List,
	Callable,
	Tuple, 
	Dict,
	Final,
	Literal,
	TypeAlias,
	Generator,
	Optional,
)
from _blockbp.classes import MessageModel, BlocksConType

# Numeric stuff:
import numpy as np
from numpy.linalg import norm, svd, qr
from numpy import zeros, ones, array, tensordot, sqrt, diag, dot, \
	reshape, transpose, conj, eye, trace, pi, exp, sinh, log
from scipy.linalg import expm

# Bad practice:
from sys import exit

from ncon import ncon

try:
    from mpi4py import MPI
except ImportError as e:
    pass  # module doesn't exist, deal with it.
    MPI=None

PYX=1
try:
    import pyx
except ImportError as e:
    pass  # module doesn't exist, deal with it.
    PYX=0


import ctypes
import time

# For smart iterations:
import itertools

# ============================================================================ #
#|                             Helper Types                                   |#
# ============================================================================ #


# ============================================================================ #
#|                               Constants                                    |#
# ============================================================================ #


# ============================================================================ #
#|                            Inner Functions                                 |#
# ============================================================================ #
def _standard_super_edge_name(
	ind1:int, 
	ind2:int
) -> str :
	return f"se:{min(ind1, ind2)}-{max(ind1, ind2)}"

def _all_possible_couples(N:int) -> Generator[Tuple[int, int], None, None]:
	for i, j  in itertools.product(range(N), repeat=2):
		yield i, j



def _contraction_order_per_direction(dir:str, ib:int, jb:int, n:int, N:int) -> Tuple[List[Any], float]:
	sw=[]

	num_blocks = N//n  # how many blocks in a side
	if num_blocks==1:
		def _tuple(type_:str, value:int, edge_side:str="")-> tuple:
			if type_=="e":
				val3 = edge_side
			elif type_=="v":
				assert edge_side==""
				val3 = None
			else:
				raise ValueError(f"Not a supported type_ {type_}")
			return (type_, value, val3)

		get_edge_from_tensor_coordinates_ = get_edge_from_tensor_coordinates_1block

	else:
		def _tuple(type_:str, value:int, edge_side:str="")-> tuple:
			return (type_, value)

		get_edge_from_tensor_coordinates_ = get_edge_from_tensor_coordinates

	def _tuple(type_:str, value:int, edge_side:str="")-> tuple:
		return (type_, value)

	
	# Left sedge
	if dir == 'L':
		angle = pi
		for i in range(ib*n, ib*n+n):
			j=jb*n + n-1
			e = get_edge_from_tensor_coordinates_(i,j,'R',N)
			sw = sw + [_tuple('e', e, "R")]

		mode='bottom->top'
		for j in range(jb*n+n-1, jb*n-1, -1):

			if mode=='top->bottom':

				for i in range(ib*n, ib*n+n):
					sw = sw + [_tuple('v', i*N+j)]
					if i==ib*n:
						e = get_edge_from_tensor_coordinates_(i,j,'U',N)
						sw = sw + [_tuple('e',e, "U")]

				i = ib*n+n-1
				e = get_edge_from_tensor_coordinates_(i,j,'D',N)
				sw = sw + [_tuple('e',e, "D")]

				mode = 'bottom->top'
			else:

				for i in range(ib*n+n-1, ib*n-1, -1):
					sw = sw + [_tuple('v', i*N+j)]
					if i==ib*n+n-1:
						e = get_edge_from_tensor_coordinates_(i, j, 'D', N)
						sw = sw + [_tuple('e',e, "D")]

				i = ib*n
				e = get_edge_from_tensor_coordinates_(i,j,'U',N)
				sw = sw + [_tuple('e',e, "U")]

				mode = 'top->bottom'

	# Right sedge
	elif dir == 'R':
		angle = 0
		for i in range(ib*n, ib*n+n):
			j=jb*n
			e = get_edge_from_tensor_coordinates_(i,j,'L',N)
			sw = sw + [_tuple('e',e, "L")]

		mode = 'bottom->top'

		for j in range(jb*n, jb*n+n):

			if mode=='top->bottom':
				for i in range(ib*n, ib*n+n):
					sw = sw + [_tuple('v', i*N+j)]
					if i==ib*n:
						e = get_edge_from_tensor_coordinates_(i,j,'U',N)
						sw = sw + [_tuple('e',e, "U")]

				i = ib*n+n-1
				e = get_edge_from_tensor_coordinates_(i,j,'D',N)
				sw = sw + [_tuple('e',e, "D")]

				mode = 'bottom->top'

			else:

				for i in range(ib*n+n-1, ib*n-1, -1):
					sw = sw + [_tuple('v', i*N+j)]
					if i==ib*n+n-1:
						e = get_edge_from_tensor_coordinates_(i,j,'D',N)
						sw = sw + [_tuple('e',e, "D")]

				i = ib*n
				e = get_edge_from_tensor_coordinates_(i,j,'U',N)
				sw = sw + [_tuple('e',e, "U")]

				mode = 'top->bottom'

	# Up sedge
	elif  dir == 'U':
		angle = pi/2
		for j in range(jb*n, jb*n+n):
			i=ib*n+n-1
			e = get_edge_from_tensor_coordinates_(i,j,'D',N)
			sw = sw + [_tuple('e',e, "D")]

		mode = 'right->left'

		for i in range(ib*n+n-1, ib*n-1, -1):

			if mode=='left->right':

				for j in range(jb*n, jb*n+n):
					sw = sw + [_tuple('v', i*N+j)]
					if j==jb*n:
						e = get_edge_from_tensor_coordinates_(i,j,'L',N)
						sw = sw + [_tuple('e',e, "L")]

				j = jb*n+n-1
				e = get_edge_from_tensor_coordinates_(i,j,'R',N)
				sw = sw + [_tuple('e',e, "R")]

				mode = 'right->left'

			else:
				for j in range(jb*n+n-1, jb*n-1, -1):
					sw = sw + [_tuple('v', i*N+j)]
					if j==jb*n+n-1:
						e = get_edge_from_tensor_coordinates_(i,j,'R',N)
						sw = sw + [_tuple('e',e, "R")]

				j = jb*n
				e = get_edge_from_tensor_coordinates_(i,j,'L',N)
				sw = sw + [_tuple('e',e, "L")]

				mode = 'left->right'

	# Down sedge
	elif dir == 'D':
		angle = 3*pi/2
		for j in range(jb*n, jb*n+n):
			i=ib*n
			e = get_edge_from_tensor_coordinates_(i,j,'U',N)
			sw = sw + [_tuple('e',e, "U")]

		mode = 'right->left'

		for i in range(ib*n, ib*n+n):

			if mode=='left->right':

				for j in range(jb*n, jb*n+n):
					sw = sw + [_tuple('v', i*N+j)]
					if j==jb*n:
						e = get_edge_from_tensor_coordinates_(i,j,'L',N)
						sw = sw + [_tuple('e',e, "L")]

				j = jb*n+n-1
				e = get_edge_from_tensor_coordinates_(i,j,'R',N)
				sw = sw + [_tuple('e',e, "R")]

				mode = 'right->left'

			else:
				for j in range(jb*n+n-1, jb*n-1, -1):
					sw = sw + [_tuple('v', i*N+j)]
					if j==jb*n+n-1:
						e = get_edge_from_tensor_coordinates_(i,j,'R',N)
						sw = sw + [_tuple('e',e, "R")]

				j = jb*n
				e = get_edge_from_tensor_coordinates_(i,j,'L',N)
				sw = sw + [_tuple('e',e, "L")]

				mode = 'left->right'
	else:
		raise ValueError(f"'{dir}' not a possible direction.")

	return (sw, angle)



# ============================================================================ #
#|                          Declared Functions                                |#
# ============================================================================ #


def all_boundary_tensors(
	N:int,  # Num tensors in edge of lattice
	side: Literal['L', 'R', 'U', 'D']
) -> Generator[int, None, None]:
	for i in range(N):
		if side == "L":
			yield i*N
		elif side == "R":
			yield i*N + (N-1)
		elif side == "U":
			yield i
		elif side == "D":
			yield i + (N-1)*N
		else:
			raise ValueError("'{side}' not a supported input")


def all_possible_sides()->Generator[str, None, None]:
	for side in ['L', 'R', 'U', 'D']:
		yield side

def check_boundary_tensor(
	ind:int, # index of node  
	N:int    # linear size of lattice
)->List[str]: # Edges L R U D  if tensor is touching the related boundary

	def _tensor_in_boundary(side:str)->bool:
		return ind in all_boundary_tensors(N, side)
		# Can be done more efficiently if really needed

	return [side for side in all_possible_sides() if _tensor_in_boundary(side)]



def get_neighbor_from_coordinates(
	i:int,  # row
	j:int,  # column
	side:Literal['L', 'R', 'U', 'D'],
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
	raise ValueError("'{side}' not a supported input")

def get_neighbor_from_index(
	ind:int,  # index of node  
	side:Literal['L', 'R', 'U', 'D'],
	N:int   # linear size of lattice
)->int:
	"""
	Calculde the idx of the neighboring blocks
	"""
	i, j = get_coordinates_from_index(ind, N)
	return get_neighbor_from_coordinates(i, j, side, N)

def get_coordinates_from_index(
	ind:int,  # index of node  
	N:int   # linear size of lattice
) -> Tuple[int, int]:
	
	quotient, remainder = divmod(ind, N)
	return quotient, remainder

def get_index_from_coordinates(
	i:int,  # row
	j:int,  # column
	N:int   # linear size of lattice
) -> int:
	return i*N + j

def get_edge_from_tensor_index_1block(
	ind:int,
	side:Literal['L', 'R', 'U', 'D'],
	N:int
)->str:
	row, column = get_coordinates_from_index(ind, N)
	return get_edge_from_tensor_coordinates_1block(row, column, side, N)

def get_edge_from_tensor_index(
	ind:int,
	side:Literal['L', 'R', 'U', 'D'],
	N:int
)->int:
	"""
	Get the index of an edge in the double-layer PEPS.

	Input Parameters:
	------------------
	ind  --- index of the vertex to which the edge belong.	        
	side --- The side of the edge. Either 'L', 'R', 'U', 'D'
	N    --- Linear size of the lattice
	"""
	row, column = get_coordinates_from_index(ind, N)
	return get_edge_from_tensor_coordinates(row, column, side, N)

def get_edge_from_tensor_coordinates_1block(
	i:int,
	j:int,
	side:Literal['L', 'R', 'U', 'D'],
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
	this = get_index_from_coordinates(i, j, N)
	neighbor = get_neighbor_from_coordinates(i, j, side, N)

	if side=='L' and j==0:
		return f"L{i}"

	if side=='R' and j==N-1:
		return f"R{i}"

	if side=='D' and i==N-1:
		return f"D{j}"

	if side=='U' and i==0:
		return f"U{j}"

	return f"{min(this,neighbor)}-{max(this,neighbor)}"

def get_edge_from_tensor_coordinates(
	i:int,
	j:int,
	side:Literal['L', 'R', 'U', 'D'],
	N:int
)->int:
	"""
	Get the index of an edge in the double-layer PEPS.


	Input Parameters:
	------------------
	i,j --- location of the vertex to which the edge belong.
	        i=row, j=column. i=0 ==> upper row.

	side --- The side of the edge. Either 'L', 'R', 'U', 'D'

	N    --- Linear size of the lattice
	"""

	if side=='L':
		return i*N + ((j-1)%N) + 1

	if side=='R':
		return i*N + j + 1

	if side=='D':
		return i*N + j + N*N + 101

	if side=='U':
		return ( (i-1)%N )*N + j + N*N+101

	raise ValueError("'{side}' not a supported input")


def get_block_params(ib, jb, n, N):
	"""
	Creates the blockBP parameters of a block in the sqaure grid. This
	includes
	 (*) List of vertices
	 (*) List of super edges connected to it
	 (*) The bcon of the block --- block contraction order

	Input Parameters:
	------------------
	ib, jb --- The indices of the block --- each is a number 0 => N/n-1
	n      --- Linear size of a block
	N      --- Linear size of the grid

	OUTPUT:
	--------
	 v_list, se_edges, bcon

	"""

	num_blocks = N//n  # how many blocks in a side
	block_id = ib*num_blocks + jb


	# ====================================================================
	# 1. Create the list of tensors in the block
	# ====================================================================
	v_list=[]
	for i in range(n):
		for j in range(n):

			i1 = ib*n + i
			j1 = jb*n + j

			v_list.append( i1*N + j1)

	# ====================================================================
	# 2. Create the list of its adjacent super-edges (sedges). Every
	#    sedge label is of the form 'se:<bi>-<bj>', where bi<bj. For
	#    example,'se:1-2', etc...
	#
	#    <bi>, <bj> are the indices of the blocks. Like vertices, the
	#    blocks are numbered from left->right, up->down.
	# ====================================================================

	if num_blocks==1:
		Lse = "se:L"
		Rse = "se:R"
		Use = "se:U"
		Dse = "se:D"

	else:
		# Calculde the idx of the neighboring blocks
		Lb = ib*num_blocks + (jb - 1) % num_blocks
		Rb = ib*num_blocks + (jb + 1) % num_blocks
		Ub = ((ib - 1) % num_blocks)*num_blocks + jb
		Db = ((ib + 1) % num_blocks)*num_blocks + jb

		Lse = _standard_super_edge_name(block_id, Lb) 
		Rse = _standard_super_edge_name(block_id, Rb) 
		Use = _standard_super_edge_name(block_id, Ub) 
		Dse = _standard_super_edge_name(block_id, Db) 

	se_edges = [(Lse,-1), (Rse,+1), (Use,+1), (Dse,-1)]
	directions = ['L', 'R', 'U', 'D']


	# ====================================================================
  	# 3. Create the contraction order.
	# ====================================================================

	# We want the contraction order to be as "snake-like" as possible
	# to minimize the contraction time of bubblecon.
	bcon : List[tuple] = []

	for dir in all_possible_sides():
		sw, angle = _contraction_order_per_direction(dir, ib, jb, n, N)
		bcon.append((sw, angle))

	return v_list, se_edges, bcon


#
# --------------- create_periodic_random_2D_PEPS  ----------------------
#

def create_periodic_random_2D_PEPS(\
	N:int,  # Num tensors in edge of lattice
	n:int,  # Num tensors in edge of block
	D:int=1, 
	d:int=2,
	same_tensors_seed:Optional[int]=None
) -> Tuple[ \
	List[np.ndarray], 				# PEPS_tensors_list, 
	List[np.ndarray], 				# T_list, 
	List[List[str]], 				# edges_list, 
	List[Tuple[int, ...]],  		# pos_list, 
	List[List[int]], 				# blocks_v_list, 
	Dict[str, List[int]], 			# sedges_dict, 
	List[List[Tuple[str, int]]],	# sedges_list, 
	List[BlocksConType]  			# blocks_con_list
]: 
	"""

	Creates an initial random PEPS on a 2D square grid with periodic
	boundary conditions, together with all the needed blockBP structure.

	This includes:
	1. PEPS_tensors_list --- The PEPS tensors of the random PEPS.

	2. T_list,           --- The fused ket-bra version of the PEPS tensors
	                         which constitute the TN on which blockBP runs.

	3. edges_list        --- The contraction relation between the T_list
	                         tensors

	3. pos_list,         --- The (x,y) positions of every T_list tensor

	4. blocks_v_list,    --- The list of blocks. Each block defined
	                         by the list of T_list tensors that make

	5. sedges_dict,      --- A dictonary that defines the super-edges
	                         between blocks.

	6. sedges_list,      --- The list of super-edges of every block

	7. blocks_con_list   --- The contraction order inside each block.


	"""

	N2 = N*N  # total number of spins
	num_blocks = N//n  # how many blocks in a side

	if num_blocks>2:
		_get_edge_from_coordinates_func = get_edge_from_tensor_coordinates
		_get_edge_from_index_func = get_edge_from_tensor_index
	elif num_blocks==2:
		print("Error in create_periodic_random_2D_PEPS !!!")
		raise NotImplementedError(
			f"BlockBP cannot handle a setup of {num_blocks} x {num_blocks} blocks "+\
			"in periodic b.c., since it will cause double super-edges "+\
			"between blocks. There must be at least 3 blocks on each side."
		)
	elif num_blocks==1:
		_get_edge_from_coordinates_func = get_edge_from_tensor_coordinates_1block
		_get_edge_from_index_func = get_edge_from_tensor_index_1block
	else:
		raise NotImplementedError("Not supporter number of blocks {Nb}")

	## same_tensors_seed controls if all the tensors should be the same:
	if same_tensors_seed is not None:
		assert isinstance(same_tensors_seed, int)
		all_tensors_same = True
		rs = np.random.RandomState( np.random.MT19937( same_tensors_seed ) )

	else:
		all_tensors_same = False
		rs = np.random.RandomState()  # default seed from clock
	


	PEPS_tensors_list = []  # list of random tensors 
	T_list = []  # list of their double-edge tensors
	edges_list = []  # list of edges. For an (i,j) vertex, we have:
	pos_list = []  # list of positions

	T : np.ndarray   # for type-checker 
	T2 : np.ndarray  # for type-checker

	for i, j in _all_possible_couples(N):

		# Create random tensors:
		if i==j==0 or not all_tensors_same:			
			T = rs.normal(size=[d, D, D, D, D]) \
				+ 1j*rs.normal(size=[d, D, D, D, D])
			T = T/norm(T)
			T2 = fuse_tensor(T)

			PEPS_tensors_list.append(T)
			T_list.append(T2)

		if not i==j==0 and all_tensors_same:
			PEPS_tensors_list.append( T.copy() )
			T_list.append( T2.copy() )



		# Right-Edge: = i*N+j + 1
		# Up-Edge:    = i*N+j + N**2 + 101
		eL = _get_edge_from_coordinates_func(i,j,'L', N)
		eR = _get_edge_from_coordinates_func(i,j,'R', N)
		eD = _get_edge_from_coordinates_func(i,j,'D', N)
		eU = _get_edge_from_coordinates_func(i,j,'U', N)
		edges_list.append([eL, eR, eU, eD])

		# Create the list of positions
		pos_list.append( (j*1.0, (N-i)*1.0))

	# Go over the blocks, and for each block create its vertices list,
	# its list of adjacent super-edges, and its contraction list
	blocks_v_list=[]
	sedges_list=[]
	blocks_con_list=[]
	sedges_dict = {}

	for ib, jb in _all_possible_couples(num_blocks):

		v_list, se_list, bcon = get_block_params(ib, jb, n, N)

		blocks_v_list.append(v_list)
		sedges_list.append(se_list)
		blocks_con_list.append(bcon)

		if num_blocks==1:			
			for side in ['L', 'R', 'U', 'D']:
				sedges_dict["se:"+side] = [_get_edge_from_index_func(i, side, N) for i in all_boundary_tensors(N, side)]
		
		else:
			# Fill the super-edges dictionary. Every sedge label
			# is of the form 'se:<bi>-<bj>', where bi<bj. For example,
			# 'se:1-2', etc...
			# Unluess there is only 1 block

			# Horizontal
			jb2 = (jb+1) % num_blocks

			se1 = ib*num_blocks + jb
			se2 = ib*num_blocks + jb2

			se_label = _standard_super_edge_name(se1, se2)

			j = jb*n + n-1
			se_edges=[]
			for i in range(ib*n, ib*n+n):
				e = _get_edge_from_coordinates_func(i,j,'R',N)
				se_edges.append(e)

			sedges_dict[se_label] = se_edges

			# Vertical
			ib2 = (ib+1) % num_blocks

			se1 = ib*num_blocks + jb
			se2 = ib2*num_blocks + jb

			se_label = _standard_super_edge_name(se1, se2)
			se_label = f"se:{min(se1, se2)}-{max(se1, se2)}"

			i = ib*n + n-1
			se_edges=[]
			for j in range(jb*n, jb*n+n):
				e = _get_edge_from_coordinates_func(i,j,'D',N)
				se_edges.append(e)

			sedges_dict[se_label] = se_edges	


	return PEPS_tensors_list, T_list, edges_list, pos_list, \
		blocks_v_list, sedges_dict, sedges_list, blocks_con_list





#
# -------------------- horiz_2body_envs ------------------------
#


def horiz_2body_envs(mpL, mpR, mid_T_list, i0, i1, \
	D_trunc, eps=None, mode='LR'):
	"""

	Given a left and right MPSs, together with a square grid TN made of
	two columns, calculates the 2-body environment of the horizontal
	bonds between columns i0 and i1.

	The function is equivalent to running the ITE.get_2body_env on all
	the environemnts between i0->i1 --- but should be much more efficient.

	The overall TN that we contract looks is of the following shape:

	mpL    mid-T   mpR
	+-- --+----+-- --+
	|     |    |     |
	+-- --+----+-- --+ <== i0 row
	|     |    |     |
	+-- --+----+-- --+
	|     |    |     |
	+-- --+----+-- --+ <== i1 row
	|     |    |     |
	+-- --+----+-- --+
	|     |    |     |
	+-- --+----+-- --+

	We assume that the top and bottom rows of the mid part are by
	themselves part of an MPS (so the have only 3 legs L--mid--R

	Parameters:
	------------

	mpL, mpR --- the left/right MPSs

	mid_T_list --- The tensors that make up the mid part (the two columns).
	               The tensors are arranged as usual: the tensor at
	               row i, column j is tensor number 2*i + j

	               The top row/bottom tensors come from MPSs so their
	               legs have the form [L,d,R]

	               The bulk tensors have the form [L,R,U,D]

	i0,i1  --- Specify the range [i0,i1] of rows for which we want
	           to calculate the env.

	D_trunc, eps --- the usual bubblecon contraction parameters

	mode --- Either 'LR' or 'full'. Specify the form of the output
	         environment, just as get_2body_env from the ITE module.


	OUTPUT --- A list of the 2body environemnts on rows i0->i1.

	"""


	#
	# First trim mpL, mpR
	#

	bmpslib.trim_mps(mpL)
	bmpslib.trim_mps(mpR)

	n = mpL.N  # Number of rows

	L, R, U, D = pi, 0, pi/2, 3*pi/2  # 4 angles of legs

	#
	# We now take mpL, mpR and mid_T_list and define a tensor-network
	# to be contracted using bubblecon.
	#

	V = 4*n    # V is where the indexing of the vertical legs start


	#
	# Add the first row (which comes from an MPS)
	#

	T_list = [mpL.A[0], mid_T_list[0], mid_T_list[1], mpR.A[n-1] ]

	e_list = [[1,V+1], [2,V+2,1], [3,V+3,2], [V+4, 3]]

	angles_list = [ [R, D], [R, D, L], [R, D, L], [D, L] ]

	#
	# Add the n-2 bulk rows
	#

	for i in range(n-2):

		T_list = T_list + [mpL.A[i+1], mid_T_list[2*i+2], \
			mid_T_list[2*i+3], mpR.A[n-2-i]]

		e_list = e_list + [[V+1+i*4, 4+i*3, V+5+i*4], \
			[4+i*3, 5+i*3, V+2+i*4, V+6+i*4], \
			[5+i*3, 6+i*3, V+3+i*4, V+7+i*4], \
			[V+8+i*4, 6+i*3, V+4+i*4] ]

		angles_list = angles_list + [[U, R, D], [L, R, U, D], \
			[L, R, U, D], [D, L, U]]

	#
	# Add the bottom row (which comes from an MPS)
	#

	T_list = T_list + [mpL.A[n-1], mid_T_list[2*(n-1)], \
		mid_T_list[2*n-1], mpR.A[0]]

	e_list = e_list + [[V+1+4*(n-2), 1+(n-1)*3], \
		[1+(n-1)*3, V+2+4*(n-2), 2+(n-1)*3],
		[2+(n-1)*3, V+3+4*(n-2), 3+(n-1)*3],
		[3+(n-1)*3, V+4+4*(n-2)]]

	angles_list = angles_list + [ [U,R], [L,U,R], [L,U,R], [L,U]]


	#
	# Now that we defined the TN, we contract it from top->bottom and
	# bottom->top to get a list of upper/lower MPSs in the range of
	# the [i0,i1] rows that would enable us to calculate the horizontal
	# 2body envs.
	#


	# -------------------------------------------------------------
	# Contract the TN up->bottom to get the upper Umps_list
	# -------------------------------------------------------------

	sw_order = list(range(4*i1))
	sw_angle = D

	break_points = list(range(i0*4-1, (i1-1)*4, 4))

	Umps_list = bubblecon.bubblecon(T_list, e_list, angles_list, \
		sw_angle, sw_order, D_trunc, break_points=break_points)

	#
	# Trim the top MPSs
	#

	for i in range(len(Umps_list)):
		bmpslib.trim_mps(Umps_list[i])



	# -------------------------------------------------------------
	# Contract the TN bottom->up to get the lower Dmps_list
	# -------------------------------------------------------------

	sw_order = list(range(4*n-1, i0*4-1, -1))
	sw_angle = U
	break_points = list(range((n-i1-1)*4-1, (n-i0-1)*4, 4))



	Dmps_list = bubblecon.bubblecon(T_list, e_list, angles_list, \
		sw_angle, sw_order, D_trunc, break_points=break_points)

	#
	# Trim the bottom MPSs
	#

	for i in range(len(Dmps_list)-1):
		bmpslib.trim_mps(Dmps_list[i])


	#
	# ==================================================================
	#  Now that we have the top and buttom MPSs, we can calculate the
	#  2-body horizontal envs
	# ==================================================================
	#

	envs_list = []

	for i in range(i0, i1+1):
		mpU = Umps_list[i-i0]
		mpD = Dmps_list[i1-i]

		#
		# Calculate the left env using ncon
		#

		T_list = [mpU.A[2], mpU.A[3], mpL.A[i], mpD.A[0], mpD.A[1]]
		e_list = [[-100, -1, 1], [1,2], [2,-2,3], [3,4], [4,-3,-101]]

		env_L = ncon(T_list, e_list)

		# open up the ket-bra legs of env_L

		shape = env_L.shape
		DU2, DL2, DD2 = shape[0], shape[1], shape[2]

		DU = int(sqrt(DU2))
		DL = int(sqrt(DL2))
		DD = int(sqrt(DD2))

		env_L = env_L.reshape([DU, DU, DL, DL, DD, DD, shape[3], shape[4]])

		#
		# Calculate the right env using ncon
		#

		T_list = [mpU.A[1], mpU.A[0], mpR.A[n-1-i], mpD.A[3], mpD.A[2]]
		T_list = [mpD.A[2], mpD.A[3], mpR.A[n-1-i], mpU.A[0], mpU.A[1]]
		e_list = [ [-101,-1,1], [1,2], [2,-2,3], [3,4], [4,-3,-100]]

		env_R = ncon(T_list, e_list)

		# open up the ket-bra legs of env_R

		shape = env_R.shape
		DD2, DR2, DU2 = shape[0], shape[1], shape[2]

		DU = int(sqrt(DU2))
		DR = int(sqrt(DR2))
		DD = int(sqrt(DD2))

		env_R = env_R.reshape([DD, DD,  DR, DR, DU, DU, shape[3], shape[4]])

		#
		# If mode=='LR', then every environment is a tuple (env_L, env_R).
		# If mode=='full', we contract env_L with env_R
		#

		if mode=='LR':
			envs_list.append( (env_L, env_R) )
		else:
			full_env = tensordot(env_L, env_R, axes=([6,7], [6,7]))
			envs_list.append( full_env )


	return envs_list



#
# -------------------- vert_2body_envs ------------------------
#


def vert_2body_envs(mpU, mpD, mid_T_list, j0, j1, \
	D_trunc, eps=None, mode='LR'):

	"""
	Same as horiz_2body_envs, but for vertical 2-body environments.
	Therefore the input is a TN that is given as upper MPS + lower MPS
	+ bulk tensors that are made of two rows of tensors

	Graphically, the total TN looks like:

	        j0                  j1
	        |                   |
	        V                   V
	+---+---+---+---+---+---+---+---+---+---+---+  mpU
	|   |   |   |   |   |   |   |   |   |   |   |

	|   |   |   |   |   |   |   |   |   |   |   |
	+---+---+---+---+---+---+---+---+---+---+---+
	|   |   |   |   |   |   |   |   |   |   |   |   mid_T
	+---+---+---+---+---+---+---+---+---+---+---+
	|   |   |   |   |   |   |   |   |   |   |   |

	|   |   |   |   |   |   |   |   |   |   |   |
	+---+---+---+---+---+---+---+---+---+---+---+  mpD


	Parameters:
	------------

	mpU, mpD --- the upper/lower MPSs

	mid_T_list --- The tensors that make up the mid part (the two rows).
	               The tensors are arranged as usual: the tensor at
	               row i, column j is tensor number n*i + j

	               The left/right columns tensors come from MPSs so their
	               legs have the form [L,d,R]

	               The bulk tensors have the form [L,R,U,D]

	j0,j1  --- Specify the range [j0,j1] of columns for which we want
	           to calculate the env.

	D_trunc, eps --- the usual bubblecon contraction parameters

	mode --- Either 'LR' or 'full'. Specify the form of the output
	         environment, just as get_2body_env from the ITE module.


	OUTPUT --- A list of the 2body environemnts on rows j0->j1.


	"""

	#
	# Instead of performing all the calculation, we rotate the system
	# anti-clockwise and call the horiz_2body_envs
	#
	# Mapping:
	# -----------
	# mpU -> mpL
	# mpD -> mpR
	#
	# In mid_T_list: (i,j) -> (n-1-j, i)
	#    in the bulk, the tensor indices trans as: L->D, D->R, R->U, U->L
	#
	#

	n = mpU.N

	new_mid_T_list = [None]*len(mid_T_list)

	for j in range(n):
		for i in range(2):

			T = mid_T_list[j+n*i]
			#
			# Only transpose tensors in the bulk (the extreme come from
			# MPSs and as such they transform unchanged)
			#
			if j>0 and j<n-1:
				new_T = T.transpose([2,3,1,0])
			else:
				new_T = T.copy()

			new_mid_T_list[(n-1-j)*2 + i] = new_T

	envs_list = horiz_2body_envs(mpU, mpD, new_mid_T_list, \
		n-1-j1, n-1-j0, D_trunc, eps, mode)


	#
	# reverse the order of the envs, since in the rotated frame,
	# the top most env comes from the right most env in the orig
	# frame.
	#


	return envs_list[::-1]


#
# -------------------- apply_horiz_block_gates ------------------------
#

def apply_horiz_block_gates(BTN, mid_PEPS, mode, block_gates=None, \
	Dmax=None, D_trunc=None, D_trunc2=None, eps=None):
	"""

	Either calculates the 2-body RDMS of the horizontal terms in a BTN
	(from which the energy can be calculated), or applies a set of
	horizontal gates at these locations.

	The BTN is assumed to be a n\times n PEPS. The horizontal locations
	where the gates are applied or where the RDMS are calculated are
	given by

	                   (i,j)--(i,j+1) for i=1,..,n-2

	with j=1,...,n-2 (either even or odd).


	Tunction first contracts the BTN from left to right and from right to
	left to find vertical MPSs. Then it sends these MPSs to the
	horiz_2body_envs function to calculate the 2-body environments of
	the horizontal places where we want to update.

	Finally, it either sends these environments together with the PEPS
	tensors to:
	 a) ITE.apply_2local_gate function to update the PEPS, or
	 b) ITE.rho_ij to calculate the 2-body RDMs


	Input Parameters:
	-------------------

	BTN --- The square grid BTN

	mid_PEPS --- The list of n^2 PEPS of the BTN arraned such that
	             the tensor at (i,j) is number i*n+j (i=col, j=row)

	mode --- Either 'H-even' or 'H-odd' or 'RDMs'. The first two
	         correspond to whether to apply the gates for even or odd j.
	         The third is for only calculating the RDMs (no update takes
	         place)

	block_gates --- A list of the horizontal gates to apply in the order
	                at which they are applied. If mode='H-odd', then
	                the order is:

	                (1,1)--(1,2) => (2,1)--(2,2) => (3,1)--(3,2) => ...
					 ... => (1,3)--(1,4) => (2,3)--(2,4) => ...

					        Similarly, if mode=='H-even', the order is
	                (1,2)--(1,3) => (2,3)--(2,3) => (3,2)--(3,3) => ...
					 ... => (1,4)--(1,5) => (2,4)--(2,5) => ...


	Dmax --- The maximal allowed bond dimension of the resultant PEPS

	D_trunc, eps --- bubblecon parameters for the BTN contraction

	OUTPUT: If mode=='RDMs', return a list of the 2body RDMS. Otherwise,
					None is returned. When mode=='H-even' or 'H-odd', the new
					PEPS tensors are updated in-place of the mid_PEPS tensors
					list.

	"""

	log=False

	RDMS_list=[]

	n = int(sqrt(len(mid_PEPS)))
	k = n-2

	(T_list, edges_list, angles_list, bcon) = BTN


	#
	# ==============================================================
	# Contract the BTN from left to right to find the k-1 left
	# vertical MPSs
	# ==============================================================
	#

	break_points = []

	# Add the left-most message-MPS vertices #
	sw_order = list(range(n**2, n**2 + n))

	# Add the first column (j=0)

	sw_order = sw_order + [0, n*n + 3*n-1]
	sw_order = sw_order + list(range(n, n*(n-1)+1, n))
	sw_order = sw_order + [n*n+3*n]

	# Add the k-2 columns

	for j in range(k-2):

		break_points = break_points + [len(sw_order)-1]

		# add top vertex that belongs to the top MPS message at j+1
		sw_order = sw_order + [n*n+3*n-2-j]

		# add the bulk vertices at j+1
		sw_order = sw_order + list(range(j+1, j+2+(n-1)*n,n))

		# add the j+1 vertex of the bottom MPS
		sw_order = sw_order + [n*n+3*n+1+j]

	sw_angle = 0

	if log:
		print("Get left MPSs for horizontal gates update")

	Lmps_list = bubblecon.bubblecon(T_list, edges_list, angles_list, \
			sw_angle, swallow_order=sw_order, D_trunc=D_trunc, \
			D_trunc2=D_trunc2, break_points=break_points)

	# ==============================================================
	# Contract the BTN from right to left to find the k-1 right
	# vertical MPSs
	# ==============================================================

	break_points = []

	# Add the right-most message-MPS vertices #
	sw_order = list(range(n**2+n, n**2 + 2*n))

	# Add the top vertices of the second to last col
	sw_order = sw_order + [n-1, n*n+2*n]

	# Add the 0->k-1 vertices in the second to last col
	sw_order = sw_order + list(range(2*n-1,n*n, n))

	# Add the bottom two vertex of the second to last col
	sw_order = sw_order + [n*n+4*n-1]

	# Now add the k-2 columns
	for j in range(k-2):

		break_points = break_points + [len(sw_order)-1]
	
		# Add the column n-j-2
		sw_order = sw_order + [n*n+2*n + j+1]
		sw_order = sw_order + list(range(n-j-2, n*n-j-1, n))
		sw_order = sw_order + [n*n+4*n-2-j]


	sw_angle = pi

	if log:
		print("Get right MPSs for horizontal gates update")


	Rmps_list = bubblecon.bubblecon(T_list, edges_list, angles_list, \
			sw_angle, swallow_order=sw_order, D_trunc=D_trunc, \
			D_trunc2=D_trunc2, break_points=break_points)


	# Now go over the even or odd columns, calculate
	# the 2-body environment rows i=1->n-2 and columns (j,j+1) and either
	# apply the gates or calculate the RDMs

	# The index of the gate in the block_gates list.
	gid = 0

	envs_list=[]

	for j in range(1,n-2):

		if j%2==0 and mode=='H-odd' or j%2==1 and mode=='H-even':
			continue

		# Create a TN from the tensors at columns j,j+1, calculate the
		# 2-body env at rows 1->n-2

		# Add the two top tensors
		mid_T_list = [T_list[n*n+3*n-(j+1)], T_list[n*n+3*n-(j+2)]]

		# Add the bulk
		for l in range(n):
			mid_T_list = mid_T_list + [T_list[j+l*n], T_list[j+1+l*n]]

		# Add the bottom tensors
		mid_T_list = mid_T_list + [ T_list[n*n+3*n+j], T_list[n*n+3*n+1+j] ]

		# Check that bubblecon's break-points worked correctly:		
		assert isinstance(Lmps_list, list)
		assert isinstance(Rmps_list, list)

		mpL = Lmps_list[j-1]
		mpR = Rmps_list[k-1-j]

		i0 = 2
		i1 = k+1

		if log:
			print(f"Calculate env at i0,i1={i0,i1}")

		# Calculate the horizontal environments
		envs = horiz_2body_envs(mpL, mpR, mid_T_list, i0,i1, \
			D_trunc, eps, mode='LR')

		if log:
			print("=> done")

		# Now go over all the horizontal environments that were calculated
		# and use them to either apply the horizontal gates or calculate
		# the RDMs.

		for i in range(1, n-1):

			# Left/Right environments of gate (i,j)--(i,j+1)
			(env_L, env_R) = envs[i-1]


			# Re-arrange the legs of the PEPS tensors such that the virtual
			# leg that connects them will appear first after the physical leg:
			#
			#           0   1   2  3   4
			#          [d, DL, DR, DU, DD]
			#
			#                 ||
			#                 \/
			#
			#           0   1   2  3   4
			# Ti legs: [d, DR, DU, DL, DD]  (on the left)
			# Tj legs: [d, DL, DD, DR, DU]  (on the right)

			Ti = mid_PEPS[i*n + j].transpose([0,2,3,1,4])
			Tj = mid_PEPS[i*n + j+1].transpose([0,1,4,2,3])

			if mode=='RDMs':

				#
				# If mode=='RDMs', then calculate the 2body RDM instead of
				# updating the PEPS
				#
				rho = ITE.rho_ij(Ti, Tj, env_L, env_R)

				RDMS_list.append(rho)

			else:

				#
				# mode=='H-even' or 'H-odd' so update the PEPS by applying the
				# horizontal gates.
				#

				g = block_gates[gid]

				gid += 1

				if log:
					print(f"\n Applying horizontal gate T[{i},{j}]--T[{i},{j+1}] ")

				new_Ti, new_Tj = ITE.apply_2local_gate(g, Dmax, Ti, Tj, \
					env_L, env_R)

				norm_new_Ti = norm(new_Ti)
				norm_new_Tj = norm(new_Tj)

				if norm_new_Ti < 1e-10 or norm_new_Ti>10e10:
					print("apply-horiz-block: Extreme Ti norm: ", norm_new_Ti)

				if norm_new_Tj < 1e-5 or norm_new_Tj>10e5:
					print("apply-horiz-block: Extreme Tj norm: ", norm_new_Tj)

				new_Ti = new_Ti/norm_new_Ti
				new_Tj = new_Tj/norm_new_Tj

				mid_PEPS[i*n + j]   = new_Ti.transpose([0,3,1,2,4])
				mid_PEPS[i*n + j+1] = new_Tj.transpose([0,1,3,4,2])

	if mode=='RDMs':
		return RDMS_list




#
# -------------------- apply_vertical_block_gates ------------------------
#


def apply_vertical_block_gates(BTN, mid_PEPS, mode, block_gates=None, \
	Dmax=None, D_trunc=None, D_trunc2=None, eps=None, RDMS=False):

	"""

	Same as apply_horiz_block_gates, but for veritcal gates.

	Either applies a list of 2-local gate to the vertical 2-body bonds in
	a square grid BTN, or calculates the 2body RDMs. The BTN is assumed
	to be a n\times n PEPS. The gates are applied to bonds of the form
	(i,j)--(i+1,j) for j=1,..,n-2 and i=1,...,n-2 (either even or odd).

	The function first contracts the BTN from top to buttom and from
	buttom to top to find horizontal MPSs. Then it sends
	these MPSs to the vert_2body_envs function to calculate the 2-body
	environments of the vertical places where we want to update.
	Finally, it either sends these environments, together with the PEPS
	tensors, to ITE.apply_2local_gate or ITE.rho_ij.


	Input Parameters:
	-------------------

	BTN --- The square grid BTN

	mid_PEPS --- The list of n^2 PEPS of the BTN arraned such that
	             the tensor at (i,j) is number i*n+j (i=col, j=row)

	mode --- Either 'V-even' or 'V-odd', 'RDMs'. Whether to apply the gates
	         for even or odd i, or simply just calculate the vertical
	         RDMs

	block_gates --- A list of the gates to apply, ordered at the same
	                order at which they are applied. If mode='V-odd', then
	                the order is:

	                (1,1)--(2,1) => (1,2)--(2,2) => (1,3)--(2,3) => ...
					 ... => (3,1)--(4,1) => (3,2)--(4,2) => ...

					        Similarly, if mode=='V-even', the order is
	                (2,1)--(3,1) => (2,2)--(3,2) => (2,3)--(3,3) => ...
					 ... => (4,1)--(5,1) => (4,2)--(5,2) => ...

					        Both orders are the order at which the gates are
					        applied.

	Dmax --- The maximal allowed bond dimension of the resultant PEPS

	D_trunc, eps --- bubblecon parameters for the BTN contraction


	OUTPUT: If mode=='RDMs', return the 2body RDMS. Otherwise, None is
	        returned. If mode=='V-even', 'V-odd', the new PEPS tensors
	        are updated in-place of the mid_PEPS tensors list.


	"""


	log=False

	RDMS_list=[]


	n = int(sqrt(len(mid_PEPS)))
	k = n-2

	(T_list, edges_list, angles_list, bcon) = BTN

	#
	# ==============================================================
	# Contract the BTN from top to buttom to find the k-1 upper
	# horizontal MPSs
	# ==============================================================
	#

	break_points = []

	# Add the upper message-MPS vertices #
	sw_order = list(range(n**2+2*n, n**2 + 3*n))

	# Add the second row
	sw_order = sw_order + [0, n*n]
	sw_order = sw_order + list(range(1,n)) + [n*n + 2*n-1]

	# Add the k-2 rows

	for i in range(k-2):

		break_points = break_points + [len(sw_order)-1]

		# add left-most vertex that belongs to the left MPS message at i+1
		sw_order = sw_order + [n*n+i+1]

		# add bulk vertices at row i+1
		sw_order = sw_order + list(range(n*(i+1), n*(i+1)+n))

		# add theright-most vertex in row i+1
		sw_order = sw_order + [n*n + 2*n-i-2]


	sw_angle = 3*pi/2

	if log:
		print("get upper MPS for veritcal gates update")

	Umps_list = bubblecon.bubblecon(T_list, edges_list, angles_list, \
			sw_angle, swallow_order=sw_order, D_trunc=D_trunc, \
			D_trunc2=D_trunc2, break_points=break_points)


	#
	# ==============================================================
	# Contract the BTN from bottom to top to find the k-1 lower
	# horizontal MPSs
	# ==============================================================
	#


	break_points = []

	# Add the bottom message-MPS vertices
	sw_order = list(range(n*n + 3*n, n*n+4*n))


	# Add the first row from below

	sw_order = sw_order + [n*(n-1), n*n+n-1]
	sw_order = sw_order + list(range(n*(n-1)+1, n*n)) + [n*n+n]


	# Add the k-2 rows

	for i in range(k-2):

		break_points = break_points + [len(sw_order)-1]

		#
		# Add row n-2-i
		#
		sw_order = sw_order + [n*n+n-2-i]
		sw_order = sw_order + list(range(n*(n-i-2), n*(n-i-2)+n))
		sw_order = sw_order + [n*n+n+i+1]


	sw_angle = pi/2

	if log:
		print("get lower MPS for veritcal gates update")


	Dmps_list = bubblecon.bubblecon(T_list, edges_list, angles_list, \
			sw_angle, swallow_order=sw_order, D_trunc=D_trunc, \
			D_trunc2=D_trunc2, break_points=break_points)


	#
	# Now go over the rows i and calculate the 2-body environment in
	# column j=1->n-2 and rows (i,i+1)
	#

	gid = 0

	for i in range(1, n-2):

		if i%2==0 and mode=='V-odd' or i%2==1 and mode=='V-even':
			continue


		#
		# Create a TN from the tensors at rows i,i+1, calculate the
		# 2-body env at columns 1->n-2
		#


		# Add the upper row i

		mid_T_list = [T_list[n*n+i]] + T_list[i*n : (i+1)*n] \
		 + [T_list[n*n+2*n-1-i]]

		# Add the lower row i+1

		mid_T_list = mid_T_list + [T_list[n*n+1+i]] \
			+ T_list[(i+1)*n : (i+2)*n] \
			+ [T_list[n*n+2*n-2-i]]

		mpU = Umps_list[i-1]
		mpD = Dmps_list[k-1-i]

		j0 = 2
		j1 = k+1

		if log:
			print(f"Calculate envs for j0,j1=({j0,j1})")

		envs = vert_2body_envs(mpU, mpD, mid_T_list, j0,j1, \
			D_trunc, eps, mode='LR')

		if log:
			print("=> done")

		#
		# Now go over all the vertical environments and either apply the
		# vertical gates or calculate the vertical 2-body RDMs.
		#

		for j in range(1, n-1):

			#
			# Get the Up/Down environment of the pair (i,j)--(i+1,j)
			#

			(env_U, env_D) = envs[j-1]


			#
			# Ti is the upper, Tj is the lower. Re-arrange the legs of the
			# tensors such that:
			#
			#           0   1   2  3   4
			#          [d, DL, DR, DU, DD]
			#
			#                 ||
			#                 \/
			#
			#           0   1   2  3   4
			# Ti legs: [d, DD, DR, DU, DL]  (top tensor)
			# Tj legs: [d, DU, DL, DD, DR]  (bottom tensor)
			#

			Ti = mid_PEPS[i*n + j].transpose([0,4,2,3,1])
			Tj = mid_PEPS[(i+1)*n + j].transpose([0,3,1,4,2])


			if mode=='RDMs':

				#
				# Calculate the 2body RDM instead of updating the PEPS
				#

				rho = ITE.rho_ij(Ti, Tj, env_U, env_D)
				RDMS_list.append(rho)

			else:

				#
				# If mode=='V-even' or 'V-odd', apply the vertical gates and
				# update the PEPS.
				#

				g = block_gates[gid]

				gid += 1


				if log:
					print(f"\n Applying vertical gate T[{i},{j}]--T[{i+1},{j}] ")

				new_Ti, new_Tj = ITE.apply_2local_gate(g, Dmax, Ti, Tj, \
					env_U, env_D)

				norm_new_Ti = norm(new_Ti)
				norm_new_Tj = norm(new_Tj)

				if norm_new_Ti < 1e-10 or norm_new_Ti>10e10:
					print("apply-vert-block: Extreme Ti norm: ", norm_new_Ti)

				if norm_new_Tj < 1e-5 or norm_new_Tj>10e5:
					print("apply-vert-block: Extreme Tj norm: ", norm_new_Tj)

				new_Ti = new_Ti/norm_new_Ti
				new_Tj = new_Tj/norm_new_Tj


				mid_PEPS[i*n + j]     = new_Ti.transpose([0,4,2,3,1])
				mid_PEPS[(i+1)*n + j] = new_Tj.transpose([0,2,4,1,3])



	if mode=='RDMs':
		return RDMS_list



#
# ----------------------  get_horiz_gates  ------------------------
#

def get_horiz_gates(n, dt, H) -> List[np.ndarray] :

	"""

	Given a local Hamiltonian H, and imaginary time evolution dt, create
	a list of imaginary time gates g=e^{-dt h} for all the horizontal
	bonds in the square grid.

	The order of the output gates is:
	(0,0)--(0,1) => (0,1)--(0,2) => ... => (0,n-1)--(0,0) =>
	(1,0)--(1,1) => (1,1)--(1,2) => ... => (1,n-1)--(1,0) =>
	.
	.
	.

	Input Parameters:
	------------------
	n --- Linear size of the square grid

	dt --- Imaginary time period (can also be complex)

	H --- The local Hamiltonian function. Must be a ref to a function
	      that expects parameters of the form H(n, i1,j2, i2,j2)
	      and can handle periodic boundary conditions.

	OUTPUT:
	--------

	The list of 2-local gates e^{-dt h}.


	"""


	horiz_g_list=[]

	for i in range(n):
		for j in range(n):

			h = H(n, i, j, i, j+1)
			g = ITE.g_from_exp_h(h, dt)

			horiz_g_list.append(g)


	return horiz_g_list



#
# ----------------------  get_vertical_gates  ------------------------
#

def get_vertical_gates(n, dt, H) -> List[np.ndarray]:

	"""

	Given a local Hamiltonian H, and imaginary time evolution dt, create
	a list of imaginary time gates g=e^{-dt h} for all the vertical
	bonds in the square grid.

	The order of the output gates is:
	(0,0)--(1,0) => (0,1)--(1,1) => ... => (0,n-1)--(1,n-1) =>
	(1,0)--(2,0) => (1,1)--(2,1) => ... => (1,n-1)--(2,n-1) =>
	.
	.
	.
	(n-2,0)--(n-1,0) => (n-2,1)--(n-1,1) => ... => (n-2,n-1)--(n-1,n-1)


	Input Parameters:
	------------------
	n --- Linear size of the square grid

	dt --- Imaginary time period (can also be complex)

	H --- The local Hamiltonian function. Must be a ref to a function
	      that expects parameters of the form H(n, i1,j2, i2,j2)
	      and can handle periodic boundary conditions.

	OUTPUT:
	--------

	The list of 2-local gates e^{-dt h}.


	"""


	vertical_g_list=[]

	for i in range(n):
		for j in range(n):

			h = H(n, i, j, i+1, j)
			g = ITE.g_from_exp_h(h, dt)

			vertical_g_list.append(g)

	return vertical_g_list




#
# ---------------------- scale_leg ---------------------------------
#

def scale_leg(T, C, leg):
	"""

	Contract a leg of a tensor with some matrix C.

	Input parameters:
	------------------
	T - the tensor
	C - the scaling matrix
	leg - the number of the leg (0,1,2,...)

	OUTPUT:
	--------
	The rescaled tensor

	"""

	if len(C.shape)==1:
		#
		# Its a vector - then make it into a matrix
		#
		B = diag(C)
	else:
		B = C

	#
	# We move the leg we contract to the end, contract, and then return
	# it back to its place.
	#

	#
	# Move the leg to the end:
	#
	sh = T.shape
	ell = len(sh)

	perm = list(range(ell))
	perm[ell-1] = leg
	perm[leg] = ell-1

	T = T.transpose(perm)

	#
	# Contract the last leg with B
	#

	T = tensordot(T, B, axes=([ell-1],[0]))


	#
	# Return the legs to their original ordering
	#
	T = T.transpose(perm)

	return T


#
# ---------------------- random_Q ---------------------
#

def random_Q(D):
	"""

	Create a random orthogonal matrix Q of dimension D.

	This is done by first creating a random normal matrix and then
	performing a QR decomposition.

	"""

	A = np.random.normal(size=[D,D])

	Q,R = np.linalg.qr(A)

	return Q




#
# ------------------ randomize_PEPS_gauge ---------------------------
#

def randomize_PEPS_gauge(PEPS_T_list):
	"""

	Given a PEPS, change its local gauge randomly. This means go over
	all edges, and apply Q*Q^T for some random orthogonal Q, and absorb
	Q, Q^T in each leg.

	This is done to improve convergence of blockBP if we reach a cycle.

	"""

	N2=len(PEPS_T_list)
	N = int(sqrt(N2))


	#
	# Go over the PEPS tensors and rescale their legs.
	#

	for i in range(N):
		for j in range(N):

			#
			# For T at (i,j), we randomize the gauge with (i,j+1)
			# and with (i+1, j)
			#


			#
			#  ======== Horizontal legs =========
			#

			i1,j1 = i,j
			i2,j2 = i,(j+1) % N

			T1,T2 = PEPS_T_list[i1*N+j1], PEPS_T_list[i2*N+j2]

			# shape = d, D_L, D_R, D_U, D_D
			D = T1.shape[2]

			if D>1:
				Q = random_Q(D)

				new_T1 = scale_leg(T1, Q, 2)    # right leg
				new_T2 = scale_leg(T2, Q, 1)    # left leg

				PEPS_T_list[i1*N+j1], PEPS_T_list[i2*N+j2] = new_T1, new_T2

			#
			#  ======== Vertical legs =========
			#

			i1,j1 = i,j
			i2,j2 = (i+1)%N, j

			T1,T2 = PEPS_T_list[i1*N+j1], PEPS_T_list[i2*N+j2]

			# shape = d, D_L, D_R, D_U, D_D
			D = T1.shape[4]

			if D>1:
				Q = random_Q(D)

				new_T1 = scale_leg(T1, Q, 4)    # down leg
				new_T2 = scale_leg(T2, Q, 3)    # up leg

				PEPS_T_list[i1*N+j1], PEPS_T_list[i2*N+j2] = new_T1, new_T2










#
# --------------------------  shift   -----------------------
#


def shift(
	T_list:List[np.ndarray], 
	del_i:int, 
	del_j:int
) -> List[np.ndarray]:

	#
	# If shift size is 0,0 --- there's nothing to do --- return the
	# original list.
	#
	if (del_i,del_j) == (0,0):
		return T_list

	N2 = len(T_list)
	N = int(sqrt(N2))

	shift_T_list : List[np.ndarray] = [None]*N2

	for i in range(N):
		for j in range(N):

			shift_i = (i + del_i) % N
			shift_j = (j + del_j) % N

			shift_T_list[shift_i*N + shift_j] = T_list[i*N + j]

	return shift_T_list


#
# --------------------------  print_PEPS_shape   -----------------------
#

def print_PEPS_shape(T_list):

	n2 = len(T_list)
	n = int(sqrt(n2))

	for i in range(n):
		shape_str = ''
		for j in range(min(10,n)):
			T = T_list[i*n+j]
			shape_str = shape_str + f"{T.shape} "
		print(shape_str)



#
# --------------------------  Tdist  ----------------------------------
#

def Tdist(tenA,tenB):
	"""

		Calculates the distance between two tensors of the same shape,
		defined as
		  1 - <A|B>/(|A|*|B|)

		mpA, mpB are given as two bmpslib MPS objects

	"""

	A = tenA.flatten()
	B = tenB.flatten()

	A = A/norm(A)
	B = B/norm(B)

	overlap = np.dot(A,conj(B))

	d = 1-abs(overlap)

	return d


#
# --------------------------  mps_dist  ----------------------------------
#

def mps_dist(mpA,mpB):

	"""

		Calculates the distance between two MPSs, defined as
		  1 - <A|B>/(|A|*|B|)

		mpA, mpB are given as two bmpslib MPS objects

	"""

	AA = bmpslib.mps_inner_product(mpA, mpA,conjB=True)
	AB = bmpslib.mps_inner_product(mpA, mpB,conjB=True)
	BB = bmpslib.mps_inner_product(mpB, mpB,conjB=True)


	d = 1-abs(AB)/sqrt(AA*BB)

	d = d.real

	return d




#
# ------------------------   update_blocks   ---------------------------
#

"""

Given an N-by-N PEPS and a set of horizontal and vertical gates,
together with the structural variables needed to run the blockBP
algorithm, the function applies the gates via a full update procedure.
The gate set is applied to the bulk of every block.

The PEPS and the gate lists are assumed to be in periodic order.

The gates in the bulk of every block are divided into 4 sets:
horizontal-even-gates, horizontal-odd-gates, vertical-odd-gates and
vertical-even-gates. Every such set is updated simulaniously in all
blocks. Before each such update, the blockBP is run, and its messages
are used to calculate the local evnironments for each full update.

Input Parameters:
------------------------

(*) PEPS_T_list:

		The PEPS tensors. Every tensor is of the form
		[d, D_L, D_R, D_U, D_D]. The tensor at (i,j)
		(col i, row j) appears at place [i*M+j] in the list.

(*) h_g_list, v_g_list:

    A list of Horizontal and Vertical 2-body gates. Every gate
    is a 4-legs tensors of the form [i1,j1; i2,j2].

    The h_g_list is ordered as follows:
    At the [i*N+j] place is the gate (i,j)--(i,j+1)

    Similarly, for the v_g_list, at the [i*N+j] is (i,j)--(i+1,j)

(*) target_D:
		The maximal final bond dimension of the PEPS after update.

(*) edges_list, pos_list, ... :
		Structure parameters for the blockBP algorithm.


		OUTPUT: PEPS_T_list, m_list, blob
		-------

"""


def update_blocks(PEPS_T_list, h_g_list, v_g_list, target_D, \
	edges_list, pos_list, blocks_v_list, sedges_dict, \
	sedges_list, blocks_con_list, blob=None, m_list=None, \
	classical = False, \
	BP_D_trunc=None, BP_D_trunc2=None, BP_eps=None,
	D_trunc=None, D_trunc2=None, eps=None, max_iter=50, BP_delta=1e-8, \
	Lx=1e9, Ly=1e9, mpi_comm=None):

	isMPI = mpi_comm is not None

	log=False

	NON_CONVERGE_TRIALS = 3

	#
	# Find the linear dimension of the lattice (N) and of the block (n)
	#

	N2 = len(PEPS_T_list)
	N = int(sqrt(N2))  # Number of sites on each side of the lattice

	n2 = N2//len(blocks_v_list)
	n = int(sqrt(n2))  # Number of sites on each side of a block

	Nb = N//n          # Number of blocks on each side of the lattice

	Nb2 = Nb*Nb


	if m_list is not None:
		print("\n\n")
		print("   * * *  Entring update_blocks with a non empty m_list  * * *")
		print("\n\n")

	if log:
		print("\n\n\n")
		print(f" Entering update_blocks N={N}   n={n}   Nb={Nb}")
		print( "=========================================================\n")

	#
	# Create 4 lists of horizontal/vertical odd/even gates for each
	# block. Gates in each list appear in the order at which they are
	# applied in apply_horiz_block_gates or apply_vertical_block_gates
	#

	H_even_block_gates : List[List[List[np.ndarray]]] = [[None]*Nb for k in range(Nb)]
	H_odd_block_gates  : List[List[List[np.ndarray]]] = [[None]*Nb for k in range(Nb)]
	V_even_block_gates : List[List[List[np.ndarray]]] = [[None]*Nb for k in range(Nb)]
	V_odd_block_gates  : List[List[List[np.ndarray]]] = [[None]*Nb for k in range(Nb)]

	for ib in range(Nb):
		for jb in range(Nb):

			H_even_block_gates[ib][jb] = []
			H_odd_block_gates[ib][jb]  = []
			V_even_block_gates[ib][jb] = []
			V_odd_block_gates[ib][jb]  = []

			for j1 in range(1,n-2):
				for i1 in range(1, n-1):

					i = ib*n + i1
					j = jb*n + j1

					g = h_g_list[i*N + j]

					if j1%2==0:
						H_even_block_gates[ib][jb].append(g)
					else:
						H_odd_block_gates[ib][jb].append(g)

			for i1 in range(1, n-2):
				for j1 in range(1, n-1):

					i = ib*n + i1
					j = jb*n + j1

					g = v_g_list[i*N + j].copy()

					if i1%2==0:
						V_even_block_gates[ib][jb].append(g)
					else:
						V_odd_block_gates[ib][jb].append(g)

	#
	# If we're on MPI mode, send initial update-block info to slave
	# blocks.
	#
	# Informatation includes: truncation info + the gates of the block.
	#

	if isMPI:

		for ib in range(Nb):
			for jb in range(Nb):

				ijb = ib*Nb + jb

				block_info = (D_trunc, target_D, \
					H_even_block_gates[ib][jb], H_odd_block_gates[ib][jb], \
					V_even_block_gates[ib][jb], V_odd_block_gates[ib][jb])

				if log:
					print(f"Master: sending initial update-block info to "
						f"block {ijb} {ib,jb}")

				#
				# First command them to recive the inital update-block info, and
				# then send the info itself.
				#

				command=1
				mpi_comm.send(command, dest=ijb, tag=1)

				mpi_comm.send(block_info, dest=ijb, tag=2)

				if log:
					print("Master: sent.")


	#
	# =================================================================
	#
	#     NOW APPLY THE 4 SETS OF GATES: H-odd, V-odd, H-even, V-even
	#
	# =================================================================
	#

	update_modes = ['H-odd', 'V-odd', 'H-even', 'V-even']

	for mode in update_modes:

		#
		# To apply the gates in mode, we first run blockbp to get the BTNs
		# from which we calculate the local environments needed for optimal
		# truncation (full-update).
		#

		#
		# Create the blockBP T_list by contracting the ket-bra PEPS tensors
		# along the physical legs, which creates the double-edge tensors
		#

		T_list = [ITE.fuse_tensor(T) for T in PEPS_T_list]

		#
		# First, run the blockBP algorithm to get the updated BTNs
		#

		if log:
			print("\n")
			print(f"At mode {mode}. Entering blockBP")

		if isMPI:
			#
			# Send all the slaves a message to run the blockbp in slave
			# mode (i.e., run the MPIblock function)
			#

			for ib in range(Nb2):
				if log:
					print(f"Master: sending block {ib} a command to go into blockbp")
				command=10
				mpi_comm.send(command, dest=ib, tag=1)
				if log:
					print("Master: sent.")

			if log:
				print("\n\n")
				print("Master: going into blockbp in MPI mode\n")

		m_list_was_not_empty = (m_list is not None)

		#
		# We first run the blockBP algorithm with m_list, which might be
		# non-empty; this might accelerate the convergence (if m_list is
		# close to the fixed point), but can also lead to non-converging
		# cycles.
		#
		t1 = time.time()
		m_list, blob = blockbp.blockbp(T_list, edges_list, pos_list, blocks_v_list, sedges_dict, sedges_list, blocks_con_list, 
			D_trunc=BP_D_trunc, 
			max_iter=max_iter, 
			delta=BP_delta, 
			Lx=Lx, Ly=Ly, 
			initial_m_mode=MessageModel.UNIFORM_QUANTUM,  
			PP_blob=blob, 
			m_list=m_list, 
			mpi_comm=mpi_comm
		)
		t2 = time.time()

		eps = blob['final-error']
		iter_no = blob['final-iter']

		print("\n")
		print(f"Mode {mode}  blockBP finished in {int(t2-t1)} secs "\
			f"using {iter_no} iterations and error {eps}\n")

		if log:
			print(f"=> done with err={eps} using {iter_no} iterations.")

		if eps>BP_delta and m_list_was_not_empty:
			#
			# If it did not converge with non-empty initial messages, then
			# first try again with uniform messages and m_list=None.
			#
			print(f"BP DID NOT CONVERGE AT MODE={mode} (eps={eps}). "\
				f"WITH NON-EMPTY m_list. TRYING AGAIN WITH UNIFORM MESSAGES:")

			if isMPI:
				#
				# Send all the slaves a message to run the blockbp in slave
				# mode (i.e., run the MPIblock function)
				#

				for ijb in range(Nb2):
					if log:
						print(f"Master: sending block {ijb} a command to go into blockbp")
					command=10
					mpi_comm.send(command, dest=ijb, tag=1)

					if log:
						print("Master: sent.")

				if log:
					print("\n\n")
					print("Master: going into blockbp in MPI mode\n")

			t1 = time.time()
			m_list, blob = blockbp.blockbp(T_list, edges_list, pos_list, blocks_v_list, sedges_dict, sedges_list, blocks_con_list, 
				D_trunc=BP_D_trunc, 
				max_iter=max_iter, 
				delta=BP_delta, 
				Lx=Lx, Ly=Ly, 
				initial_m_mode=MessageModel.UNIFORM_QUANTUM, 
				m_list=None, 
				mpi_comm=mpi_comm
			)
			t2 = time.time()

			eps = blob['final-error']
			iter_no = blob['final-iter']
			print(f"=> done in {int(t2-t1)} secs with err={eps} using {iter_no} iterations.")


		if eps>BP_delta:

			#
			# If we did not converge, then try again BP with after we've
			# randomized the local gauge of the PEPS. Try this for
			# NON_CONVERGE_TRIALS times. Hopefully, one of these trials will
			# converge.
			#

			for i in range(NON_CONVERGE_TRIALS):

				print(f"BP DID NOT CONVERGE AT MODE={mode} (eps={eps}). "\
					f"TRYING AGAIN AFTER LOCAL GAUGE RANDOMIZATION AND "\
					f"D_trunc={BP_D_trunc}. ATTEMPT No. {i} ")

				if isMPI:
					#
					# Send all the slaves a message to run the blockbp in slave
					# mode (i.e., run the MPIblock function)
					#

					for ijb in range(Nb2):
						if log:
							print(f"Master: sending block {ijb} a command to go into blockbp")
						command=10
						mpi_comm.send(command, dest=ijb, tag=1)

						if log:
							print("Master: sent.")

					if log:
						print("\n\n")
						print("Master: going into blockbp in MPI mode\n")

				#
				# Randomize the PEPS
				#

				randomize_PEPS_gauge(PEPS_T_list)
				T_list = [ITE.fuse_tensor(T) for T in PEPS_T_list]


				#
				# Now run the BP again
				#

				t1 = time.time()
				m_list, blob = blockbp.blockbp(
					T_list, edges_list, pos_list, blocks_v_list, sedges_dict, sedges_list, blocks_con_list, 
					D_trunc=BP_D_trunc, 
					max_iter=max_iter, 
					delta=BP_delta, 
					Lx=Lx, Ly=Ly, 
					initial_m_mode='UQ', 
					m_list=None, 
					mpi_comm=mpi_comm
				)
				t2 = time.time()

				eps = blob['final-error']
				iter_no = blob['final-iter']
				print(f"=> done in {int(t2-t1)} secs with err={eps} using {iter_no} iterations.")

				if eps<BP_delta:
					break


		#
		# If at this point we still did not converge, then there's nothing
		# much to do. So we exit update_block, and hopefully the next
		# random shift of the grid will yield a convergent blockBP
		#
		if eps>=BP_delta:

			print(f"ERROR: squareITE.py: blockBP is unable to converge " \
					f"even after {NON_CONVERGE_TRIALS} trials. Exiting " \
					f"update_blocks")

			return PEPS_T_list, m_list, blob


		# ============================================================
		#    At this point we have a converged set of BP messages.
		#    So now we can use them to update the bulk tensors.
		# ============================================================


		BTN_list = blob['BTN_list']

		#
		# Now go over all BTNs and update the gates that correspond
		# to the mode
		#

		if log:
			print("\nGoing over the blocks:")
			print("---------------------------")

		for ib in range(Nb):
			for jb in range(Nb):

				ijb = ib*Nb + jb  # the block number

				if log:
					print(f"At block {ijb} {ib,jb}")

				BTN = BTN_list[ijb]

				#
				# Create a list of the PEPS tensors in the BTN
				#
				mid_PEPS = []
				for i1 in range(n):
					for j1 in range(n):
						i = ib*n + i1
						j = jb*n + j1

						mid_PEPS.append(PEPS_T_list[i*N+j])

				if isMPI:
					#
					# send the mid_PEPS to the slave.
					#

					if log:
						print(f"Master: sending block {ijb} a command to perform "
							"block-update")
					command=20
					mpi_comm.send(command, dest=ijb, tag=1)
					if log:
						print("Master: sent.")

					block_info = (mode, BTN, mid_PEPS)
					mpi_comm.send(block_info, dest=ijb, tag=2)

				else:



					if mode == 'H-even' or mode == 'H-odd':
						#
						# Apply the horizontal gates in the block
						#

						if mode == 'H-odd':
							block_gates = H_odd_block_gates[ib][jb]
						else:
							block_gates = H_even_block_gates[ib][jb]

						apply_horiz_block_gates(BTN, mid_PEPS, mode, \
							block_gates, Dmax=target_D, D_trunc=D_trunc)


					if mode == 'V-even' or mode == 'V-odd':
						#
						# Apply the vertical gates in the block
						#

						if mode == 'V-odd':
							block_gates = V_odd_block_gates[ib][jb]
						else:
							block_gates = V_even_block_gates[ib][jb]

						apply_vertical_block_gates(BTN, mid_PEPS, mode, \
							block_gates, Dmax=target_D, D_trunc=D_trunc)

					#
					# Now update the global PEPS with the PEPS tensors of the block
					#
					for j1 in range(n):
						for i1 in range(n):

							i = ib*n + i1
							j = jb*n + j1

							PEPS_T_list[i*N + j] = mid_PEPS[i1*n + j1]


		if isMPI:

			if log:
				print("\n\n ===== Gathering mid-PEPS from slave blocks =====\n\n")


			#
			# Now gather the mid-PEPS info from all slaves and update
			# the main PEPS
			#
			for ib in range(Nb):
				for jb in range(Nb):

					ijb = ib*Nb + jb  # the block number

					if log:
						print(f"Master: recieving mid-PEPS from block {ijb}")
					mid_PEPS = mpi_comm.recv(source=ijb, tag=3)
					if log:
						print("Master: got it.")

					for j1 in range(n):
						for i1 in range(n):

							i = ib*n + i1
							j = jb*n + j1

							PEPS_T_list[i*N + j] = mid_PEPS[i1*n + j1]






	return PEPS_T_list, m_list, blob




#
# --------------------------- BP_RDMs --------------------------------
#

def BP_RDMs(PEPS_T_list, shifts, T_list, \
	edges_list, pos_list, blocks_v_list, sedges_dict, \
	sedges_list, blocks_con_list,
	BP_D_trunc=None, BP_D_trunc2=None,
	D_trunc=None, D_trunc2=None, Lx=1e9, Ly=1e9, BP_delta=1e-4, \
	mpi_comm=None):

	"""

	Runs the blockBP algorithm to calculate the set of 2-body vertical
	and horizontal RDMS. The function is given the TN structure + the
	blockBP parameters. In addition, it is given a set of (x,y) shifts
	with which to run the blockBP. The resultant RDMs are the average
	RDMs over each one of those shifts.

	Input Parameters
	------------------

	PEPS_T_list --- list of the PEPS tensors ordered as [N*i+j]

	shifts: determines the shifts of the PEPS on which the blockBP is
	        run. There are 3 possibilities:
	        (1) shift=k (an integer), then randomly pick k shifts.
	        (2) shift=[(x0, y0), (x1, y1), ...] --- explicit list of
	            shifts to use
	        (3) 'all' --- use all possible shifts. If the block size
	            is n by n, then there are n^2 possible shifts


	T_list, edges_list, pos_list, blocks_v_list, sedges_dict,
	sedges_list, blocks_con_list, BP_D_trunc=None, BP_D_trunc2=None,
	Lx=1e9, Ly=1e9, BP_delta=1e-4 --- standard blockBP parameter.

	D_trunc=None, D_trunc2=None --- bubblecon truncation parameters
	                                for calculating the RDMs (after
	                                blockBP converged and produced its
	                                messages)

	mpi_comm --- A possible MPI object (when executed in MPI mode)


	OUTPUT:
	--------

	H_RDMs, V_RDMs --- A list of horizontal and vertical 2-body RDMs.
	    Each list is ordered by [N*i + j] where the (i,j) entry is:
	    (*) (i,j)--(i,j+1) RDM in the horizontal list
	    (*) (i,j)--(i+1,j) RDM in the vertical list




	"""


	log = True


	isMPI = mpi_comm is not None

	if log:
		if isMPI:
			print("\n")
			print("Entering BP_RDMs in MPI mode\n")
		else:
			print("\n")
			print("Entering BP_RDMs in non-MPI mode\n")


	#
	# Extract the global TN parameters: number of sites on each side (N)
	# and number of sites on each block side (n)
	#

	N2 = len(PEPS_T_list)
	N = int(sqrt(N2))  # Number of sites on each side of the lattice

	n2 = N2//len(blocks_v_list)
	n = int(sqrt(n2))  # Number of sites on each side of a block

	Nb = N//n          # Number of blocks on each side of the lattice

	Nb2 = Nb*Nb        # Total number of blocks


	#
	# If T_list is empty (None), then create it from PEPS_T_list
	# by contracting a ket-bra pair along the physical leg
	#

	if T_list is None:
		T_list = [ITE.fuse_tensor(T) for T in PEPS_T_list]




	#
	# Analyze the shifts parameter. Can be either an integer, a list or
	# a string.
	#
	if type(shifts) is int:
		#
		# If shifts=k, the use k random shifts
		#

		k = shifts
		shifts=[]
		for ell in range(k):
			i_shift = np.random.randint(0,n)
			j_shift = np.random.randint(0,n)

			shifts.append( (i_shift, j_shift) )

	elif shifts=='all':
		#
		# If shifts=='all' then use all possible shifts.
		#
		shifts=[]
		for i_shift in range(n):
			for j_shift in range(n):
				shifts.append( (i_shift, j_shift) )

	else:

		#
		# In this case, we expect shifts to be the explicit list of shifts
		# of the form [(x0,y0), (x1,y1), ...]
		#

		if type(shifts) is not list:
			print("Error in BP_energy - shifts should be either an intger or "\
				"a list of shifts ")
			exit(1)


	#
	# Initialize the lists of RDMs and a counter that counts for each RDM
	# how many times it has been calculated (by different shifts)
	#
	H_RDMs = [None]*N2
	V_RDMs = [None]*N2

	H_counter = [0]*N2
	V_counter = [0]*N2


	#
	# ========================== Main Loop ===============================
	#
	#         Go over all shifts, and calculate the RDMs in the
	#         bulk of each block for each shift. Average the RDMs over
	#         the different shifts.
	#
	# ====================================================================
	#


	for (i_shift, j_shift) in shifts:

		if log:
			print("\n\n")
			print(f"==> At shift {i_shift, j_shift} ")

		#
		# First, shift the TN and the RDMs list according to
		# (i_shift, j_shift)
		#

		PEPS_T_list = shift(PEPS_T_list, i_shift, j_shift)
		T_list =  shift(T_list, i_shift, j_shift)

		V_RDMs = shift(V_RDMs, i_shift, j_shift)
		H_RDMs =  shift(H_RDMs, i_shift, j_shift)
		V_counter = shift(V_counter, i_shift, j_shift)
		H_counter =  shift(H_counter, i_shift, j_shift)

		#
		# Now run blockBP. If we're on MPI mode then first tell the
		# slaves to go into blockBP mode.
		#

		if isMPI:
			for ijb in range(Nb2):
				if log:
					print(f"Master: sending block {ijb} a command to go into blockbp")
				command=10
				mpi_comm.send(command, dest=ijb, tag=1)
				if log:
					print("Master: sent.")

			if log:
				print("\n\n")
				print("Master: going into blockbp in MPI mode\n")

		m_list, blob = blockbp.blockbp(T_list, edges_list, pos_list, \
			blocks_v_list, sedges_dict, sedges_list, blocks_con_list, \
			D_trunc=BP_D_trunc, max_iter=10, delta=BP_delta, Lx=Lx, Ly=Ly, \
			initial_m_mode='RQ', mpi_comm=mpi_comm)

		eps = blob['final-error']
		iter_no = blob['final-iter']

		if log:
			print(f"=> done with err={eps} using {iter_no} iterations.")

		BTN_list = blob['BTN_list']

		#
		# Now go over all BTNs and update the gates that correspond
		# to the mode.
		#

		if log:
			print("\nCalculating the Blocks RDMs:")
			print("--------------------------------")


		#
		# Define empty lists to hold the RDMs of each block
		#

		RDMs_H_list = [None]*Nb2
		RDMs_V_list = [None]*Nb2


		if isMPI:
			#
			# We're in MPI mode. So first tell each block to find its RDMs.
			# Then collect all RDMs from blocks into the lists
			# RDMs_H_list, RDMs_V_list
			#

			for ib in range(Nb):
				for jb in range(Nb):

					ijb = ib*Nb + jb  # the block number

					if log:
						print(f"At block {ijb} {ib,jb}")

					BTN = BTN_list[ijb]

					#
					# Create a list of the PEPS tensors in the BTN. We need this
					# list to calculate the RDMs in the bulk of the block.
					#
					mid_PEPS = []
					for i1 in range(n):
						for j1 in range(n):
							i = ib*n + i1
							j = jb*n + j1

							mid_PEPS.append(PEPS_T_list[i*N+j])

					#
					# Send the BTN + mid_PEPS to the MPI slave.
					#

					command=30
					mpi_comm.send(command, dest=ijb, tag=1)

					mpi_comm.send((BTN, mid_PEPS, D_trunc, D_trunc2), dest=ijb, tag=2)

			#
			# Now get the different RDMs from the slaves
			#

			for ib in range(Nb):
				for jb in range(Nb):

					ijb = ib*Nb + jb  # the block number

					if log:
						print(f"Master: recieving RDMs from block {ijb}")
					(RDMs_H_list[ijb], RDMs_V_list[ijb]) = mpi_comm.recv(source=ijb, tag=3)
					if log:
						print("Master: got it.")




		else:
			#
			# So we're not on MPI mode. So go sequentially over all blocks
			# and for each block calculate its RDMs and put them in the lists
			# RDMs_H_list, RDMs_V_list
			#

			for ib in range(Nb):
				for jb in range(Nb):

					ijb = ib*Nb + jb  # the block number

					if log:
						print(f"At block {ijb} {ib,jb}")

					BTN = BTN_list[ijb]

					#
					# Create a list of the PEPS tensors in the BTN. We need this
					# list to calculate the RDMs in the bulk of the block.
					#
					mid_PEPS = []
					for i1 in range(n):
						for j1 in range(n):
							i = ib*n + i1
							j = jb*n + j1

							mid_PEPS.append(PEPS_T_list[i*N+j])

					#
					# Now use the BTN and the PEPS of the block to calculate the
					# bulk RDMs.
					#

					RDMs_H_list[ijb] = apply_horiz_block_gates(BTN, mid_PEPS, \
						mode='RDMs', D_trunc=D_trunc, D_trunc2=D_trunc2)

					RDMs_V_list[ijb] = apply_vertical_block_gates(BTN, mid_PEPS, \
						mode='RDMs', D_trunc=D_trunc, D_trunc2=D_trunc2)

		#
		# Once we have the RDMs of each block, add them to the global
		# RDMs list
		#

		for ib in range(Nb):
			for jb in range(Nb):

				ijb = ib*Nb + jb  # the block number

				#
				# Add the Horizontal RDMs to the global array
				#

				k=0
				for j1 in range(1,n-2):
					for i1 in range(1, n-1):

						i = ib*n + i1
						j = jb*n + j1

						rho = RDMs_H_list[ijb][k]
						k += 1

						H_counter[i*N+j] += 1

						if H_RDMs[i*N+j] is None:
							H_RDMs[i*N+j] = rho
						else:
							H_RDMs[i*N+j] += rho

				#
				# Add the Vertical RDMs to the global array
				#

				k=0
				for i1 in range(1, n-2):
					for j1 in range(1, n-1):

						i = ib*n + i1
						j = jb*n + j1

						rho = RDMs_V_list[ijb][k]
						k += 1

						V_counter[i*N+j] += 1

						if V_RDMs[i*N+j] is None:
							V_RDMs[i*N+j] = rho
						else:
							V_RDMs[i*N+j] += rho

		#
		# Once we calculated the RDMs in the shifted TN, bring back all
		# the lists to their unshifted-state.
		#

		PEPS_T_list = shift(PEPS_T_list, -i_shift, -j_shift)
		T_list =  shift(T_list, -i_shift, -j_shift)

		V_RDMs = shift(V_RDMs, -i_shift, -j_shift)
		H_RDMs =  shift(H_RDMs, -i_shift, -j_shift)
		V_counter = shift(V_counter, -i_shift, -j_shift)
		H_counter =  shift(H_counter, -i_shift, -j_shift)

	#
	#   ===================   End of Main loop  ==========================
	#


  #
  # Now calculate the average RDMs by dividing the sums of the RDMs
  # by the counter (the number of RDMs added at each site)
  #
	for ij in range(N2):
		if H_counter[ij]>0:
			H_RDMs[ij] /= H_counter[ij]

		if V_counter[ij]>0:
			V_RDMs[ij] /= V_counter[ij]


	return H_RDMs, V_RDMs



#
# ------------------------- bmps_RDMs ---------------------------------
#

def bmps_RDMs(T_list, Dp=None) -> Tuple[
	List[np.ndarray],
	List[np.ndarray]
]:
	"""

	Given an open square PEPS use the boundary MPS to calculate the
	horizontal and vertical 2-body RDMs. This is actually a wrapper to the
	bmpslib function calculate_PEPS_2RDM.

	Input Parameters:
	------------------
	T_list --- The list of PEPS tensors. Each tensor should be of the
	           form [d, D_L, D_R, D_U, D_D]

	           The order of the tensors is [N*i + j] (row i, column j)

	           On the boundaries, the virtual bond dimension should be 1.


	Dp --- The truncation bond for the boundary MPS. If omitted than
	       2D^2 is used, where D is the maximal virtual bond dimension
	       in the PEPS.

	OUTPUT:
	---------

	(H_RDMs, V_RDMs) --- Two lists of 2-Body RDMS (Horizontal and
	                     Vertical). Each RDM is of the form [i1,j1; i2,j2]
	                     and contains N^2 RDMs. The non-existant RDMs are
	                     set to None --- for example, (i,N-1)-->(i,N)

	                     The order of the RDMs in the lists is by the
	                     usual rule [i*N+j].

	                     (*) In the H_RDMs, the [i*N+j] entry is the
	                         (i,j)--(i,j+1) RDM.

	                     (*) In the V_RDMs, the [i*N+j] entry is the
	                         (i,j)--(i+1,j) RDM.



	"""

	Log=False

	#
	# Get the geometrical dimension of the TN. We expect an N\times N
	# PEPS, so the overall number of tensors is N^2
	#
	N2 = len(T_list)
	N = int(sqrt(N2))

	#
	# Build a bmpslib PEPS object from the PEPS tensors
	#
	PEPS = bmpslib.PEPS(N,N)

	for i in range(N):
		for j in range(N):
			PEPS.set_site(T_list[i*N+j], i, j)

	#
	# If Dp is not given, the take it to be 2D^2, where D is the maximal
	# bond dimension in the PEPS
	#
	if Dp is None:
		Dp = 1
		for T in T_list:
			D = max(list(T.shape[1:]))
			D2 = 2*D*D
			if D2>Dp:
				Dp = D2

	#
	# Invoke the bmpslib.calculate_PEPS_2RDM function to do the heavy
	# lifting
	#
	RDMs_list = bmpslib.calculate_PEPS_2RDM(PEPS, Dp, log=Log)


	#
	# Copy the RDMs into two lists, horizontal RDMs and Vertical RDMs
	#
	H_RDMs = [None]*N2
	V_RDMs = [None]*N2


	rdm_idx=0

	for i in range(N):
		for j in range(N-1):
			rho = RDMs_list[rdm_idx]

			H_RDMs[i*N+j] = rho

			rdm_idx += 1


	for j in range(N):
		for i in range(N-1):
			rho = RDMs_list[rdm_idx]

			V_RDMs[i*N+j] = rho

			rdm_idx += 1

	return H_RDMs, V_RDMs



#
# --------------------------  PEPS_energy  -----------------------------
#

def PEPS_energy(
	H:Callable[[int, int, int, int, int], np.ndarray], 
	H_RDMs, 
	V_RDMs
) -> float:
	"""

	Calculate the global energy of a 2-local Hamiltonian H using a list
	of its 2-body RDMs (which can be calculated either by BP_RDMs or by
	bmps_RDMs).


	Input Parameters:
	------------------

	H --- A ref to the Hamiltonian function. Should be of the form
	      H(n, i1, j1, i2, j2) and output the 2-local operator of
	      the enegy between (i1,j1)--(i2,j2)

	(H_RDMs, V_RDMs) --- Two lists of 2-Body RDMS (Horizontal and
	                     Vertical). Each RDM is of the form [i1,j1; i2,j2].

	                     The order of the RDMs in the lists is by the
	                     usual rule [i*N+j].

	                     (*) In the H_RDMs, the [i*N+j] entry is the
	                         (i,j)--(i,j+1) RDM.

	                     (*) In the V_RDMs, the [i*N+j] entry is the
	                         (i,j)--(i+1,j) RDM.

	OUTPUT: the total energy E = <psi|H|psi>


	"""

	N2 = len(H_RDMs)
	N = int(sqrt(N2))

	energy = 0.0

	#
	# Calculate the energy of the horizontal interactions
	#
	for i in range(N):
		for j in range(N-1):
			rho = H_RDMs[i*N+j]

			if rho is not None:
				h = H(N, i,j, i,j+1)
				e = tensordot(rho, h, axes=([0,1,2,3], [0,1,2,3]))
				energy += e.real


	#
	# Calculate the energy of the vertical interactions
	#
	for j in range(N):
		for i in range(N-1):
			rho = V_RDMs[i*N+j]

			if rho is not None:
				h = H(N, i,j, i+1,j)
				e = tensordot(rho, h, axes=([0,1,2,3], [0,1,2,3]))
				energy += e.real

	return energy







#
# --------------------------  draw_E_map  -----------------------------
#

def draw_E_map(H, H_RDMs, V_RDMs, fname, minE=None, maxE=None, title=''):


	if PYX==0:
		print("\n\n Error in draw_PEPS_energy! pyx module not found. ")
		print(" Energy map is not created\n")
		return


	N2 = len(H_RDMs)
	N = int(sqrt(N2))

	#
	# Calculate the energy of the horizontal interactions
	#

	Eden=[]


	for i in range(N):
		for j in range(N):
			rho = H_RDMs[i*N+j]

			if rho is not None:
				h = H(N, i,j, i,j+1)
				e = tensordot(rho, h, axes=([0,1,2,3], [0,1,2,3]))
				energy = e.real

				print(f"E[{i},{j}] = {energy}")

				Eden.append( [j,N-i, energy])

	fname_H = 'Horiz-Emap-' + fname

	ca = pyx.graph.axis.linear(min=minE, max=maxE, title="$E$")

	g = pyx.graph.graphxy(height=8, width=8,
                  x=pyx.graph.axis.linear(min=0, max=N-1, title=r"$j$"),
                  y=pyx.graph.axis.linear(min=0, max=N-1, title=r'$i$'))
	g.plot(pyx.graph.data.points(Eden, x=1, y=2, color=3, title="Horizontal E"),
       [pyx.graph.style.density(gradient=pyx.color.gradient.Jet, \
				coloraxis=ca)])

	c = pyx.canvas.canvas()
	c.insert(g, [pyx.trafo.translate(0,-1.5)])
	c.text(4, 8, title, [pyx.text.halign.boxcenter])
	c.text(4, 7, r"Horizontal Energy", [pyx.text.halign.boxcenter])
	c.writePDFfile(fname_H)


	#
	# Calculate the energy of the vertical interactions
	#

	Eden=[]

	for i in range(N):
		for j in range(N):
			rho = V_RDMs[i*N+j]

			if rho is not None:
				h = H(N, i,j, i+1,j)
				e = tensordot(rho, h, axes=([0,1,2,3], [0,1,2,3]))
				energy = e.real

				Eden.append( [j,N-i, energy])

	fname_V = 'Vert-Emap-' + fname

	g = pyx.graph.graphxy(height=8, width=8, \
                  x=pyx.graph.axis.linear(min=0, max=N-1, title=r"$j$"),\
                  y=pyx.graph.axis.linear(min=0, max=N-1, title=r'$i$')) \

	g.plot(pyx.graph.data.points(Eden, x=1, y=2, color=3, \
		title="Vertical E"), \
		[pyx.graph.style.density(gradient=pyx.color.gradient.Jet, \
		coloraxis=ca)])

	c = pyx.canvas.canvas()
	c.insert(g, [pyx.trafo.translate(0,-1.5)])
	c.text(4, 8, title, [pyx.text.halign.boxcenter])
	c.text(4, 7, r"Vertical Energy", [pyx.text.halign.boxcenter])
	c.writePDFfile(fname_V)








#
# --------------------------  MPI_slave  -----------------------------
#

def MPI_slave(mpi_comm, master):

	mpi_rank = mpi_comm.Get_rank()
	mpi_size = mpi_comm.Get_size()

	log = False

	if mpi_rank > master-1:
		#
		# master has the highest rank; any process above it is un-needed.
		#
		return

	if log:
		print("\n")
		print(f"At MPI slave {mpi_rank}")

	#
	# block number is equal to mpi process index (mpi rank)
	#

	ijb = mpi_rank

	#
	# Get inital info for the block
	#



	continue_slave_loop=True

	while continue_slave_loop:

		#
		# Get Master's command
		#

		if log:
			print(f"block {ijb}: getting master command")
		command = mpi_comm.recv(source=master, tag=1)
		if log:
			print(f"block {ijb}: got command = {command}")

		if command==10:
			if log:
				print(f"block {ijb}: going into blockbp.MPIblock")
			blockbp.MPIblock(mpi_comm)

			if log:
				print(f"block {ijb}: came back from blockbp.MPIblock")

		if command==1:
			if log:
				print(f"block {ijb}: getting inital update_blocks info")

			(D_trunc, target_D, H_even_block_gates, H_odd_block_gates, \
					V_even_block_gates, V_odd_block_gates) \
						=  mpi_comm.recv(source=master, tag=2)

			if log:
				print(f"block {ijb}: got initial info")


		if command==20:
			if log:
				print(f"block {ijb}: getting update_blocks info")
			(mode, BTN, mid_PEPS) = mpi_comm.recv(source=master, tag=2)

			if log:
				print(f"block {ijb}: got it. mode={mode}")

			#
			# Perform the gates update in the block
			#

			if mode=='H-even':
				apply_horiz_block_gates(BTN, mid_PEPS, mode, \
					H_even_block_gates, Dmax=target_D, D_trunc=D_trunc)

			elif mode=='H-odd':
				apply_horiz_block_gates(BTN, mid_PEPS, mode, \
					H_odd_block_gates, Dmax=target_D, D_trunc=D_trunc)

			elif mode=='V-even':
				apply_vertical_block_gates(BTN, mid_PEPS, mode, \
					V_even_block_gates, Dmax=target_D, D_trunc=D_trunc)

			elif mode=='V-odd':
				apply_vertical_block_gates(BTN, mid_PEPS, mode, \
					V_odd_block_gates, Dmax=target_D, D_trunc=D_trunc)

			#
			# Send the resultant mid_PEPS to the master
			#

			mpi_comm.send(mid_PEPS, dest=master, tag=3)


		if command==30:
			#
			# Calculate the block RDMs (used in BP_RDMs)
			#

			if log:
				print(f"block {ijb}: getting BP_RDMs info")
			(BTN, mid_PEPS, D_trunc, D_trunc2) = mpi_comm.recv(source=master, tag=2)

			if log:
				print(f"block {ijb}: got it. mode={mode}")

			RDMs_H_list = apply_horiz_block_gates(BTN, mid_PEPS, \
				mode='RDMs', D_trunc=D_trunc, D_trunc2=D_trunc2)

			RDMs_V_list = apply_vertical_block_gates(BTN, mid_PEPS, \
				mode='RDMs', D_trunc=D_trunc, D_trunc2=D_trunc2)

			#
			# Send the resultant RDMs to the master
			#

			mpi_comm.send( (RDMs_H_list, RDMs_V_list), dest=master, tag=3)



		if command==100:
			continue_slave_loop = False


	return






def _main_test():
	from _solve_heisenberg import solve_heisenberg
	solve_heisenberg()

if __name__ == "__main__":
	_main_test()