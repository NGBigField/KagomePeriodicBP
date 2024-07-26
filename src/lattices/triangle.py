# Import types used in the code:
from lattices._common import Node, sorted_boundary_nodes, plot_lattice
from lattices.directions import LatticeDirection, BlockSide, Direction, DirectionError
from lattices.edges import edges_dict_from_edges_list, EdgesDictType
from _error_types import LatticeError, OutsideLatticeError

# For type hinting:
from typing import Generator, Final, Iterable, TypeGuard, TypeAlias, NamedTuple
from dataclasses import dataclass

# some of our utils:
from utils import tuples, lists, strings

# For caching results:
import functools 

# for math:
import numpy as np

# For datastructures required in some algorithms:
from queue import Queue


class TriangularLatticeError(LatticeError): ...


CACHED_ORDERED_BOUNDARY_EDGES_PER_LATTICE_SIZE : Final[dict[int, list[str]]] = {}


@functools.cache
def total_vertices(N:int) -> int:
	"""
	Returns the total number of vertices in the *bulk* of a hex 
	TN with linear parameter N
	"""
	return 3*N*N - 3*N + 1


@functools.cache
def linear_size_from_total_vertices(NT:int) -> int:
	## Solving a quadratic roots formula:
	# Two options:
	for sign in [+1, -1]:
		N = ( 3 + sign*np.sqrt(9 - 12*(1 - NT)) )/6 
		if N<0:
			continue
		N = int(N)
		if NT!=total_vertices(N):
			continue
		return N
	raise ValueError("No solution")

@functools.cache
def center_vertex_index(N):
    i = num_rows(N)//2
    j = i
    return get_vertex_index(i, j, N)


@functools.cache
def num_rows(N):
	return 2*N-1


def row_width(i, N):
	"""
	Returns the width of a row i in a hex TN of linear size N. i is the
	row number, and is between 0 -> (2N-2)
	"""
	if i<0 or i>2*N-2:
		return 0
	
	return N+i if i<N else 3*N-i-2

def _get_neighbor_coordinates_in_direction_no_boundary_check(i:int, j:int, direction:LatticeDirection, N:int)->tuple[int, int]:
	## Simple L or R:
	if direction==LatticeDirection.L:  
		return i, j-1
	if direction==LatticeDirection.R:  
		return i, j+1

	## Row dependant:
	middle_row_index = num_rows(N)//2   # above or below middle row

	if direction==LatticeDirection.UR: 
		if i <= middle_row_index: 
			return i-1, j
		else: 
			return i-1, j+1
	
	if direction==LatticeDirection.UL:
		if i <= middle_row_index:
			return i-1, j-1
		else:
			return i-1, j
		
	if direction==LatticeDirection.DL:
		if i < middle_row_index:
			return i+1, j
		else:
			return i+1, j-1
		
	if direction==LatticeDirection.DR:
		if i < middle_row_index:
			return i+1, j+1
		else:
			return i+1, j
		
	TriangularLatticeError(f"Impossible direction {direction!r}")


def check_boundary_vertex(index:int, N)->list[BlockSide]:
	on_boundaries = []

	# Basic Info:
	i, j = get_vertex_coordinates(index, N)
	height = num_rows(N)
	width = row_width(i,N)
	middle_row_index = height//2

	# Boundaries:
	if i==0:
		on_boundaries.append(BlockSide.U)
	if i==height-1:
		on_boundaries.append(BlockSide.D)
	if j==0: 
		if i<=middle_row_index:
			on_boundaries.append(BlockSide.UL)
		if i>=middle_row_index:
			on_boundaries.append(BlockSide.DL)
	if j == width-1:
		if i<=middle_row_index:
			on_boundaries.append(BlockSide.UR)
		if i>=middle_row_index:
			on_boundaries.append(BlockSide.DR)
	
	return on_boundaries


def get_neighbor_coordinates_in_direction(i:int, j:int, direction:LatticeDirection, N:int)->tuple[int, int]:
	i2, j2 = _get_neighbor_coordinates_in_direction_no_boundary_check(i, j, direction, N)

	if i2<0 or i2>=num_rows(N):
		raise OutsideLatticeError()
	
	if j2<0 or j2>=row_width(i2, N):
		raise OutsideLatticeError()
	
	return i2, j2


def get_neighbor_by_index_and_direction(ind:int, direction:LatticeDirection, N:int) -> int:
	i, j = get_vertex_coordinates(ind, N)
	return get_neighbor(i, j, direction, N)


def get_neighbor(i:int, j:int, direction:LatticeDirection, N:int)->int:	
	i2, j2 = get_neighbor_coordinates_in_direction(i, j, direction, N)
	return get_vertex_index(i2, j2, N)


def all_neighbors(index:int, N:int)->Generator[tuple[Node, LatticeDirection], None, None]:
	i, j = get_vertex_coordinates(index, N)
	for direction in LatticeDirection.all_in_counter_clockwise_order():
		try: 
			neighbor = get_neighbor(i, j, direction, N)
		except OutsideLatticeError:
			continue
		yield neighbor, direction


@functools.cache
def get_vertex_coordinates(index, N)->tuple[int, int]:
	running_index = 0 
	for i in range(num_rows(N)):
		width = row_width(i, N)
		if index < running_index + width:
			j = index - running_index
			return i, j
		running_index += width
	raise TriangularLatticeError("Not found")


def get_vertex_index(i:int, j:int, N:int) -> int:
	"""
	Given a location (i,j) of a vertex in the hexagon, return its 
	index number. The vertices are ordered left->right, up->down.
	
	The index number is a nunber 0->NT-1, where NT=3N^2-3N+1 is the
	total number of vertices in the hexagon.
	
	The index i is the row in the hexagon: i=0 is the upper row.
	
	The index j is the position of the vertex in the row. j=0 is the 
	left-most vertex in the row.
	"""
	
	# Calculate Aw --- the accumulated width of all rows up to row i,
	# but not including row i.
	if i==0:
		Aw = 0
	else:
		Aw = (i*N + i*(i-1)//2 if i<N else 3*N*N -3*N +1 -(2*N-1-i)*(4*N-2-i)//2)
		
	return Aw + j


@functools.cache
def get_center_vetex_coordinates(N:int) -> tuple[int, int]:
	i = num_rows(N)//2
	j = row_width(i, N)//2
	return i, j

		
@functools.cache
def get_center_vertex_index(N):
	i, j = get_center_vetex_coordinates(N)
	return get_vertex_index(i, j, N)


def get_node_position(i,j,N):
	w = row_width(i, N)
	x = N - w + 2*j	
	y = N - i
	return x, y


def get_edge_index(i,j,side,N):
	"""
	Get the index of an edge in the triangular PEPS.
	
	Given a vertex (i,j) in the bulk, we have the following rules for 
	labeling its adjacent legs:
	
	       (i-1,j-1)+2*NT+101 (i-1,j)+NT+101
	                         \   /
	                          \ /
	               ij+1 ------ o ------ ij+2
	                          / \
	                         /   \
                   ij+NT+101  ij+2*NT+101
                   
  Above ij is the index of (i,j) and we denote by (i-1,j-1) and (i-1,j)
  the index of these vertices. NT is the total number of vertices in the
  hexagon.
  
  Each of the 6 boundaries has 2N-1 external legs. They are ordered 
  counter-clockwise and labeled as:
  
  d1, d2, ...   --- Lower external legs
  dr1, dr2, ... --- Lower-Right external legs
  ur1, ur2, ... --- Upper-Right external legs
  u1, u2, ...   --- Upper external legs
  ul1, ul2, ... --- Upper-Left external legs
  dl1, dl2, ... --- Lower-Left external legs
  

	Input Parameters:
	------------------
	i,j --- location of the vertex to which the edge belong.
	        i=row, j=column. i=0 ==> upper row, j=0 ==> left-most column.

	side --- The side of the edge. Either 'L', 'R', 'UL', 'UR', 'DL', 'DR'

	N    --- Linear size of the lattice
	
	OUTPUT: the label
	"""

	# The index of the vertex
	ij = get_vertex_index(i,j,N)

	# Calculate the width of the row i (how many vertices are there)
	w = row_width(i, N)
	
	# Total number of vertices in the hexagon
	NT = total_vertices(N)
		
	if side=='L':
		if j>0:
			e = ij
		else:
			if i<N:
				e = f'ul{i*2+1}'
			else:
				e = f'dl{(i-N+1)*2}'
				
	if side=='R':
		if j<w-1:
			e = ij+1
		else:
			if i<N-1:
				e = f'ur{2*(N-1-i)}'
			else:
				e = f'dr{2*(2*N-2-i)+1}'  
				
	if side=='UL':
		if i<N:
			if j>0:
				if i>0:
					e = get_vertex_index(i-1,j-1,N) + 2*NT + 101
				else:
					# i=0
					e = f'u{2*N-1-j*2}'
			else:
				# j=0
				if i==0:
					e = f'u{2*N-1}'
				else:
					e = f'ul{2*i}'
					
		else:
			# so i=N, N+1, ...
			e = get_vertex_index(i-1,j,N) + 2*NT + 101
				
	if side=='UR':
		if i<N:
			if j<w-1:
				if i>0:
					e = get_vertex_index(i-1,j,N) + NT + 101
				else:
					# i=0
					e = f'u{2*N-2-j*2}'
			else:
				# j=w-1
				if i==0:
					e = f'ur{2*N-1}'
				else:
					e = f'ur{2*N-1-2*i}'
		else:
			e = get_vertex_index(i-1,j+1,N) + NT + 101
			
	if side=='DL':
		if i<N-1:
			e = ij + NT + 101
		else:
			# so i=N-1, N, N+1, ...
			if j>0:
				if i<2*N-2:
					e = ij + NT + 101
				else:
					# i=2N-2 --- last row
					e = f'd{2*j}'
			else:
				# j=0
				if i<2*N-2:
					e = f'dl{(i-N+1)*2+1}'
				else:
					e = f'dl{2*N-1}'
					
	if side=='DR':
		if i<N-1:
			e = ij + 2*NT + 101
		else:
			# so i=N-1, N, ...
			if j<w-1:
				if i<2*N-2:
					e = ij + 2*NT + 101
				else:
					# i=2N-2 --- last row
					e = f'd{2*j+1}'
			else:
				# so we're on the last j
				if i<2*N-2:
					e = f'dr{(2*N-2-i)*2}'
				else:
					# at i=2*N-2 (last row)
					e = f'd{2*N-1}'
				
	return e


def create_hex_dicts(N):
	"""
	Creates two dictionaries for a two-way mapping of two different 
	indexing of the bulk vertices in the hexagon TN. These are then used 
	in the rotate_CW function to rotate the hexagon in 60 deg 
	counter-clock-wise.
	
	The two mappings are:
	-----------------------
	
	ij --- The ij index that is defined by row,col=(i,j). This integer
	       is calculated in get_vertex_index()
	       
	(n, side, p). Here the vertices sit on a set of concentrated hexagons.
	              n=1,2, ..., N is the size of the hexagon
	              side = (d, dr, ur, u, ul, dl) the side of the hexagon
	                      on which the vertex sit
	              p=0, ..., n-2 is the position within the side (ordered
	                        in a counter-clock order
	              
	              For example (N,'d',1) is the (i,j)=(2*N-2,1) vertex.
	              
	              
	The function creates two dictionaries:
	
	ij_to_hex[ij] = (n,side, p)
	
	hex_to_ij[(n,side,p)] = ij
	              
	              
	
	Input Parameters:
	-------------------
	
	N --- size of the hexagon.
	
	OUTPUT:
	--------
	
	ij_to_hex, hex_to_ij --- the two dictionaries.
	"""
	
	hex_to_ij = {}
	ij_to_hex = {}
	
	sides = ['d', 'dr', 'ur', 'u', 'ul', 'dl']
	
	i,j = 2*N-1,-1


	# To create the dictionaries, we walk on the hexagon counter-clockwise, 
	# at radius n for n=N ==> n=1. 
	#
	# At each radius, we walk 'd' => 'dr' => 'ur' => 'u' => 'ul' => 'dl'

	for n in range(N, 0, -1):
		i -= 1
		j += 1
		
		#
		# (i,j) hold the position of the first vertex on the radius-n 
		# hexagon
		# 
		
		for side in sides:
			
			for p in range(n-1):
				
				ij = get_vertex_index(i,j,N)
				
				ij_to_hex[ij] = (n,side,p)
				hex_to_ij[(n,side,p)] = ij
				
				if side=='d':
					j +=1
					
				if side=='dr':
					j +=1
					i -=1
					
				if side=='ur':
					j -=1
					i -=1
					
				if side=='u':
					j -= 1
					
				if side=='ul':
					i += 1
					
				if side=='dl':
					i += 1
		

	#
	# Add the centeral node (its side is '0')
	#
	
	ij = get_vertex_index(i,j,N)
	
	ij_to_hex[ij] = (n,'0',0)
	hex_to_ij[(n,'0',0)] = ij
	
	return ij_to_hex, hex_to_ij


def rotate_CW(N, ijs, ij_to_hex, hex_to_ij):
	
	"""
	
	Takes a list of vertices on the hexagon TN and rotates it 60 degrees
	clockwise.
	
	
	Input Parameters:
	------------------
	
	N --- Hexagon size (radius)
	
	ijs --- The list of vertices
	
	ij_to_hex, hex_to_ij --- two dictionaries used to switch between the
	                         usual (ij) indexing to the internal hexagon
	                         indexing. This is used for the rotation.
	
	OUTPUT:
	--------
	
	new_ijs --- A corresponding list of rotated vertices.
	
	
	Notes: the vertices can be of any type, bulk vertices or external
	       MPS vertices.
	
	
	"""
	
	
	
	NT = total_vertices(N)
	
	new_ijs = []   # Initialize the new vertices list
	
	for ij in ijs:
		
		#
		# Check if ij is a bulk vertex or if it is an external MPS vertex.
		#
		
		if ij>= NT:
			#
			# Rotate an MPS vertex
			# 
			if ij>=NT+10*N-5:
				#
				# We're on the Up-Left edge --- so we rotate to the upper edge
				#
				
				new_ij = ij - 10*N + 5
				
			else:
				#
				# We're other edge --- so rotate clockwise by adding 2N-1
				#
				new_ij = ij + 2*N-1
				
		else:
			#
			# Rotate a bulk vertex using the dictionaries. Once we know
			# the (n,side,p) of a vertex, all we need to do is rotate side
			# to its clockwise neighbor.
			#
			
			(n,side, p) = ij_to_hex[ij]
			
			if side=='u':
				side = 'ur'
			elif side=='ur':
				side = 'dr'
			elif side=='dr':
				side = 'd'
			elif side=='d':
				side = 'dl'
			elif side=='dl':
				side = 'ul'
			elif side=='ul':
				side = 'u'
				
			new_ij = hex_to_ij[(n,side,p)]
			
		new_ijs.append(new_ij)
		
	return new_ijs
			



def rotate_ACW(N, ijs, ij_to_hex, hex_to_ij):
	
	"""
	
	Takes a list of vertices on the hexagon + MPSs TN and rotates it 
	60 degrees anti-clockwise.
	
	
	Input Parameters:
	------------------
	
	N --- Hexagon size (radius)
	
	ijs --- The list of vertices
	
	ij_to_hex, hex_to_ij --- two dictionaries used to switch between the
	                         usual (ij) indexing to the internal hexagon
	                         indexing. This is used for the rotation.
	
	OUTPUT:
	--------
	
	new_ijs --- A corresponding list of rotated vertices.
	
	
	Notes: the vertices can be of any type, bulk vertices or external
	       MPS vertices.
	
	
	"""
	
	
	
	NT = total_vertices(N)
	
	new_ijs = []   # Initialize the new vertices list
	
	for ij in ijs:
		
		#
		# Check if ij is a bulk vertex or if it is an external MPS vertex.
		#
		
		if ij>= NT:
			#
			#     -------------    Rotate an MPS vertex  ------------------
			# 
			if ij>=NT+10*N-5:
				#
				# We're on the Down-Left edge --- so we rotate to the bottom edge
				#
				
				new_ij = ij - 10*N + 5
				
			else:
				#
				# We're other edge --- so rotate anti-clockwise by adding 2N-1
				#
				new_ij = ij + 2*N-1
				
		else:
			#     -------------    Rotate a bulk vertex  ------------------
			#
			# Rotate a bulk vertex using the dictionaries. Once we know
			# the (n,side,p) of a vertex, all we need to do is rotate side
			# to its clockwise neighbor.
			#
			
			(n,side, p) = ij_to_hex[ij]


			if side=='d':
				side = 'dr'
			elif side=='dr':
				side = 'ur'
			elif side=='ur':
				side = 'u'
			elif side=='u':
				side = 'ul'
			elif side=='ul':
				side = 'dl'
			elif side=='dl':
				side = 'd'
				
			new_ij = hex_to_ij[(n,side,p)]
			
		new_ijs.append(new_ij)
		
	return new_ijs


@dataclass
class _EdgeDataForShifting:
	vertex_ind : int
	edge_ind : int
	crnt_name : str
	new_name : str  = ""


def _is_boundary_edge(edge:str|int) -> TypeGuard[str]:
	if isinstance(edge, int):
		return False
	assert isinstance(edge, str)
	assert edge[0].isalpha
	return True


def _split_boundary_edge_name_to_side_and_order(edge_name:str) -> tuple[str, int]:
	c0 : str = edge_name[0]
	c1 : str = edge_name[1]
	# Derive side name string:
	assert c0.isalpha, "First string in boundary edge must be a 'd' or 'u'"
	if c1.isalpha():  
		side_name = c0+c1  # second char is also alphabetical 
		order = edge_name[2:]
	else: 			  
		side_name = c0
		order = edge_name[1:]

	return side_name, int(order)


def _boundary_edge_order_by_num_order(edge_data:_EdgeDataForShifting) -> int:
	side_name, order = _split_boundary_edge_name_to_side_and_order(edge_data.crnt_name)
	return order


def _boundary_edge_order_by_side(edge_data:_EdgeDataForShifting) -> int:
	side_name, order = _split_boundary_edge_name_to_side_and_order(edge_data.crnt_name)
	# direction by order:
	for i, side in enumerate(BlockSide.all_in_counter_clockwise_order()):
		crnt_side_name = str(side).casefold()
		if crnt_side_name == side_name:
			return i
	raise ValueError("Not found")



def _shift_boundary_edges_clockwise(edges_list:list[list[str|int]], N:int) -> list[str]:
	""" First, find all the edges and keep their order in the following format:

	During the algorithm, a list is created and sorted containing boundary edges:
	[(i_1, j_1, e_1, n_1), (i_2, j_2, e_2, n_2), ..., (i_N, j_N, e_N, n_N)]
	Where i is the index of the vertex
	and j is the index of its leg where the edge is.
	and e is the name of the edge

	Returns a list with all ordered edges
	"""
	## Define basic data:
	# How many edges per edge do we expect:
	num_outer_edges = 2*N-1

	## Define helper functions:

	## Search for all boundary edges:
	boundary_edges = [
		[_EdgeDataForShifting(vertex_ind=i, edge_ind=j, crnt_name=edge) for j, edge in enumerate(edges) if _is_boundary_edge(edge)] 
		for i, edges in enumerate(edges_list)
	]
	boundary_edges = lists.join_sub_lists(boundary_edges)

	## Sort edges:
	# Sort by number:
	boundary_edges.sort(key=_boundary_edge_order_by_num_order)
	# sort by direction:
	boundary_edges.sort(key=_boundary_edge_order_by_side)

	## Cyclicly assign new names according to next in order:
	for prev, crnt, next in lists.iterate_with_periodic_prev_next_items(boundary_edges):
		crnt.new_name = next.crnt_name
	
	## Change names:
	for edge_data in boundary_edges:
		i = edge_data.vertex_ind
		j = edge_data.edge_ind
		## Apply on original data-structure:
		edges_list[i][j] = edge_data.new_name 
	
	## For output:
	return [edge_data.new_name for edge_data in boundary_edges]
	
	
    

def create_triangle_lattice(N)->list[Node]:

	"""
	The structure of every node in the list is:
	
	T[d, D_L, D_R, D_UL, D_UR, D_DL, D_DR]
	
	With the following structure
	
                       (3)    (4)
                        UL    UR
                          \   /
                           \ /
                   (1)L  ---o--- R(2)
                           / \
                          /   \
                        DL     DR
                       (5)    (6)

	"""

	NT = total_vertices(N)

	if N<2 and False:
		print("Error in create_random_trianlge_PEPS !!!")
		print(f"N={N} but it must be >= 2 ")
		exit(1)		


	#
	# Create the list of edges. 
	#
	edges_list = []
	h = num_rows(N)
	for i in range(h):
		w = row_width(i,N)

		for j in range(w):
			eL  = get_edge_index(i,j,'L' , N)
			eR  = get_edge_index(i,j,'R' , N)
			eUL = get_edge_index(i,j,'UL', N)
			eUR = get_edge_index(i,j,'UR', N)
			eDL = get_edge_index(i,j,'DL', N)
			eDR = get_edge_index(i,j,'DR', N)

			edges_list.append([eL, eR, eUL, eUR, eDL, eDR])

	#
	# Rotate edges at boundary clockwise.
	# 	Done because the canonical form in the original triangular blockBP
	#	code, does not the shape of the block in this code:
	#
	boundary_edges = _shift_boundary_edges_clockwise(edges_list, N)  
	CACHED_ORDERED_BOUNDARY_EDGES_PER_LATTICE_SIZE[N] = boundary_edges

	#
	# Create the list of nodes:
	#
	index = 0
	nodes_list = []
	for i in range(2*N-1):
		w = row_width(i,N)
		for j in range(w):
			n = Node(
				index = index,
				pos = get_node_position(i, j, N),
				edges = edges_list[index],
				directions=[LatticeDirection.L, LatticeDirection.R, LatticeDirection.UL, LatticeDirection.UR, LatticeDirection.DL, LatticeDirection.DR]
			)
			nodes_list.append(n)
			index += 1


	return nodes_list


def all_coordinates(N:int)->Generator[tuple[int, int], None, None]:
	for i in range(num_rows(N)):
		for j in range(row_width(i, N)):
			yield i, j


def _unit_vector_rotated_by_angle(vec:tuple[int, int], angle:float)->tuple[float, float]:
	x, y = vec
	angle1 = np.angle(x+1j*y)
	new_angle = angle1+angle
	new_vec = np.cos(new_angle), np.sin(new_angle)
	new_vec /= new_vec[0]
	from utils import numerics
	return new_vec


def unit_vector_corrected_for_sorting_triangular_lattice(direction:Direction)->tuple[float, float]:
	if isinstance(direction, LatticeDirection):
		match direction:
			case LatticeDirection.R :	return (+1,  0)
			case LatticeDirection.L :	return (-1,  0)
			case LatticeDirection.UL:	return (-1, +1)
			case LatticeDirection.UR:	return (+1, +1)
			case LatticeDirection.DL:	return (-1, -1)
			case LatticeDirection.DR:	return (+1, -1)
	elif isinstance(direction, BlockSide):
		match direction:
			case BlockSide.U :	return ( 0, +1)
			case BlockSide.D :	return ( 0, -1)
			case BlockSide.UR:	return (+1, +1)
			case BlockSide.UL:	return (-1, +1)
			case BlockSide.DL:	return (-1, -1)
			case BlockSide.DR:	return (+1, -1)
	else:
		raise TypeError(f"Not a supported typee")


def sort_coordinates_by_direction(items:Iterable[tuple[int, int]], direction:Direction, N:int)->list[tuple[int, int]]:
	# unit_vector = direction.unit_vector  # This basic logic break at bigger lattices
	unit_vector = unit_vector_corrected_for_sorting_triangular_lattice(direction)
	def key(ij:tuple[int, int])->float:
		i, j = ij[0], ij[1]
		pos = get_node_position(i, j, N)
		return tuples.dot_product(pos, unit_vector)  # vector dot product
	return sorted(items, key=key)
	

@functools.cache
def vertices_indices_rows_in_direction(N:int, major_direction:BlockSide, minor_direction:LatticeDirection)->list[list[int]]:
	""" arrange nodes by direction:
	"""
	## Arrange indices by position relative to direction, in reverse order
	coordinates_in_reverse = sort_coordinates_by_direction(all_coordinates(N), major_direction.opposite(), N)

	## Bunch vertices by the number of nodes at each possible row (doesn't matter from wich direction we look)
	list_of_rows = []
	for i in range(num_rows(N)):

		# collect vertices as much as the row has:
		row = []
		w = row_width(i, N)
		for _ in range(w):
			item = coordinates_in_reverse.pop()
			row.append(item)
		
		# sort row by minor axis:
		sorted_row = sort_coordinates_by_direction(row, minor_direction, N)
		indices = [get_vertex_index(i, j, N) for i,j in sorted_row]
		list_of_rows.append(indices)

	return list_of_rows


def _assign_boundaries_to_nodes(lattice:list[Node]) -> None:
	N = linear_size_from_total_vertices(len(lattice))
	for node in lattice:
		boundaries = check_boundary_vertex(node.index, N)
		node.boundaries = set(boundaries)


def _sorted_boundary_edges(lattice:list[Node], boundary:BlockSide) -> list[str]:
	# Basic info:
	N = linear_size_from_total_vertices(len(lattice))
	_expected_num_outgoing_edges = 2*N-1

	## Get all boundary nodes:
	boundary_nodes = sorted_boundary_nodes(lattice, boundary)
	assert len(boundary_nodes)==N
	
	## Get all edges in order
	boundary_edges_set : set[tuple[str, int]] = set()
	for edge in CACHED_ORDERED_BOUNDARY_EDGES_PER_LATTICE_SIZE[N]:
		side, order = _split_boundary_edge_name_to_side_and_order(edge)
		if side==boundary.__str__().casefold():
			boundary_edges_set.add((side, order))
	# Sort by oder and keep list of name+order:
	boundary_edges : list[str] = [f"{_tuple[0]}{_tuple[1]}" for _tuple in sorted(boundary_edges_set, key=lambda _tuple: _tuple[1])]
	assert _expected_num_outgoing_edges == len(boundary_edges_set)
	
	return boundary_edges


def _find_node_by_outer_edge(lattice:list[Node], edge:str) -> tuple[Node, int]:
	"""Looks for a node that has a certain outgoing edge.
	Return a node and the index of the leg associated with the edge
	"""
	for node in lattice:
		if edge in node.edges:
			return node, node.edges.index(edge)
	raise LatticeError("Node not found in lattice")


def _new_name_for_periodic_edge(edge1:str, edge2:str) -> str:
	d1, o1 = _split_boundary_edge_name_to_side_and_order(edge1)
	d2, o2 = _split_boundary_edge_name_to_side_and_order(edge2)
	if d1[0]=="d":
		return f"{d1}-{d2}-{o1}"
	elif d2[0]=="d":
		return f"{d2}-{d1}-{o2}"
	else:
		raise ValueError("Not a valid option")


def _kagome_connect_boundary_edges_periodically(lattice:list[Node], edge1:str, edge2:str) -> None:
	## Get basic data:
	n1, e1 = _find_node_by_outer_edge(lattice, edge1)
	n2, e2 = _find_node_by_outer_edge(lattice, edge2)
	new_edge_name = _new_name_for_periodic_edge(edge1, edge2)

	## Change edge name for both nodes:
	lattice[n1.index].edges[e1] = new_edge_name
	lattice[n2.index].edges[e2] = new_edge_name


def change_boundary_conditions_to_periodic(lattice:list[Node]) -> list[Node]:
	## Make sure that all boundary nodes are marked:
	_assign_boundaries_to_nodes(lattice)

	## connect per two opposite faces:
	ordered_half_list = list(BlockSide.all_in_counter_clockwise_order())[0:3]
	for block_side in ordered_half_list:
		boundary_edges          = _sorted_boundary_edges(lattice, boundary=block_side)
		opposite_boundary_edges = _sorted_boundary_edges(lattice, boundary=block_side.opposite())
		opposite_boundary_edges.reverse()

		for edge1, edge2 in zip(boundary_edges, opposite_boundary_edges, strict=True):
			_kagome_connect_boundary_edges_periodically(lattice, edge1, edge2)

	return lattice


def _shift_periodically_in_direction_given_periodic_lattice(
	periodic_lattice:list[Node], edges:EdgesDictType, given_permutation_list:list[int], direction:LatticeDirection
) -> list[int]:

	## Help functions:
	def get_new_node_in_direction(node:Node) -> int:
		edge = node.get_edge_in_direction(direction)
		_indices = edges[edge]
		if   _indices[0] == node.index:	return _indices[1]
		elif _indices[1] == node.index:	return _indices[0]
		else:
			raise LatticeError("Impossible situation")
		
	def _get_new_index(crnt_index:int) -> int:
		node = periodic_lattice[crnt_index]
		return get_new_node_in_direction(node)

	## Produce output:
	permutation_list = [_get_new_index(index) for index in given_permutation_list]
	return permutation_list


def _common_periodic_shifting_inputs(N:int) -> tuple[list[Node], EdgesDictType, list[int]]:
	## Create periodic triangular lattice:
	open_lattice = create_triangle_lattice(N)
	periodic_lattice = change_boundary_conditions_to_periodic(open_lattice)
	# plot_lattice(periodic_lattice, periodic=True)

	## Collect a comfortable dict of all edges:
	edges = edges_dict_from_edges_list([node.edges for node in periodic_lattice])

	## The starting permutation is each node pointing to itself
	num_nodes = total_vertices(N)
	null_permutation = list(range(num_nodes))  # Permutation from node to itself

	return periodic_lattice, edges, null_permutation


def shift_periodically_in_direction(N:int, direction:LatticeDirection) -> list[int]:
	"""Return the permutation list of of a triangular lattice when shifting periodically in a direction.

	Args:
		N (int): Size of lattice
		direction (LatticeDirection): Direction in which to move the lattice.

	Returns:
		list[int]: Permutation list
	"""
	return _shift_periodically_in_direction_given_periodic_lattice(
		*_common_periodic_shifting_inputs(N), direction
	)


	


class _PermutationLatticeWalkItem(NamedTuple):
	"""Each item keeps the index of the current node position, Together with the
	permutation the brought us from the cetner vertex to this node position. 
	"""
	node_index : int
	permutation : list[int]

	def __repr__(self) -> str:
		return f"{self.node_index}"


class _PermutationsQueue():
	__slots__ = "_queue", "_indices"

	def __init__(self) -> None:
		self._queue : Queue[_PermutationLatticeWalkItem] = Queue()
		self._indices : set[int] = set()

	def get(self) -> _PermutationLatticeWalkItem:
		item = self._queue.get()
		self._indices.remove(item.node_index)
		return item
	
	def put(self, item:_PermutationLatticeWalkItem) -> None:
		if item in self:
			assert ValueError("Item already in queue")
		self._indices.add(item.node_index)
		return self._queue.put(item)
	
	@property
	def indices(self)->Iterable[int]:
		return self._indices
	
	@property
	def empty(self) -> bool:
		if len(self._indices)==0:
			assert len(self._queue.queue)==0
			return True
		else:
			assert len(self._queue.queue)!=0
			return False
	
	def __contains__(self, item:int|_PermutationLatticeWalkItem) -> bool:
		if isinstance(item,_PermutationLatticeWalkItem):
			index = item.node_index
		elif isinstance(item, int):
			index = item
		else:
			raise TypeError("Not a supported type")
		return index in self._indices
	
	def _ordered_indices(self) -> list[int]:
		return [item.node_index for item in self._queue.queue]
	
	def __repr__(self) -> str:
		return f"{self._ordered_indices()}"
		
	def __str__(self) -> str:
		return f"_PermutationQueue with {self._queue.qsize()} item: "+self.__repr__()
	

def _lattice_walk_add_movie_frame(ax, lattice:list[Node], visited_nodes:set[int], lattice_walk_queue:_PermutationsQueue) -> tuple:
	from matplotlib import pyplot as plt	
	from utils import visuals

	if ax is None:
		ax = plt.subplot(1,1,1)
	elif isinstance(ax, plt.Axes):
		ax.cla()
	fig = ax.figure


	## Basic data:
	N = linear_size_from_total_vertices(len(lattice))

	## Plot lattice:
	plot_lattice(
		lattice, periodic=True,
		node_color="black",
		node_size=20,
		edge_style="y-",
		with_edge_names=False 
	)

	def _plot_node(ind:int, color:str) -> None:
		i, j = get_vertex_coordinates(ind, N)
		x, y = get_node_position(i, j, N)
		plt.scatter(x, y, s=300, c=color, zorder=2)

	## Add frontier:
	for i in lattice_walk_queue.indices:
		_plot_node(i, "blue")

	## Add visited nodes:
	for i in visited_nodes:
		_plot_node(i, "red")

	return fig, ax


def all_periodic_lattice_shifting_permutation(N:int, _movie:bool=False, _print:bool=False) -> Generator[list[int], None, None]:
	"""Get all permutation of the triangular lattice that shift nodes altogether in a periodic manner

	Args:
		N (int): Linear size of lattice

	This is achieved with a breadth-first graph walk of the entire lattice.
	To track that we really explored the entire graph, we imagine that the center node is where we start,
	and each neighboring node that we arrived to, we shift the graph periodically in its direction.
	We keep a set with all the nodes that we visited, to make sure nothing is visited twice.
	"""	
	## Visuals:
	if _movie:
		from utils.visuals import VideoRecorder, draw_now
		from matplotlib import pyplot as plt 
		movie = VideoRecorder()
		ax = plt.subplot(1,1,1)
		draw_now()

	## Helper functions and types:
	visited_nodes : set[int] = set()
	lattice_walk_queue  = _PermutationsQueue()
	periodic_lattice, edges, null_permutation = _common_periodic_shifting_inputs(N)

	## Start with the current position (The NULL permutation)
	crnt = _PermutationLatticeWalkItem(node_index=get_center_vertex_index(N), permutation=null_permutation) 
	lattice_walk_queue.put(crnt)

	## For each node in waiting list
	while not lattice_walk_queue.empty:
		if _print:
			print(lattice_walk_queue)
		## Get item and continue if already visited here
		crnt = lattice_walk_queue.get()
		if crnt.node_index in visited_nodes:
			continue

		## Mark node as visited and send permutation:
		visited_nodes.add(crnt.node_index)
		yield crnt.permutation

		# plt.subplot(1,1,1)
		# _plot_shifting_lattice(periodic_lattice, crnt.permutation)

		## For each direction, yield the new permutation:
		for direction in LatticeDirection.all_in_counter_clockwise_order():
			## look for next node:
			try:
				next_index = get_neighbor_by_index_and_direction(crnt.node_index, direction, N)
			except LatticeError:
				# No need to explore beyond the block:
				continue

			## Avoid from computing shifts for places we've already visited or are queued:
			if next_index in lattice_walk_queue or next_index in visited_nodes:
				continue

			## get permutation based on the permutation that brough us here:
			next_permutation = _shift_periodically_in_direction_given_periodic_lattice(
				periodic_lattice, edges, crnt.permutation, direction
			)
			next = _PermutationLatticeWalkItem(node_index=next_index, permutation=next_permutation)
			lattice_walk_queue.put(next)

		## frame for movie:
		if _movie:
			fig, ax = _lattice_walk_add_movie_frame(ax, periodic_lattice, visited_nodes, lattice_walk_queue)
			movie.capture(fig)
			draw_now()

	## Write Movie
	if _movie:
		movie.write_video(f"Lattice-Walk N={N}")


def _plot_shifting_lattice(lattice:list[Node], permutation_list:list[int]) -> None:
	from utils import visuals        
	from matplotlib import pyplot as plt
	not_all_the_way = 0.9
	positions = [node.pos for node in lattice]

	plot_lattice(
		lattice, periodic=True,
		node_color="black",
		node_size=20,
		edge_style="k--",
		with_edge_names=False 
	)

	colors = visuals.color_gradient(len(permutation_list))
	for (prev_i, next_i), color in zip(enumerate(permutation_list), colors, strict=True):
		x1, y1 = positions[prev_i]
		x2, y2 = positions[next_i]
		plt.text(x1, y1, f"{prev_i}")
		plt.arrow(
			x1, y1, (x2-x1)*not_all_the_way, (y2-y1)*not_all_the_way, 
			width=0.05,
			color=color,
			zorder=0
		)
	print("Done plotting shifting")
