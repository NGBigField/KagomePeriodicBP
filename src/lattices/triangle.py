from lattices._common import NodePlaceHolder, LatticeError, OutsideLatticeError
from tensor_networks.directions import DL, DR, R, UR, UL, L, Direction, U, D 
from tensor_networks.directions import lattice_directions_in_standard_order
from typing import Generator


class TriangularLatticeError(LatticeError): ...


def total_vertices(N):
	"""
	Returns the total number of vertices in the *bulk* of a hex 
	TN with linear parameter N
	"""
	return 3*N*N - 3*N + 1


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

def _get_neighbor_coordinates_in_direction_no_boundary_check(i:int, j:int, direction:Direction, N:int)->tuple[int, int]:
	## Simple L or R:
	if direction==L:  
		return i, j-1
	if direction==R:  
		return i, j+1

	## Row dependant:
	middle_row_index = num_rows(N)//2   # above or below middle row

	if direction==UR: 
		if i <= middle_row_index: 
			return i-1, j
		else: 
			return i-1, j+1
	
	if direction==UL:
		if i <= middle_row_index:
			return i-1, j-1
		else:
			return i-1, j
		
	if direction==DL:
		if i < middle_row_index:
			return i+1, j
		else:
			return i+1, j-1
		
	if direction==DR:
		if i < middle_row_index:
			return i+1, j+1
		else:
			return i+1, j
		
	TriangularLatticeError(f"Impossible direction {direction!r}")


def check_boundary_vertex(index:int, N)->list[Direction]:
	on_boundaries = []

	# Basic Info:
	i, j = get_vertex_coordinates(index, N)
	height = num_rows(N)
	width = row_width(i,N)
	middle_row_index = height//2

	# Boundaries:
	if i==0:
		on_boundaries.append(U)
	if i==height-1:
		on_boundaries.append(D)
	if j==0: 
		if i<=middle_row_index:
			on_boundaries.append(UL)
		if i>=middle_row_index:
			on_boundaries.append(DL)
	if j == width-1:
		if i<=middle_row_index:
			on_boundaries.append(UR)
		if i>=middle_row_index:
			on_boundaries.append(DR)
	
	return on_boundaries



def get_neighbor_coordinates_in_direction(i:int, j:int, direction:Direction, N:int)->tuple[int, int]:
	i2, j2 = _get_neighbor_coordinates_in_direction_no_boundary_check(i, j, direction, N)

	if i2<0 or i2>=num_rows(N):
		raise OutsideLatticeError()
	
	if j2<0 or j2>=row_width(i2, N):
		raise OutsideLatticeError()
	
	return i2, j2


def get_neighbor(i:int, j:int, direction:Direction, N:int)->tuple[int, int]:	
	i2, j2 = get_neighbor_coordinates_in_direction(i, j, direction, N)
	return get_vertex_index(i2, j2, N)


def all_neighbors(index:int, N:int)->Generator[tuple[NodePlaceHolder, Direction], None, None]:
	i, j = get_vertex_coordinates(index, N)
	for direction in lattice_directions_in_standard_order():
		try: 
			neighbor = get_neighbor(i, j, direction, N)
		except OutsideLatticeError:
			continue
		yield neighbor, direction


def get_vertex_coordinates(index, N)->tuple[int, int]:
	running_index = 0 
	for i in range(num_rows(N)):
		width = row_width(i, N)
		if index < running_index + width:
			j = index - running_index
			return i, j
		running_index += width
	raise TriangularLatticeError("Not found")


def get_vertex_index(i,j,N):
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
		

def get_node_position(i,j,N):
	w = row_width(i, N)
	x = N - w + 2*j	
	y = 2*N - i
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
			

    

def create_triangle_lattice(N)->list[NodePlaceHolder]:

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
	for i in range(2*N-1):
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
	# Create the list of nodes:
	#
	index = 0
	nodes_list = []
	for i in range(2*N-1):
		w = row_width(i,N)
		for j in range(w):
			n = NodePlaceHolder(
				index = index,
				pos = get_node_position(i, j, N),
				edges = edges_list[index],
				directions=[L, R, UL, UR, DL, DR]
			)
			nodes_list.append(n)
			index += 1


	return nodes_list



