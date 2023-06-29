
## Numpy imports:
import numpy as np
from numpy.linalg import norm
from numpy import pi, conj, array, tensordot

## Common packages
from lib import bmpslib
from lib.bubblecon import bubblecon
# from bubblecon import bubblecon

from lib.ITE import fuse_tensor, hermitize_a_message, g_from_exp_h, \
	apply_2local_gate, rho_ij, invert_permutation, \
	get_a_random_mps_message, open_mps_env

from lib.bmpslib import mps_inner_product, mps

import time


################
## Constants:
#################

MESSAGES_SIDE_ORDER = ['D', 'R', 'U', 'L']

Rang, Uang, Lang, Dang = 0, 0.5*pi, pi, 1.5*pi

UR_ang, UL_ang, DL_ang, DR_ang = 0.25*pi, 0.75*pi, 1.25*pi, 1.75*pi

MPS_LEG_ANGLES = dict(
	D=[Lang, Uang, Rang],
	R=[Dang, Lang, Uang],
	U=[Rang, Dang, Lang],
	L=[Uang, Rang, Dang]
)

PRINT_SPACE = "            "





def get_core_indices(N:int, n:int)->list[int]:
	
	"""
	
	Given a "big" square lattice of size NxN and a small square lattice
	of size nxn in its center, give a list of the indices of the 
	vertices of the small lattice within the big lattice.
	
	This is used when calculating the small square TN. There, we need 
	to swallow the tensors of the big square without the ones of the
	small square.
	
	Input Parameters:
	-----------------
	
	N, n --- the linear size of the big/small square lattices
	
	
	Output:
	---------
	
	A list of the indices of the small square
	
	
	"""
	core_indices = []
	for i in range(n):
		for j in range(n):
			core_indices.append( \
				_get_index_from_coordinates(i+(N-n)//2, j+(N-n)//2, N))
				
	return core_indices
	



def create_random_square_PEPS(N, D=1, d=2):

	"""

	Internal Tensor structure:
	===========================

	The structure of every tensor in the list is:

	T[d, D_L, D_R, D_U, D_D, D_R]

	With the following structure

                          (3)U
                            |
                            |
                            |
                 (1)L  -----o----- R(2)
                            |
                            |
                            |
                          (4)D


  Indexing of tensors in the grid:
  ================================

  The tensors are ordered left->right, up->down. For example, an N=3
  system will look like:

        |     |     |
     -- 0 --- 1 --- 2 --
        |     |     |
     -- 3 --- 4 --- 5 --
        |     |     |
     -- 6 --- 7 --- 8 --
        |     |     |

  Labeling of edges:
  ==================

  (*) Every internal edge between Tensor i and Tensor j is labeled by
      '{i}-{j}', assuming i<j.

  (*) As shown in the N=3 example below, external edges are labeled as:

       L0, L1, L2, ...
       R0, R1, R2, ...
       U0, U1, U2, ...
       D0, D1, D2, ...

       U0     U1    U2
        |     |     |
  L0 -- o --- o --- o -- R0
        |     |     |
  L1 -- o --- o --- o -- R1
        |     |     |
  L2 -- o --- o --- o -- R2
        |     |     |
        D0    D1    D2



	"""

	NT = N*N # total vertices


	#
	# ----------------------------------------------------------------
	#                 I. CREATE THE BULK TENSORS
	# ----------------------------------------------------------------
	#

	#
	# Create the list of random tensors and their double-edge tensors
	#

	PEPS_tensors_list = []

	for i in range(N):
		for j in range(N):


			T = np.random.uniform(size=[d]+[D]*4) \
				+ 1j*np.random.normal(size=[d]+[D]*4)

			T = T/norm(T)

			PEPS_tensors_list.append(T)


	#
	# Create the list of edges.
	#

	edges_list = []
	for i in range(N):
		for j in range(N):
			eL = get_edge_name(i, j, "L", N)
			eR = get_edge_name(i, j, "R", N)
			eU = get_edge_name(i, j, "U", N)
			eD = get_edge_name(i, j, "D", N)
			edges_list.append([eL, eR, eU, eD])

	#
	# Create the list of angles
	#
	angles_list = [ [Lang, Rang, Uang, Dang] for _ in range(NT)]

	return PEPS_tensors_list, edges_list, angles_list




def get_smallhex_2env(T_list, e_list, angles_list, env_names, \
	bmps_Chi
)  -> tuple[
	dict[str, np.ndarray],
	dict[str, np.ndarray]
]:

	log = True

	opt_method='high'

	is_ket = [True]*4 + [False]*8

	envs_list = {}

	if 'h1' in env_names:

		#
		# h1 is the env of (0,0)--(0,1)
		#

		sw_down = [11,4,5,6,3,2]
		mps_down = bubblecon(T_list, e_list, angles_list, \
				bubble_angle=0.0, swallow_order=sw_down, \
				D_trunc=bmps_Chi, ket_tensors=is_ket)

		#
		# Trim it and swallow the edge tensors in their neighboring tensors
		#
		bmpslib.trim_mps(mps_down)

		mps_down.A[1] = tensordot(mps_down.A[0],mps_down.A[1], \
			axes=([1],[0]))

		mps_down.A[2] = tensordot(mps_down.A[2], mps_down.A[3], \
			axes=([2],[0]))


		mps_env = [T_list[9], T_list[10], mps_down.A[1], mps_down.A[2], \
			T_list[7], T_list[8]]

		envs_list['h1'] = mps_env



	if 'h2' in env_names:

		#
		# h2 is the env of (1,0)--(1,1)
		#

		sw_up = [7,8,9,10,0,1]
		mps_up = bubblecon(T_list, e_list, angles_list, \
				bubble_angle=pi, swallow_order=sw_up, \
				D_trunc=bmps_Chi, ket_tensors=is_ket)

		#
		# Trim it and swallow the edge tensors in their neighboring tensors
		#
		bmpslib.trim_mps(mps_up)

		mps_up.A[1] = tensordot(mps_up.A[0],mps_up.A[1], \
			axes=([1],[0]))

		mps_up.A[2] = tensordot(mps_up.A[2], mps_up.A[3], \
			axes=([2],[0]))


		mps_env = [mps_up.A[2], T_list[11], T_list[4], T_list[5], T_list[6],
			mps_up.A[1]]

		envs_list['h2'] = mps_env


	if 'v1' in env_names:

		#
		# v1 is the env of (0,0)--(1,0)
		#

		sw_right = [5,6,7,8,1,3]
		mps_right = bubblecon(T_list, e_list, angles_list, \
				bubble_angle=pi/2, swallow_order=sw_right, \
				D_trunc=bmps_Chi, ket_tensors=is_ket)

		#
		# Trim it and swallow the edge tensors in their neighboring tensors
		#
		bmpslib.trim_mps(mps_right)

		mps_right.A[1] = tensordot(mps_right.A[0],mps_right.A[1], \
			axes=([1],[0]))

		mps_right.A[2] = tensordot(mps_right.A[2], mps_right.A[3], \
			axes=([2],[0]))


		mps_env = [mps_right.A[2], T_list[9], T_list[10], T_list[11], \
			T_list[4], mps_right.A[1]]

		envs_list['v1'] = mps_env


	if 'v2' in env_names:

		#
		# v2 is the env of (0,1)--(1,1)
		#

		sw_left = [9, 10, 11, 4, 2, 0]
		mps_left = bubblecon(T_list, e_list, angles_list, \
				bubble_angle=3*pi/2, swallow_order=sw_left, \
				D_trunc=bmps_Chi, ket_tensors=is_ket)

		#
		# Trim it and swallow the edge tensors in their neighboring tensors
		#
		bmpslib.trim_mps(mps_left)

		mps_left.A[1] = tensordot(mps_left.A[0],mps_left.A[1], \
			axes=([1],[0]))

		mps_left.A[2] = tensordot(mps_left.A[2], mps_left.A[3], \
			axes=([2],[0]))


		mps_env = [T_list[7], T_list[8], mps_left.A[1], mps_left.A[2], \
			T_list[5], T_list[6]]

		envs_list['v2'] = mps_env


	#
	# We now "open" all MPS envs. This means that we unfuse all the
	# ket-bra mid legs of the  MPS tensors into a ket leg and a bra leg.
	# Overall, every tensor in the MPS env will now have 4 legs:
	# [D_L, D_ket, D_bra, D_R]
	#
	envs_list_open = {}
	for env in env_names:
		envs_list_open[env] = open_mps_env( envs_list[env].copy() )

	return envs_list, envs_list_open





def two_way_contraction_to_core(
	N, n,
	full_T_list, full_e_list, full_angles_list,
	bmps_chi, opt_method,
	mpi_comm=None
):
	
	NT = N*N # Total number of tensors in the large square lattice
	half_n = n//2

	#
	# II. Contract the big square TN from up->middle and down->middle.
	#     The two resultant MPSs are then contracted on the sides to
	#     create the small square TN env.
	#

	#
	# Find the list of indices of core tensors. They are going to be
	# substracted from the up->down and down->up contraction orders
	#

	core_indices = get_core_indices(N, n)

	half_core_size = n*n//2 # we assume that n is even

	#
	# Get upper indices for contraction. We use the contraction order
	# for creating the down message, which goes Up -> Bottom, and take
	# its first half
	#
	# It is made of the N tensors of the up MPS, and the N/2 lines of
	# the bulk, where each line is made of N bulk tensors + 2 MPS tensors
	# from both sides.
	#

	# from testsing import plot_network
	# plot_network(full_T_list, full_e_list, full_angles_list, N=N, verbose=False)


	sw_full, sw_angle = get_cont_order(N, 'D', n)

	sw_U_half = sw_full[0:(N + (N//2)*(N+2))]

	#
	# Remove the upper half core indices from the swallowing list
	#
	for core_index in core_indices[:half_core_size] :
		sw_U_half.remove(core_index)

	#
	# Similarly, get the contraction order for the lower-half MPS
	# using the Bottom->Up contraction order of the Up message
	#

	sw_full, sw_angle = get_cont_order(N, 'U', n)
	sw_D_half = sw_full[0:(N + N//2 * (N+2))]

	#
	# Remove the lower half core indices:
	#

	for core_index in core_indices[half_core_size:] :
		sw_D_half.remove(core_index)

	#
	# When contracting the full square TN, the expanded core tensors are
	# kets while the boundary MPSs are contracted ket-bra:
	#
	is_ket = [True]*NT + [False]*4*N

	if mpi_comm is None:
		#
		# Calculate two MPSs in SERIAL mode
		# ----------------------------------------------------
		#


		upper_mps = bubblecon(full_T_list, full_e_list, full_angles_list, \
			bubble_angle=pi/2, swallow_order=sw_U_half, D_trunc=bmps_chi, \
			opt=opt_method, ket_tensors=is_ket)


		lower_mps = bubblecon(full_T_list, full_e_list, full_angles_list, \
			bubble_angle=3*pi/2, swallow_order=sw_D_half, D_trunc=bmps_chi, \
			opt=opt_method, ket_tensors=is_ket)

	else:
		raise NotImplementedError("No MPI mode yet")


	#
	# III. Contract the upper/lower MPS to create an edge tensor from the
	#      left and from the right, creating LTensor, RTensor.
	#

	LTensor=None

	jD = (N-n)//2 + 1
	for j in range(0, jD):
		LTensor = bmpslib.updateCLeft(LTensor,
			upper_mps.A[-j-1].transpose([2,1,0]), \
			lower_mps.A[j])

	#
	# LTensor legs are [i_up, i_down]. Absorb it in the first tensor of
	# mps_D_half that going to appear in the small square TN
	#

	lower_mps.A[jD] = tensordot(LTensor, lower_mps.A[jD], axes=([1],[0]))


	#
	# Now create RTensor, which is the contraction of the right part
	# of the up/down MPSs
	#

	RTensor=None

	jU = (N-n)//2 + 1
	for j in range(0, jU):
		RTensor = bmpslib.updateCRight(RTensor,
			upper_mps.A[j].transpose([2,1,0]), \
			lower_mps.A[-1-j])
	#
	# Absorb the RTensor[i_up, i_down] in the first tensor of mps_up_half
	#
	upper_mps.A[jU] = tensordot(RTensor, upper_mps.A[jU], axes=([0],[0]))


	#
	# IV. We now have all tensors we need to define the small TN.
	#     It consists of the original TN in the small square + the MPS
	#     tensors that surround it.
	#


	#
	# first create the skeleton of the TN using the random PEPS function.
	#

	small_T_list, small_e_list, small_angles_list = \
		create_random_square_PEPS(n, 1, 1)

	# plot_network(small_T_list, small_e_list, small_angles_list, N=n)

	#
	# Now replace the random tensors of the small TN with the tensor from
	# the full TN
	#

	i0_full, j0_full = (N-n)//2, (N-n)//2

	for i in range(n):
		for j in range(n):

			# the (i,j) in the large hexagon are given by
			# i+N-n, j+N-n

			ind_large = _get_index_from_coordinates(i0_full+i, j0_full+j, N)
			ind_small = _get_index_from_coordinates(i,j, n)

			small_T_list[ind_small] = full_T_list[ind_large]

	#
	# Add the surrounding MPS tensors in the edge order D, R, U, L
	#

	env_tensors = lower_mps.A[(jD+n//2):(jD+2*n)] \
		+ upper_mps.A[jU:(jU+2*n)] + lower_mps.A[jD:(jD+n//2)]


	small_T_list += env_tensors

	#
	# Add the edges labels of the MPS tensors
	#

	i_mps = 0  # A running index for the MPS edges

	for side in MESSAGES_SIDE_ORDER:
		for ell in range(n):

			#
			# Calculate the edge labels
			#

			ij, i, j = _get_message_neighbour_in_lattice(ell, side, n)
			bulk_edge_name = get_edge_name(i, j, side, n)

			i_mps_next = (i_mps + 1) % (4*n)
			es = [f'env{i_mps}', bulk_edge_name, f'env{i_mps_next}']

			i_mps += 1

			#
			# Calculate the edge angles
			#

			angles = MPS_LEG_ANGLES[side]

			#
			# Add the edge & angles to the TN lists
			#
			small_e_list.append(es)
			small_angles_list.append(angles)


	return small_T_list, small_e_list, small_angles_list



#
# --------------------- calc_small_square_TN  ------------------------
#

def calc_small_square_TN(N, T_list, e_list, angles_list,
	mps_list, bmps_chi, mpi_comm, n=2,
	extra_config:dict=None
):

	"""

	Takes an NxN square TN + its 4 MPS messages and calculates the TN
	of a small nxn square in the middle of the big square.

	We use this "small TN" to calculate the local environments of the
	edges in the unit cell in order perform optimal truncation in the
	full-update procedure.

	The small TN is made of the nxn tensors in the middle of the full TN,
	together with a periodic MPS that forms the environment. The perioidic
	MPS is made of edges ordered by D->R->U->L, where the first tensor
	in the env is the left-most tensor of the D edge.

	All in all the small TN is made of nxn ket bulk tensors + 4*n ket-bra
	tensors that make up the periodic MPS env.

	In order to calculate the periodic MPS, we contract the "big TN" from
	Up->Bottom to the middle of the TN, and then from Bottom->Up up to the
	middle. In both cases, we do not cotract the tensors of the small-TN.
	The periodic MPS env is then calculated from these two MPSs.

	Input Parameters:
	------------------
	N --- Linear size of the big square TN

	T_list, e_list, angles_list --- TN parameters of the big square TN

	mps_list --- List of the incoming MPS messages (usually come from
	             blockBP), ordered by [D, R, U, L]

	bmps_chi --- The truncation Chi used in bubblecon to calculate the
	             small TN

	mps_comm --- An MPI object if we want to run it in MPI mode

	n --- Linear size of the small TN



	Output:
	----------
	A taple describing the small-TN:

  (small_T_list, small_e_list, small_angles_list)



	"""

	opt_method='high'  # bubblecon optimization level

	#
	# I. Add the boundary MPSs to the large tensor-network, which is
	#    then contracted to find the MPS boundary of the small hexagon
	#

	full_T_list, full_e_list, full_angles_list = add_messages_to_square(
		T_list, e_list, angles_list, mps_list)

	if isinstance(extra_config, dict) and "two_ways_contraction" in extra_config and extra_config["two_ways_contraction"] is True :
		return _four_way_contraction_to_core(
			N, n,
			full_T_list, full_e_list, full_angles_list,
			bmps_chi, opt_method,
			mpi_comm
		)
	else:
		return two_way_contraction_to_core(
			N, n,
			full_T_list, full_e_list, full_angles_list,
			bmps_chi, opt_method,
			mpi_comm
		)




def _get_message_neighbour_in_lattice(ind, side:str, N):

	"""

	Given the index of an MPS message tensor, return the index and
	(i,j) coordinates of its neighboring bulk tensor.

	The MPS tensors are ordered anti-clockwise, starting from D->R->U->L
	so that index 0 in the D MPS means the bottom left-most tensor.

	Input Parameters:
	-----------------
	ind --- the index of the MPS tensor (0, 1, ..., N-1)
	side --- which lattice side ('D', 'R', 'U', L')


	Output:
	--------
	The index and (i,j) of the neighboring tensor in the bulk.


	"""


	if side == "U":
			i, j = 0, N-ind-1

	elif side == "D":
		i, j = N-1, ind

	elif side == "L":
			i, j = ind, 0

	else:
		i, j = N-ind-1, N-1

	i_neighbour = _get_index_from_coordinates(i,j, N)

	return i_neighbour, i, j



#
# -------------------   _get_coordinates_from_index  -------------------
#
def _get_coordinates_from_index(
	ind:int,  # index of node
	N:int   # linear size of lattice
) -> tuple[int, int]:

	"""

	Given an index of a tensor in the bulk, return its (i,j) coordinates.

	The relation between the index and the (i,j) coordinates is:

	ind = i*N + j

	"""

	quotient, remainder = divmod(ind, N)

	return quotient, remainder


#
# -------------------  _get_index_from_coordinates-  -------------------
#
def _get_index_from_coordinates(i:int,  # row
	j:int,  # column
	N:int   # linear size of lattice
	) -> int:

	"""

	Given the (i,j) coordinates of a tensor, return its index:

	ind = i*N + j

	"""

	return i*N + j


def _get_neighbor_from_coordinates(
	i:int,  # row
	j:int,  # column
	side:str,  # ['L', 'R', 'U', 'D']
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




#
# -------------------  get_edge_name  -------------------
#

def get_edge_name(
	i:int,
	j:int,
	side:str,  # ['L', 'R', 'U', 'D']
	N:int
)->str:
	"""

	Given a (i,j) coordinate of a bulk tensor, together with a side
	(L,R,U,D), get the label of its edge on that side.

	External edges are of the form
       L0, L1, L2, ...
       R0, R1, R2, ...
       U0, U1, U2, ...
       D0, D1, D2, ...

  Internal edges between tensors i<j are of the form {i}-{j}.


	Input Parameters:
	------------------
	i,j --- location of the vertex to which the edge belong.
	        i=row, j=column. i=0 ==> upper row.

	side --- The side of the edge. Either 'L', 'R', 'U', 'D'

	N    --- Linear size of the lattice
	"""

	#
	# See first if its an external (boundary) edge
	#
	if side=='L' and j==0:
		return f"L{i}"

	if side=='R' and j==N-1:
		return f"R{i}"

	if side=='D' and i==N-1:
		return f"D{j}"

	if side=='U' and i==0:
		return f"U{j}"

	#
	# So its an internal edge
	#

	this = _get_index_from_coordinates(i, j, N)
	neighbor = _get_neighbor_from_coordinates(i, j, side, N)

	return f"{min(this,neighbor)}-{max(this,neighbor)}"





#
# ----------------------- get_initial_mps_messages --------------------
#

def get_initial_mps_messages(N, D, mode='UQ'):

	"""

	Set initial MPS messages for the square lattice. Messages can be 
	either random PSD product states (mode='RQ') or a contraction of the 
	ket-bra (mode='UQ')

	Paramters:
	-----------

	N    --- Linear size of square lattice
	D    --- bond dimension
	mode --- either 'UQ' or 'RQ'

	OUTPUT:
	-------

	mps_list - a list of 4 MPSs


	"""

	
	mps_list = []
	
	for side in range(4):

		mp = get_a_random_mps_message(N, D, mode)
		
		mps_list.append(mp)

	return mps_list



def rotate_ACW(T_ind_list, N):

	"""

	Given a list of indices of tensors (either bulk or MPS tensors),
	output a corresponding list of a system that is rotated in 90 deg
	anti-clockwise.

	This is useful when trying to turn a bubblecon swallowing order from
	one direction to another.


	Input Parameters:
	-------------------

	T_ind_list --- A list of the indices of the tensors
	N --- Linear size of the system


	Output:
	--------

	The list of the indices of tensors in the rotated system

	"""


	ACW_T_ind_list = []

	for ind in T_ind_list:

		if ind<N*N:
			#
			# So we're on an internal tensor.
			#
			# Perform anti-clockwise 90 deg rotation:
			#
			#            (i,j) ==> (N-j-1, i)
			#

			i,j = _get_coordinates_from_index(ind, N)
			ACW_ind = _get_index_from_coordinates(N-j-1, i, N)

			ACW_T_ind_list.append(ACW_ind)

		else:
			#
			# So we're an external MPS tensor. Essentially, just add N to it
			# and if we're pass the last MPS tensor, substract 4N
			#

			ACW_ind = (ind-N*N + N) % (4*N) + N*N

			ACW_T_ind_list.append(ACW_ind)

	return ACW_T_ind_list



def get_cont_order(N, mode, n=2)->tuple[list[int], float]:

	"""

	Get a contraction order of the square TN, either for the creation
	of a BP message, or for the creation of the small square net.

	Input Parameters:
	------------------

	N --- Linear size of the square
	mode --- Either 'D', 'R', 'U', 'L' for the creation of an *outgoing*
	         Down, Right, Up, Left message, or 'half-U', 'half-D'
	         for the upper/lower halfs used for calculating the
	         environment of the small-square TN.

	n = None --- the linear size of the small TN


	Output:
	--------
	The desired contraction order.

	"""


	#
	# ================================================================
	#  First calculate the swallowing order for calculating the
	#  Upper outgoing MPS message. We go like a snake from bottom-left
	#  upward.
	#
	#  Once we have this swallowing order, we calculate the rest of
	#  the contraction orders by rotating the system using the
	#  rotate_ACW() function.
	# =================================================================

	U_cont_order = list(range(N*N, N*N+N)) # lower MPS

	#
	# Lowest row above bottom MPS
	#
	U_cont_order = U_cont_order + [N*N-1, N*N+N] + \
		list(range(N*N-2, N*(N-1)-1, -1)) + [N*N+4*N-1]


	#
	# Add the rest of the rows N-2 ==> 0 like a snake
	#

	direction = 'LR'

	for i in range(N-2, -1, -1):

		row_LR_order = [N*N+3*N + i] + list(range(i*N, i*N+N)) \
				+ [N*N + N + N-1-i]


		if direction=='LR':

			#
			# Go from Left to Right
			#

			U_cont_order += row_LR_order

			direction = 'RL'

		else:
			#
			# Go from Right to Left
			#

			row_LR_order.reverse()
			U_cont_order += row_LR_order

			direction = 'LR'


	sw_order = U_cont_order

	if mode == 'U':
		return sw_order, pi/2

	if mode == 'L':
		sw_order = rotate_ACW(sw_order, N)
		return sw_order, pi


	if mode == 'D':
		sw_order = rotate_ACW(sw_order, N)
		sw_order = rotate_ACW(sw_order, N)
		return sw_order, 3*pi/2


	if mode == 'R':
		sw_order = rotate_ACW(sw_order, N)
		sw_order = rotate_ACW(sw_order, N)
		sw_order = rotate_ACW(sw_order, N)
		return sw_order, 0.0
	
	raise ValueError(f"mode={mode!r}")


def add_messages_to_square(T_list, e_list, angles_list, \
	mps_list):

	"""

	Given a square TN and a set of 4 incoming MPS messages, create one
	big TN

	The MPS internal edges are labeled by

	   'bd0', 'bd1', 'bd2', ...

	in a counter-clock fashion, starting from the left-most tensor of the
	bottom message.


	The mid legs, which connect to the hex TN are labeled according
	to the external legs labeling of the PEPS:

       L0, L1, L2, ...
       R0, R1, R2, ...
       U0, U1, U2, ...
       D0, D1, D2, ...


	Input Parameters:
	-----------------
	T_list, e_list, angles_list --- the square TN

	mps_list --- list of the 4 incoming MPS messages, ordered
	             counter-clockwise by: D, R, U, L

	OUTPUT:
	--------
	new_T_list, new_e_list, new_angles_list ---
	   the parameters of the fused TN

	"""


	#
	# Extract the linear size of the TN from the length of the MPS messages
	#


	N = mps_list[0].N


	#
	# Add the boundary MPSs to the large tensor-network, which is
	# then contracted to find the MPS boundary of the small hexagon
	#
	#
	# We're adding them in the following order:
	#
	#        D => R => U => L
	#
	#    The vertices are ordered in a clockwise order.
	#



	#
	# ell is the index of the internal edges of the MPS messages, which
	# are labed by 'bd0', 'bd1', ..., 'bd{4N-1}' in counterclock fashion.
	#
	ell = 0

	new_T_list = T_list.copy()
	new_e_list = e_list.copy()
	new_angles_list = angles_list.copy()

	#
	# Loop over the 4 sides. Each side has an MPS which we add
	# to the TN. Each MPS message has N tensors.
	#
	for k in range(4):

		side, mp = MESSAGES_SIDE_ORDER[k], mps_list[k]
		angles = MPS_LEG_ANGLES[side]

		for i_mps in range(N):

			#
			# Get coordinates of bulk tensor in the lattice:
			#
			i_neighbor, i, j = _get_message_neighbour_in_lattice(i_mps, side, N)
			bulk_edge_name = get_edge_name(i, j, side, N)
			assert bulk_edge_name in e_list[i_neighbor]

			#
			# Derive dimensions and edge-names:
			#
			A = mp.A[i_mps].copy()
			DL, Dmid, DR = A.shape[0], A.shape[1], A.shape[2]
			edges = [f"bd{ell-1}", bulk_edge_name, f"bd{ell}"]

			#
			# When we're on the edges, the tensors have only 2 legs.
			#
			if i_mps==0:
				A = A.reshape([Dmid, DR])
				my_angles = [angles[1], angles[2]]
				my_edges  = [edges[1] , edges[2]]

			elif i_mps==N-1:
				A = A.reshape([DL, Dmid])
				my_angles = [angles[0], angles[1]]
				my_edges  = [edges[0] , edges[1]]
			else:
				my_angles = [angles[0], angles[1], angles[2]]
				my_edges  = [edges[0] , edges[1] , edges[2] ]

			#
			# Add the tensor, its edges and agles to the TN lists.
			#
			new_T_list.append(A)
			new_e_list.append(my_edges)
			new_angles_list.append(my_angles)

			ell += 1

		ell -= 1

	return new_T_list, new_e_list, new_angles_list


#
# --------------------- square_blockBP  ------------------------
#

def square_blockBP(N, square_T_list, square_e_list, square_angles_list, \
	mps_list=None, D_trunc=None, max_iter=100, delta=1e-5, \
	max_random_trials=15,
	mpi_comm = None):

	"""

	Perform blockBP for a single-square block. Return the converged MPS
	messages

	Input Parameters:
	------------------

	N - Linear size of the square TN (it is a N x N square)

	square_T_list, square_e_list, square_angles_list --- the square TN

	mps_list --- possible initial MPS messages. The messages in the list
	             are ordered as [D, R, U, L]. If None is given,
	             then starting the BP iterations with random messages.

	D_trunc --- Possible truncation. If omitted D^2 is used, where D
	            is the bond dimension of the hex TN

	max_iter --- maximal number of iterations

	delta --- Convergence criteria for the BP iterations.

	max_random_trials --- If we don't converge on the first trial
	          then try max_random_trials runs with random initial MPS
	          messages.

	mpi_comm --- An MPI communication buffer when running in MPI


	OUTPUT:
	-------

	A list of the 4 converged MPS messages in the order [D,R,U,L]


	"""


	D = square_T_list[0].shape[1]

	if D_trunc is None:
		D_trunc = D*D

	trial = 0

	if mps_list is None:
		#
		# Create random initial conditions
		#

		mps_list = get_initial_mps_messages(N, D, 'RQ')

		trial = 1

		#
		# Make them canonical and normalized
		#

		for mp in mps_list:
			mp.left_canonical_QR()
			N1 = mp.N
			mp.set_site(mp.A[N1-1]/norm(mp.A[N1-1]), N1-1)


	#
	# set the is_ket flags. The square TN are ket tensors, whereas the
	# tensors of the MPS messages are ket-bra tensors
	#

	is_ket = [True]*(N*N) + [False]*(4*N)


	#
	# Create a list of contraction orders that corresponds to the
	# list of edges.
	#


	super_edges = ['D', 'R', 'U', 'L']

	cont_order = []
	for mode in super_edges:
		sw_order, sw_angle = get_cont_order(N, mode)
		cont_order.append( (sw_order, sw_angle) )





	#
	#    <<<<<<<<<<<<<    Main BP Loop    >>>>>>>>>>>>>>>
	#

	er0 = 0.1
	was_below_er0 = False

	converged=False

	while not converged and (trial <= max_random_trials):

		for t in range(max_iter):

			# print(f"square-blockBP: entering round {t}")

			#
			# Calculate out going messages
			#


			err = None
			out_mps_list = []



			if mpi_comm is None:

				#
				# ================= SERIAL blockBP ========================
				#

				#
				# Add the MPS messages to the square TN
				#

				T_list, e_list, angles_list = \
					add_messages_to_square(square_T_list, square_e_list, \
						square_angles_list, mps_list)



				#
				# Go over the edges, and for each edge calculate its out-going
				# MPS message. Put it in the out_mps_list.
				#
				for k in range(4):

					sw_order, sw_angle = cont_order[k]

					mp = bubblecon(T_list, e_list, angles_list,\
						bubble_angle=sw_angle, swallow_order=sw_order, D_trunc=D_trunc, \
						opt='high', ket_tensors=is_ket)

					mp = hermitize_a_message(mp)

					#
					# Make the MPS canonical and noramlize so that we can compare to
					# the previous iteration
					#
					mp.left_canonical_QR()
					N1 = mp.N
					mp.set_site(mp.A[N1-1]/norm(mp.A[N1-1]), N1-1)

					out_mps_list.append(mp)


			#
			# Create a copy of the old (incoming) MPS list, and map the
			# outgoing messages to the incoming messages list
			#

			old_mps_list = mps_list.copy()


			# 0   1  2  3
			# D,  R, U, L

			mps_list[0] = out_mps_list[2]   # D <-> U
			mps_list[1] = out_mps_list[3]   # R <-> L
			mps_list[2] = out_mps_list[0]   # U <-> D
			mps_list[3] = out_mps_list[1]   # L <-> R

			#
			# Calculate the average distance between the new MPS messsages
			# and the old MPS messages
			#

			err = 0.0
			for k in range(4):

				mp = mps_list[k]
				old_mp = old_mps_list[k]

				inP = mps_inner_product(mp, old_mp, conjB=True)

				err0 = 2-2*inP.real

				if err is None:
					err = err0
				else:
					err += err0

			err = err/4

			if err<er0:
				was_below_er0 = True


			print(PRINT_SPACE+f"Err({t}) = {err}")

			if err < delta:
				print(PRINT_SPACE+f"Err({t})<{delta}. Exiting blockBP")
				break

			if err>1.0 and was_below_er0:
				print(PRINT_SPACE+f"BAD blockBP CONVERGENCE: Err({t}) > 1.0 even though "\
					f"it was already < {er0}. Exiting blockBP")

				was_below_er0 = False

				break

		#
		#
		#

		converged = (err < delta)  #type: ignore

		if not converged and trial<=max_random_trials:
			#
			# use other inital mps messages
			#
			print(PRINT_SPACE+f"blockBP did not converge at trial {trial} !!!\n\n")

			if trial==0:
				mps_list = get_initial_mps_messages(N, D, 'UQ')
				print(PRINT_SPACE+"Trying again with uniform initial mps messages\n")
			else:
				mps_list = get_initial_mps_messages(N, D, 'RQ')
				print(PRINT_SPACE+"Trying again with random initial mps messages\n")

			#
			# Make them canonical and normalized
			#

			for mp in mps_list:
				mp.left_canonical_QR()
				N1 = mp.N
				mp.set_site(mp.A[N1-1]/norm(mp.A[N1-1]), N1-1)

			trial += 1



	if not converged:
		print("\n\n")
		print(f"blockBP did not converge after {max_random_trials}")
		print("Quitting!")
		exit(1)

	return mps_list


#
# ------------------------ robust_BP ----------------------------------
#

def robust_BP(N, T_list, e_list, angles_list,
	prev_BP_mps_messages, D_trunc, max_iter, delta, mpi_comm)->list[mps]:
		
	"""
	
	Calls the blockBP function square_blockBP in a robust way that
	captures linear algbera errors. If such errors occurs once --- 
	it is ignored and BP is re-started.
	
	Input Parameters: same as squareBP
  -----------------
  
  Output: the converged MPS BP messages
	-------
	
	"""
	
	try:

	
		BP_mps_messages = square_blockBP(N, T_list, e_list, angles_list, \
			mps_list=prev_BP_mps_messages, \
			D_trunc=D_trunc, max_iter=max_iter, delta=delta, \
			mpi_comm=mpi_comm)


	except np.linalg.LinAlgError as e:
		print("\n")
		print("* * * Linear-Algebra error in hex_blockBP * * *")
		print("    '{e}'")
		print("\n")
		print("==>  Trying again with initial random messages\n\n")

		BP_mps_messages = square_blockBP(N, T_list, e_list, angles_list, \
			mps_list=None, \
			D_trunc=D_trunc, max_iter=max_iter, delta=delta, \
			mpi_comm=mpi_comm)
			
	return BP_mps_messages



def edges_dict_from_edges_list(edges_list:list[list[str]])->dict[str, tuple[int, int]]:
    vertices = {}
    for i, i_edges in enumerate(edges_list):
        for e in i_edges:
            if e in vertices:
                (j1,j2) = vertices[e]
                vertices[e] = (i,j1)
            else:
                vertices[e] = (i,i)
    return vertices     



def plot_network(
    t_list : list[np.ndarray],
    e_list : list[list[str]], 
    a_list : list[list[float]],
    N=None,
    verbose : bool = True
)-> None:
	
    ## Special imports:
    from matplotlib import pyplot as plt
    plt.figure()
    plt.show(block=False)

    ## Constants:
    edge_color ,alpha ,linewidth = 'gray', 0.5, 3
    angle_color, angle_linewidth, angle_dis = 'green', 2, 0.2
    
    ## Derive basic data:
    if N is None:
        N = len(t_list)
    N2 = N*N
    num_items = len(t_list)
    assert len(t_list) == len(e_list) == len(a_list)

    if num_items>N2:
        has_messages = True
    else:
        has_messages = False


    ## Define helper functions:
    average = lambda lst: sum(lst) / len(lst)

    def _pos_from_index(ind)->tuple[int, int]:
        if ind<N2:
            i, j = _get_coordinates_from_index(ind, N)
            x = j
            y = N - i - 1
            return x, y
        # Message tensor
        i_mps = ind-N2
        frac, i_mps = divmod(i_mps, N)
        side = MESSAGES_SIDE_ORDER[frac]
        i_neigbor, _, _ = _get_message_neighbour_in_lattice(i_mps, side, N)
        neighbor_pos = _pos_from_index(i_neigbor)
        if side == "U":	return (neighbor_pos[0]  , neighbor_pos[1]+1)
        if side == "D":	return (neighbor_pos[0]  , neighbor_pos[1]-1)
        if side == "L":	return (neighbor_pos[0]-1, neighbor_pos[1]  )
        if side == "R":	return (neighbor_pos[0]+1, neighbor_pos[1]  )
        raise ValueError("Not an option")

    def _edge_dim(edge_name:str) -> int:
        tensors_indices = e_dict[edge_name]
        dims = []
        for i_tensor in tensors_indices:
            edges = e_list[i_tensor]
            leg_index = edges.index(edge_name)
            tensor_shape = t_list[i_tensor].shape
            if len(tensor_shape)==5:
                # this is a peps
                dim = tensor_shape[leg_index+1]
                dim *= dim
            elif len(tensor_shape)==4:
                # this is a fused-peps
                dim = tensor_shape[leg_index]
            else:
                # this is an mps
                if not has_messages:
                    raise ValueError("This is impossible")
                dim = tensor_shape[leg_index]
            dims.append(dim)

        d1, d2 = dims[0], dims[1]
        assert d1==d2
        return d1


    def _check_on_boundary(edge_name:str)->str|None:
        # if has_messages:
        # 	return None
        tensors_indices = e_dict[edge_name]
        if tensors_indices[0]!=tensors_indices[1]:
            return None		
        i, j = _get_coordinates_from_index(tensors_indices[0], N)
        if j==0		and "L" in edge_name: 	return "L"
        if j==N-1	and "R" in edge_name: 	return "R"
        if i==N-1	and "D" in edge_name: 	return "D"
        if i==0		and "U" in edge_name:	return "U"

        raise ValueError("Not an option")


    all_edges = []
    for edges in e_list:
        for edge in edges:
            if edge not in e_list:
                all_edges.append(edge)

    core_indices = get_core_indices(N, n=2)


    e_dict = edges_dict_from_edges_list(e_list)
    pos_list = [_pos_from_index(i) for i in range(num_items)]

    def _edge_positions(edge_name:str) -> tuple[list[int], list[int]]:
        tensors_indices = e_dict[edge_name]

        on_boundary = _check_on_boundary(edge_name)
        if on_boundary is None:
            x_vec = [pos_list[tensor_ind][0] for tensor_ind in tensors_indices]
            y_vec = [pos_list[tensor_ind][1] for tensor_ind in tensors_indices]
            return x_vec, y_vec

        x, y = pos_list[tensors_indices[0]]
        delta = 1
        if on_boundary == "L":
            x_vec = [x, x-delta]
            y_vec = [y, y]
        elif on_boundary == "R":
            x_vec = [x, x+delta]
            y_vec = [y, y]
        elif on_boundary == "U":
            x_vec= [x, x]
            y_vec= [y, y+delta]
        elif on_boundary == "D":
            x_vec = [x, x]
            y_vec = [y, y-delta]
        else: 
            raise ValueError(f"Not a valid option dir={dir}")    

        return x_vec, y_vec
        
        
        
    # Plot nodes:
    for i, pos in enumerate(pos_list):
        node_name = f"{i}"
        x, y = pos
        if i in core_indices:
            node_color = 'blue'
        else:
            node_color = 'red'
        plt.scatter(x, y, c=node_color)
        text = f"{node_name}"
        plt.text(x, y, text)


    # Plot edges:
    for edge_name in all_edges:

        ## Gather info:
        x_vec, y_vec = _edge_positions(edge_name)
        edge_dim = _edge_dim(edge_name)      
        
        plt.plot(x_vec, y_vec, color=edge_color, alpha=alpha, linewidth=linewidth )
        if verbose:
            plt.text(
                average(x_vec), average(y_vec), f"{edge_name}:\n{edge_dim}",
                fontdict={'color':'darkorchid', 'size':10 }
            )			

    # Plot angles of all edges per node:
    for i_node, (angles, origin, edges) in enumerate(zip(a_list, pos_list, e_list, strict=True)):
        assert len(angles)==len(edges)
        if not verbose:
            continue
        for i_edge, angle in enumerate(angles):
            dx, dy = angle_dis*np.cos(angle), angle_dis*np.sin(angle)
            x1, y1 = origin
            x2, y2 = x1+dx, y1+dy
            plt.plot([x1, x2], [y1, y2], color=angle_color, alpha=alpha, linewidth=angle_linewidth )
            plt.text(x2, y2, f"{i_edge}", fontdict={'color':'olivedrab', 'size':10 } )	

    plt.show(block=False)

