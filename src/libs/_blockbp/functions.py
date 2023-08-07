# ============================================================================ #
#|                                 imports                                    |#
# ============================================================================ #

if __name__ == "__main__":
	import sys, pathlib
	sys.path.append(pathlib.Path(__file__).parent.parent.__str__())

# Numeric operations and constants:
import numpy as np
from numpy import pi

# Import mps functions:
from tensor_networks.tensors.mps import (
    init_mps_classical,
    init_mps_quantum,
	MPS,
)

# for typing hints:
from typing import (
	Tuple,
	List,
	Any,
	Dict,
)

# Use our other types and classes:
from _blockbp.errors import MPIError

# Our other common types:
from _blockbp.classes import (
	MessageModel,
	BlocksConType,
)

# BlockBP configuration:
from _blockbp.config import USE_SPECIAL_CASE_1BLOCK

from lib.bubblecon import bubblecon




# ============================================================================ #
#|                            Inner Functions                                 |#
# ============================================================================ #




# ============================================================================ #
#|                           Declared Functions                               |#
# ============================================================================ #

def outside_m(BTN, Dtrunc, Dtrunc2, eps):
	"""
	Updates the messages of a block.

	It gets a block tensor network (BTN), which includes the block
	and all its incoming messages, and use it to calculate all the
	*outgoing* messages.

	Input Parameters:
	-------------------

	BTN: A block-tensor-network object. This is 4-tuple of the form
			 (T_list, edges_list, angles_list, bcon)
			 where: T_list, edges_list, angles_list describe the TN of
			 the block + incoming MPS messages.

			 Then bcon is a list of contraction orders that are used to
			 calculate all outgoing MPS messages.

	Dtrunc, Dtrunc2, eps:
	     bubblecon contraction accuracy parameters.

	"""

	(T_list, edges_list, angles_list, bcon) = BTN


	out_mp_list = []

	k = len(bcon) # number of super-edges of the block
	for i in range(k):

		sw_order, bubble_angle = bcon[i]

		#
		# Calculate the outgoing mps message
		#
		mp = bubblecon(T_list, edges_list, angles_list, bubble_angle,\
			swallow_order=sw_order, D_trunc=Dtrunc, D_trunc2=Dtrunc2, eps=eps, opt='high')

		out_mp_list.append(mp)

	return out_mp_list

def initial_messages(
	initial_m_mode:MessageModel, 
	blocks:Dict[str, Tuple[int, int]], 
	sedges_list:List[List[Tuple[str, int]]], 
	mpsD: List[List[List[int]]],
	log:bool, 
	num_blocks:int, 
) -> List[List[MPS]]:

	if isinstance(initial_m_mode, MessageModel):
		pass
	elif isinstance(initial_m_mode, str):
		initial_m_mode = MessageModel(initial_m_mode)
	else:
		raise TypeError(f"Expected `initial_m_mode` to be of type <str> or <MessageModel> enum. Got {type(initial_m_mode)}")

	if log:
		print(f"Preparing initial MPS messages in '{initial_m_mode}' mode")

	if USE_SPECIAL_CASE_1BLOCK and num_blocks==1:
		m_matrix  = [ dict() ] 		
	else:
		m_matrix  = [ [None]*num_blocks for i in range(num_blocks)] 

	for i in range(num_blocks):
     

		for (sedge_name,order) in sedges_list[i]:
			(ii,jj) = blocks[sedge_name]
			j = (jj if jj != i else ii)
   
			if USE_SPECIAL_CASE_1BLOCK and num_blocks==1:
				crnt_dims = mpsD[0][sedge_name]	
			else:
				crnt_dims = mpsD[i][j]	

			if log:
				if USE_SPECIAL_CASE_1BLOCK and num_blocks==1:
					print(f"Preparing initial message {sedge_name}")
				else:
					print("Preparing initial message {}->{}".format(i,j))

			if initial_m_mode==MessageModel.UNIFORM_CLASSIC:
				# Initial messages are uniform probability dist'
				mp = init_mps_classical(crnt_dims, random=False)

			elif initial_m_mode==MessageModel.RANDOM_CLASSIC:
				# Initial messages are random probability dist'
				mp = init_mps_classical(crnt_dims, random=True)

			elif initial_m_mode==MessageModel.UNIFORM_QUANTUM:
				# Initial messages are uniform quantum density matrices
				mp = init_mps_quantum(crnt_dims, random=False)

			elif initial_m_mode==MessageModel.RANDOM_QUANTUM:
				# Initial messages are random |v><v| quantum states
				mp = init_mps_quantum(crnt_dims, random=True)

			else:
				raise ValueError("Not a valid option")
	
			if USE_SPECIAL_CASE_1BLOCK and num_blocks==1:
				m_matrix[0][sedge_name] = mp
			else:
				m_matrix[i][j] = mp
	
	return m_matrix


def calculate_angles(pos_list, edges_list, vertices, Lx=1e9, Ly=1e9):
	"""
		Go over the list of tensors and calculate for each tensor leg its
		angle (on the plane) using the position of the tensors on the plane.
		The angles are always between 0 -> 2\pi defined by

		x = cos(theta),   y = sin(theta)

		Each tensor has a list of angles in accordance to the order of its
		legs, and the output is a list of these lists (according to the
		order in which the tensors appear in neighbors_list).

		Input Parameters:
		-------------------
		neighbors_list --- A list of lists: [N_0, N_1, N_2, ...]
											 N_0 is the list of the neighbors of tensor T_0
											 arranged according to indices in T_0. N_1 are
											 for T_1 etc.. External legs are specified by
											 a negative integer.

		int_pos_list --- a list of the tupples (x,y) position of each of the
		                 tensors that appear in neighbors_list

		ext_pos_list ---	A list of (x,y) tupples of the position of the
											ends of each of the external legs.

		Lx, Ly       ---  If given, they specify the unit cell size in
		                  a periodic boundary condition setup
	"""

	angles_list=[]

	# Go over all tensors in the system
	assert len(pos_list) == len(edges_list)
	for i, (pos, e_list) in enumerate( zip(pos_list, edges_list) ):

		x, y = pos # The location of the tensor T_i

		# Now go over all the legs and calculate their angles
		angles = []
		for e in e_list:

			(ii, jj) = vertices[e]

			j = (jj if ii==i else ii)

			(xj, yj) = pos_list[j]

			dx,dy = xj-x,yj-y

			if dx>Lx/2:
				dx = dx - Lx

			if dx<-Lx/2:
				dx = dx + Lx

			if dy>Ly/2:
				dy = dy - Ly

			if dy<-Ly/2:
				dy = dy + Ly


			theta = np.angle(dx + 1j*dy) % (2*pi)

			angles.append(theta)

		angles_list.append(angles)

	return angles_list


def check_mpi(has_mpi:bool, mpi_comm, num_blocks:int, log:bool) -> None:
	"""		
		Get the total number of MPI processes (mpi_size) and the index
		of the current process. Since the current process invoked blockbp,
		its index should be equal to nb (the index of all other processes 
		should match their block number, which is in the range 0->(nb-1)
	"""
	if has_mpi:
		mpi_rank = mpi_comm.Get_rank()
		mpi_size = mpi_comm.Get_size()
		
		if mpi_size != num_blocks+1:
			raise MPIError(
				f"blockbp was called with {num_blocks} blocks, but "
				f"mpi_size={mpi_size} (the number of MPI processes should "
				f"be {num_blocks+1}"
			)
		
		if mpi_rank != num_blocks:
			raise MPIError(
				f"blockbp has been invoked in MPI mode, but its rank "
				f"(the MPI index of the blockbp process) is {mpi_rank}, while "
				f"it should be {num_blocks} (the total number of blocks)"
			)

	if log and has_mpi:
		print("Entering blockBP in MPI MODE")
		print("=================================")
		print(f"MPI rank (current MPI process ID):        {mpi_rank}")
		print(f"MPI size (total number of MPI processes): {mpi_size}")
		print("\n\n")
	
def derive_data_structures(
	log:bool,
	num_tensors : int,
	num_blocks : int,
	edges_list : List[List[int]],
	sedges_list : List[List[Tuple[str, int]]],
	sedges_dict : Dict[str, List[int]],
	pos_list : List[Tuple[int, ...]],
	Lx : float,
	Ly : float,
	T_list : List[np.ndarray],
	blocks_v_list : List[List[int]],
	blocks_con_list : List[BlocksConType]
) -> Tuple[
	Dict[str, Tuple[int, int]],  # blocks
	List[List[List[int]]],  # mps_mapping
	List[List[List[int]]],  # mpsD
	List[Tuple[list, list, list, BlocksConType]]  # BTN_list
]:
	"""
	                       PRE-PROCESSING
	
	  If PP_blob is not given, we need some pre-processing to find:
	  the following objects:
	
	  1. blocks      --- A double list giving the super-edge label of two adjacent blocks
	
	  2. mps_mapping --- A double list [i][j] that tells us how every
	                     tensor in the MPS of the B_i -> B_j maps to a tensor in BTN_j
	
	  3. BTN_list    --- The list of BTNs
	
	 If PP_blob is given, we only need to update the tensors in the
	 BTN_list
	"""

	# 1. Create a dictionary that tells the vertices of each edge. The
	#    keys of the dictionary are the edges labels, and the values
	#    are tuples (i,j), where i,j are the vertices connected by it.

	vertices = {}

	for i in range(num_tensors):
		i_edges = edges_list[i]
		for e in i_edges:

			if e in vertices:
				(j1,j2) = vertices[e]
				vertices[e] = (i,j1)
			else:
				vertices[e] = (i,i)
	
	if log:
		print("vertices of edges:")
		print(vertices)	

	# Now do the same thing for the blocks and the super-edges
	blocks = {}
	
	for i in range(num_blocks):
		i_sedges = sedges_list[i]
		for (e,ty) in i_sedges:

			if e in blocks:
				(j1,j2) = blocks[e]
				blocks[e] = (i,j1)
			else:
				blocks[e] = (i,i)

	if log:
		print("blocks of super-edges:")
		print(blocks)

	if num_blocks==1 and USE_SPECIAL_CASE_1BLOCK:
		_super_edge_empty_mapping = lambda : [ dict() ]
		def _super_edge_mapping_setvalue( i, j, label, new_val, mapping): 
			mapping[0][label] = new_val
			return mapping
		def _super_edge_mapping_getvalue( i, j, label, mapping): 
			return mapping[0][label]
	else:
		_super_edge_empty_mapping = lambda : [ [None]*num_blocks for i in range(num_blocks)]
		def _super_edge_mapping_setvalue( i, j, label, new_val, mapping): 
			mapping[i][j] = new_val
			return mapping
		def _super_edge_mapping_getvalue( i, j, label, mapping): 
			return mapping[i][j]

	# 2. Find the angles_list --- the angles of each leg in each vertex
	#    using the positions of the vertices
	angles_list = calculate_angles(pos_list, edges_list, vertices, Lx, Ly)

	# 3a. Create a skeleton of a double list mps_mapping[i][j] that
	#    tells us how every tensor in the mps of the B_i -> B_j maps to
	#    a tensor in BTN_j
	# if num_tensors==1 then this mapping is a dict
	mps_mapping = _super_edge_empty_mapping()

	# 3b. Create a skeleton double list mpsD[i][j] that tells
	#    us how the dimnsion of the legs in the i->j super-edge. This
	#    only needed when m_list is empty and we need to start the BP
	#    with some initial mps messages.
	# if num_tensors==1 then this mapping is a dict
	mpsD = _super_edge_empty_mapping()

	# ================================================
	# 4. Creating the block tensor-networks (BTN) list
	# ================================================

	BTN_list = []

	for bi in range(num_blocks):
		if log:
			print("\n Creating BTN #{}".format(bi))
			print(   "-------------------")

		# Create T1_list, edges1_list, angles1_list using the vertices
		# that define the BTN and the global lists
		v_list = blocks_v_list[bi]

		if log:
			print("v-list: ", v_list)

		T1_list = [T_list[v] for v in v_list]
		edges1_list = [edges_list[v] for v in v_list]
		angles1_list = [angles_list[v] for v in v_list]

		# Create a vertices dictionary that serves as a map. It maps
		# the vertices in the global TN using to their corresponding
		# Ids in the BTN. It also maps the external edges to the block
		# to the MPS veritces in the block.
		#
		# The vertices in the global TN create a key of the form ('v', id)
		# with the value of the local ID of the tensor.
		#
		# Similarly, edges in the global TN create a key ('e', id)
		vertices_dict : Dict[ Tuple[str, int] | Tuple[str, int, str|None] , int ]
		if num_blocks==1 and USE_SPECIAL_CASE_1BLOCK:
			vertices_dict = {('v',v, None): i for i, v in enumerate(v_list) }
		else:
			vertices_dict = {('v',v): i for i, v in enumerate(v_list) }

		v = len(v_list)  # index of next vertex to add
		emps = 0              # running index of MPS edges to add

		# Now go over all super edges and add them to the BTN
		k = len(sedges_list[bi])

		for se_j in range(k):

			(se_name, se_order) = sedges_list[bi][se_j]
			if num_blocks==1:
				block_side_name = se_name[-1]

			(bi1, bj1) = blocks[se_name]
			bj = (bj1 if bi1==bi else bi1)

			# We now create a list e_list of the edges in the super-edge.
			#
			# It is more easy to work with counter-clockwise order. So if
			# se_order = +1, then we reverse the list.
			''' optional:
				edge_list = sedges_dict[se_name]
				if se_order==1:
					edge_list.reverse()
			'''

			if se_order==-1:
				e_list = sedges_dict[se_name]
			else:
				e_list1 = sedges_dict[se_name]
				l = len(e_list1)
				e_list = []
				for q in range(l):
					e_list.append(e_list1[l-1-q])

			se_pos_list = []
			mp_mapping = []
			mpD = []

			# First pass - calculate the positions of the mid vertices
			#              and add them to the legs->vert dictionary.
			#              These are the vertices of the MPS that are located
			#              in the middle of the external edges.
			le = len(e_list)
			for e in e_list:
				(i,j) = vertices[e]

				# make sure i is inside B and j outside
				if j in v_list:
					if num_blocks!=1:
						assert i not in v_list
					i,j = j,i

				T1_list.append(None)  # add a place holder for BTN vertex v
				if num_blocks==1 and USE_SPECIAL_CASE_1BLOCK:
					vertices_dict[('e',e, block_side_name)] = v
				else:
					vertices_dict[('e',e)] = v
				mp_mapping.append(v)

				# Find the dimension of the e=(i,j) leg and put it in the mpsD list:
				i_ind = edges_list[i].index(e)  # the position of the leg e in T_i
				D = T_list[i].shape[i_ind]
				mpD.append(D)

				# The position of the new vertex is in the middle of the external
				# leg. We keep its position, as well as the position of vertex
				# i inside B so that later in the second pass we can calculate
				# the angles of the mps edges.
				(xi,yi) = pos_list[i]
				(xj,yj) = pos_list[j]

				# (xj,yj) is the position of the vertex in the other block.
				# if we're in periodic boundary-conditions then we might
				# need to shift its x or y by a unit cell.
				if xj-xi > Lx/2:
					xj -= Lx

				if xj-xi < -Lx/2:
					xj += Lx

				if yj-yi > Ly/2:
					yj -= Ly

				if yj-yi < -Ly/2:
					yj += Ly


				x,y = 0.5*(xi+xj), 0.5*(yi+yj)
				se_pos_list.append( (x,y,xi,yi) )

				v += 1


			mps_mapping = _super_edge_mapping_setvalue(bi, bj, se_name, mp_mapping, mps_mapping)
			mpsD        = _super_edge_mapping_setvalue(bi, bj, se_name, mpD, mpsD)


			# Second pass - add the mps edges
			for ie, e1 in enumerate(e_list):
				# e1 is the the external leg
				# get the mps vertex:
				
				if USE_SPECIAL_CASE_1BLOCK and num_blocks==1:
					i = vertices_dict[('e',e1, block_side_name)]  
				else: 
					i = vertices_dict[('e',e1)]  

				# Add the mps edge connecting ie, ie+1
				eR = 'm{}'.format(emps)

				(x,y,xi,yi) = se_pos_list[ie]

				theta_mid = np.angle(xi-x + 1j*(yi-y)) % (2*pi)


				if ie==le-1:
					edges1_list.append([eL,e1])
					angles1_list.append([theta_L, theta_mid])

				else:
					(x2,y2,xi2,yi2) = se_pos_list[ie+1]

					theta_R = np.angle(x2-x + 1j*(y2-y)) % (2*pi)

					emps +=1

					if ie==0:
						edges1_list.append([e1, eR])
						angles1_list.append([theta_mid, theta_R])
					else:
						edges1_list.append([eL, e1, eR])
						angles1_list.append([theta_L, theta_mid, theta_R])



					eL = eR
					theta_L = (theta_R + pi) % (2*pi)

				if log:
					print("added edges {} and angles {}".format(edges1_list[-1], angles1_list[-1]))

		# Final calculation for the BTN: translate the swallowing-order
		# from the ('e',id) and ('v',id) notation to the actual vertices
		# in the BTN. Use the vertices_dict mapping for that.

		# get the contraction lists of the block. We make a copy of the 
		# list because it is going to be changed and we don't want to 
		# change the input parameter blocks_con_list.
		bcon = blocks_con_list[bi].copy()

		# k is the number of super-edges adjacent to the block, which
		# is equal to the number of swallowing orders.
		for se_j in range(k):
			(sw_order, bubble_angle) = bcon[se_j]
			(se_name, se_order) = sedges_list[bi][se_j]

			sw_order1 = [vertices_dict[q] for q in sw_order]


			
			''' DEBUG:
				from utils.strings import formatted
				for s, s1 in zip(sw_order, sw_order1):
					print(
						formatted(str(s), width=12) + formatted(s1, width=12)
					)
			'''


			bcon[se_j] = (sw_order1, bubble_angle)

		BTN = (T1_list, edges1_list, angles1_list, bcon)

		BTN_list.append(BTN)

	if log:
		print("\n\n\n")
		print("MPS MAPPING:")

		if num_blocks==1:
			for se_j in range(k):
				(se_name, se_order) = sedges_list[bi][se_j]
				crnt_dim = _super_edge_mapping_getvalue(0, 0, se_name, mpsD)
				crnt_mps = _super_edge_mapping_getvalue(0, 0, se_name, mps_mapping)
				print(f"mapping  sedge_name '{se_name}' -> {crnt_mps}   Dims={crnt_dim}")				
		else:
			for i in range(num_blocks):
				for j in range(num_blocks):
					print("mapping {}->{} = {}   Dims={}".format(i,j,mps_mapping[i][j], mpsD[i][j]))				
    

	return blocks, mps_mapping, mpsD, BTN_list


def MPIblock(mpi_comm):
	"""
	
	A slave function for handling a block in MPI mode. 
	
	This function should be called by an MPI process with mpi-rank 
	that matches the block index. 
	
	The function then waits to orders from its master (the MPI process
	that runs blockbp). It first gets its BTN from the master. Then it
	goes into a loop in which:
	  (1) Calculates the outgoing messages
	  (2) Sends the outgoing messages to the neighboring blocks (via MPI)
	  (3) Recives incoming messages from the neighboring blocks
	  (4) Calculates the distance between the incoming messages and 
	      the old incoming messages from the previous round.
	  (5) Send the distance to the master, and wait to its command
	      (whether to continue with the BP loop or exit)
	      
	      
	  Input Parameters:
	  -------------------
	  
	  mpi_comm --- The MPI communication block. Its rank should match
	               the block index.
	
		OUTPUT: None.
		--------------
	
	"""

	log = False
	
	ib = mpi_comm.Get_rank()
	master = mpi_comm.Get_size()-1 
	
	t0 = time.time()
		
	
	if log:
		print("\n\n")
		print("=========================================================")
		print(f"MPIblock: Entering for block {ib} T={int(time.time()-t0)}")
		print("=========================================================")
		print("\n")
	
	(BTN, neighbors, mps_list, mps_mapping, D_trunc, D_trunc2, eps, damping) \
		= mpi_comm.recv(source=master, tag=0)
	
	if log:
		print(f"Block {ib}: got neighbors info from master: {neighbors}  T={int(time.time()-t0)}")
	

	T1_list, edges1_list, angles1_list, bcon = BTN

	N_neighbors = len(neighbors)

	#
	# Assign the mps-messages tensors to their places inside the BTN
	#

	for k in range(N_neighbors):
		jb = neighbors[k]
		mapping = mps_mapping[jb][ib]
		mp = mps_list[k]
		
		for q in range(mp.N):
			T = mp.A[q]

			if q==0:
				sh = T.shape
				Tnew = T.reshape([sh[1], sh[2]])
			elif q==mp.N-1:
				sh = T.shape
				Tnew = T.reshape([sh[0], sh[1]])
			else:
				Tnew = 1.0*T

			T1_list[mapping[q]] = Tnew
	
	if log:
		print(f"Block {ib}: Done mapping mps->BTN. Going to main block BP loop  T={int(time.time()-t0)}")
	
	run_BP = True
	
	
	#
	# ===================== MAIN SLAVE BP LOOP ==========================
	#
	while run_BP:
		
		#
		# 1. Calculate outgoing messages
		#
		
		if log:
			print("\n")
			print(f"Block {ib}: calculating outside messages. T={int(time.time()-t0)}")
			
		out_m_list = outside_m(BTN, D_trunc, D_trunc2, eps)
		
		if log:
			print(f"Block {ib}: ==> done. T={int(time.time()-t0)}")


		#
		# 2. Normalize the messages and send to neighbors
		#

		reqs_list = []
		for k in range(N_neighbors):
			jb = neighbors[k]
			
			mps_message = out_m_list[k]  # this is the new ib->jb message
			#
			# Make it left canonical, and normalize the right-most tensor
			#

			mps_message.left_canonical_QR()
			N1 = mps_message.N
			mps_message.set_site(mps_message.A[N1-1]/norm(mps_message.A[N1-1]), N1-1)
			
			#
			# Update the outgoing list with the normalized MPS
			#
			
			out_m_list[k] = mps_message
			
			#
			# Send the mps message to the k'th neighbor and get his message.
			#
			# We send it in a non-blocking manner so that message transfer
			# between blocks is as fast as possible.
			#
			
			if log:
				print(f"Block {ib}: Sending MPS {ib}->{jb}  T={int(time.time()-t0)}")
			
			req = mpi_comm.isend(mps_message, dest=jb, tag=2)
			reqs_list.append(req)
			
			if log:
				print(f"Block {ib}: MPS sent. T={int(time.time()-t0)}")
				
		
		#
		# 3. Get the neighbors messages, update the BTN and calculate the 
		#    difference between old messages and new messages
		#
		
		err = 0.0
		
		for k in range(N_neighbors):
			jb = neighbors[k]

			if log:
				print(f"Block {ib}: Receving MPS {jb}->{ib}  T={int(time.time()-t0)}")
			
			incoming_mps = mpi_comm.recv(source=jb, tag=2)

			if log:
				print(f"Block {ib}: Got it.  T={int(time.time()-t0)}")

			#
			# 4. Calculate the difference between the old incoming message and
			# the new incoming message
			#

			inP = bmpslib.mps_inner_product(incoming_mps, mps_list[k], \
				conjB=True)

			err0 = 2-2*inP.real
			err += err0

			#
			# Update the messages list
			#

			mps_list[k] = incoming_mps  # update the jb->ib mps message

			#
			# Update the BTN tensors
			#
			mapping = mps_mapping[jb][ib]
			for q in range(incoming_mps.N):
				T = incoming_mps.A[q]

				if q==0:
					sh = T.shape
					Tnew = T.reshape([sh[1], sh[2]])
				elif q==N1-1:
					sh = T.shape
					Tnew = T.reshape([sh[0], sh[1]])
				else:
					Tnew = 1.0*T

				if damping is None or T1_list[mapping[q]].shape != Tnew.shape:
					T1_list[mapping[q]] = 1.0*Tnew
				else:
					T1_list[mapping[q]] = damping*T1_list[mapping[q]] \
						+ (1-damping)*Tnew
				
		#
		# 5. Send the error and number of neighbors to the master
		#
		
		if log:
			print(f"Block {ib}: Sending error to master  T={int(time.time()-t0)}")
			
		mpi_comm.send( (err, N_neighbors), dest=master, tag=5)
		
		if log:
			print(f"Block {ib}: message sent  T={int(time.time()-t0)}")
		
		
		#
		# Now get the command from the master whether to continue with 
		# the BP loop or exit
		#
		
		if log:
			print(f"Block {ib}: Getting master response. T={int(time.time()-t0)}")
			
		run_BP = mpi_comm.recv(source=master, tag=6)
		
		if log:
			print(f"Block {ib}: Got it.  T={int(time.time()-t0)}")
		
	#
	# End of the BP loop for the block. Send the master the latest 
	# BTN and outgoing messages
	#
	
	mpi_comm.send( (BTN, out_m_list), dest=master, tag=7)
		
	return 


if __name__ == "__main__":
	from src.scripts.blockbp_test import mean_z_from_blockbp
	mean_z_from_blockbp()
