#----------------------------------------------------------
#
# Module: blockbp- Boundary MPS Belif-Propagation library
#
#
# History:
# ---------
#
# 19-Sep-2021 (Itai) Initial version.
#
# 20-Nov-2021 (Itai) Added support for periodic BC (Lx, Ly parameters
#                    in blockbp, as well as in calculate_angles)
#
#
# 12-Dec-2021 (Itai) Renamed create_random_PSD_mps --> init_mps_quantum
#                    Added/updated documentation to init_mps_quantum,
#                    trim_mp, outside_m, blockbp
#
# 31-Dec-2021 (Itai) Removed un-needed imports
#
# 5-Feb-2022  (Itai) Removed the trim_mp function and moved it
#                    (as trim_mps) to bmpslib
#
# 4-Mar-2022  (Itai) changed bmpslib.mps_inner_prodcut -> 
#                    bmpslib.mps_inner_product (to match a fix in 
#                    bmpslib)
#
# 7-Apr-2022  (Itai) In blockbp, when blob=None, then when creating
#                    the BTN, do a hard copy of the block_con_list,
#                    for otherwise this list is changed and cannot 
#                    be used if calling blockbp again with blob=None.
# 
# 8-Apr-2022  (Itai) Added two fields to the final blob in blockBP:
#                    'final-error', 'final-iter', which store the 
#                    final error in the iteration and the final 
#                    iteration number.
#
#
# 17-Apr-2022 (Itai) Add the "random" parameter to init_mps_quantum.
#                    If True then the MPS is a random (PSD).
#
# 5-May-2022 (Itai)  Added MPI support. This is done via mpi4py. 
#                    When calling blockbp in MPI mode, every block
#                    has an MPI process. In addition, there's a master
#                    process that invokes blockbp. The other (block) 
#                    procecess call MPIblock.
#
# 6-May-2022 (Itai)  In MPIblock, use damp only when old and new
#                    tensors have the same shape
#
#
# 7-May-2022 (Itai)  Fix damp also in blockbp. Also, add abs()
#                    to the error calculation.
#
# 28-May-2022 (Itai) In blockbp, removed the waiting for the message to 
#                    be sent to other neighbors. It is un-needed because
#                    at any round in any case we wait for a response
#                    from the master.
#
# 28-May-2022 (Itai) In blockbp, replaced the input parameter classical
#                    with initial_m_mode, which can take 4 options:
#                    'UQ' (uniform quantum), 'RQ' (random quantum), 
#                    'UC' (uniform classical), 'RC' (random classical)
#
# 28-May-2022 (Itai) In blockbp, add the input parameter 'damping' 
#                    for possible damping in the BP messages to 
#                    facilitate the convergence. Default value is None.
#----------------------------------------------------------
#


import numpy as np

from numpy.linalg import norm

from numpy import  sqrt, pi

from sys import exit

from ItaysModules import bmpslib
from ItaysModules.bubblecon import bubblecon

import time


#
# -------------------------- calculate_angles --------------------------
#

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

	#
	# Go over all tensors in the system
	#
	n = len(edges_list)
	for i in range(n):

		e_list = edges_list[i]

		(x,y) = pos_list[i] # The location of the tensor T_i

		#
		# Prepare an empty list of angles
		#
		k = len(e_list)
		angles = []

		#
		# Now go over all the legs and calculate their angles
		#
		for e in e_list:

			(ii,jj) = vertices[e]

			j = (jj if ii==i else ii)

			(xj,yj) = pos_list[j]

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







#
# ----------------------  init_mps_quantum  -----------------------
#


def init_mps_quantum(D_list, random=False):
	"""

	Creates an initial MPS message.

	The MPS is just a product state, and each state is the vectorization
	of the ID or a random |v><v|. This corresponds, in the double edge 
	model, to a trivial ket-bra contraction or a contraction with a 
	random |v>.

	Input Parameters:
	-------------------

	D_list --- A list "physical" dimensions of the MPS. Because we are
						 in the double-edge model, then each leg is the
						 fusion of two legs, and so its dimension must be of the
						 form D=d^2.

	           Since the MPS describes a product state, the shape of
	           the i'th tensor will be [1,D_i,1]  (right/left bond dimension
	           are 1)
	           
	random --- Whether or not the initial MPS is random (or a trivial
	           identity in double-edge formalism)

	Output:
	--------
	An MPS object which is left-canonical.


	"""

	N = len(D_list)
	mp = bmpslib.MPS(N)

	for i in range(N):
		D = D_list[i]
		d = int(sqrt(D))
		if D!= d*d:
			print("Error: trying to create a random PSD mps with D={}, which "\
				"does not have an integer square root".format(D))
			exit(1)

		if random:
			A = np.random.normal(size=d)
			A = A/norm(A)
			A2 = np.tensordot(A,A,0)
		else:
			A2 = np.eye(d)
			
		A2 = A2.reshape(-1)

		mp.set_site(A2.reshape([1,D,1]), i)
	#
	# Make it left-canonical and normalize
	#
	
	mp.left_canonical_QR()
	
	mp.set_site(mp.A[N-1]/norm(mp.A[N-1]), N-1)

	return mp




#
# ---------------------- init_mps_classical  -----------------------
#

def init_mps_classical(D_list, random=False):
	"""

	Initializes an MPS message for TNs representing classical systems

	In this case, the MPS represents a product state of classical
	probability distributions: either uniform (random=False) or random
	(random=True)

	Input parameters:
	-----------------

	D_list --- A list of the "physical" leg dimensions, which is simply
						 the dimension of each message.
						 
	random --- Whether to use a random prob dist.

	Output:
	-----------
	An MPS object which is left-canonical.


	"""
	N = len(D_list)
	mp = bmpslib.MPS(N)

	for i in range(N):
		D = D_list[i]
		
		if random:
			A = np.random.uniform(size=D)
		else:
			A = np.ones(D)  # Create a uniform probability dist'
			
		A = A/norm(A)
		
		mp.set_site(A.reshape([1,D,1]), i)

	mp.left_canonical_QR()
	mp.set_site(mp.A[N-1]/norm(mp.A[N-1]), N-1)

	return mp


#
# ----------------------  outside_m  -----------------------
#


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




#
# ------------------------- MPIblock  ---------------------------
#
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

	


#
# ------------------------- blockbp  ---------------------------
#


def blockbp(T_list, edges_list, pos_list, \
	blocks_v_list, sedges_dict, \
	sedges_list, blocks_con_list, \
	PP_blob=None, m_list=None, \
	initial_m_mode = 'UQ', \
	D_trunc=None, D_trunc2=None, eps=None, max_iter=100, delta=1e-8, 
	damping = None, Lx=1e9, Ly=1e9, mpi_comm=None):

	"""

  A belief propagation contraction algorithm for blocks of tensors
  in a 2D tensor network.

  Input Parameters
  -------------------

  (*) T_list, edges_list:

      ncon style encoding of the TN. Note that, unlike in ncon, here
      edges labels can be any string/integer. Also, there should not
      be an open edges.

  (*) pos_list:

      An array of (x,y) tuples encoding the (x,y) position of every
      tensor in the network

  (*) blocks_v_list:

			list of lists [v1_list, v2_list, ...]
			where every v<i>_list is a list of tensors that make up the
			i'th block.

	(*) sedges_dict:
			a dictionary specifying the super edges. The keys are the labels
			of the superedge. The values are lists of the labels of the edges
			that make up the super edge. For example, in
			{'A':[1,4,5], 'B':[2,3], ...}
			we have a super-edge 'A' made up of egdes 1,4,5, and a super-edge
			'B' made up of edges [2,3]

	(*) sedges_list:
			The equivalent of the edges_list to the case of super-edges.
			The list contains an element for each block. This element is the
			list of super-edges of that block. The list of super-edges is made
			of 2-tuples (se_name, order) where se_name is the label of the
			super-edge and order is a +1 or -1 integer that tells if edges in
			the super edge are ordered in a clockwise order with respect to
			the block (order=+1) or anti clockwise (order=-1)

	(*)	blocks_con_list:
			A list telling the contraction order of the bubblecon algorithm
			when used to calculate outgoing mps messages of each block.

			Each item in the list correspond to a block. For each block we
			have a contraction orders - one for each super-edge. The
			contraction order of a super-edge tells bubblecon how to calculate
			the out-going message of that super-edge. It is an ordered list
			of the vertices that participate in the contraction. There are
			the original veritces of the block, plus the veritces of the
			incomping mps messages. Each vertex is specified by a taple
			(type,id), where type='v' or type='e' for an original block vertex
			of for an incoming mps vertex. For a type='e' vertex, the id is
			actually the label of the edge that is connected to the MPS vertex.

	(*) PP_blob=None (OPTIONAL):
			This is an *optional* dictionary that contains a lot of
			pre-processing data. If specified, it can save some pre-processing
			computations. The blob is automatically returned by blockbp, and
			so it can be given as an input for the next run of blockbp as long
			as the topological data has not changed (only the values of the
			entries of the tensors).

	(*) m_list=None (OPTIONAL):
			A list of incoming MPS messages. If supplied then the BP iteration
			starts with these messages. This is a double list where
			m_list[i][j] contains the block i --> block j MPS message

			m_list is also returned form the blockbp function so that it can
			be used as an input in the next iteration.

	(*) initial_m_mode = 'UQ' (OPTIONAL):
			Describes the type of initial messages used when m_list is not
			given. There are 4 possibilities:
			1) UQ --- Uniform quantum. The initial messages are 
			          \delta_{\alpha\beta} (i.e., the identity)
			2) RQ --- The initial messages are product states of |v><v|, where
			          |v> is a random unit vector
			3) UC --- For when the TN represents a classical probability.  
			          in this case the initial message is simply the uniform
			          prob.
			4) RC --- For the TN represents a classical prob. Then the inital
			          messages are product of random prob dist.
			          
	(*) D_trunc=None, D_trunc2=None, eps=None  (OPTIONAL):
			Optional bubblecon accuracy truncation parameters for the block
			contraction.

	(*) max_iter=100 (OPTIONAL):
			An upperbound for the number of iterations. The BP loop terminates
			even if no convergence was achieved when the number of iterations
			passes this limit.

	(*)	delta=1e-8 (OPTIONAL):
			A convergence threshold for the messages.
			For every mps message at time t, we calculate its overlap with the
			previous message (at time t-1), and from it calculate the distance:
			|| psi_t - psi_{t-1}||^2 = 2 - 2Re(<psi_t | psi_{t+1}>)

			Then we calculate the average distance, and stop the BP iteration
			when it is under the delta threshold.
			
	(*) damping=None (OPTIONAL). 
	    Add a possible damping to the mps messages to facilitate the 
	    convergence in challanging cases. When damping is not None, 
	    the t+1 messages are given as:
	    (1-damping)*MPS_{t+1} + daming*MPS_t

	(*)	Lx=1e9, Ly=1e9 (OPTIONAL):
			The X,Y periods of the lattice.
			
	(*) mpi_comm --- A possible MPI communication object, if the
	                 the blockBP algorithm is to be run in MPI mode.


	RETURNS:
	---------

	m_list, PP_blob

	Note that the PP_blob contains also the block-tensor networks (BTNs)
	from which it is easy to calculate local expectation values.








	"""

	log = False
	
	MPI = mpi_comm is not None
	
	n = len(T_list) # number of vertices
	nb = len(blocks_v_list) # number of blocks

	
	if MPI:
		#
		# Get the total number of MPI processes (mpi_size) and the index
		# of the current process. Since the current process invoked blockbp,
		# its index should be equal to nb (the index of all other processes 
		# should match their block number, which is in the range 0->(nb-1)
		#
		mpi_rank = mpi_comm.Get_rank()
		mpi_size = mpi_comm.Get_size()
		
		if mpi_size != nb+1:
			print("Error in blockbp!")
			print(f"blockbp was called with {nb} blocks, but "
				f"mpi_size={mpi_size} (the number of MPI processes should "
				f"be {nb+1}" )
			exit(1)
		
		if mpi_rank != nb:
			print("Error in blockbp!")
			print(f"blockbp has been invoked in MPI mode, but its rank "
				f"(the MPI index of the blockbp process) is {mpi_rank}, while "
				f"it should be {nb} (the total number of blocks)")
			exit(1)


	if log and MPI:
		print("Entering blockBP in MPI MODE")
		print("=================================")
		print(f"MPI rank (current MPI process ID):        {mpi_rank}")
		print(f"MPI size (total number of MPI processes): {mpi_size}")
		print("\n\n")
			

	if log:
		print("Number of vertices n={}     Number of blocks nb={}".format(n,nb))
		


	#
	# ================================================================
	#
	#                        PRE-PROCESSING
	#
	#   If PP_blob is not given, we need some pre-processing to find:
	#   the following objects:
	#
	#   1. blocks      --- A double list giving the super-edge label
	#                      of two adjacent blocks
	#
	#   2. mps_mapping --- A double list [i][j] that tells us how every
	#                      tensor in the MPS of the B_i -> B_j maps to
	#                      a tensor in BTN_j
	#
	#   3. BTN_list    --- The list of BTNs
	#
	#
	#  If PP_blob is given, we only need to update the tensors in the
	#  BTN_list
	#
	# ================================================================
	#

	if PP_blob is None:

		#
		# 1. Create a dictionary that tells the vertices of each edge. The
		#    keys of the dictionary are the edges labels, and the values
		#    are tuples (i,j), where i,j are the vertices connected by it.
		#

		vertices = {}

		for i in range(n):
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

		#
		# Now do the same thing for the blocks and the super-edges
		#
		blocks = {}

		for i in range(nb):
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



		#
		# 2. Find the angles_list --- the angles of each leg in each vertex
		#    using the positions of the vertices
		#


		angles_list = calculate_angles(pos_list, edges_list, vertices, Lx, Ly)

		#
		# 3a. Create a skeleton of a double list mps_mapping[i][j] that
		#    tells us how every tensor in the mps of the B_i -> B_j maps to
		#    a tensor in BTN_j
		#

		mps_mapping = [ [None]*nb for i in range(nb)]


		#
		# 3b. Create a skeleton double list mpsD[i][j] that tells
		#    us how the dimnsion of the legs in the i->j super-edge. This
		#    only needed when m_list is empty and we need to start the BP
		#    with some initial mps messages.
		#

		mpsD = [ [None]*nb for i in range(nb)]

		#
		# ================================================
		# 4. Creating the block tensor-networks (BTN) list
		# ================================================
		#

		BTN_list = []

		for bi in range(nb):
			if log:
				print("\n\n Creating BTN #{}".format(bi))
				print(    "-------------------\n")

			#
			# Create T1_list, edges1_list, angles1_list using the vertices
			# that define the BTN and the global lists
			#

			v_list = blocks_v_list[bi]

			if log:
				print("v-list: ", v_list)

			T1_list = [T_list[v] for v in v_list]
			edges1_list = [edges_list[v] for v in v_list]
			angles1_list = [angles_list[v] for v in v_list]

			#
			# Create a vertices dictionary that serves as a map. It maps
			# the vertices in the global TN using to their corresponding
			# Ids in the BTN. It also maps the external edges to the block
			# to the MPS veritces in the block.
			#
			# The vertices in the global TN create a key of the form ('v', id)
			# with the value of the local ID of the tensor.
			#
			# Similarly, edges in the global TN create a key ('e', id), where
			# now
			#
			vertices_dict = {}
			for i in range(len(v_list)):
				vertices_dict[('v',v_list[i])] = i


			v = len(v_list)  # index of next vertex to add

			emps = 0              # running index of MPS edges to add

			#
			# Now go over all super edges and add them to the BTN
			#

			k = len(sedges_list[bi])

			for se_j in range(k):

				(se, se_order) = sedges_list[bi][se_j]

				(bi1, bj1) = blocks[se]
				bj = (bj1 if bi1==bi else bi1)


				#
				# We now create a list e_list of the edges in the super-edge.
				#
				# It is more easy to work with counter-clockwise order. So if
				# se_order = +1, then we reverse the list.
				#

				if se_order==-1:
					e_list = sedges_dict[se]
				else:
					e_list1 = sedges_dict[se]
					l = len(e_list1)
					e_list = []
					for q in range(l):
						e_list.append(e_list1[l-1-q])

				se_pos_list = []
				mp_mapping = []
				mpD = []

				#
				# First pass - calculate the positions of the mid vertices
				#              and add them to the legs->vert dictionary.
				#              These are the vertices of the MPS that are located
				#              in the middle of the external edges.

				le = len(e_list)

				for ie in range(le):

					e = e_list[ie]
					(i,j) = vertices[e]

					# make sure i is inside B and j outside
					if j in v_list:
						i,j = j,i


					T1_list.append(None)  # add a place holder for BTN vertex v

					vertices_dict[('e',e)] = v

					mp_mapping.append(v)

					# Find the dimension of the e=(i,j) leg and put it in the mpsD
					# list.

					i_pos = edges_list[i].index(e)  # the position of the leg e
																					# in T_i

					D = T_list[i].shape[i_pos]

					mpD.append(D)

					#
					# The position of the new vertex is in the middle of the external
					# leg. We keep its position, as well as the position of vertex
					# i inside B so that later in the second pass we can calculate
					# the angles of the mps edges.
					#
					(xi,yi) = pos_list[i]
					(xj,yj) = pos_list[j]

					#
					# (xj,yj) is the position of the vertex in the other block.
					# if we're in periodic boundary-conditions then we might
					# need to shift its x or y by a unit cell.
					#

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


				mps_mapping[bj][bi] = mp_mapping
				mpsD[bj][bi] = mpD

				#
				# Second pass - add the mps edges
				#
				for ie in range(le):
					e1 = e_list[ie]   # the external leg
					i = vertices_dict[('e',e1)]  # the mps vertex

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
						print("added edges {} and angles {}".format(edges1_list[-1], \
							angles1_list[-1]))

			#
			# Final calculation for the BTN: translate the swallowing-order
			# from the ('e',id) and ('v',id) notation to the actual vertices
			# in the BTN. Use the vertices_dict mapping for that.
			#

			#
			# get the contraction lists of the block. We make a copy of the 
			# list because it is going to be changed and we don't want to 
			# change the input parameter blocks_con_list.
			#
			bcon = blocks_con_list[bi].copy()

			# k is the number of super-edges adjacent to the block, which
			# is equal to the number of swallowing orders.


			for se_j in range(k):
				(sw_order, bubble_angle) = bcon[se_j]

				sw_order1 = [vertices_dict[q] for q in sw_order]


				bcon[se_j] = (sw_order1, bubble_angle)

			BTN = (T1_list, edges1_list, angles1_list, bcon)

			BTN_list.append(BTN)

		if log:
			print("\n\n\n")
			print("MPS MAPPING:")

			for i in range(nb):
				for j in range(nb):
					print("mapping {}->{} = {}   Dims={}".format(i,j,mps_mapping[i][j], mpsD[i][j]))

	else:
		#
		# So PP_blob is given. Extract its fields and update BTN_list
		#

		blocks = PP_blob['blocks']
		mps_mapping = PP_blob['mps_mapping']
		mpsD = PP_blob['mpsD']
		BTN_list = PP_blob['BTN_list']

		#
		# Update the BTN tensors with tensors from T_list
		#
		for ib in range(nb):

			v_list = blocks_v_list[ib]

			(T1_list, edges1_list, angles1_list, bcon) = BTN_list[ib]

			for i in range(len(v_list)):
				v = v_list[i]
				T1_list[i] = T_list[v]








	#
	# ================================================================
	#
	#                      INITIALIZING THE MAIN LOOP
	#
	# ================================================================
	#



	#
	# If m_list is empty (no initial messages are given), then start with
	# either random MPS messages or messages that represent unifom density
	# matrices or uniform probability dist.
	#

	if m_list is None:

		if log:
			print("\n")
			print(f"Preparing initial MPS messages in {initial_m_mode} mode")
			print("\n")

		m_list = [ [None]*nb for i in range(nb)]

		for i in range(nb):

			for (se,order) in sedges_list[i]:
				(ii,jj) = blocks[se]
				j = (jj if jj != i else ii)

				if log:
					print("Preparing initial message {}->{}".format(i,j))

				if initial_m_mode=='UC':
					#
					# Initial messages are uniform probability dist'
					#
					mp = init_mps_classical(mpsD[i][j], random=False)
				elif initial_m_mode=='RC':
					#
					# Initial messages are random probability dist'
					#
					mp = init_mps_classical(mpsD[i][j], random=True)
				elif initial_m_mode=='UQ':
					#
					# Initial messages are uniform quantum density matrices
					#
					mp = init_mps_quantum(mpsD[i][j], random=False)
				else:
					#
					# Initial messages are random |v><v| quantum states
					#					
					mp = init_mps_quantum(mpsD[i][j], random=True)

				m_list[i][j] = mp




	if MPI:
		
		#
		# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
		#                 
		#                  MPI BP
		#
		# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
		#
		
		
		blocks_neighbors = [None]*nb

		for i in range(nb):
			
			neighbors = []
			mps_list = []
			
			for (se,order) in sedges_list[i]:
				(ii,jj) = blocks[se]
				j = (jj if jj != i else ii)
				
				neighbors.append(j)
				
				mps_list.append(m_list[j][i])
				
			blocks_neighbors[i] = neighbors
			
			if log:
				print(f"block {i}: neighbors={neighbors}")
			
			block_info = (BTN_list[i], neighbors, mps_list, mps_mapping, \
				D_trunc, D_trunc2, eps, damping) 
				
			mpi_comm.send(block_info, dest=i, tag=0)
		
		if log:
			print("Finished sending all initial info to the block clients")
		
		err = 1.0
		iter_no = 0
		run_BP = True
		
		while run_BP:
			
			iter_no += 1

			
			#
			# Get the errors
			#
			
			err = 0.0
			S = 0
			for i in range(nb):
				(err1, S1) = mpi_comm.recv(source=i, tag=5)
				err += err1
				S += S1
				
				
			err = sqrt(abs(err))/S

			if log:
				print("")
				print("==> MPI blockbp iter {}: err = {}".format(iter_no,err))
				
			#
			# Decide whether or not we continue in the BP loop and upate
			# the blocks clients about it
			#
			
			run_BP = (err>delta and iter_no < max_iter)
			
			for i in range(nb):
				mpi_comm.send(run_BP, dest=i, tag=6)
			
			
		
		#
		# End of BP loop. Get the updated BTNs from the blocks clients,
		# as well as the last outgoing messages
		#

		if log:
			print("\n\n\n\n ========== DONE MPI BP LOOP ===============\n\n\n")
			
			print("Gathering the updated BTNs & outgoing messages from block "
				"clients")
		
		for ib in range(nb):
			(BTN, out_m_list) = mpi_comm.recv(source=ib, tag=7)
			
			BTN_list[ib] = BTN
			neighbors = blocks_neighbors[ib]
			
			for k in range(len(neighbors)):
				jb = neighbors[k]
				m_list[ib][jb] = out_m_list[k]
				
	else:

		#
		# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
		#                 
		#                  NON-MPI BP
		#
		# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
		#

		#
		# Assign the mps tensors to their places inside the block TNs
		#

		for i in range(nb):

			(T1_list, edges1_list, angles1_list, bcon) = BTN_list[i]

			for (se,order) in sedges_list[i]:
				(ii,jj) = blocks[se]
				j = (jj if jj != i else ii)

				mp = m_list[j][i]

				mapping = mps_mapping[j][i]

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


		err = 1.0
		iter_no = 0

		#
		# ================================================================
		#
		#                      MAIN BP LOOP
		#
		# ================================================================
		#

		if log:
			print("\n\n")
			print("Entering main blockbp loop")
			print("===============================\n")


		while err>delta and iter_no < max_iter:

			iter_no += 1

			if log:
				print("\n\n\n")
				print("      <<<<<<<<<<<<<<<  At round {}  >>>>>>>>>>>>>>\n".format(iter_no))

			err = 0.0
			S = 0

			#
			# Go over all vertices, and for each vertex find its outgoing
			# messages from its incoming messages
			#

#			for ib in range(nb):
			for ib in np.random.permutation(nb):

				if log:
					print("\n\n ==> Calculating messages of block {}".format(ib))


				BTN = BTN_list[ib]

				out_m_list = outside_m(BTN, D_trunc, D_trunc2, eps)


				#
				# Normalize the messages and update the main list of messages
				#

				k = len(sedges_list[ib])

				for lb in range(k):
					(se,order) = sedges_list[ib][lb]
					(ib1,jb1) = blocks[se]
					jb = (ib1 if ib1 !=ib else jb1)

					mps_message = out_m_list[lb]  # this is the new ib->jb message
					#
					# Make it left canonical, and normalize the right-most tensor
					#

					mps_message.left_canonical_QR()
					N1 = mps_message.N
					mps_message.set_site(mps_message.A[N1-1]/norm(mps_message.A[N1-1]), N1-1)

					#
					# Calculate the difference between the old message and the
					# new message
					#

					inP = bmpslib.mps_inner_product(mps_message, m_list[ib][jb], \
						conjB=True)

					err0 = 2-2*inP.real
					err += err0
					S += 1

					#
					# Update the messages list
					#

					m_list[ib][jb] = mps_message  # update the ib->jb mps message

					#
					# Update the Block Tensors to which ib sends messages
					#
					(T1_list, edges1_list, angles1_list, bcon) = BTN_list[jb]
					mapping = mps_mapping[ib][jb]
					for q in range(N1):
						T = mps_message.A[q]

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
			# The error is the average L_2 distance divided by the total number
			# of coordinates if we stack all messages as one huge vector
			#

			
			err = sqrt(abs(err))/S

			if log:
				print("")
				print("==> blockbp iter {}: err = {}".format(iter_no,err))




	#
	# Prepare the PP_blob before existing.
	#

	PP_blob = {}
	PP_blob['sedges_list'] = sedges_list
	PP_blob['blocks'] = blocks
	PP_blob['mps_mapping'] = mps_mapping
	PP_blob['mpsD'] = mpsD
	PP_blob['BTN_list'] = BTN_list
	
	PP_blob['final-error'] = err
	PP_blob['final-iter'] = iter_no



	return m_list, PP_blob


