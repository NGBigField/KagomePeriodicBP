# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)

# For type hints:
from typing import(
	List,
	Dict, 
	Tuple,
	Any,
 	Optional,
)

# Numpy:
import numpy as np
from numpy.linalg import norm
from numpy import  sqrt, pi


# Use our utils:
from utils import (
	errors,
)

# get helper functions:
from _blockbp.functions import (
	check_mpi,
	outside_m,
	derive_data_structures,
	initial_messages,
)

# get common error types:
from _blockbp.errors import (
	BlockBPError
)

# get common types and classes:
from _blockbp.classes import(
	MessageModel,
	BlocksConType,
)

# BlockBP configuration:
from _blockbp.config import USE_SPECIAL_CASE_1BLOCK

# Import MPS tools:
from tensor_networks.tensors.mps import MPS
from lib import bmpslib




# ============================================================================ #
#|                             Inner  Functions                               |#
# ============================================================================ #

def _blockbp_sequential(
	m_matrix : List[List[MPS]|Dict[str, MPS]],
	mps_mapping : List[List[List[int]]],  # A double list [i][j] that tells us how every
	                     				  # tensor in the MPS of the B_i -> B_j maps to a tensor in BTN_j
	sedges_list : List[List[Tuple[str, int]]],
	blocks : Dict[str, Tuple[int, int]],  # A double list giving the super-edge label of two adjacent blocks
	num_blocks,
	BTN_list : List[Tuple[list, list, list, BlocksConType]] , # each element is (T_list, edges_list, angles_list, bcon)
	damping,
	D_trunc,
	D_trunc2,
	eps,
	delta : float,
	max_iter : int,
	log : bool
) -> Tuple[
	List[List[MPS]],  # m_list
	float,  # err
	int  # iter_no
]:
	
	# Assign the mps tensors to their places inside the block TNs
	for ib in range(num_blocks):

		(T1_list, edges1_list, angles1_list, bcon) = BTN_list[ib]

		for (sedge_name, order) in sedges_list[ib]:
			(ii, jj) = blocks[sedge_name]
			jb = (jj if jj != ib else ii)

			if USE_SPECIAL_CASE_1BLOCK and num_blocks==1:
				mp = m_matrix[0][sedge_name]
				mapping = mps_mapping[0][sedge_name]
			else:
				mp = m_matrix[jb][ib]
				mapping = mps_mapping[jb][ib]

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

				mapped_ind = mapping[q]
				assert isinstance(mapped_ind, int)
				assert mapped_ind<len(T1_list)
				T1_list[mapped_ind] = Tnew


	err = 1.0
	iter_no = 0

	# ================================================================
	#                      MAIN BP LOOP
	# ================================================================
	if log:
		print("\n")
		print("Entering main blockbp loop")
		print("===============================\n")


	while err>delta and iter_no < max_iter:

		iter_no += 1

		if log:
			print("\n      <<<<<<<<<<<<<<<  At round {}  >>>>>>>>>>>>>>".format(iter_no))

		err = 0.0
		S = 0

		# Go over all vertices, and for each vertex find its outgoing
		# messages from its incoming messages

		# for ib in range(num_blocks):
		for ib in np.random.permutation(num_blocks):

			if log:
				print(" ==> Calculating messages of block {}".format(ib))


			BTN = BTN_list[ib]

			out_m_list = outside_m(BTN, D_trunc, D_trunc2, eps)

			# Normalize the messages and update the main list of messages
			k = len(sedges_list[ib])

			for lb in range(k):
				(sedge_name,order) = sedges_list[ib][lb]
				(ib1,jb1) = blocks[sedge_name]
				jb = (ib1 if ib1 !=ib else jb1)

				mps_message = out_m_list[lb]  # this is the new ib->jb message

				# Make it left canonical, and normalize the right-most tensor

				mps_message.left_canonical_QR()
				N1 = mps_message.N
				mps_message.set_site(mps_message.A[N1-1]/norm(mps_message.A[N1-1]), N1-1)

				# Calculate the difference between the old message and the new message
				inP = bmpslib.mps_inner_product(mps_message, m_matrix[ib][jb], \
					conjB=True)

				err0 = 2-2*inP.real
				err += err0
				S += 1

				# Update the messages list
				m_matrix[ib][jb] = mps_message  # update the ib->jb mps message

				# Update the Block Tensors to which ib sends messages
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

		# The error is the average L_2 distance divided by the total number
		# of coordinates if we stack all messages as one huge vector
		
		err = sqrt(abs(err))/S

		if log:
			print("==> blockbp iter {}: err = {}".format(iter_no,err))

	return m_matrix, err, iter_no

def _blockbp_with_mpi(
	mpi_comm,
	m_list : List[List[MPS]],
	mps_mapping : List[List[List[int]]],
	sedges_list : List[List[Tuple[str, int]]],
	blocks : Dict[str, Tuple[int, int]],
	num_blocks,
	BTN_list,
	damping,
	D_trunc,
	D_trunc2,
	eps,
	delta,
	max_iter,
	log : bool,
) -> Tuple[
	List[List[MPS]],  # m_list
	float,  # err
	int,  # iter_no
]:

	blocks_neighbors = [None]*num_blocks

	for i in range(num_blocks):
		
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
		for i in range(num_blocks):
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
		
		for i in range(num_blocks):
			mpi_comm.send(run_BP, dest=i, tag=6)
		
	#
	# End of BP loop. Get the updated BTNs from the blocks clients,
	# as well as the last outgoing messages
	#

	if log:
		print("\n\n\n\n ========== DONE MPI BP LOOP ===============\n\n\n")
		
		print("Gathering the updated BTNs & outgoing messages from block "
			"clients")
	
	for ib in range(num_blocks):
		(BTN, out_m_list) = mpi_comm.recv(source=ib, tag=7)
		
		BTN_list[ib] = BTN
		neighbors = blocks_neighbors[ib]
		
		for k in range(len(neighbors)):
			jb = neighbors[k]
			m_list[ib][jb] = out_m_list[k]

	return m_list, err, iter_no
				

# ============================================================================ #
#|                            Declared Functions                              |#
# ============================================================================ #

def blockbp(
	T_list : List[np.ndarray], 
	edges_list : List[List[int]], 
	pos_list : List[Tuple[int, ...]], 
	blocks_v_list : List[List[int]], 
	sedges_dict : Dict[str, List[int]], 
	sedges_list : List[List[Tuple[str, int]]], 
	blocks_con_list : List[BlocksConType], 
	PP_blob : Optional[Dict[str, Any]] = None, 
	m_list=None, 
	initial_m_mode : MessageModel = MessageModel.UNIFORM_QUANTUM, 
	D_trunc=None, 
	D_trunc2=None, 
	eps=None, 
	max_iter : int = 100, 
	delta : float =1e-8, 
	damping = None, 
	Lx : float = 1e9, 
	Ly : float = 1e9, 
	mpi_comm=None,
	log:bool=True,
) -> Tuple[
	List[List[MPS]], # m_list
	Dict[str, Any]   # PP_blob
]: 
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
			incomping mps messages. Each vertex is specified by a tuple
			(type,id), where type='v' or type='e' for an original block vertex
			or for an incoming mps vertex. For a type='e' vertex, the id is
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

	# Derive basic info:
	has_mpi = mpi_comm is not None
	num_tensors = len(T_list) 
	num_blocks = len(blocks_v_list)
	
	check_mpi(has_mpi, mpi_comm, num_blocks, log)

	if log:
		print("Number of vertices n={}     Number of blocks nb={}".format(num_tensors, num_blocks))
		

	if PP_blob is None:
		blocks, mps_mapping, mpsD, BTN_list = derive_data_structures( 
			log, num_tensors, num_blocks, edges_list, sedges_list, sedges_dict, 
			pos_list, Lx, Ly, T_list, blocks_v_list, blocks_con_list
		)

	else:
		# So PP_blob is given. Extract its fields and update BTN_list
		blocks = PP_blob['blocks']
		mps_mapping = PP_blob['mps_mapping']
		mpsD = PP_blob['mpsD']
		BTN_list = PP_blob['BTN_list']

		# Update the BTN tensors with tensors from T_list
		for ib in range(num_blocks):

			v_list = blocks_v_list[ib]
			(T1_list, edges1_list, angles1_list, bcon) = BTN_list[ib]

			for i in range(len(v_list)):
				v = v_list[i]
				T1_list[i] = T_list[v]

	# ================================================================
	#
	#                      INITIALIZING THE MAIN LOOP
	#
	# ================================================================

	# If m_list is empty (no initial messages are given), then start with
	# either random MPS messages or messages that represent unifom density
	# matrices or uniform probability dist.
	if m_list is None:
		m_list = initial_messages(initial_m_mode, blocks, sedges_list, mpsD, log, num_blocks)

	if has_mpi:		
		m_list, err, final_iter = _blockbp_with_mpi( 
			mpi_comm, m_list, mps_mapping, sedges_list, blocks, num_blocks, 
			BTN_list, damping, D_trunc, D_trunc2, eps, delta, max_iter, log
		)

	else:
		m_list, err, final_iter = _blockbp_sequential( 
			m_list, mps_mapping, sedges_list, blocks, num_blocks, 
			BTN_list, damping, D_trunc, D_trunc2, eps, delta, max_iter, log
		)

	# Prepare the PP_blob before exiting:
	PP_blob = {}
	PP_blob['sedges_list'] = sedges_list
	PP_blob['blocks'] = blocks
	PP_blob['mps_mapping'] = mps_mapping
	PP_blob['mpsD'] = mpsD
	PP_blob['BTN_list'] = BTN_list
	
	PP_blob['final-error'] = err
	PP_blob['final-iter'] = final_iter

	return m_list, PP_blob



# ============================================================================ #
#                                  main                                        #
# ============================================================================ #

def _main_test():
	from _solve_heisenberg import solve_heisenberg
	solve_heisenberg()

def main():
	_main_test()

if __name__ == "__main__":
	main()