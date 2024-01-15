########################################################################
#
#   bubblecon.py --- A library providing bmpsncon --- an ncon replacement
#                   using the boundary MPS method for 2D tensor networks.
#
#
#   Log:
#   ------
#
#   18-Sep-2021: Changed name to bubblecon (from bmpsncon). Removed
#                the requirement that the root vertex has an external
#                leg.
#
#
#   19-Nov-2021: Changed the text of an error message of mis-match in 
#                the MPS indices vs the in-legs: add the vertex
#                number that is being swallowed
#
#   26-Feb-2022: Add the break_points optional input parameter to 
#                bubblecon.
#
#   28-Apr-2022: Access the bmpslib MPS object via set_site (so that
#             we also update the order)
#
#
#   10-Dec-2022: Added ket_tensors parameter to bubblecon. Removed the
#                ncon dependence (from swallow_T) and replaced it with
#                the native numpy @ operator, which makes the code much
#                faster.
#                
#   14-Dec-2022: Added the functions fuse_T_in_legs, fuse_MPS_legs, 
#                together with the ability to fuse the in-legs in 
#                swallow-T before they are being contracted to the MPS. 
#                This will allow for different recipes of contraction, 
#                some for optimal speed and some for optimal memeory. 
#                Currently, there is an ad-hock recipe which fused the 
#                first 3 legs together and then the rest of the legs.
#
#   5-Feb-2023: Added separate_exp flag to bubblecon. Default = False.
#               if True *and* if resultant TN is a scalar, then return
#               the taple val,exp where the actual result is val*10^exp
#
#               This is useful when the resultant value is either very 
#               large or very small and we want to prevent a float number 
#               overflow.
#
#               To accomplish that we use the a new feature in bmpslib, 
#               and set the flag nr_bulk=True in reduceD(). This makes 
#               the MPS normalized and the overall scale is saved in the
#               MPS class in two variables nr_mantissa, nr_exp
#                
#
#   6-Feb-2023: Fixed a bug in swallow_T that appeared when swallowing
#               a tensor T with no out legs
#
#   10-Nov-2023: Add the swallow_ket_T function which replaces swallow_T
#                when T is a ket tensor.
#
#   14-Jan-2024: Improved code documentation in various places, as well
#                as error & info messages. No actual change to the code.
#
#   14-Jan-2024: Removed the opt='low' functionality from bubblecon, 
#                since it was never really used and did not give any 
#                advantage. Accordingly, removed the tensor_to_MPS_SVD 
#                function.
#
#   ===================================================================
#
#

import numpy as np
import scipy as sp

from scipy.linalg import sqrtm, polar, expm

from numpy.linalg import norm
from numpy import sqrt, tensordot, array, eye, zeros, ones, pi, conj

from libs import bmpslib

from utils.prints import ProgressBar

from _error_types import BubbleConError


#
# --------------------------- fuse_tensor  -----------------------------
#

def fuse_tensor(T):
	"""

		Given a PEPS tensor T of the form [d, D1, D2, ...],
		contract it with its conjugate T^* along the physical leg (bond d),
		and fuse all the matching ket-bra pairs of the virtual legs to a
		single double-layer leg.

		The resultant tensor is of the form: [D1**2, D2**2, D3**2, ...]

	"""

	n = len(T.shape)

	T2 = tensordot(T,conj(T), axes=([0],[0]))

	#
	# Permute the legs:
	# [D1, D2, ..., D1^*, D2^*, ...] ==> [D1, D1^*, D2, D2^*, ...]
	#
	perm = []
	for i in range(n-1):
		perm = perm + [i, i+n-1]

	T2 = T2.transpose(perm)

	#
	# Fuse the ket-bra pairs: [D1, D1^*, D2, D2^*, ...] ==> [D1^2, D2^2, ...]
	#

	dims = [T.shape[i]**2 for i in range(1,n)]

	T2 = T2.reshape(dims)

	return T2



#
# ----------------------- id_tensor --------------------------------
#

def id_tensor(D_left, D_mid, D_right):
	"""

	Takes an ID tensor of shape (N,N) and turns it into an MPS
	tensor of the shape (D_left, D_mid, D_right), where either:

	  (*) N = D_left*D_mid = D_right

	    Or

	  (*) N = D_left = D_mid*D_right

	 The input parameters (D_left, D_mid, D_right) must therefore satisfy
	 one of these conditions.

	 This function is used as a subroutine in the tensor_to_mps function
	 that turns a general tensor into an MPS tensor-network.

	 Parameters:
	 -------------
	 D_left, D_mid, D_right --- the dimensions of the MPS tensor


	Returns:
	--------

	An MPS tensor of the shape [D_left, D_mid, D_right]

	"""

	if D_left==D_mid*D_right:
		T = eye(D_left)
	else:
		T = eye(D_right)

	T = T.reshape([D_left, D_mid, D_right])

	return T



#
# ----------------------- tensor_to_MPS_ID --------------------------------
#

def tensor_to_MPS_ID(T):

	"""

	Takes a general tensor of shape [i_0, i_1, ...., i_{n-1}] and turns
	it into an MPS with physical legs of dimensions [i_0, i_1, ...., i_{n-1}]
	that correspond to the original tensor. The algorithmic idea is from
	the paper https://arxiv.org/abs/2101.04125 of Christopher T. Chubb.

	It does not use any fancy SVD machinary; instead it simply "flattens"
	the original tensor into an MPS shape where the mid tensor is the
	original tensor with some of the left legs fused into one leg and
	some of the right legs into another (and one leg is the middle physical
	leg). This flattening uses a reshaping of the ID tensor.


	Parameters:
	--------------

	T - a general tensor of shape [i_0, ..., i_{n-1}]

	Returns:
	----------
	An MPS object

	"""

	dims = T.shape

	n = len(dims)

	# The total dimension of the tensor = d_0*d_1*...
	totalD = T.size

	mp = bmpslib.mps(n)

	#
	# k_mid is the index of the leg which is in the middle. 
	# The tensor of this leg
	# will carry all the information of T. The other tensors are simply
	# reshaped ID tensors.
	#
	
	k_mid = n//2
	
	#
	# If n is an even number then k_mid can be either n//2 or n//2-1. We
	# choose the option which will give us the lowest bond dimension
	#

	if n%2==0:
		totalD_L = np.prod(dims[:k_mid])
		if totalD_L**2>totalD:
			k_mid -= 1


	#
	# ===== Add the ID tensors on the left
	#

	DL = 1
	for i in range(k_mid):
		Dmid = dims[i]
		DR = DL*Dmid
		A = id_tensor(DL, Dmid, DR)

		mp.set_site(A,i)
		DL *= Dmid


	#
	# ===== Add the (reshaped) T tensor on the middle (k_mid index)
	#

	Dmid = dims[k_mid]
	A = T.reshape([DL, Dmid, totalD//(DL*Dmid)])

	mp.set_site(A, k_mid)


	#
	# ===== Add the ID tensors on the right
	#

	DR = 1
	for i in range(n-1,k_mid,-1):
		Dmid = dims[i]
		DL = DR*Dmid
		A = id_tensor(DL, Dmid, DR)

		mp.set_site(A,i)
		DR *= Dmid

	return mp



#
# ---------------------------- max_bond ----------------------------
#

def max_bond(mp, T, i0, i1, out_legs):
	"""
	
		Given an MPS mp and a tensor T that is going to be swallowed, 
		calculates the maximal bond of the new MPS in the segment where 
		T was swallown. 
		
		This function is used to decide if an MPS has to be compressed 
		before swallowing a tensor (this happens when D_trunc2 is given 
		in bubblecon).
		
		Parameters:
		--------------
		mp 				--- the MPS object
		T  				--- the tensor to be swallown
		i0,i1   	--- the location of the MPS legs with which T is contracted
		out_legs	--- a list of the output legs of the tensor (the legs that
		             are not contracted with the MPS)
		             
		Returns: the maximal bond number of the new lets in the new MPS.
		
	
	"""
	
	DL = mp.A[i0].shape[2]
	DR = mp.A[i1].shape[0]

	if out_legs:
		
		mid_Ds = [T.shape[i] for i in out_legs]
		
		k = len(out_legs)
		
		for i in range(k//2):
			DL *= mid_Ds[i]
			DR *= mid_Ds[k-1-i]
			
	return max(DL,DR)




#
# -------------------------- fuse_T_in_legs ----------------------------
#

def fuse_T_in_legs(T, legs_subsets_list):
	"""
	
	Given a tensor to be swallowed, it fuses some of the in-legs
	together.
	
	Input Parameters:
	------------------
	
	T --- The tensor. It must be of the form 
	      [in_leg_1, in_leg_2, ..., in_leg_k, out_leg1, out_leg2, ...]
	      
	legs_subsets_list --- A list of lists that specifies a partition of
	                      the in legs. Each list is a subset of 
	                      neighboring in-legs to be fused together.
	                      For example, [ [0,1],[2],[3,4]] will fuse
	                      0,1 legs together and [3,4] legs together.
	                      
	                      
	OUTPUT:
	-------
	
	The fused tensor.

	"""
	
	T_dims = list(T.shape)

	#
	# We calculate the dimensions of the fused legs, and then use a singe
	# reshape to fuse them.
	#
	
	Ds = []
	total_in_legs_no=0
	for leg_subset in legs_subsets_list:
		D = np.prod([T_dims[i] for i in leg_subset])
		Ds.append(D)
		total_in_legs_no += len(leg_subset)
		
	Ds = Ds + T_dims[total_in_legs_no:]
		
	return T.reshape(Ds)
		
	
#
# -------------------------- fuse_MPS_legs ----------------------------
#

def fuse_MPS_legs(mp, legs_subsets_list, i0):
	"""
	
	Fuses the legs of an MPS segment that is about to swallow a tensor. 
	This function is to be used in conjunction with fuse_Tin.
	
	Input Parameters:
	------------------
	
	mp --- the MPS
	
	legs_subsets_list --- A list of lists that specifies a partition of
	                      the in legs. Each list is a subset of 
	                      neighboring in-legs to be fused together.
	                      For example, [ [0,1],[2],[3,4]] will fuse
	                      i0,i0+1 legs together and [i0+3,i0+4] legs
	                      together.
	
	
	i0 --- The left most leg with which the tensor is going to be 
	       contracted.
	       
	OUTPUT:
	--------
	
	A fused MPS. The returend MPS only contains the tensors in [i0,i1],
	i.e., the tensors that participate in the swallowing.
	
	"""
	
	k = len(legs_subsets_list) # How many fused legs
	
	fused_mp = bmpslib.mps(k)
	
	#
	# Go over the legs subsets and fuse each subset of legs
	#
	for ell in range(k):
		legs = legs_subsets_list[ell]
		
		A = mp.A[legs[0]+i0]

		if len(legs)>1:
		#
		# There's only a reason to swallow if there is two or more legs
		#
			DL, Dmid, DR = A.shape[0], A.shape[1], A.shape[2]
			
			A = A.reshape([DL*Dmid, DR])
			
			for i in legs[1:]:
				dims = mp.A[i+i0].shape
				DR = dims[2]
				A = A @ mp.A[i+i0].reshape([dims[0], dims[1]*dims[2]])
				A = A.reshape([A.size//dims[2], dims[2]])
				
			A = A.reshape([DL, A.size//(DL*DR), DR])
		
		fused_mp.set_site(A,ell)
		
	return fused_mp
			
	
	




#
# ---------------------------- swallow_ket_T ----------------------------
#

def swallow_ket_T(mp, ket_T, i0, i1, in_legs, out_legs, \
	D_trunc=None, eps=None):

	"""

	Contracts a ket tensor with an MPS, and turns the resulting TN back into
	an MPS. This way, the tensor is "swallowed" by the MPS.
	
	This function is equivalent to swallow_T only that here T is a ket 
	tensor instead of a ket-bra tensor. It is useful when T has a large
	bond dimension D, and so contracting first the ket to the MPS and 
	only then the bra is much more efficient (memory and complexity) than
	first contracting the ket-bra and then swallowing the combined tensor.

	The tensor has to be contracted with a *contiguous* segment of
	physical legs in the MPS

	NOTE: calculation is done directly on the input MPS. So there is no
	      need for output MPS

	Input parameters:
	------------------

	mp --- The MPS (given as an MPS object from bmpslib.py).

	       This MPS also holds the resultant merged MPS.

	ket_T --- The ket tensor to be swallowed. We assume that its first
	          leg is the physical leg

	i0, i1 --- Specify the contiguous segment of physical legs in the MPS
	           to be contracted with T. This includes all legs
	           i0,i0+1,..., i1

	in_legs --- the indices of the input legs in T to be contracted with
	            the MPS. The order here matters, as it has correspond to
	            the order of the legs in [i0,i1]. The dimension of each
	            input leg must be equal to the dimension of the
	            corresponding physical in the MPS. 

	out_legs --- the indices of the output legs in T, ordered in the way
	             they will appear in the output MPS. (this list can be
	             empty)

  NOTE: The indices inside in_legs and out_legs are defined as if 
        we're in the regular swallow_T case, and there is no physical
        leg. So if in_legs = [0,2,3], this means that we actually
        refer to legs [1,3,4] in the ket_tensor T.

	OUTPUT:    NONE
	---------------
"""

	log = False
	#
	# Number of legs in the ket tensor that we swallow. 
	#
	# Note, we expect: n_legs = n_in_legs + n_out_legs + 1
	#
	
	n_legs = len(ket_T.shape)   # no. of legs, including the physical leg.
	n_in_legs = len(in_legs)    # no. of in legs
	n_out_legs = len(out_legs)  # no. of out legs
	
	#
	# ===================================================================
	# STEP 1: prepare T for swallowing. 
	#         (*) Move the phys ket leg to the end of the tensor
	#         (*) Premute the tensor legs according to in_legs, out_legs
	#         (*) Fuse all the out legs (including the phys leg, which 
	#             is the last)
	# ===================================================================
	#
	
	#
	# First, move the physical leg to the end of the tensor and add it 
	# to the out legs list
	#
	
	T = ket_T.transpose(list(range(1,n_legs)) + [0])
	
	out_legs = out_legs + [n_legs-1] # the physical leg is now the last leg
	

	#
	# Permute T indices so that their order matches in_legs, out_legs
	#

	if log:
		print("\n\n")
		print("Entring swallow_ket_T with D_trunc={}".format(D_trunc))
		print("=======================================\n\n")

		print(f"T has {len(T.shape)} legs: 1 physical + {len(in_legs)} "\
			f"input + {len(out_legs)} output. MPS swallowing range: "\
			f" i0={i0}  i1={i1}")
		inl = [T.shape[i] for i in in_legs]
		outl = [T.shape[i] for i in out_legs]
		print("Dims in-ket-legs: ", inl, "   Dims of out-ket-legs: ", outl )
		
		print("in ket legs: ", in_legs)
		print("out ket legs: ", out_legs)
		
		print(f"Permuting to: {in_legs + out_legs} (last is the phys leg)")

	T0 = T.transpose(in_legs+out_legs)

	#
	# Save a copy of the current bra tensor, which will be used later.
	#
	
	bra_T0 = conj(T0) 

	#
	# The dims of the out legs of T0 (including the last one, which is the
	# physical leg)
	#

	out_legs_shape = list(T0.shape[len(in_legs):])
	 
	#
	# Fuse all ket out-legs together 
	#

	D_out = np.prod(out_legs_shape)
	T0 = T0.reshape(list(T0.shape[:len(in_legs)])+[D_out])
	
	#
	# T0 is now of the following form:
	# 0, 1, 2, n_in_legs-1 --- the in-ket legs
	# n_in_legs            --- the fused out legs (including the phys leg)
	#
	
	
	
	#
	# 1. Start with the left-most tensor in the MPS segment. Seperate its
	#    mid leg to a ket and a bra.
	#

	sh = mp.A[i0].shape
	mpsT = mp.A[i0].reshape([sh[0], T0.shape[0], T0.shape[0], sh[2]])
		

	#
	# ===================================================================
	# STEP 2: Contract T0 with the first MPS tensor along its ket leg
	# ===================================================================
	#
	
	A = tensordot(mpsT, T0, axes=([1], [0]))
	
	#
	# Transpose the legs of A to this order:
	#
	# 1. MPS left-leg
	# 2. MPS right-leg
	# 3. Remaining legs of T
	# 4. Out-leg 
	# 5. bra MPS leg
	#
	
	A = A.transpose([0,2] + list(range(3, 3+n_in_legs)) + [1])
	
	#
	# ===================================================================
	# STEP 3: Loop over the rest of the MPS tensors and contract their
	#         ket part to the ket part of T0 (which is now part of A)
	#      
	# ===================================================================
	#
		
	k = len(A.shape)
	
	for i in range(i0+1, i1+1):
		
		#
		# separate the to ket-bra the mid leg of the MPS. The dim of each
		# ket/bra should be the dim of the corresponding leg in A --- leg
		# no. 2
		#
		sh = mp.A[i].shape
		
		mpsT = mp.A[i].reshape([sh[0], A.shape[2], A.shape[2], sh[2]])
		
		#
		# contract it to A along the left leg and the ket leg 
		#
		A = tensordot(mpsT, A, axes=([0,1], [1,2]))
		
		#
		# Transpose A to the following order:
		# 1. MPS left-leg
		# 2. MPS right-leg
		# 3. Remaining legs of T
		# 4. Fused-out-legs
		# 5. Previous bra legs
		# 6. This MPS bra leg
				
		A = A.transpose([2,1] + list(range(3, k)) + [0])
		

	#
	# ===================================================================
	# STEP 4: To the resulting contraction of the MPS and the ket-T
	#         contract now the bra-T. After that unfuse the ket-out
	#         legs and refuse them with their corresponding bra legs
	# ===================================================================
	#
	
	#
	# At this point all the MPS tensors were contracted to T and formed
	# one huge tensor of the following structure:
	# 0.   MPS left-leg
	# 1.   MPS right-leg
	# 2.   Fused out-legs of T (including phy leg)
	# 3... bra legs of the MPS (according to their original order)
	#
	# We now need to contract the bra legs with the bra tensor conj(T0)
	#
	
	A = tensordot(A, bra_T0, \
		axes=(list(range(3, 3+n_in_legs)), list(range(n_in_legs))))
	
	#
	# Shape of resultant A:
	# 0. MPS left-leg
	# 1. MPS right-leg
	# 2. Fused ket out legs of T (including ket phy leg)
	# 3. Unfused bra out legs of T (including bra phy leg)
	#
	
	#
	# Unfuse the out legs (both ket and bra)
	#
	
	sh = A.shape
	A = A.reshape([sh[0], sh[1]] + out_legs_shape + out_legs_shape)
	
	
	#
	# Shape of A:
	# ============
	#
	# 0                             --- MPS left-leg
	# 1                             --- MPS right-leg
	# 2 -> 1+n_out_legs             --- ket out legs
	# 2+n_out_legs                  --- ket phys leg 
	# 3+n_out_legs-> 2+2*n_out_legs --- bra out legs
	# 3+2*n_out_legs                --- bra phys leg
	#
	
	#
	# Contract the phys ket leg to the phys bra leg
	#
	
	A = trace(A, axis1=2+n_out_legs, axis2=3+2*n_out_legs)

	#
	# Shape of A:
	# ============
	#
	# 0                             --- MPS left-leg
	# 1                             --- MPS right-leg
	# 2 -> 1+n_out_legs             --- ket out legs
	# 2+n_out_legs-> 1+2*n_out_legs --- bra out legs
	#
		
	#
	# Now move each out bra leg next to its out ket leg, and fust them
	# together
	#
	
	perm = [0,1]
	
	sh = A.shape
	new_sh = [sh[0], sh[1]]
	
	for i in range(n_out_legs):
		perm = perm + [2+i, 2+n_out_legs+i]
		new_sh.append(out_legs_shape[i]**2)
		
	
	A = A.transpose(perm)
	A = A.reshape(new_sh)

	#
	# Shape of A:
	# ============
	#
	# 0                             --- MPS left-leg
	# 1                             --- MPS right-leg
	# 2 -> 1+n_out_legs             --- ket-bra out legs
	#
	
	#
	# Move the MPS right-leg to the end of the tensor
	#
	
	perm = [0] + list(range(2, 2+n_out_legs)) + [1]
	
	A = A.transpose(perm)


	#
	# ================================================================
	#  STEP 5:  Reshape the resulting tensor into an MPS using the
	#           tensor_to_mps function
	# ================================================================
	#


	if n_out_legs==0:
		#
		# If there are no out-legs then A has just two legs --- one to
		# the left and one to the right. We can therefore absorb it
		# in either the MPS tensor to its left or the MPS tensor to its
		# right. Its better to absorb it into the MPS tensor with the
		# highest D so that the new MPS tensor will have a smaller D.
		#

		if log:
			print("T has no out-legs so A has only two leg! shape: ", A.shape)

		if i0==0 and i1==mp.N-1:

			# In such case A is just a scalar shaped as a tensor[1,1]
		  # In this case, we create an MPS with a single trivial tensor
		  # of the shape [1,1,1]

		  mp.A = [A.reshape([1,1,1])]
		  mp.Corder=[None]
		  mp.N = 1
		  return

		#
		# Explicitly handle the two edge cases when i0=0 or i1=N-1
		#
		if i0==0:
			mp.A = mp.A[i1+1:]
			mp.Corder=mp.Corder[i1+1:]
			mp.N = len(mp.A)
			mp.set_site(tensordot(A, mp.A[0], axes=([1],[0])), 0)
			return

		if i1==mp.N-1:
			mp.A = mp.A[:i0]
			mp.Corder = mp.Corder[:i0]
			mp.N = len(mp.A)
			mp.set_site(tensordot(mp.A[-1], A, axes=([2],[0])), -1)
			return

		#
		# So its a regular case in the bulk. Then absorb A along the leg
		# with the largest bond dimension.
		#

		mp.A = mp.A[:i0] + mp.A[i1+1:]
		mp.Corder = mp.Corder[:i0] + mp.Corder[i1+1:]
		
		mp.N = len(mp.A)
		
		if A.shape[0]<A.shape[1]:
			#
			# Absorb to the right
			#
			mp.set_site(tensordot(A, mp.A[i0], axes=([1],[0])), i0)
		else:
			#
			# Absorb to the left
			#
			mp.set_site(tensordot(mp.A[i0-1], A, axes=([2],[0])), i0-1)

		return

	#
	# If there are out_legs, then we can turn A into a small MPS and
	# then combine it with the main MPS.
	#

	if D_trunc is None:
		A_mp = tensor_to_MPS_ID(A)
	else:
		A_mp = tensor_to_MPS_SVD(A, D_trunc, eps)

	#
	# The left-most and right-most legs of A_mp are of 
	# dimension 1, and the left most and right most physical legs are 
	# of the dimension of the left/right indices of A (which is the 
	# contraction of T and the MPS segment). So we need to 'chop off'
	# the left-most and right-most matrices in the sub-MPS, by absorbing
	# them into the MPS matrices next to them. This way, the resultant 
	# MPS will have two OPEN legs to the left/right with the right 
	# dimensionality.
	
	if log:
		print("Orig A_mp shape: ", A_mp.mps_shape())

	#
	# Absorb the left-most matrix A_0. Recal that it is of the form
	# [1,D_L, D]. So turn it into a matrix [D_L, D] and abosorb it
	# into A_1, so that A_1 will be of the form [D_L, XXX, XXX]
	#
	AL = A_mp.A[0]
	AL = AL.reshape(AL.shape[1], AL.shape[2])  
	A_mp.set_site(tensordot(AL, A_mp.A[1], axes=([1],[0])), 1)

	#
	# Absorb the right-most matrix A_{N-1}. Recal that it is of the form
	# [D, D_R, 1]. So turn it into a matrix [D,D_R] and abosorb it
	# into A_{N-2}, so that A_{N-2} will be of the form [XXX,XXX,D_R]
	#
	AR = A_mp.A[A_mp.N-1]
	AR = AR.reshape(AR.shape[0], AR.shape[1])
	A_mp.set_site(tensordot(A_mp.A[A_mp.N-2], AR, axes=([2],[0])), A_mp.N-2)


	A_mp.A = A_mp.A[1:(A_mp.N-1)]
	A_mp.Corder = A_mp.Corder[1:(A_mp.N-1)]
	A_mp.N -= 2

	if log:
		print("New A_mp shape: ", A_mp.mps_shape())



	#
	# ================================================================
	#  STEP 3:  Merge the two MPSs into one
	# ================================================================
	#


	#
	# Merge mp_A into mp
	#

	if log:
		print("\n\n")
		print("Merging:")
		print("Original mp shape: ", mp.mps_shape())

	mp.A = mp.A[0:i0]+A_mp.A + mp.A[i1+1:]
	mp.Corder = mp.Corder[0:i0]+A_mp.Corder + mp.Corder[i1+1:]

	mp.N = mp.N - n_in_legs + n_out_legs

	if log:
		print("New mp shape: ", mp.mps_shape())






#
# ---------------------------- swallow_T ----------------------------
#

def swallow_T(mp, T, i0, i1, in_legs, out_legs, D_trunc=None, eps=None):

	"""

	Contracts a tensor with an MPS, and turns the resulting TN back into
	an MPS. This way, the tensor is "swallowed" by the MPS.

	The tensor has to be contracted with a *contiguous* segment of
	physical legs in the MPS

	NOTE: calculation is done directly on the input MPS. So there is no
	      need for output MPS

	Input parameters:
	------------------

	mp --- The MPS (given as an MPS object from bmpslib.py).

	       This MPS also holds the resultant merged MPS.

	T  --- The tensor to be swallowed

	i0, i1 --- Specify the contiguous segment of physical legs in the MPS
	           to be contracted with T. This includes all legs
	           i0,i0+1,..., i1

	in_legs --- the indices of the input legs in T to be contracted with
	            the MPS. The order here matters, as it has correspond to
	            the order of the legs in [i0,i1]. The dimension of each
	            input leg must be equal to the dimension of the
	            corresponding physical in the MPS.

	out_legs --- the indices of the output legs in T, ordered in the way
	             they will appear in the output MPS. (this list can be
	             empty)


	OUTPUT:    NONE
	---------------

	"""

	log = False


	#
	# ================================================================
	#  STEP 1:  Turn the MPS segment that participates in the 
	#           contraction into a matrix. Do the same for the tensor we
	#           swallow. Then contract them using the regular matrix
	#           multiplication of numpy (when the dimension over which
	#           we contract is not too large. Otherwise, we contract
	#           a group of legs by a group of legs).
	# ================================================================
	#

	#
	# First permute T indices so that their order matches in_legs, out_legs
	#

	if log:
		print("\n\n")
		print("Entring swallow_T with D_trunc={}".format(D_trunc))
		print("=======================================\n\n")

		print("T has {} legs: {} input + {} output. i0={}  i1={}".format(\
			len(T.shape), len(in_legs), len(out_legs), i0, i1))
		inl = [T.shape[i] for i in in_legs]
		outl = [T.shape[i] for i in out_legs]
		print("Dims in-legs: ", inl, "     Dims of out-legs: ", outl )
		
		print("in legs: ", in_legs)
		print("out legs: ", out_legs)
		
		print("Permuting to: ", in_legs + out_legs)

	T0 = T.transpose(in_legs+out_legs)
	

	#
	# The dims of the out legs of T0
	#
	
	out_legs_shape = T0.shape[len(in_legs):]
	

	#
	# If there are any out-legs then fuse them together
	#
	
	if out_legs:
		D_out = np.prod(out_legs_shape)
		T0 = T0.reshape(list(T0.shape[:len(in_legs)])+[D_out])
	
	#
	# Fuse the in legs into subsets. Currently we use an ad-hock
	# method (to be changed later)
	#
	
	k=len(in_legs)
	
	if k<=3 or T0.size < 500000:
		#
		# If we have 1->3 in legs, or the tensor is not too big --- don't 
		# bother --- just fuse them all together.
		#
		
		fusing_subsets = [list(range(k))]
		
	else:
		#
		# If 4 or more, fuse the first 3 together, and the rest of the 
		# in legs together.
		#
		
		fusing_subsets = [ [0,1,2], list(range(3,k))]
		
	
		
	#
	# Number of fused legs
	#
	fused_legs_n = len(fusing_subsets)

	#
	# Now fuse the MPS segment and T0 in a similar way
	#
	
	fused_mp = fuse_MPS_legs(mp, fusing_subsets, i0)
	T0 = fuse_T_in_legs(T0, fusing_subsets)
	
	
	# 
	# Save the Left/Right dimensions of the fused MPS for later use.
	#
	DL = fused_mp.A[0].shape[0]              # Left-most  virtual leg
	DR = fused_mp.A[fused_legs_n-1].shape[2] # Right-most virtual leg
		
	
	#
	# 1. contract the first in-leg of the tensor to the first fused 
	#    MPS tensor 
	#

	mpsT = fused_mp.A[0]

	# The dims of the legs of the first MPS tensor
	
	Dmid1, DR1 = mpsT.shape[1], mpsT.shape[2]
	
	# Turn the first MPS tensor into a matrix
	mpsT = mpsT.transpose([0,2,1]) # ===> mpsT = [DL, DR1, Dmid]
	mpsT = mpsT.reshape([DL*DR1, Dmid1])
	
	# The tensor we swallow, T0, is given as T[in-legs, out-legs]. 
	# we turn it into a matrix [in-leg1, rest-of-legs] to be contracted
	# with the mid1 leg of mpsT
	
	T0_shape = T0.shape
	DT = T0.size
	
	T0 = T0.reshape([T0_shape[0], DT//T0_shape[0]])
	
	# Contract the first leg
	
	A = mpsT @ T0 
	
	#
	# A has the shape [DL+DR1, D_2+D_3+...+D_k+legs-out]
	#
	
	
	#
	# Separate the first DL+DR1 leg of A so that it has the shape
	#
	#     [DL, DR1, (rest of legs-in) + legs-out]  
	#
	# But if there are no out-legs and no more legs-in, then reshape
	# it into [DL, DR1] = [DL, DR]
	#
	
	if fused_legs_n==1 and not out_legs:
		A = A.reshape([DL, DR1])
	else:
		A = A.reshape([DL, DR1, A.size//(DL*DR1)])  
		
	#
	# We continue only if there are more legs to contract.
	#
	
	
	if fused_legs_n>1:

		# Move DL to the end of the tensor
		A = A.transpose([1,2,0])  

		# Now A is of the form [DR1, legs-in-legs-out, DL]
		
		# Fuse DL with the legs-in-legs-out
		
		A = A.reshape([DR1, A.size//DR1])
		
		# Now A is of the form [DR1, legs-in-legs-out-DL]

	
		for i in range(1, fused_legs_n):

			# We now fuse the i'th fused-MPS tensor

			mpsT = fused_mp.A[i]
			
			DL_i, Dmid_i, DR_i = mpsT.shape[0], mpsT.shape[1], mpsT.shape[2]

			# At this point A is of the form [DL_i, legs-in-legs-out-DL]

			
			# Unfuse the next leg from legs-in-legs-out
							
			A = A.reshape([DL_i*Dmid_i, A.size//(DL_i*Dmid_i)])
			
			mpsT = mpsT.transpose([2,0,1]) # mpsT = [DR_i, DL_i, Dmid_i]
			mpsT = mpsT.reshape([DR_i, DL_i*Dmid_i])
			
			# contract
			A = mpsT @ A
			
			# Now A is of the form [DR_i, legs-in-legs-out-DL]
		
		
		#
		# After the main contraction loop A is of the form
		# [DR, fused-legs-out+DL]. We want to move DL back to the start 
		# to be the first leg
		#
		
		if out_legs:
			A = A.reshape([DR, A.size//(DR*DL), DL])
			A = A.transpose([2,0,1])
			#
			# Now A is of the form [DL, DR, fused-legs-out]. 
			#
		else:
			#
			# if there are no out legs, then A is in the form
			# [DR, DL] ==> so we just permute the order of the legs 
			
			A = A.transpose([1,0])
		
	
	if log:
		print("=> Contraction done")
	

	#
	# If there are out legs then unfuse them and send DR to be the right
	# most leg
	#
	if out_legs:
		A = A.reshape([DL, DR] + list(out_legs_shape))
		A = A.transpose([0] + list(range(2,2+len(out_legs_shape))) + [1])
	

	#
	# ================================================================
	#  STEP 2:  Reshape the resulting tensor into an MPS using the
	#           tensor_to_mps function
	# ================================================================
	#


	if not out_legs:
		#
		# If there are no out-legs then A has just two legs --- one to
		# the left and one to the right. We can therefore absorb it
		# in either the MPS tensor to its left or the MPS tensor to its
		# right. Its better to absorb it into the MPS tensor with the
		# highest D so that the new MPS tensor will have a smaller D.
		#

		if log:
			print("T has no out-legs so A has only two leg! shape: ", A.shape)

		if i0==0 and i1==mp.N-1:

			# In such case A is just a scalar shaped as a tensor[1,1]
		  # In this case, we create an MPS with a single trivial tensor
		  # of the shape [1,1,1]

		  mp.A = [A.reshape([1,1,1])]
		  mp.Corder=[None]
		  mp.N = 1
		  return

		#
		# Explicitly handle the two edge cases when i0=0 or i1=N-1
		#
		if i0==0:
			mp.A = mp.A[i1+1:]
			mp.Corder=mp.Corder[i1+1:]
			mp.N = len(mp.A)
			mp.set_site(tensordot(A, mp.A[0], axes=([1],[0])), 0)
			return

		if i1==mp.N-1:
			mp.A = mp.A[:i0]
			mp.Corder = mp.Corder[:i0]
			mp.N = len(mp.A)
			mp.set_site(tensordot(mp.A[-1], A, axes=([2],[0])), -1)
			return

		#
		# So its a regular case in the bulk. Then absorb A along the leg
		# with the largest bond dimension.
		#

		mp.A = mp.A[:i0] + mp.A[i1+1:]
		mp.Corder = mp.Corder[:i0] + mp.Corder[i1+1:]
		
		mp.N = len(mp.A)
		
		if A.shape[0]<A.shape[1]:
			#
			# Absorb to the right
			#
			mp.set_site(tensordot(A, mp.A[i0], axes=([1],[0])), i0)
		else:
			#
			# Absorb to the left
			#
			mp.set_site(tensordot(mp.A[i0-1], A, axes=([2],[0])), i0-1)

		return


	#
	# If there are out_legs, then we can turn A into a small MPS and
	# then combine it with the main MPS.
	#

	if D_trunc is None:
		A_mp = tensor_to_MPS_ID(A)
	else:
		A_mp = tensor_to_MPS_SVD(A, D_trunc, eps)

	#
	# The left-most and right-most legs of A_mp are of 
	# dimension 1, and the left most and right most physical legs are 
	# of the dimension of the left/right indices of A (which is the 
	# contraction of T and the MPS segment). So we need to 'chop off'
	# the left-most and right-most matrices in the sub-MPS, by absorbing
	# them into the MPS matrices next to them. This way, the resultant 
	# MPS will have two OPEN legs to the left/right with the right 
	# dimensionality.
	
	if log:
		print("Orig A_mp shape: ", A_mp.mps_shape())

	#
	# Absorb the left-most matrix A_0. Recal that it is of the form
	# [1,D_L, D]. So turn it into a matrix [D_L, D] and abosorb it
	# into A_1, so that A_1 will be of the form [D_L, XXX, XXX]
	#
	AL = A_mp.A[0]
	AL = AL.reshape(AL.shape[1], AL.shape[2])  
	A_mp.set_site(tensordot(AL, A_mp.A[1], axes=([1],[0])), 1)

	#
	# Absorb the right-most matrix A_{N-1}. Recal that it is of the form
	# [D, D_R, 1]. So turn it into a matrix [D,D_R] and abosorb it
	# into A_{N-2}, so that A_{N-2} will be of the form [XXX,XXX,D_R]
	#
	AR = A_mp.A[A_mp.N-1]
	AR = AR.reshape(AR.shape[0], AR.shape[1])
	A_mp.set_site(tensordot(A_mp.A[A_mp.N-2], AR, axes=([2],[0])), A_mp.N-2)


	A_mp.A = A_mp.A[1:(A_mp.N-1)]
	A_mp.Corder = A_mp.Corder[1:(A_mp.N-1)]
	A_mp.N -= 2

	if log:
		print("New A_mp shape: ", A_mp.mps_shape())



	#
	# ================================================================
	#  STEP 3:  Merge the two MPSs into one
	# ================================================================
	#


	#
	# Merge mp_A into mp
	#

	if log:
		print("\n\n")
		print("Merging:")
		print("Original mp shape: ", mp.mps_shape())

	mp.A = mp.A[0:i0]+A_mp.A + mp.A[i1+1:]
	mp.Corder = mp.Corder[0:i0]+A_mp.Corder + mp.Corder[i1+1:]

	mp.N = mp.N - len(in_legs) + len(out_legs)

	if log:
		print("New mp shape: ", mp.mps_shape())






#
# --------------------------- bubblecon  -------------------------------
#

def bubblecon(T_list, edges_list, angles_list, bubble_angle,\
	swallow_order, D_trunc=None, D_trunc2=None, eps=None, opt='high', \
	break_points=[], ket_tensors=None, separate_exp=False):

	"""
	
	Given an open 2D tensor network, this function contracts it and 
	returns and MPS that approximates it. The physical legs of the MPS 
	correspond to the open edges of the original tensor network.
	
	The function uses an adaptive version of the boundary-MPS algorithm
	to calculate the resulting MPS. As such, one can use a maximal bond
	and singular-value thresholds to perform a truncation during the 
	contraction, thereby keeping the memory consumption from exploding.
	
	Input Parameters
	-----------------
	
	T_list --- 	A list [T_0,T_1,T_2, ...] of the tensors that participate in 
							the TN
							
	edges_list --- A list of the edges that are connected to each tensor.
	               This is a list of lists. For each tensor 
	               T_{i_0, i_1, ..., i_{k-1}} we have a list 
	               [e_0, e_1, ..., e_{k-1}], where the e_j are labels 
	               that define the edges (legs). The e_j labels are either
	               positive or negative integers: positive integers are
	               internal legs and negative are external legs.
	                   	                   
	angles_list --- A list of lists. For each tensor, we have a list 
									of angles that describe the angle on the plane of 
									each leg. The angles are numbers in [0,2\pi).
									
	bubble_angle ---	A number [0,2\pi) that describes the initial 
										orientation of the bubble as it swallows 
										the first vertex. Imagine the initial bubble as a 
										very sharp arrow pointing in some direction \theta
										as it swallows the root vertex. Then 
										bubble_angle=\theta. Swallowing the first vertex 
										turns it into an MPS. The legs on that MPS are sorted
										according to the angle they form with the imaginary 
										arrow.
	
	swallow_order ---	A list of vertices which the bubble is swallowing.
										It can also be just a single vertex, in which case
										this is the root vertex. It can also be omitted, 
										in which case the root vertex is taken to be 0.
										
										If not vertices (except for the root vertex) are
										given, the TN is swallowed by starting from the
										root vertex, and then going over the current MPS 
										from left to right and swalling the first tensor
										we encouter.
										
	D_trunc, D_trunc2	---	Truncation bonds. If omitted, no truncation is 
										done. If only D_trunc is given then truncation
										to D_trunc is done after every swallowing of a tensor.
										
										If also D_trunc2>D_trunc is given then truncation
										to D_trunc is done *before* swallowing a tensor 
										whenever the largest bond dimension in the MPS is
										larger than D_trunc2. This enables higher precision
										in several situations --- but it highly depends on
										the swallowing order.
										
										The parameter D_trunc2 is only considered when 
										opt='high'.
	
	
	eps	--- Singular value truncation threshold. Keep only singular values
	        greater than or equal to eps. If omitted, all singular values
	        are taken.
	        
	opt	---	Optimization level. Disabled. Only accepts opt='high'
					
					
	break_points --- An optional list of steps *after* which a copy of the
	                 current MPS will be saved and outputed later. For 
	                 example, if break_points=[1,3] then the output of the
	                 function will be a list of 3 MPSs: the bubble MPS 
	                 that is obtained after the contraction of the second 
	                 tensor (idx=1), the bubble MPS after the contraction
	                 of the 4th tensor (idx=3) and the bubble MPS at the 
	                 end of the contraction.
	                 
	                 If omitted (break_points=[]), then only the final
	                 bubble MPS is returned.
	                 
	ket_tensors = None  --- An optional lists of boolean variables that
	                        indicate which of the corresponding tensors is
	                        given as ket tensors (ket_tensors[i]=True),
	                        and which as a ket-bra tensor 
	                        (ket_tensors[i]=False).
	                        
	                        ket tensors are the usal PEPS tensors, where
	                        the first leg is the physical leg. 
	                        
	                        ket-bra tensors are the contraction of a 
	                        ket tensor with its bra along the physical leg
	                        which results in squaring of the dimension of
	                        all the remaining virtual legs. 
	                        
	                        It might be useful to use ket tensors when
	                        their ket-bra takes too much memory. Then
	                        the contraction of the physical legs is done
	                        inside bubblecon, and this can lead to a much
	                        smaller memory footprint.
	                        
	                        If not specified, then all tensors are assumed
	                        to be ket-bra tensors.
	                        
	separate_exp = False --- If True, and if the resulting TN is a scalar,
	                         then return the result as a taple (val, exp)
	                         so that the actual result is val*10^exp
	                         
	                         This is useful when the result is either 
	                         very large or very close to 0. Separating
	                         it into exp and val then prevents overflow. 
	                        
		

	"""


	log=False  # Whether or not to print debuging messages
	
	if opt != 'high':
		print("bubblecon error: opt parameter can only be set to 'high'")
		exit(1)

	n=len(T_list)  # How many vertices are in the TN
	
	#
	# If ket_tensors is not specified, we assume all tensors are 
	# ket-bra tensors (double-layer PEPS with no physical legs).
	#
	if ket_tensors is None:
		ket_tensors = [False]*n
	
	#
	# First, create a dictonary in which the edges are the keys and their
	# pair of veritces is the value. For internal edges, the value of the 
	# dictonary is (i,j), where i,j are the vertices connected by it. 
	# For negative edges it is (i,i).
	#
	
	vertices = {}
	
	for i in range(n):
		i_edges = edges_list[i]
		for edge in i_edges:
						
			if edge in vertices:
				(j1,j2) = vertices[edge]
				vertices[edge] = (i,j1)
			else:
				vertices[edge] = (i,i)
				
				
	if log:
		print("\n\n\n\nvertices of dictionary:")
		print(vertices)

	root_id = swallow_order[0]

	if log:
		print("\n")
		print("root_id is ", root_id)
		print("\n")

	#
	# Transpose the indices of T[root_id] according to their relative
	# angle with the bubble swallowing angle in a clockwise fashion.
	#

	root_angles = array(angles_list[root_id])
	root_edges = edges_list[root_id]

	k = len(root_edges)

	if log:
		print("root edges: ", root_edges)
		print("root angles: ", root_angles)
		print("bubble angle: ", bubble_angle)

	#
	# Calculate the rotated_angles array. These are the angles of the
	# different legs with respect to the inverse of bubble angle. For
	# example, if bubble angle=0 and our leg has angle=0.8\pi then
	# the inverse of the bubble angle is \pi and the rotated angle is 
	# +0.2\pi .
	#
	rotated_angles = (bubble_angle + pi - root_angles) % (2*pi)

	if log:
		print("rotated angles: ", rotated_angles)

	#
	# Now re-arrange the indices of the root tensor according to the 
	# rotated angles. This way, the leg that has the smallest angle with
	# the inverse of the bubble angle comes first on the leg, 
	# and so forth.
	#
	L = [(rotated_angles[i],i, root_edges[i]) for i in range(k)]
	L.sort()
	sorted_angles, permutation, sorted_edges = zip(*L)

	if log:
		print("root tensor permutation: ", permutation)
		print("sorted angles: ", sorted_angles)
		print("sorted edges: ", sorted_edges)

	#
	# Prepare the initial tensor. If it is a ket tensor, turn 
	# it into a ket-bra tensor by fusing its physical leg.
	#
	
	T_root = T_list[root_id]
	
	if ket_tensors[root_id]:
		
		if log:
			print("root tensor is ket. Turning it into ket-bra")
			
		T_root = fuse_tensor(T_list[root_id])
				
	T_root = T_root.transpose(permutation)

	#
	# Turn the root tensor it into an MPS
	#

	mp = tensor_to_MPS_ID(T_root)
	
	if D_trunc2 is None and D_trunc is not None:
		mp.reduceD(D_trunc, eps, nr_bulk=True)
			
		
	#
	# Define the swallowed_veritces set 
	#

	swallowed_vertices = {root_id}

	#
	# Define the mp_edges_list which bookkeeps the edge of every leg 
	# in the MPS. It is used to know which legs of the MPS should be 
	# contracted with the swallen tensor
	#

	mp_edges_list = list(sorted_edges)


	#
	# ===================================================================
	#            MAIN LOOP: swallow the rest of the tensors
	# ===================================================================
	#

	if log:
		print("\n\n\n")
		print("================= START OF MAIN LOOP ===================\n")

	more_tensors_to_swallow = True

	#
	# l is a running index on the swallow_order list, which points to the
	# vertex we just swallowed.
	#
	
	l=0
	
	mp_list = []

	while more_tensors_to_swallow:
		
		#
		# See if we reached a break point, and in such case add the 
		# current bubble MPS to an output list.
		#
		if l in break_points:
			mp_list.append(mp.copy())
			if log:
				print("")
				print(f"=> Break Point {l}: adding a copy of the current MPS "\
					"to the output MPS list")
				print("")
				
		
		

		# 
		# Now go to swallow the next vertex
		#
		
		l += 1
		v = swallow_order[l]

		more_tensors_to_swallow = (l<len(swallow_order)-1)

		#
		# ========= Swallowing the vertex v
		#

		if log:
			
			if ket_tensors[v]:
				label_s = "(ket tensor)"
			else:
				label_s = "(ket-bra tensor)"
				
			print(f"\n\n---------------- Swallowing v={v} {label_s} -------------------\n")
			print("mp_edges_list: ", mp_edges_list)

		#
		# To swallow v we need 3 pieces of information:
		#
		# 1. the locations (i0,i1) of the MPS legs that are contracted to v
		#
		# 2. The list in_legs of indices of v that is to be contracted with 
		#    the MPS, ordered according to their appearance in the MPS
		#
		# 3. The list out_legs of indices of v that are not contracted and
		#    will become part of the new MPS. These have to be ordered 
		#    according to their angles.
		#

		v_edges = edges_list[v]
		v_angles = array(angles_list[v])  # the angles of the legs of tensor v
		k = len(v_edges)

		if log:
			print(f"The edges  of {v} are: {v_edges}")
			print(f"The angles of {v} are: {v_angles}")

		#
		# First, find (i0,i1). We do that by creating the v_mps_legs list.
		# It is made of taples of the form (i,e), where:
		#
		# (*) i = the index of an MPS leg that points to v
		# (*) e = the edge in v along which it is connected.
		#
		# By construction, the list is sortted by the index i.
		#
		
		v_mps_legs = [(i,e) for (i,e) in enumerate(mp_edges_list) \
			if v in vertices[e]]
			
		if not v_mps_legs:
			print(f"bubblecon error: could not swallow vertex {v} because "\
				"there are no MPS legs that are connected to it.")
			print("Current mps legs are: ", mp_edges_list)
			exit(1)
			
		i0 = v_mps_legs[0][0]
		i1 = v_mps_legs[-1][0]
		
		if log:
			print(f"Swallowing tensor {v} into MPS legs range {i0} ==> {i1}")

		#
		# Use v_mps_legs to find in_legs --- the locations of the legs in
		# the v tensor that are connected to the MPS, sorted by their 
		# appearence in the MPS
		#
		
		in_legs = [v_edges.index(e) for (i,e) in v_mps_legs]
		
		if log:
			print("in-legs: ", in_legs)
			
		if len(in_legs) != i1-i0+1:
			print(f"bubblecon error: while trying to swallow vertex {v}: " \
				f"the [i0={i0},i1={i1}] range in the MPS does" \
			" not match the number in-legs={len(in_legs)}. Perhaps it is "\
			"not contiguous")
			exit(1)
		
		#
		# Define out_legs as the complement set of in_legs, and then sort 
		# these legs (if there are any) according to their angle. The 
		# angle is calculated with respect to the first leg in in_legs
		# (any other in leg there would also be fine)
		#

		out_legs1 = list( set(range(k)) - set(in_legs) )
		
		if len(out_legs1)>1:
			rotated_v_angles = (v_angles[in_legs[0]]*ones(k) - v_angles + 2*pi) % (2*pi)

			if log:
				print("rotated angles: ", rotated_v_angles)

			L = [(rotated_v_angles[i],i) for i in out_legs1]
			L.sort()

			sorted_angles, out_legs = zip(*L)
			out_legs = list(out_legs)
		else:
			out_legs=out_legs1

		if log:
			print("out-legs: ", out_legs)

		#
		# Now swallow tensor v. If its a ket tensor, turn it into a ket-bra.
		#
		
		tensor_to_swallow = T_list[v]
		
		
		#
		# Here we pass D_trunc=None and eps=None to the swallow_T routine
		# (simply by omitting them) so that the swallowing is exact. 
		# Truncation is done later on the *entire* boundary-MPS
		#

		if log:
			print("")
			print("Swallowing the tensor into the MPS\n ")


		if D_trunc2 is not None:

			
			max_D = max_bond(mp, tensor_to_swallow, i0, i1, out_legs)
			if max_D>D_trunc2 and D_trunc is not None:
				
				if log:
					print("\n")
					print(f" ====> max_bond > D_trunc2={D_trunc2}. Truncating "
						"it to D_trunc={D_trunc}  <====\n\n")
					
				mp.reduceD(D_trunc, eps, nr_bulk=True)
				
			
			
		if ket_tensors[v]:
			swallow_ket_T(mp, tensor_to_swallow, i0, i1, in_legs, out_legs)
		else:
			swallow_T(mp, tensor_to_swallow, i0, i1, in_legs, out_legs)
		
		if log:
			print("=> done.")
			print("")
		
		
		
		if D_trunc2 is None and D_trunc is not None:
			if log:
				print("Performing reduceD")
				
			mp.reduceD(D_trunc, eps, nr_bulk=True)
			
			if log:
				print("=> done.")
				print("")
				

		if log:
			print("new MPS shape: ", mp.mps_shape())

		#
		# Update the set of swallowed vertices
		#

		swallowed_vertices.add(v)

		#
		# Update the mp_edges_list
		#

		v_out_edges_list = [v_edges[i] for i in out_legs]

		mp_edges_list = mp_edges_list[:i0] + v_out_edges_list \
			+ mp_edges_list[(i1+1):]
			
	


	if log:
		print("\n\n ========== END OF LOOP =========\n")
		print("Final target + source of legs: ")
		print("mp-edges-list: ", mp_edges_list)
		print("mp shape: ", mp.mps_shape())


	#
	# If there are no legs and then the TN is just a scalar. If, 
	# in addition there are no break_points then return the scalar that 
	# is defined by the MPS instead of returning the full mps object.
	#
	if not mp_edges_list and not mp_list:
		mpval = mp.A[0][0,0,0]
		
		#
		# If the separate_exp flag is on, then we separate the 
		# 10-based exponent from the result
		#
		if separate_exp:
			mpval *= mp.nr_mantissa
			return mpval, mp.nr_exp
		
		return mpval*mp.overall_factor()
		
	if D_trunc2 is not None and D_trunc is not None:
		mp.reduceD(D_trunc, eps, nr_bulk=True)
	
	if mp_list != []:
		#
		# If we had some break points, then add the final MPS to it and
		# return the MPS list.
		#
		mp_list.append(mp)
		return mp_list
		

	return mp







