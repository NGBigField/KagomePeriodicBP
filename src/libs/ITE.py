#----------------------------------------------------------
#
#                           ITE.py
#
#  Functions to perform Imaginary Time Evolution (ITE) with
#  blockBP
#
#
# History:
# --------
#
# 30-Mar-2022  Fixed a bug in reduced_env in which Di_red was assumed
#              to be d*D instead of min(d*D, Di1*Di2*...)  (and the 
#              same for Dj_red. This caused a bug when d*D was bigger
#              than Di1*Di2*... --- for example when there was only
#              one external leg to Ti.
#
# 15-Feb-2023  Added the option to use periodic MPS environment in 
#              reduced_env, apply_2local_gate, rho_ij
#
# 12-Apr-2023  Add the function hermitize_a_message to make a double
#              layer MPS message hermitian.
#
#              Also add warnnings in reduced_env and rho_ij if the 
#              tensors there are not sufficiently Hermitian (as 
#              defined by the Hermicity thereshold HERMICITY_ERR)
#
# 14-Nov-2023  Fixed a bug in rho_ij --- in the mps_env case, the Aj
#              contraction had typo that used i instead of j
#
#
# 6-Dec-2023   Used pseudo invert in calculating Li_inv, Lj_inv 
#              in the reduced_env function (they might not be invertible)
#              and also use it for solving Ni ai=b, Nj aj=b in the 
#              ALS_optimization if Ni or Nj are not invertible. This is
#              done via the new function robust_solve.
#
#              Finally, in ALS_optimization, explicitly do nothing 
#              if D<= D_max
#
#
# 6-Dec-2023   Yet another improvement to robust_solve: use pinv
#              whenever the norm of x >> norm(N)*norm(b)
#
# 12-Dec-2023  1) Change robust_solve. No longer using pinv. Instead, 
#                 whenever there's a problem simply regularize N by 
#                 adding eps*ID to it.
#
#              2) Add a log flag to apply_2local_gate, together with
#                 few log messages
#
#              3) Fixed a colossal bug in reduced_env:
#                 for some reason, removing small e.v. was done by
#
#                    wpos_idx = np.where(w>trunc_pos_eps/w[-1])
#                 
#                 instead of:
#
#                    wpos_idx = np.where(w>trunc_pos_eps*w[-1])
#                 
# 14-Dec-2023  Add a norm-0 normalization of new_Ti, new_Tj at the end
#              of apply_2local_gate
#
#
#
# 12-Feb-2024  Add Yigal's hack for faster robust solve. Introduce
#              a new constant NTHRESH such that if matrix dim > NTHRESH
#              then solve the linear equation using scipy.linalg.lstsq
#
# 21-Feb-2024  In apply_2local_gate check first if g is (approximately) 
#              a tensor product of two gates. If so, then no truncation 
#              is needed; apply each gate seperately. 
#
#----------------------------------------------------------


import numpy as np
import scipy

import pickle

from numpy.linalg import norm, svd, qr

from numpy import zeros, ones, array, tensordot, sqrt, diag, conj, \
	eye, trace, pi, exp, isnan
	
from scipy.linalg import expm

from sys import exit

from libs import bmpslib
from libs import bubblecon
from libs.ncon import ncon

from _error_types import ITEError

HERMICITY_ERR = 1e-5
PINV_THRESH = 1e-8
ROBUST_THRESH = 1e8

#
# If the dimension is > NTHRESH, we run robust_solve using least-square.
# Otherwise using the usual Guassian elimination.
#

NTHRESH = 1024

#
# ---------------------- hermitize_a_message  --------------------------
#

def hermitize_a_message(mpA):
	
	"""
	
	Take a MPS which represents an outgoing blockBP message of a 
	double-layer PEPS, and makes it Hermitian:
	
	The MPS tensor is actually an MPO (because of the fused ket-bra legs)
	so we turn it into an Hermitian MPS using the following steps:
	1. Turn the MPS into an MPO (unfuse the "physical" legs)
	    A_{I_1, I_2, ...} => A_{ (i_1,j_1), (i_2,j_2), ... }
	    
	2.  Calculate it's dagger
	    A^dagger = (A_{ (j_1,i_1), (j_2,i_2), ... })^*
	    
	3.  Fuse it back into an MPS B
	
	4. Return 0.5(A + B)
	
	
	Input Parameters:
	------------------
	mpA --- the double-layer MPS message
	
	Output:
	--------
	An Hermitian MPS
	
	
	"""
	
	N = mpA.N
	
	mpB = bmpslib.mps(N)
	
	#
	# Go over the tensors of A, and define the tensors of B to be 
	# the dagger of these tensors.
	#
	# While in the loop, calculate Dmax --- the maximal bond dimension of
	# A, to be used in the end to bring back the bond dimension to its
	# original value.
	#
	
	
	Dmax = 0  
	for i in range(N):
		A2 = mpA.A[i]
		
		DL, d2, DR  = A2.shape[0], A2.shape[1], A2.shape[2]
		
		d = int(sqrt(d2))
		A = A2.reshape([DL, d, d, DR])
		Atran = conj(A.transpose([0,2,1,3]))
		
		A2tran = Atran.reshape(DL, d2, DR)
		
		mpB.set_site(A2tran, i)
		
		if DL > Dmax:
			Dmax = DL
	
	#
	# mpC := 0.5 mpA + 0.5 mpB
	#
	mpC = bmpslib.add_two_MPSs(mpA, 0.5, mpB, 0.5)
	
	mpC.reduceD(Dmax)
	
	return mpC
	
	
	
		
	

#
# ----------------------- get_initial_mps_messages --------------------
#

def get_a_random_mps_message(N, D, mode='UQ'):

	"""

	Get an initial MPS message for the blockBP algorithm. MPS Messages 
	can be either random PSD product states (mode='RQ') or an identity 
	contraction of the ket-bra (mode='UQ')

	Paramters:
	-----------

	N    --- Linear size of MPS 
	D    --- bond dimension
	mode --- either 'UQ' or 'RQ'

	OUTPUT:
	-------

	An random MPS message 


	"""

	D2 = D*D

	mp = bmpslib.mps(N)

	for ell in range(N):

		if mode == 'UQ':
			#
			# For mode 'UQ' we create a simple ket-bra contraction
			#
			Tketbra = np.eye(D)
			
		else:
			#
			# For mode 'RQ' we create random PSDs
			#
			A = np.random.normal(size=[D,D])
			Tketbra = A@A.T
			Tketbra = Tketbra/norm(Tketbra)

		Tketbra = Tketbra.reshape([1,D2,1])

		A = np.random.normal(size=[D*D*D,D*D*D])+ 1j*np.random.normal(size=[D*D*D,D*D*D])
		Tketbra = A@conj(A.T)
		Tketbra = Tketbra/norm(Tketbra)
		Tketbra = Tketbra.reshape([D,D,D,D,D,D])
		Tketbra = Tketbra.transpose([0,3, 1,4, 2,5])
		Tketbra = Tketbra.reshape([D*D,D*D,D*D])

		if ell==0:
			Tketbra = Tketbra[0, :, :].reshape([1,D*D, D*D])

		if ell==N-1:
			Tketbra = Tketbra[:, :, 0].reshape([D*D,D*D,1])

		mp.set_site(Tketbra, ell)

	mp.left_canonical_QR()
	
	return mp
	
	
	


#
# --------------------- invert_permutation  ------------------------
#

def invert_permutation(permutation):

	"""

	Given a permutation in the form of a list of integers, find its
	inverse permutation. This is useful when permuting the indices
	of a tensor T_i to prepare it for imaginary time update (using the
	apply_2local_gate function) or for calculating the 2-body RDM using
	rho_ij function.
	
	Input Parameters:
	-------------------
	
	 permutation --- A list of n integers from 0-->n-1 that captures
	                 a permutation of these numbers. So there are
	                 exactly n numbers and every number between 0-->n-1
	                 appears exactly once.
	                 
	                 
	Output:
	--------
	
	A list of integers giving its inverse
	                 
	                 
	

	"""

	inv = np.zeros_like(array(permutation))
	
	inv[permutation] = np.arange(len(inv), dtype=inv.dtype)

	return list(inv)





#
# ------------------------ get_2body_env  ------------------------------
#

def get_2body_env(mps_i, i0, i1, mps_j, j0, j1, mode='LR'):

	"""

	Calculates the 2-body environment of two neighboring sites i, j from
	the two MPSs that surround the environment.

	The 2-body environment is a set of tensors that surrounds i and j.

	The two MPSs (called mps_i, mps_j) are the result of splitting the
	closed planar PEPS TN in the middle of the edge that connectes i and j. 
	This defines two TNs. The two MPSs are the contraction of each TN
	after removing either the i or j tensor. To calculate the environment,
	the two MPS are contracted along the legs that don't touch i or j. 
	This yields two tensors T_up, T_down. The environment is the tensors 
	of the MPSs that touch i,j, together with T_{up}, T_{down}.

	Input Parameters:
	-----------------

	mps_i  --- The MPS of the half TN in the side of tensor T_i
	i0, i1 --- The range [i0,i1] of the legs in MPS-i that are contracted 
	           with T_i

	mps_j, j0, j1 --- same as above, but just for T_j

	mode --- The way the environment is encoded:

	         (*) 'mps'  -- as MPS periodic MPS tensors, starting with the
	                     tensors of T_i, followed by the tensors of T_j

	         (*) 'LR'   -- As two tensors: the envL tensor which is the
	                      contraction of all the MPS tensors on the left
	                      that are connected to T_i, and the envR tensor
	                      on the right

	         (*) 'full' -- The full environment as a single tensor. The
	                       indices of that tensors are first the indices
	                       of T_i, followed by the indices of T_j.

	OUTPUT:
	-------

	The 2-body environment. Depending on the input parameter 'mode',
	this can be either given as:
	(*) A periodic MPS                                (mode = 'mps')
	(*) Two tensors env_i, env_j for the parts of i,j (mode='LR')
	(*) One big tensor                                (mode='full')


	"""

#	(T_list, edges_list, angles_list, bcon) = BTN


	#
	# Remove the degenerate D=1 leg on the left/right sides of the MPSs
	#
	
	bmpslib.trim_mps(mps_i)
	bmpslib.trim_mps(mps_j)

	#
	# Get the number of legs in both MPSs
	#
	n_i = mps_i.N
	n_j = mps_j.N

	#
	# Use ncon to create the upper tensor T_up: the begining of MPS_i is 
	# contracted with the end of MPS_j
	#

	T1_list = [mps_i.A[0], mps_j.A[n_j-1]]

	edges1_list = [[1,2], [3,1]]

	for i in range(1,i0):
		T1_list = T1_list + [mps_i.A[i], mps_j.A[n_j-1-i]]
		edges1_list = edges1_list + [[i*3-1, i*3+1, i*3+2], [i*3+3, i*3+1, i*3]]

	edges1_list[-1][0]=-2
	edges1_list[-2][-1]=-1

	T_up = ncon(T1_list, edges1_list)


	#
	# Create the lower tensor T_down: the end of MPS_i is contracted with
	# the begining of MPS_j
	#

	T1_list = [mps_i.A[n_i-1], mps_j.A[0]]

	edges1_list = [[2,1], [1,3]]

	for i in range(1,j0):
		T1_list = T1_list + [mps_i.A[n_i-1-i], mps_j.A[i]]
		edges1_list = edges1_list + [[i*3+2, i*3+1, i*3-1], [i*3, i*3+1, i*3+3]]

	edges1_list[-1][-1]=-2
	edges1_list[-2][0]=-1

	T_down = ncon(T1_list, edges1_list)

	#
	# Now we contract T_up with the first tensor of MPS_i and contract
	# T_down with the first tensor of MPS_j
	#

	A = mps_i.A[i0]
	save_mps_i = A
	A = tensordot(T_up, A, axes=([0],[0]))
	mps_i.A[i0] = A

	A = mps_j.A[j0]
	save_mps_j = A
	A = tensordot(T_down, A, axes=([1],[0]))
	mps_j.A[j0] = A


	#
	# Create an "open" version of the environment tensors in which the
	# double-legs are de-fused, so that we can contract them with the
	# single-layer legs of T_i, T_j
	#

	Ti_env = []
	for k in range(i0, i1+1):
		T = mps_i.A[k]
		Tshape = T.shape
		q = int(sqrt(Tshape[1]))

		Ti_env.append(T.reshape([ Tshape[0], q, q, Tshape[2]]))

	Tj_env = []
	for k in range(j0, j1+1):
		T = mps_j.A[k]
		Tshape = T.shape
		q = int(sqrt(Tshape[1]))

		Tj_env.append(T.reshape([ Tshape[0], q, q, Tshape[2]]))


	#
	# Return the input MPSs to the usual, untrimmed shape, and restore
	# mps_i.A[i0] and mps_j.A[j0]
	#
	
	mps_i.A[i0] = save_mps_i
	mps_j.A[j0] = save_mps_j
	
	bmpslib.untrim_mps(mps_i)
	bmpslib.untrim_mps(mps_j)


	#
	# If mode = 'mps' then simply return the MPS tensors of the closed
	# environment, which are Ti_env + Tj_env.
	#

	if mode=='mps':
		env_T_list = Ti_env + Tj_env
		return env_T_list

	#
	# Then the mode is either LR or full. In any case, we first
	# contract all the MPS tensors of T_i (env_i) and all the MPS tensors
	# of T_j (env_j) together. If the mode is 'full', we further contract
	# these two tensors together to create the full environment.
	#


	#
	# ---------------------------------------------------------
	# Create env_i (the contraction of the MPS tensors that
	# are connected to T_i)
	# ---------------------------------------------------------
	#

	env_i_legs=[ [-100,-3,-4,1] ]  # the legs of first MPS-i tensor

	#
	# The legs of the bulk MPS-i tensors
	#
	for k in range(1,i1-i0):
		env_i_legs.append([k, -(2*k+3), -(2*k+4), k+1])
  
	#
	# The legs of the last MPS-i tensor
	#
	k += 1
	env_i_legs.append([k, -(2*k+3), -(2*k+4), -101])


	env_i = ncon(Ti_env, env_i_legs)



	#
	# ---------------------------------------------------------
	# Create env_j (the contraction of the MPS tensors that
	# are connected to T_j)
	# ---------------------------------------------------------
	#

	env_j_legs=[ [-101,-3,-4,1] ]  # the legs of first MPS-j tensor

	#
	# The legs of the bulk MPS-j tensors
	#
	for k in range(1,j1-j0):
		env_j_legs.append([k, -(2*k+3), -(2*k+4), k+1])

	#
	# The legs of the last MPS-j tensor
	#
	k += 1
	env_j_legs.append([k, -(2*k+3), -(2*k+4), -100])

	env_j = ncon(Tj_env, env_j_legs)


	if mode=='LR':
		return env_i, env_j

	#
	# If we got up to here then mode=='full' - so return the environment
	# as a single huge tensor
	#

	full_env = tensordot(env_i, env_j, axes=([2*(i1-i0+1),2*(i1-i0+1)+1], \
		[2*(j1-j0+1),2*(j1-j0+1)+1]))


	return full_env




#
# --------------------------- rho_ij  ----------------------------------
#

def rho_ij(Ti, Tj, env_i=None, env_j=None, mps_env=None):
	"""

	Given two neighboring tensors PEPS T_i,T_j, together with their
	environment env_i, env_j, it returns the RDM.

	env_i, env_j are full tensor (not MPS tensors list).

	The legs of Ti are [i0,i1,i2,i3,...]
	where:
	i0        --- physical leg
	i1        --- The leg that is contracted with Tj
	i2,i3,... --- The rest of the legs, ordered in accordance with env_i,
	              which is determined by the bubble MPS that created it.
	              
	              Specifically, if Ti is on the left and Tj is on the right
	              then i2, i3, ... are ordered from top in a counter clockwise
	              order, and j2, j3, ... are ordered from bottom in 
	              counter-clockwise order.
	             
	              i2           i6
	           i3  |           |   j5
	            \  |           |  /
               \ |           | /
        i4 ----- O ========= O ----- j4
               / |           | \
              /  |           |  \
             i5  |           |   j3
                 i6          j2


	Input Parameters:
	-----------------
	Ti, Tj --- The PEPS tensors of the neighboring sites
	
	env_i, env_j --- The environment of the (i,j) pair given as 
	                 half environments of sites i,j (mode='LR' in 
	                 get_2body_env)
	                 
	mps_env      --- The environment of the (i,j) given as a list of 
	                 tensors in a periodic MPS, which starts at the
	                 first upper leg of T_i and continues counter-clockwise
	                 (mode 'mps' in get_2body_env)


	OUTPUT:
	--------
	rho[i0,i1, j0,j1] --- The 2body RDM,  where i0,i1 are the ket-bra legs
	                      of i and j0,j1 are ket-bra legs of j


	"""

	if mps_env is None and env_i is None and env_j is None:
		print("Error in ITE.rho_ij: either env_i, env_j or mps_env should")
		print("be given (all of them are None)")
		exit(1)
		
	if mps_env and (env_i or env_j):
		print("Error in ITE.rho_ij:  (env_i, env_j) and mps_env cannot")
		print("be both given")
		exit(1)
		

	ni = len(Ti.shape)-2  # no. of Ti mps tensors
	nj = len(Tj.shape)-2  # no. of Tj mps tensors
	

	#
	#
	# We split on two cases: either we are given the environment in LR 
	# mode via (env_i, env_j), or in mps mode via mps_env
	#
	#

	if mps_env is None:
		#
		# ---------------------------------------------------------------
		# So the environment is given as (env_i, env_j)
		# ---------------------------------------------------------------
		#

		#
		# We find the 2body RDM by calling ncon once. To that aim, we need
		# to construct a list of tensors and the list of legs
		#

		#
		# The list of tensors:
		#
		T_list = [Ti, conj(Ti), Tj, conj(Tj), env_i, env_j]

		#
		# The list of contraction legs for each tensor. The general idea is
		# to contract with the following order:
		#
		# T_i + env_i ==> T*_i + env_i ==> T_j + env_j ==> T*_j + env_j
		#
		# The physical legs are (-1,-2) = (i0,i1)  and (-3,-4) = (j0,j1)
		#

		e_list = [ [-1,100] + list(range(1,ni+1)) ]            # Ti
		e_list = e_list + [ [-2,101] + list(range(21, ni+21))] # conj(Ti)

		e_list = e_list + [[-3, 100] + list(range(51, nj+51))] # Tj
		e_list = e_list + [[-4, 101] + list(range(71, nj+71))] # conj(Tj)

		#
		# env_i legs
		#
		env_i_es = []
		for k in range(ni):
			env_i_es = env_i_es + [k+1, k+21]
		env_i_es = env_i_es + [90,40]
		e_list.append(env_i_es)

		#
		# env_j legs
		#
		env_j_es = []
		for k in range(nj):
			env_j_es = env_j_es + [k+51, k+71]
		env_j_es = env_j_es + [90,40]
		e_list.append(env_j_es)

		rho = ncon(T_list, e_list)
		
		
	else:
		#
		# ---------------------------------------------------------------
		# So the environment is given as a periodic mps
		# ---------------------------------------------------------------
		#
				
		#
		# contraction order: 
		# 1. ket legs of T_i with the mps legs
		# 2. bra legs of T^*_i with the mps legs
		#
		#     All this is contracted into the tensor Ai
		#
		# 4. Do the same with the T_j into the tensor Aj
		#
		# 5. Contract the common legs of Ai, Aj
		#
		
		
		#
		# I. Contract Ai --- the tensors in the T_i side
		#
		
		Ti_list = [Ti, conj(Ti)] + mps_env[0:ni]
				
		Ti_legs = [-1,-3] + [2*i+1 for i in range(ni)]
		conjTi_legs = [-2,-4] + [2*i+101 for i in range(ni)]
		
		e_list = [Ti_legs, conjTi_legs]
		
		for i in range(ni):
			iL = -5 if i==0 else 2*i
			iR = -6 if i==ni-1 else 2*i+2
				
			e_list.append([iL, 2*i+1, 2*i+101, iR])
				
		
		Ai = ncon(Ti_list, e_list)


		#
		# II. Contract Aj --- the tensors in the T_j side
		#

		Tj_list = [Tj, conj(Tj)] + mps_env[ni:]
				
		Tj_legs = [-1,-3] + [2*i+1 for i in range(nj)]
		conjTj_legs = [-2,-4] + [2*i+101 for i in range(nj)]
		
		
		e_list = [Tj_legs, conjTj_legs]
		
		for i in range(nj):
			#
			# replace the iL and iR roles here to match the Ti contraction
			#
			iL = -6 if i==0 else 2*i
			iR = -5 if i==nj-1 else 2*i+2
				
			e_list.append([iL, 2*i+1, 2*i+101, iR])
				
		Aj = ncon(Tj_list, e_list)
		
		
		rho = tensordot(Ai, Aj, axes=([2,3,4,5], [2,3,4,5]))
	
	
	Hermicity = norm(rho - conj(rho.transpose([1,0,3,2])))/norm(rho)
	if Hermicity > HERMICITY_ERR:
		print("\n")
		print("* * * ITE Warning: * * *")
		print("ITE.rho_ij: The 2RDM rho_ij is not hermitian")
		print(f"                 It has hermicity {Hermicity} > {HERMICITY_ERR}.")
		print("\n\n")

#	rho = 0.5*(rho + conj(rho.transpose([1,0,3,2])))
	tr = trace(trace(rho))

	return rho/tr





#
# --------------------------- two_sites_expectation  -----------------------------
#

def two_sites_expectation(op, Ti, Tj, env_i, env_j):
	"""

	Calculate the expectation value of a 2-body operator op, defined on
	two adjacent sites i,j

	Input Parameters:
	-------------------
	op --- The 2-body operator given as array [i0,i1,j0,j1] where
	       i0,i1 = ket/bra of site i
	       j0,j1 = ket/bra of site j

	Ti,Tj --- The PEPS tensors of i,j. Their legs are assumed to
	          ordered as:
	          Ti: [i0,i1,i2,i3,...], where:
	            i0        - Physical leg
	            i1        - Leg contracted with T_j
	            i2,i3,... - Rest of the legs, ordered in accordance to
	                        env_i

	          Tj: Same as Ti

  env_i, env_j --- Half environments of sites i,j
                   (these are full tensors, not MPS tensors list)

  OUTPUT:
  -------

  <op> := Tr(op*rho_{ij})


	"""

	rho = rho_ij(Ti,Tj,env_i,env_j)

	av = tensordot(rho, op, axes=([0,1,2,3], [0,1,2,3]))

	return av



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
# ---------------------- reduced_env  ----------------------------
#

def reduced_env(Ti, Tj, env_i=None, env_j=None, mps_env=None):

	"""
	
	Calculate the reduced environment, the reduced tensors and the 
	rest of the tensors given.
	
	The reduced environment N_red is given as a contraction of a tensor
	X*X^\daggger. X has the following indices:
	
	X = [Di_red, Dj_red, DX_red]
	
	Where:
	Di_red - The legs that connects to the reduced tensor a_i
	Dj_red - The legs that connects to the reduced tensor a_j
	DX_red - The legs that connects to X^*
	
	The exact algorithm is based on Michael Lubasch, J. Ignacio Cirac, 
		and Mari-Carmen Bañuls. Phys. Rev. B, 81:165104, Apr 2010.
	and is described in the ITE notes.
	
	
	
	Input Parameters:
	-----------------
	
	Ti, Tj --- The input tensors. Assumed to be in the shape:
							Ti shape: [d, D, Di1, Di2, ...]
							Tj shape: [d, D, Dj1, Dj2, ...]
							
	env_i, env_j --- The environment tensors around Ti, Tj, assumed
	                 to be in the shape:

						env_i: [Di1, Di1^*; Di2, Di2^*; ...; Dup, Ddown]
						env_j: [Dj1, Dj1^*; Dj2, Dj2^*; ...; Dup, Ddown]
						
	mps_env --- another possible specification of the environment using
	            a periodic MPS (useful when the Ti,Tj tensors have a lot
	            of legs and are of a large size). Either this parameter 
	            is given or the (env_i, env_j) parameters are given. Both
	            cannot be given.
						
	OUTPUT:
	--------
	X, ai, aj, Ti_rest, Tj_rest
	
	With the following shapes:
	
	X:       [Di, Dj, D_red]
	ai:      [d, Di, D_red]
	aj:      [d, Dj, D_red]
	Ti_rest: [Di, D1,..., D_{k-1}, D_up, D_down]
	Tj_rest: [Dj, D1,..., D_{k-1}, D_up, D_down]

	"""

	if mps_env is None and env_i is None and env_j is None:
		print("Error in ITE.reduced_env: either env_i, env_j or mps_env should")
		print("be given (all of them are None)")
		exit(1)
		
	if mps_env and (env_i or env_j):
		print("Error in ITE.reduced_env:  (env_i, env_j) and mps_env cannot")
		print("be both given")
		exit(1)

	Ti_shape = list(Ti.shape)
	Tj_shape = list(Tj.shape)

	d = Ti_shape[0]  # Size of physical leg
	D = Ti_shape[1]  # Size of virtual leg connecting i <-> j

	#
	# Size of all legs in Ti, Tj that don't participate in the
	# contraction (i.e., they go into N_red)
	#
	Di_rest = Ti.size//(d*D)
	Dj_rest = Tj.size//(d*D)
	
	Di_red = min(Di_rest, d*D)
	Dj_red = min(Dj_rest, d*D)



	#
	# Number of external legs in Ti, Tj
	#
	n_legs_i = len(Ti_shape) - 2
	n_legs_j = len(Tj_shape) - 2

	#
	# first separate ai, aj (the reduced Ti, Tj) from the other legs 
	# using the QR decomp'
	#

	Ti_mat = Ti.reshape([d*D, Di_rest])

	Ti_rest, ai = qr(Ti_mat.T)

	Ti_rest = Ti_rest.T
	ai = ai.T
	#
	# This way Ti_mat = ai \cdot Ti_rest 
	# ai shape:      [d*D, Di_red]
	# Ti_rest shape: [Di_red, Di_rest]
	#
	
	ai = ai.reshape([d, D, Di_red])
	
	Tj_mat = Tj.reshape([d*D, Dj_rest])

	Tj_rest, aj = qr(Tj_mat.T)

	Tj_rest = Tj_rest.T
	aj = aj.T
	aj = aj.reshape([d, D, Dj_red])

	#
	# Now Tj_rest=[Dj_red, Dj_rest], aj=[d, D, Dj_red]
	#


	#
	#===========================================================
	# Calculate N_red: either from env_i, env_j or from mps_env
	#===========================================================
	#
	
	
	if mps_env is None:
		#
		# Calculate N_red from the contraction of the env_i, env_j with
		# Ti_rest and Tj_rest
		#

		#
		# reshape env_i to [Di_rest, Di_rest; Dup, Ddown]
		#

		perm = list(range(0, 2*n_legs_i, 2)) + list(range(1, 2*n_legs_i, 2))
		perm = perm + [2*n_legs_i, 2*n_legs_i+1]

		Ni = env_i.transpose(perm)
		Dup = Ni.shape[-2]
		Ddown = Ni.shape[-1]

		Ni = Ni.reshape([Di_rest, Di_rest, Dup, Ddown])

		#
		# reshape env_j to [Dj_rest, Dj_rest; Dup, Ddown]
		#

		perm = list(range(0, 2*n_legs_j, 2)) + list(range(1, 2*n_legs_j, 2))
		perm = perm + [2*n_legs_j, 2*n_legs_j+1]

		Nj = env_j.transpose(perm)

		Nj = Nj.reshape([Dj_rest, Dj_rest, Dup, Ddown])

		#
		# Now create N_red by contracting Ni+Nj+Ti_rest+Tj_rest
		#

		Ni = tensordot(conj(Ti_rest), Ni, axes=([1], [1]))
		Ni = tensordot(Ti_rest, Ni, axes=([1], [1]))

		Nj = tensordot(conj(Tj_rest), Nj, axes=([1], [1]))
		Nj = tensordot(Tj_rest, Nj, axes=([1], [1]))
		
	else:
		#
		# So the environment is given by a periodic MPS in mps_env
		#
		
		#
		# -------------------------------------
		#  Create N_i
		# -------------------------------------
		#
				
		Ti_rest = Ti_rest.reshape([Di_red] + Ti_shape[2:])
		#
		# Now Ti_rest is of the form [Di_red, i0, i1, ...]
		#		
	
		#
		# mps_env[0] is of the form [D_L, i0, i0*, D_R]
		#

		# contract the i0 leg
		Ni = tensordot(Ti_rest, mps_env[0], axes=([1], [1]))
		
		#
		# Now Ni is of the shape [Di_red, i1, i2, ..., D_L, i0*, D_R]
		# We move the D_L leg to the begining, right after Di_red, 
		# since its going to become the Dup leg of Ni
		#
		l = len(Ni.shape)
		Ni = Ni.transpose([0,l-3] + list(range(1, l-3)) + [l-2,l-1])
		
		#
		# Now Ni is of the shape [Di_red, D_L, i1, i2, ..., i0*, D_R]
		#
				
		for mp in mps_env[1:n_legs_i]:
			last_ind = len(Ni.shape)
			Ni = tensordot(Ni, mp, axes=([2, last_ind-1], [1, 0]))
			#
			# Now Ni is [Di_red, D_L, i_k, i_{k+1}, ..., i0*, i1*, ...., D_R]
			#
		
		#
		# At the end of the loop, Ni is of the shape:
		# [Di_red, D_L, i0*, i1*, ..., D_R] 
		# where D_L, D_R will become Dup, Ddown.
		#
		# It has a total of (3 + n_legs_i) legs.
		#
		# We contract the i0*, i1*, ... legs with the corresponding 
		# legs of Ti_rest* (which is of the form [Di_red*, i0*, i1*, ...])
		#
		
		Ni = tensordot(Ni, conj(Ti_rest), axes=(range(2, 2+n_legs_i), \
			range(1, 1+n_legs_i)))
			
		#
		# Ni is now of the form [Di_red, D_L, D_R, Di_red^*]. We 
		# transpose it to [Di_red, Di_red*, D_L, D_R], where
		# now D_L = Dup, D_R=Ddown
		#
		
		Ni = Ni.transpose([0, 3, 1, 2])
		

		#
		# -------------------------------------
		#  Create N_j
		# -------------------------------------
		#


		Tj_rest = Tj_rest.reshape([Dj_red] + Tj_shape[2:])
		#
		# Now Tj_rest is of the form [Dj_red, j0, j1, ...]
		#		
	
		#
		# mps_env[n_legs_i] is of the form [D_L, j0, j0*, D_R]
		#

		# contract the j0 leg
		Nj = tensordot(Tj_rest, mps_env[n_legs_i], axes=([1], [1]))
		
		#
		# Now Nj is of the shape [Dj_red, j1, j2, ..., D_L, j0*, D_R]
		# We move the D_L leg to the begining, right after Dj_red, 
		# since its going to become the Ddown leg of Nj
		#
		l = len(Nj.shape)
		Nj = Nj.transpose([0,l-3] + list(range(1, l-3)) + [l-2,l-1])
		
		#
		# Now Nj is of the shape [Dj_red, D_L, j1, j2, ..., j0*, D_R]
		#
				
		for mp in mps_env[(n_legs_i+1):]:
			last_ind = len(Nj.shape)
			Nj = tensordot(Nj, mp, axes=([2, last_ind-1], [1, 0]))
			#
			# Now Nj is [Dj_red, D_L, j_k, j_{k+1}, ..., j0*, j1*, ...., D_R]
			#

		
		#
		# At the end of the loop, Nj is of the shape:
		# [Dj_red, D_L, j0*, j1*, ..., D_R] 
		# where D_L, D_R will become Ddown, Dup.
		#
		# It has a total of (3 + n_legs_j) legs.
		#
		# We contract the j0*, j1*, ... legs with the corresponding 
		# legs of Tj_rest* (which is of the form [Dj_red*, j0*, j1*, ...])
		#
		
		Nj = tensordot(Nj, conj(Tj_rest), axes=(range(2, 2+n_legs_j), \
			range(1, 1+n_legs_j)))
			
		#
		# Nj is now of the form [Dj_red, D_L, D_R, Dj_red^*]. We 
		# transpose it to [Dj_red, Dj_red*, D_R, D_L], where
		# now D_L = Ddown, D_R=Dup
		#
		
		Nj = Nj.transpose([0, 3, 2, 1])
		
		#
		# Fuse back the legs of Ti_rest, Tj_rest
		#

		Ti_rest = Ti_rest.reshape([Di_red, Ti_rest.size//Di_red])
		Tj_rest = Tj_rest.reshape([Dj_red, Tj_rest.size//Dj_red])

	Ni = Ni/norm(Ni)
	Nj = Nj/norm(Nj)
		

	#
	# now Ni,Nj are of the shape [D_red, D*_red, Dup, Ddown]. We 
	# contract them by the Dup, Ddown legs to create N_red
	#

	Nred = tensordot(Ni, Nj, axes=([2,3],[2,3]))


	#
	# Resultant Nred is now of the form
	#     [Di_red, Di_red*; Dj_red, Dj_red*]
	#
	# We now make it into a ket-bra matrix 
	#
	# [Di_red\cdot Dj_red , Di_red* \cdot Dj_red*]

	Nred = Nred.transpose([0,2,1,3])
	Nred = Nred.reshape([Nred.shape[0]*Nred.shape[1], \
		Nred.shape[2]*Nred.shape[3]])

	Hermicity = norm(Nred-conj(Nred.T))/norm(Nred)
	
	if Hermicity > HERMICITY_ERR:
		print("\n")
		print("* * * ITE Warning: * * *")
		print("ITE.reduced_env: The reduced env tensor Nred is not hermitian")
		print(f"                 It has hermicity {Hermicity} > {HERMICITY_ERR}.")
		print("\n\n")

	#
	# Making Nred 100% Hermitian 
	#
	Nred = 0.5*(Nred + conj(Nred.T))


	#
	# and make it positive by removing its negative eigenvalues.
	#

	w, U = np.linalg.eigh(Nred)
 
	if all(w<0):
		raise ITEError("No positive eigen-values!")

	trunc_pos_eps = 1e-12

	wpos_idx = np.where(w>trunc_pos_eps*w[-1])
	
	pos_idx = wpos_idx[0][0]

	wpos = w[pos_idx:]
	U = U[:, pos_idx:]

	#
	# Now calculate X=sqrt(N_red)
	#

	sqrt_wpos = sqrt(wpos)
	

	X = U@diag(sqrt_wpos)

	#
	# Now, Nred \simeq X \cdot X^dagger  (up to the eigenvalues we 
	# discarded)
	#

	DX_red = X.shape[1]

	X = X.reshape([Di_red, Dj_red, DX_red])
	
	#
	# Final step: gauge fix the Di_red, DX_red legs of X using QR.
	#

	X_tmp = X.reshape([Di_red, Dj_red*DX_red])
	Q,Ri = qr(X_tmp.T)
	Li = Ri.T
	
	#
	# We now invert Li, but use Penrose's pseudo inverse since Li 
	# is not necessarily invertible (and might even not be square)
	#
	Li_inv = np.linalg.pinv(Li, rcond=PINV_THRESH)
		

	X_tmp = X.transpose([0,2,1])  # now its [Di_red, D_X, Dj_red]
	X_tmp = X_tmp.reshape([Di_red*DX_red, Dj_red])
	Q,Rj  = qr(X_tmp)
	
	#
	# Pseudo-invert Lj
	#
	Rj_inv = np.linalg.pinv(Rj, rcond=PINV_THRESH)
	
	#
	# Now we have our(X) = Li X_new Rj  ==> X_new = Li^{-1} X Rj^{-1}
	#

	#
	# Gauge fixing: absorb Li_inv, Rj_inv in X and Li, Rj in ai, aj
	#
	# Also, absorb Li_inv, Rj_inv in Ti_rest and Tj_rest (to keep them
	# in the same gauge)
	#
	# Recall: ai=[d, D, D_red], aj=[d, D, D_red]
	#


	X = tensordot(Li_inv, X, axes=([1],[0]))
	Ti_rest = tensordot(Li_inv, Ti_rest, axes=([1],[0]))
	ai = tensordot(ai, Li, axes=([2],[0]))

	X = tensordot(X, Rj_inv, axes=([1],[0]))
	X = X.transpose([0,2,1])
	Tj_rest = tensordot(Tj_rest, Rj_inv, axes=([0], [0]))
	Tj_rest = Tj_rest.transpose([1,0])
	aj = tensordot(aj, Rj, axes=([2],[1]))

	#
	# Unfuse the external legs of Ti_rest, Tj_rest
	#

	#
	# The dimensions of Di_red, and Dj_red might have changed during 
	# the gause fixing, so we update them.
	#
	
	Di_red = ai.shape[2]
	Dj_red = aj.shape[2]

	Ti_rest = Ti_rest.reshape([Di_red] + list(Ti.shape[2:]))
	Tj_rest = Tj_rest.reshape([Dj_red] + list(Tj.shape[2:]))

	#
	# Output tensors shape:
	# -----------------------
	#
	# X: [Di_red, Dj_red, D_X]
	# ai: [d, D, Di_red]
	# aj: [d, D, Dj_red]
	#

	return X, ai, aj, Ti_rest, Tj_rest, w



#
# ----------------------  reduced_inner_prod  -------------------------
#
def reduced_inner_prod(ai_ket, aj_ket, ai_bra, aj_bra, X):
	
	"""
	Given a reduced environment X, together with reduced ket-bra of
	a_i, a_j tensors, calculate the inner product between them:
	
	Inner=Product = Tr(X X^\dagger ai_ket ai^*_bra  aj_ket aj^*_bra) 
	
	"""

	ket = tensordot(ai_ket, X, axes=([2],[0])) # ket=[di, D, D_red(j), DX]
	ket = tensordot(aj_ket, ket, axes=([1,2],[1,2])) #ket=[dj,di,DX]

	bra = tensordot(ai_bra, X, axes=([2],[0])) # ket=[di, D, D_red(j), DX]
	bra = tensordot(aj_bra, bra, axes=([1,2],[1,2])) #ket=[dj,di,DX]

	ketbra = tensordot(ket, conj(bra), axes=([0,1,2], [0,1,2]))

	return ketbra



#
# ----------------------  truncation_distance  -------------------------
#
def truncation_distance(exact_ai, exact_aj, new_ai, new_aj, X):
	
	"""
	
	Given a reduced environment X, together with *two* pairs of reduced
	tensors a_i, a_j, calculate the distance beween the states that these
	two sets define, i.e., the square norm:
	
	||Tr(X exact_ai exact_aj) - Tr(X ai aj)||^2
	
	"""
	


	IP1 = reduced_inner_prod(exact_ai, exact_aj, exact_ai, exact_aj, X)
	IP2 = reduced_inner_prod(new_ai, new_aj, new_ai, new_aj, X)

	IP3 = reduced_inner_prod(exact_ai, exact_aj, new_ai, new_aj, X)

	dis = 2*(IP1 + IP2 - 2*IP3)/(IP1+IP2)

	return dis.real


#
# ------------------------ open_mps_env ------------------------------
#

def open_mps_env(mps_env):
	"""
	
	Given an MPS env, which is simply a list of MPS tensors 
	of the form [D_L, Dmid, D_R], which represent a periodic MPS,
	we open up the mid leg to ket-bra legs.
	
	Specifically, we assume Dmid = D*D, and reshape every tensor in the 
	list as:
	
	              [D_L, Dmid, D_R] ==> [D_L, D, D, D_R]
	
	"""
	
	
	for i in range(len(mps_env)):
		A = mps_env[i]
		Ashape = A.shape
		D2 = Ashape[1]
		D = int(np.sqrt(D2))
		assert D*D==D2, f"Fused-leg of dimension {D2} cannot be split"
		mps_env[i] = A.reshape([Ashape[0], D, D, Ashape[2]])

	return mps_env




#
# ---------------------------- Ni_env ----------------------------------
#

def Ni_env(aj_ket, aj_bra, X):
	
	"""
	Given a reduced environment X, together with ket-bra of the reduced
	tensor a_j, contract them together to obtain the local environment 
	of the i'th particle  (i.e., the TN we get if we contract ||psi||^2
	and omit a_i, a^*_i)
	
	"""
	

	# X: [D_red(i), D_red(j), D_X]
	# aj: [d, D, D_red]

	d=aj_ket.shape[0]

	ket = tensordot(aj_ket, X, axes=([2],[1])) # ket=[d, D, D_red(i), D_X]
	bra = tensordot(aj_bra, X, axes=([2],[1])) # bra=[d, D, D_red(i), D_X]
	Ni_env = tensordot(ket, conj(bra), axes=([0,3], [0,3]))

	# Now Ni_env is of the shape [D, D_red(i); D^*, D^*_red(i)]
	# We add a Kronecker delta to account for the physical legs

	Ni_env = tensordot(eye(d), Ni_env, 0)
	Ni_env = Ni_env.transpose([0,2,3,1,4,5])

	#
	# The final Ni_env is of the form
	#    [d, D, D_red(i); d^*, D^*, D^*_red(i)]
	#

	return Ni_env


#
# ---------------------------- Nj_env ----------------------------------
#

def Nj_env(ai_ket, ai_bra, X):

	"""
	Given a reduced environment X, together with ket-bra of the reduced
	tensor a_i, contract them together to obtain the local environment 
	of the j'th particle  (i.e., the TN we get if we contract ||psi||^2
	and omit a_j, a^*_j)
	
	"""

	Xt = X.transpose([1,0,2])

	Nj = Ni_env(ai_ket, ai_bra, Xt)

	return Nj

#
# -------------------------  robust_solve  ---------------------------
#

def robust_solve(N,b):
	
	"""
	
	Solve the linear equation Nx = b, where N is a square matrix in 
	a *robust* way. That is, if N turns out to be singular, or close
	to singular, then we regularize N by adding it eps*ID, where 
	eps = PINV_THRESH*norm(N) 
	
	
	"""


	#
	# We will first try to solve Nx=b using the regular linalg.solve.
	# If this doesn't work because of any error, or if we get nan as
	# answer, or if the solution we get is insanely large --- we try
	# again after regularizing N.
	#

	regularize = False
	
	Nsize = N.shape[0]

	try:

		if Nsize<=NTHRESH:
			x = np.linalg.solve(N, b)
		else:
			x_tuple = scipy.linalg.lstsq(N, b, cond=None, check_finite=False,\
				lapack_driver='gelsy')
			x = x_tuple[0]

	except:
		regularize=True


	if not regularize:
		
		if isnan(norm(x)):
			regularize=True
		
		elif norm(x)>ROBUST_THRESH*norm(b)/norm(N):
			regularize=True
		

	if regularize:
		newN = N + eye(N.shape[0])*PINV_THRESH*norm(N, ord=2)

		if Nsize<=NTHRESH:
			x = np.linalg.solve(newN, b)
		else:
			x_tuple = scipy.linalg.lstsq(newN, b, cond=None, check_finite=False,\
				lapack_driver='gelsy')
			x = x_tuple[0]

		
	return x




#
# ----------------------  ALS_optimization  ----------------------------
#

def ALS_optimization(Dmax, exact_ai, exact_aj, X, iter_max=10, 
	eps=1e-6):
		
	"""
	
	Given a reduced environment with reduced exact_ai, exact_aj tensors, 
	and a target bond dimension Dmax, perform ALS iterations to find new 
	a_i, a_j with maximal bond dimension Dmax so that the resulting 
	state (X, new_ai, new_aj) will be a good approximation of the 
	original state.
	
	The ALS procedure is based on Michael Lubasch, J. Ignacio Cirac, 
	and Mari-Carmen Bañuls. Phys. Rev. B, 81:165104, Apr 2010, and is 
	described in the ITE notes.
	
	Input Parametes:
	----------------
	
	Dmax --- Target bond dimension
	
	exact_ai, exact_aj --- The initial reduced tensors, given in the
	                       shape:
	                       
	                       exact_ai: [d, D, D_red(i)]
	                       exact_aj: [d, D, D_red(j)]
	                       
	                       here d is physical, D is the leg connecting
	                       i and j, and D_red() is the leg connecting the
	                       reduced tensors to the legs of the reduced
	                       environment X
	                       
	X --- The ket of the reduced environment (N_red = X\cdot X^*)
	      Its shape is: [Dred(i), D_red(j), D_X]
	      
	                       
	iter_max --- max allowed iterations
	
	eps --- Accuracy threshold for stopping the iteration. If
	        |distance(k) - distance(k+1)| < eps
	        then stop the iteration
	        
	        Here distance(k) is the distance between (ai_new, aj_new, X)
	        at the k'th iteration and (ai_exact, aj_exact, X)
	        (as calculated by the truncation_distance function)
	                               
	OUTPUT:
	--------
	
	The optimized ai_new, aj_new
	
	"""
		
	log=False
	
	D = exact_ai.shape[1]

	if log:
		print("\n\n")
		print(f"ALS_optimization: D={D} Dmax={Dmax} "\
			f"iter_max={iter_max} eps={eps}")


	if D<= Dmax:

		if log:
			print(f"ALS_optimization: D<=Dmax --- nothong to do...")

		new_ai = exact_ai.copy()
		new_aj = exact_aj.copy()
		return new_ai, new_aj
	

	#
	# Start with an initial guess for ai_new, aj_new.
	#
	# The initial guess is simply the first entries of the
	# exact_ai, exact_aj.
	#
	# If the exact_ai, exact_aj were calculated using SVD, then this is
	# also not so bad because it uses the largest singular values.
	#

	new_ai = exact_ai[:,0:Dmax, :]
	new_aj = exact_aj[:,0:Dmax, :]


	#
	# ============ Starting the ALS
	#

	iter_no = 0
	dist = 1e10
	delta_dist = 1

	if log:
		print(f"ALS_optimization: starting optimization loop...")

	while delta_dist>eps and iter_no < iter_max:

		#
		# Find new_ai: calculate its local environment N_i, together
		# with b, and find new_ai by solving the linear equation:
		#
		#            N_i new_ai = b 
		#

		Ni = Ni_env(new_aj, new_aj, X)  # [d, D, D_red(i); d^*, D^*, D^*_red(i)]
		Ni = Ni.reshape([Ni.shape[0]*Ni.shape[1]*Ni.shape[2], \
			Ni.shape[3]*Ni.shape[4]*Ni.shape[5]])
			
		Ni = Ni.T

		Nib = Ni_env(exact_aj, new_aj, X)
		b = tensordot(Nib, exact_ai, axes=([0,1,2],[0,1,2]))
		b = b.flatten()


		if log:
			print(f"ALS_optimization: invoking robust_solve with mat-size "\
				f"{Ni.shape[0]} x {Ni.shape[0]}")
		#
		# To solve the equation Ni ai = b, we use a "robust" alg, which
		# pseudo inverts Ni when it is not invertible.
		#
		ai = robust_solve(Ni, b)

		if log:
			print(f"ALS_optimization: robust_solve done")

		new_ai = ai.reshape(new_ai.shape)


		#
		# Find new_aj --- just like in the new_ai case.
		#

		Nj = Nj_env(new_ai, new_ai, X)  # [d, D, D_red(j); d^*, D^*, D^*_red(j)]
		Nj = Nj.reshape([Nj.shape[0]*Nj.shape[1]*Nj.shape[2], \
			Nj.shape[3]*Nj.shape[4]*Nj.shape[5]])

		Nj = Nj.T

		Njb = Nj_env(exact_ai, new_ai, X)
		b = tensordot(Njb, exact_aj, axes=([0,1,2],[0,1,2]))
		b = b.flatten()

		if log:
			print(f"ALS_optimization: invoking robust_solve with mat-size "\
				f"{Nj.shape[0]} x {Nj.shape[0]}")

		#
		# To solve the equation Nj aj = b, we use a "robust" alg, which
		# pseudo inverts Nj when it is not invertible.
		#
		aj = robust_solve(Nj, b)

		if log:
			print(f"ALS_optimization: robust_solve done")


		new_aj = aj.reshape(new_aj.shape)


		#
		# Gauge Fixing:
		# --------------
		#
		# 1. Do QR to new_ai, LQ to new_aj.
		#

		ai = new_ai.transpose([0,2,1])  # [d, D_red, D]
		ai_shape = ai.shape
		ai = ai.reshape([ai_shape[0]*ai_shape[1], ai_shape[2]])

		Qi,Ri = qr(ai)

		aj = new_aj.transpose([1,0,2])  # [D, d, D_red]
		aj_shape = aj.shape
		aj = aj.reshape([aj_shape[0], aj_shape[1]*aj_shape[2]])



		Qj,Rj = qr(aj.T)
		Qj = Qj.T
		Lj = Rj.T
		# this way aj = Lj*Qj


		# Now we have: ai = Qi*Ri  ,  aj=Lj*Qj

		#
		# 2. Final internal gauge fixing using SVD: we decompose the middle
		#    tensor Ri * Lj using SVD, and swallow U*sqrt(S) and sqrt(S)V
		#     in new_ai and new_aj.
		#

		mid_bond = Ri@Lj

		U,S,V = svd(mid_bond)

		sqrt_S = diag(sqrt(S))


		Qi = Qi@U@sqrt_S
		Qj = sqrt_S@V@Qj


		Qi = Qi.reshape(ai_shape)  # now Qi=[d, D_red, D]
		new_ai = Qi.transpose([0,2,1])

		Qj = Qj.reshape(aj_shape)  # now Qj=[D, d, D_red]
		new_aj = Qj.transpose([1,0,2])



		old_dist = dist

		dist = truncation_distance(exact_ai, exact_aj, new_ai, new_aj, X)

		delta_dist = abs(dist-old_dist)
		
		if log:
			print(f">> Iter {iter_no}: dist={dist}   delta={delta_dist}\n\n")

		iter_no += 1

	if log:
		print(f"ALS_optimization: optimization loop done with "\
			f"{iter_no} iterations and error={delta_dist}")
		print("\n\n")


	new_ai = new_ai/norm(new_ai)
	new_aj = new_aj/norm(new_aj)

	return new_ai, new_aj



#
# ---------------------- apply_2local_gate  ----------------------------
#

def apply_2local_gate(g, Dmax, Ti, Tj, env_i=None, env_j=None, \
	mps_env=None):
	
	"""
	
	Applies a 2-local gate g on two neighboring PEPS tensors T_i, T_j, 
	and then performs a bond truncation to Dmax using the reduced-tensor 
	full-update method, which performs ALS optimization.
	
	The result is two tensor Ti_new, Tj_new with maximal bond dimension 
	Dmax.
	
	Input Parameters:
	------------------
	g --- The 2-local gate to apply, given as a rank-4 array:
	      g[i0,j0; i1,j1], where i0,j0 are the ket-bra legs of particle 0
	      and i1,j1 are the ket-bra legs of particle 1.  For example, 
	      to create an X\times X gate, we use 
	      g=tensordot(sigma_X, sigma_X,0)
	      
	Dmax --- The truncation bond dimension.
	
	Ti, Tj --- The PEPS tensors on which g acts
	
	env_i, env_j --- The half environment tensors that surround Ti, Tj. 
	                 These can be calculated by get_2body_env
	                 using mode='LR'.
	                 
	mps_env --- an alternative specification of the environment using
	            a periodic MPS. Either env_i, env_j are given, or this
	            parameter is given (they cannot be both given)
	                 
	Output:
	-------
	
	The updated tensors at i,j:  Ti_new, Tj_new
	
	INDICES CONVENTION:
	--------------------
	
	The legs of the T_i, T_j tensors are sorted according to:
	
	            T_i[i_d, i_D, i_1, i_2, ..., i_{k-1}]
	
	Where: i_d = physical leg
	       i_D = virtual leg connected to the other tensor
	       i_1, ..., i_{k-1} = the other virtual legs arranged according
	                           to the order of the legs in env_i, env_j
	                           
	The legs of env_i, env_j are sorted according to:
	  
	              env_i[i1,j1; i2,j2; ..., i_{k-1},j_{k-1}, i_up, i_down]
	              
	where i1,j1, ... are the ket-bra indices of the first virtual leg etc.
	
	i_up, i_down are the legs that are connected to the other environment 
	tensor. See the ITE.pdf manual for exact description.

	
	"""
	
	log = False

	if log:
		print(f"Entering apply_2local_gate with D={Ti.shape[1]} and Dmax={Dmax}")

	if mps_env is None and env_i is None and env_j is None:
		print("Error in ITE.apply_2local_gate: either env_i, env_j or mps_env should")
		print("be given (all of them are None)")
		exit(1)
		
	if mps_env and (env_i or env_j):
		print("Error in ITE.apply_2local_gate:  (env_i, env_j) and mps_env cannot")
		print("be both given")
		exit(1)


	#
	# First see if there's anything to do: if the gate is (proportional 
	# to) the identity, we don't need to do anything
	#

	g_mat = g.transpose([0,2,1,3])
	g_mat = g_mat.reshape(g.shape[0]*g.shape[2], g.shape[1]*g.shape[3])
	
	
	sc = norm(g_mat, ord=2)  # the operator norm of g
	
	if sc<1e-15:
		print(f"Error: trying to apply a gate with operator norm={sc}")
		exit(1)

	#
	# If the gate is close to Identity --- then do nothing
	#
	if norm(g_mat-g_mat[0,0]*eye(g_mat.shape[0]))/sc < 1e-10:
		
		if log:
			print("apply_2local_gate: gate is trivial - nothing to do...")
		
		return Ti, Tj
		
	#
	# Now check if the gate is a product of two single-site gates by 
	# looking at its SVD. In such case, no truncation is needed. Just 
	# apply each gate to its tensor.
	#

	g_mat = g.reshape(g.shape[0]*g.shape[1], g.shape[2]*g.shape[3])
	
	U, s, V = np.linalg.svd(g_mat, full_matrices=False)
	
	is_product = False
	
	if s.shape[0]==0:
		is_product = True
	else:
		if s[1]/s[0]<1e-10:
			is_product = True
		

	if is_product:
		#
		# In such case g = g_i \otimes g_j. So we extract the single-body
		# gates g_i, g_j and apply them directly to T_i, T_j
		#
		if log:
			print("apply_2local_gate: gate is a product of two local gates. "\
				"No need for bond truncation.")
				
		# 
		# Find the maximal entry in abs(g) --- use it to extrac 
		# g_i, g_j
		#
		
		maxind = np.unravel_index(abs(g).argmax(), g.shape)
		
		g_i = g[:, :, maxind[2], maxind[3]]
		g_j = g[maxind[0], maxind[1], :, :]
		
		#
		# we need to rescale g_i, g_j so that their tensor product gives g.
		# There's a freedom here because its only the product that needs
		# to be equal to g
		#
		
		rescale = g[maxind[0], maxind[1], maxind[2], maxind[3]] \
			/(g_i[maxind[0], maxind[1]]*g_j[maxind[2], maxind[2]])
		
		g_i_factor = sqrt(abs(rescale))
		g_j_factor = rescale/g_i_factor
		
		g_i = g_i_factor * g_i
		g_j = g_j_factor * g_j
		
		#
		# Once we have g_i, g_j, we apply then to Ti, Tj and obtain the 
		# updated tensors
		#
		
		newTi = tensordot(g_i, Ti, axes=([1],[0]))
		newTj = tensordot(g_j, Tj, axes=([1],[0]))
		
		return newTi, newTj
			


	#
	# First calculate the reduced env tensors:
	#
	# X                --- the sqrt of the reduced env: 
	#                           N_red = X\cdot X^\dagger
	# ai, aj           --- The reduced Ti, Tj tensors
	# Ti_rest, Tj_rest --- The rest of the Ti, Tj tensors.
  #
	# Note: Ti = Ti_rest\cdot ai and similarly for j.
	#

	X, ai, aj, Ti_rest, Tj_rest, origin_eigen_vals = reduced_env(Ti, Tj, env_i, env_j, mps_env)

	#
	# The legs of the resultant tensors:
	#
	# X:  [i_red(i), i_red(j), i_X]
	# ai: [i_d, i_D, j_red(i)]
	# aj: [i_d, i_D, i_red(j)]
	# Ti_rest: [i1, j1; i2,j2; ...; i_{k-1}, j_{k-1}, i_up, i_down]
	#
	#

	d = ai.shape[0]
	D = ai.shape[1]
	Di_red = ai.shape[2]
	Dj_red = aj.shape[2]

	#
	# Now apply the gate g on ai, aj, and use SVD to get Ai, Aj.
	#
	# g shape is: [i0,i1; j0,j1]
	#

	exact_aiaj = tensordot(ai, aj, axes=([1],[1]))
	# exact_aiaj: [d, D_red(i), d, D_red(j)]
	exact_aiaj = tensordot(g, exact_aiaj, axes=([1,3], [0,2]))
	# exact_aiaj: [d_i, d_j, D_red(i) D_red(j)]
	exact_aiaj = exact_aiaj.transpose([0,2,1,3])
	# exact_aiaj: [d_i, D_red(i); d_j, D_red(j)]
	exact_aiaj = exact_aiaj.reshape([d*Di_red, d*Dj_red])
	
	U,S,V = svd(exact_aiaj, full_matrices=False)

	sqrtS = diag(sqrt(S))

	exact_ai = U@sqrtS    # shape: [d*Dred, Dp]
	exact_aj = sqrtS@V    # shape: [Dp, d*Dred]

	Dp = exact_ai.shape[1]

	exact_ai = exact_ai.reshape([d, Di_red, Dp])
	exact_ai = exact_ai.transpose([0,2,1])  # shape: [d, Dp, Dred]

	exact_aj = exact_aj.reshape([Dp, d, Dj_red])
	exact_aj = exact_aj.transpose([1,0,2]) # shape: [d, Dp, Dred]

	#
	# Find the optimal approximation to exact_ai, exact_aj using
	# ALS iterations
	#

	if log:
		print("apply_2local_gate: Entering ALS-optimization for new_ai, new_aj")

	new_ai, new_aj = ALS_optimization(Dmax, exact_ai, exact_aj, X)

	if log:
		print("apply_2local_gate: ALS-optimization done")

	#
	# Fust the reduced tensors back to the rest of the tensors to get
	# the updated full tensors
	#
	
	new_Ti = tensordot(new_ai, Ti_rest, axes=([2],[0]))
	new_Tj = tensordot(new_aj, Tj_rest, axes=([2],[0]))
	
	#
	# Noramlize new_Ti, new_Tj so that their larget entry in abs is 1
	#

	new_Ti = new_Ti/norm(new_Ti.flatten(), ord=np.inf)
	new_Tj = new_Tj/norm(new_Tj.flatten(), ord=np.inf)
	
	if log:
		print("apply_2local_gate done.")
		if new_Ti.shape[1]!=Ti.shape[1] or new_Tj.shape[1]!=Tj.shape[1]:
			print(f"apply_2local_gate: tensors changed shape: ")
			print(f"    {Ti.shape}, {Tj.shape} => {new_Ti.shape}, {new_Tj.shape}")
			print()

	return new_Ti, new_Tj, origin_eigen_vals
	

#
# ---------------------- g_from_exp_h  ----------------------------
#

def g_from_exp_h(h, dt):
	
	"""
	
	Given a 2-local Hamiltonian term, calculate its imaginary time evolved
	"gate", g := e^{-dt h}.  Note that dt can be also imaginary.
	
	Input Parameters:
	--------------------
	
	h --- The 2-local Hamiltonian term, given as a 4-legs tensors in the
	      form (i1,j1), (i2, j2)
	      
	dt --- The imaginary time evolution period
	
	OUTPUT:
	---------
	
	The gate g=e^{-dt h} in the form (i1,j1), (i2, j2).
	
	
	"""
	

	#
	# h is given as a 4-legs tensor of the form (i1,j1; i2,j2).
	# We need to turn it into a matrix (i1,i2), (j1,j2), exponentiate it
	# and then reshape and transform it back into the (i1,j1; i2,j2)
	# form.
	#
	
	d = h.shape[0]
	
	h_mat = h.transpose([0,2,1,3])
	h_mat = h_mat.reshape([d*d,d*d])
	
	g = expm(-dt*h_mat)
	
	g = g.reshape([d,d,d,d])
	g = g.transpose([0,2,1,3])
	
	return g
	


