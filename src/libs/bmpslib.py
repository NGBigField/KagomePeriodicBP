#----------------------------------------------------------
#
# Module: bmpslib - Boundary MPS library
#
#
# History:
# ---------
#
# 8-Oct-2019   -   Initial version.
#
# 10-Jul-2021  -  Improved the calculate_PEPS_2RDM function.
#                 Removed one loop by storing the intermediete bmps
#                 in a list. This increases a bit the memory consumption
#                 but speeds up a lot the running time.
#
#                 Same thing *should* be done for the
#                 calculate_PEPS_1RDM function.
#
# 27-Aug-2021  -  Improved the left/right canonical + reduceD. 
#                 (*) Added an eps parameter to control the mininal
#                     singular values that enter
#                 (*) Add left_canonical_QR, which uses the QR decomp'
#                     for a fast left-canonical result. This is used
#                     as the first pass in reduceD(). The second uses
#                     the right-canonical using the SVD, where an optional
#                     truncation is performed.
#
# 21-Oct-2021  -  Changed the limit in right_canonical, left_canonical,
#                 left_canonical_QR: in all these functions, the 
#                 procedure now happens for N>1 instead of N>2.
#
# 5-Feb-2022   -  Add the trim_mps function
#
# 26-Feb-2022  -  1. Add the copy() method to the mps class
#
#                 2. Add the reverse() method to reverse the order of
#                    the MPS tensors
#
#                 3. Fixed supid typo: 
#                    mps_inner_prodcut ==> mps_inner_product
#
# 6-Mar-2022   - Added the untrim_mps, which cancels the effect
#                of the trim_mps function.
#
#
# 28-Apr-2022 - Major improvements:
#               (*) added the list Corder[i] to the mps class, which
#                   holds the canonical ordering of A[i]. Possible values
#                   are 'L', 'C', None
#
#               (*) Added a possible range [i0,i1] for left_canonical_QR
#                   and right_canonical
#
#               (*) In right_canonical, use SVD only when there's need
#                   to truncate; otherwise, use RQ.
#
#               (*) reduceD now works locally: it identifies the minial
#                   region to perform left-canonical, and then the 
#                   minimal region to perform right-canonical + trunc.
#                   As a result, the final MPS has a mixed canonical 
#                   state. 
#
#               (*) reduceD defualt mode is 'MC' (mid-Canonical) which 
#                   reflects the above default
#
#               (*) Added the maxD function, which gives the maximal
#                   bond dimension.
#
# 28-Apr-2022 --- Added some documentaion to reduceD. Also, added
#                 the parameter canonical to the mps method set_site().
#                 It can take values 'L', 'C', or None (default)
#                 Make sure that all methods/functions in the module
#                 access the A[i] tensors in the mps class via the
#                 set_site method.
#
# 10-Dec-2022 --- Updated some comments in reduceD()
#
# 16-Dec-2022 --- Added updateCRight function, which is similar
#                 to updateCLeft, which already exists.
#                 It is used for calculating the inner product
#                 of two MPSs
#
# 5-Feb-2023  --- Add the flag nr_bulk to reduceD() and right_canonical().
#                 if True, then when changing the MPS, bulk tensors are
#                 normalized and the overall norm is absorbed in the 
#                 left-most tensor (i=0). Currently only available in 
#                 right_canonical() and reduceD() with mode='MC' or 'RC'
#
#
# 5-Feb-2023  --- Add the variables nr_mantissa, nr_exp to the mps class.
#                 These hold an overall normalization of the MPS in the 
#                 form of nr_mantissa * 10**nr_exp. So basically we
#                 want 1<abs(nr_mantissa)<10.
#
#                 In the right_canonical method, when we set nr_bulk=True,
#                 then updating the norm of A0, will triger an update
#                 of the overall scale stored in nr_mantissa, nr_exp.
#
#
#                 (*) The new method overall_factor returns that factor
#                 (*) The new method update_A0_norm normalizes A0 and
#                     updates nr_manitssa, nr_exp accordingly.
#
#                 In addition, the functions fexp, fman were added to 
#                 the module to perform the mantissa/exp calculus.
#
#                 
#  12-Apr-2023 --- Added the add_two_MPSs function that adds two MPSs
#
#  23-Oct-2023 --- Made the SVD operation in left_canonical() and 
#                  right_canonical() more resiliant to error using
#                  try: and except:
#
#  27-Dec-2023 --- in update_A0_norm and right_canonical: when 
#                  normalizing A0 using set_site then preserve its
#                  canonical order.
#
#  13-Jan-2024 --- Copy the mantissa info in the copy() and reverse()
#                  methods, and take it into account in the functions
#                  mps_inner_product, mps_sandwitch
#
#  20-Jan-2024 --- 
#  (-) Introduced the PMPS (Purification MPS) framework:
#     1. Introduced a variable mtype that can be either 'PMPS' or 'MPS'
#     2. Introduced a list of integers Ps to hold the local purification 
#        dim
#     3. Additional methods: get_P, set_P, set_mtype
#     4. Updated the mps_shape() function to include the P dim
#
#
#  (-) General methods:
#      1. set_mps_lists, get_mps_lists (for the 3 lists self.A, 
#         self.Corder, self.Ps)
#      2. resize_mps
#      3. PMPS_to_MPS
#
#  (-) In reduceD, enlarge the region where truncation is performed, 
#      by including also sites with D<=maxD, but which can yet be 
#      truncated (DL*d<DR or DL>d*DR)
#
#
#  5-Mar-2024: added the method reset_nr() to the mps class
#
#  19-Apr-2024: In the MPO class, changed the indices order to
#               [L, U, D, R]. Changed corresponding funtions: 
#               mpo_shape, applyMPO, updateCOLeft, updateCORight
#
#
#  27-Apr-2024: 1) Add the optional mode parameter to mps.copy(). If
#               mode='full-copy', then it will copy also the A[i]
#               tensors of the MPS. 
#               2) Added an output of the truncation error to 
#               the mps.right_canonical() function and via that to
#               reduceD(). Now both functions return the truncation
#               error, which is the L_2 norm of the discarded SVD
#               singular values, normalized by the total L_2 norm of all
#               the SVD singular values.
#
#
#  3-May-2024: Implemented the reduceDiter mps method, which compresses
#              the MPS using an iterative series of optimizations sweeps
#              *without* SVD (only QR). This may be much faster than
#              the SVD-based reduceD() for large bond dimensions.
#
#  4-May-2024: Fixed minor bug in iD1 selection in reduceDiter
#
#  22-May-2024: Did a small work-around in mps.reduceD(): when mode='LC', 
#               forced it to return truncation_error=None --- but this
#               should be fixed later.
#
#----------------------------------------------------------
#

"""


"""


import numpy as np
import scipy as sp
import scipy.io as sio

from scipy.sparse.linalg import eigs

from numpy.linalg import norm, svd, qr
from scipy.linalg import rq
from quimb.linalg.rand_linalg import rsvd

from numpy import zeros, ones, array, tensordot, sqrt, diag, dot, \
	reshape, transpose, conj, eye, trace, floor, log10
from sys import exit
from copy import deepcopy


############################ CLASS MPS #################################
#                                                                      #
#                                                                      #
# Main variables:                                                      #
# ----------------                                                     #
#                                                                      #
#                                                                      #
# d - physical dimension                                               #
#                                                                      #
# N - No. of sites                                                     #
#                                                                      #
# A[] - A list of the MPS matrices. Each matrix is an array            #
#       of the shape [Dleft, d, Dright]                                #
#                                                                      #
#                                                                      #
########################################################################


class mps:

	#
	# --- init
	#
	def __init__(self,N, mtype='MPS'):


		self.N = N
		
		self.A : list[np.ndarray] = [None]*N    #type: ignore
		self.Corder = [None]*N  # Canonical order of each element
		self.Ps = [1]*N
		
		self.nr_mantissa = 1.0
		self.nr_exp = 0
		
		self.set_mtype(mtype)
	
	
	#
	# --- get_mps_lists
	#
	def get_mps_lists(self):
		
		return self.A, self.Corder, self.Ps
	
		
	#
	# --- set_mps_lists
	#
	def set_mps_lists(self, A=None, Corder=None, Ps=None):
		
		"""
		
		Set the 3 arrays that contain the MPS information:
		(*) A       --- The tensors of the MPS
		(*) Corder  --- The canonical order of each tensor
		(*) Ps      --- The purification leg dim (for PMPS)
		
		Once updated, we also adjust the N (which forces all 
		lists to have the same length)
		
		"""
		
		if A is not None:
			self.A = A.copy()

		if Corder is not None:
			self.Corder = Corder.copy()
			
		if Ps is not None:
			self.Ps = Ps.copy()
			
		#
		# Now resize the MPS to have the size of A
		#
			
		N = len(self.A)
		
		self.resize_mps(N)
		

	#
	# --- get_Ps
	#
	def get_Ps(self):
		
		return self.Ps

	#
	# --- get_P
	#
	def get_P(self,i):
		
		return self.Ps[i]
		

	#
	# --- set_P
	#
	def set_P(self,P,i):
		
		self.Ps[i] = P
		



	#
	# --- resize_mps
	#
	def resize_mps(self, N):
		
		"""
		
		Resizes an MPS. This means that we need to change the size of the
		3 lists: self.A, self.Corder, self.Ps
		
		If the lists are longer than the new N --- we just cut them at N.
		
		If they are shorter --- we pad them
		
		"""
		
		self.N = N
		
		A_len = len(self.A)
		if  A_len > N:
			self.A = self.A[:N]
		else:
			self.A = self.A + [None]*(N-A_len)
		
		Corder_len = len(self.Corder)
		if  Corder_len > N:
			self.Corder = self.Corder[:N]
		else:
			self.Corder = self.Corder + [None]*(N-Corder_len)
		
		Ps_len = len(self.Ps)
		if  Ps_len > N:
			self.Ps = self.Ps[:N]
		else:
			self.Ps = self.Ps + [1]*(N-Ps_len)
		




		
	#
	# --- set_mtype
	#
	
	def set_mtype(self, mtype):
	
		if mtype not in ['MPS', 'PMPS']:
			print("bmpslib error: mps object type can be either 'MPS' or 'PMPS'")
			exit(1)

		self.mtype = mtype
		
	#
	# ------------ update_A0
	#
	
	def update_A0_norm(self):
		
		nr_A0 = norm(self.A[0])
		self.set_site(self.A[0]/nr_A0, 0, self.Corder[0])
		
		A0_exp = fexp(nr_A0)
		A0_mantissa = fman(nr_A0)
		
		self.nr_mantissa *= A0_mantissa
		self.nr_exp += A0_exp
		
		if abs(self.nr_mantissa)>= 10 or abs(self.nr_mantissa)<1:
			man_exp = fexp(self.nr_mantissa)
			
			self.nr_mantissa = fman(self.nr_mantissa)
			self.nr_exp += man_exp
			
			
	#
	# --------------- overall_factor
	#
		
	def overall_factor(self):
		return self.nr_mantissa * 10**self.nr_exp
		


	#
	# ------------ reset_nr
	#
	
	def reset_nr(self):
		"""
		
		Set the overall normalization factor of the MPS to 1
		
		"""
		
		self.nr_mantissa = 1
		self.nr_exp = 0
		

	#--------------------------   maxD   ---------------------------
	#
	# Returns the max bond dimension of the MPS
	#
	
	def maxD(self):
		D_list = [self.A[0].shape[0]] \
			+ [self.A[i].shape[2] for i in range(self.N)]
			
		return max(D_list)
		

	#	
	#--------------------------   copy   ---------------------------
	#
	def copy(self, mode='list'):
		
		"""
		
		Returns a copy of the mps class. If mode=='list' then only
		the lists are copied, but the list items are not. 
		If mode=='full-copy', then also the items in the lists (i.e., 
		the A tensors) are copied.
		
		Input Parameters:
		------------------
		mode --- Either 'list' or 'full-copy'
			
		Output
		------
		An identical mps object.
		
		"""
		
		new_mps = mps(self.N, self.mtype)
		new_mps.nr_mantissa = self.nr_mantissa
		new_mps.nr_exp = self.nr_exp
		
		new_mps.Corder = self.Corder.copy()

		if self.mtype=='PMPS':
			new_mps.Ps = self.Ps.copy()

		if mode=='list':
			new_mps.A = self.A.copy()
		else:
			for i in range(self.N):
				new_mps.A[i] = self.A[i].copy()
		
		return new_mps
	
		
	#--------------------------   reverse   ---------------------------
	#
	# Returns an MPS with the order of the tensors reversed: the first
	# tensor becomes the last etc.
	#
	#
	
	def reverse(self):
		
		new_mps = mps(self.N)
		
		new_mps.nr_mantissa = self.nr_mantissa
		new_mps.nr_exp = self.nr_exp
		
		if self.mtype=='PMPS':
			self.Ps.reverse()
		
		for i in range(self.N):
			A = self.A[self.N-1-i]
			
			new_A = A.transpose([2,1,0])  # also reverse the left/right
			                                     # indices
			Corder = self.Corder[self.N-1-i]
			if Corder=='L':
				new_Corder='R'
			elif Corder=='R':
				new_Corder='L'
			else:
				new_Corder=None
				
			new_mps.set_site(new_A, i, new_Corder)
			
		return new_mps
		

	#--------------------------  set_site   ---------------------------
	#
	#  Sets the tensor of a site in the MPS
	#
	#  We expect an array of complex numbers of the form M[D1, d, D2]
	#
	#  The D1 should match the D2 of the previous (i.e., the last)
	#  entry in the MPS.
	#
	#  Corder is an optinal parameter, specifying the local canonical
	#  order of the MPS tensor.
	#

	def set_site(self, mat, i, Corder=None):

			self.A[i] = mat.copy()
			
			self.Corder[i] = Corder


	#-------------------------    mps_shape   ----------------------------
	#
	# Returns a string with the dimensions of the matrices that make
	# up the MPS
	#
	def mps_shape(self):
		
			
		mpshape = ''

		for i in range(self.N):
			
			if self.A[i] is None:
				mpshape = mpshape + ' A_{}=None '.format(i)

			else:
				order_str ='o '
				if self.Corder[i]=='L': 
					order_str = '< '
				elif self.Corder[i]=='R':
					order_str='> '
				
				DL, d, DR = self.A[i].shape[0], self.A[i].shape[1], \
					self.A[i].shape[2]
					 
				if self.mtype=='PMPS':
					P = self.Ps[i]
					mpshape = mpshape + f" {order_str}A_{i}({DL} {d//P}*{P} {DR}) "
				else:
					mpshape = mpshape + f" {order_str}A_{i}({DL} {d} {DR}) "

		return mpshape


	#----------------------   left_canonical_QR  -------------------------
	#
	# Turns the MPS into a canonical left form using the QR decomposition
	#
	# Input parameters:
	# ------------------
	#
	# i0,i1 --- possible range of tensors [i0,i1] on which to perform.
	#           if omitted, the process is on all tensors.
	#

	def left_canonical_QR(self, i0=None, i1=None):
		if self.N<2:
			return

		if i0 is None:
			i0=0
			
		if i1 is None:
			i1=self.N-2
			
		if i1>self.N-2:
			i1 = self.N-2

		for i in range(i0,i1+1):
			
			#
			# If it is already left canonical --- then nothing to do
			#
			
			if self.Corder[i]=='L':
				continue
			

			D1 = self.A[i].shape[0]
			d  = self.A[i].shape[1]
			D2 = self.A[i].shape[2]

			M = self.A[i].reshape(D1*d, D2)

			Q,R = qr(M)
						
			myD = Q.shape[1]

			# 
			# Set A[i], and A[i+1]. A[i+1] has undefined canonical-order
			#
			
			self.set_site(Q.reshape(D1, d, myD), i, 'L')

			self.set_site(tensordot(R,self.A[i+1], axes=([1],[0])),i+1)
			

		return




	#----------------------   left_canonical   -------------------------
	#
	# Turns the MPS into a canonical left form. If maxD is specified
	# then the singualr values are truncated at this value. If the
	# given MPS is at right-canonical form, then the singular values
	# are simply the Schmidt coefficients, and the truncation is
	# optimal.
	#

	def left_canonical(self, maxD=None, eps=None, svd_emthod:str="rsvd"):
		if self.N<2:
			return		

		for i in range(self.N-1):

			D1 = self.A[i].shape[0]
			d  = self.A[i].shape[1]
			D2 = self.A[i].shape[2]

			M = self.A[i].reshape(D1*d, D2)
				
			try:
				U,S,V = _perf_svd(M, svd_emthod)
			except:
				M = M + np.random.randn(*M.shape)*norm(M)*1e-12
				U,S,V = _perf_svd(M, svd_emthod)




			if eps is not None:
				cutoff = S[0]*eps
								
				S_eff = S[S>cutoff]
			else:
				S_eff = S
			
			if maxD is None:
				myD = len(S_eff)
			else:
				myD = min(len(S_eff),maxD)

			S = S[0:myD]
			U = U[:,0:myD]
			V = V[0:myD,:]

			# Set the site as left-canonical
			self.set_site(U.reshape(D1, d, myD), i, 'L')

			V = diag(S)@V

			# Set the site with undefined canonical order
			self.set_site(tensordot(V,self.A[i+1], axes=([1],[0])), i+1)

		return





	#----------------------   right_canonical   -------------------------
	#
	# Turns the MPS into a right-canonical form. If maxD is specified
	# then the singualr values are truncated at this value. If the
	# given MPS is at left-canonical form, then the singular values
	# are simply the Schmidt coefficients, and the truncation is
	# optimal.
	#
	#
	# Input parameters:
	# -----------------
	#
	# maxD, eps --- truncation parameters
	# i0,i1     --- A possible range of tensors [i0,i1] on which to 
	#               perform the truncation.
	#
	# nr_bulk   --- If true then normalizes all the tensors it
	#               modifies, absorbing the overall norm in the
	#               left-most tensor (i=0).
	#
	#
	#  Output:
	#  -------
	#  The normalized truncation order, which is the L_2 norm of the 
	#  discarded singular values divided by the L_2 of all values.
	#
	#

	def right_canonical(self, maxD=None, eps=None, i0=None,i1=None, \
		nr_bulk = False, svd_emthod:str="rsvd"):
			
		if self.N<2:
			return

		if maxD is None:
			maxD = 10000000

		if i0 is None:
			i0=1
		if i1 is None:
			i1=self.N-1

		#
		# Holds the overall normalization (if normalize_bulk=True)
		#
		overall_norm = 1.0

		truncation_error = 0.0

		for i in range(i1,i0-1,-1):

			if np.isnan(np.sum(self.A[i])):
				print(f"Error in bmps.right_canonical!!! Tensor A[{i}] contains nan values")
				exit(1)
				
			if np.isinf(np.sum(self.A[i])):
				print(f"Error in bmps.right_canonical!!! Tensor A[{i}] contains inf values")
				exit(1)

			
			D1 = self.A[i].shape[0]
			d  = self.A[i].shape[1]
			D2 = self.A[i].shape[2]

			M = self.A[i].reshape(D1, d*D2)
			
			#
			# See if we need any truncation (D1>maxD). If needed - we use SVD,
			# otherwise, use RQ trans.
			#

			err = 0.0

			if D1>maxD or eps is not None:
				
				#
				# Use SVD for truncation
				#
				
				try:
					U,S,V = _perf_svd(M, svd_emthod)
				except:
					M = M + np.random.randn(*M.shape)*norm(M)*1e-12
					U,S,V = _perf_svd(M, svd_emthod)
				
				if nr_bulk:
					nrS = norm(S)
					S = S/nrS
					overall_norm *= nrS
				
				
				if eps is None:
					S_eff = S
				else:
					cutoff = S[0]*eps
					S_eff = S[S>cutoff]
				
				myD = min(len(S_eff),maxD)
				myD = _assert_int(myD)

				err = sqrt(sum(S[myD:]**2)/sum(S**2))
				
				S = S[0:myD]
				U = U[:,0:myD]
				V = V[0:myD,:]
				

				self.set_site(V.reshape(myD, d, D2), i, 'R')

				U = U@diag(S)

				# Set A[i-1] with an undefined canonical order
				self.set_site(tensordot(self.A[i-1], U, axes=([2],[0])), i-1)


			else:
				
				#
				# No truncation needed - use RQ decomp'
				#
				
				#
				# if it is already-right canonical, then nothing to do
				#
				
				if self.Corder[i]=='R':
					continue

				R,Q = rq(M, mode='economic')

				if nr_bulk:
					nrR = norm(R)
					R = R/nrR
					overall_norm *= nrR
#
# Alternativly, use QR
#
#				Q,R = qr(M.T)
#				Q = Q.T
#				R = R.T
							
				myD = Q.shape[0]

				self.set_site(Q.reshape(myD, d, D2), i, 'R')

				# A[i-1] is updated with an undefined canonical order
				self.set_site(tensordot(self.A[i-1], R, axes=([2],[0])), i-1)
				
			
			truncation_error += err
			
			
		#
		# if nr_bulk=True, we absorb the overall norm factor in 
		# the left-most tensor (i=0)
		#
		
		
		if nr_bulk:
			self.set_site(self.A[0]*overall_norm, 0, self.Corder[0])
			self.update_A0_norm()

		return truncation_error




	#-------------------------    reduceD   ----------------------------
	#
	# Reduce the bond dimension of the MPS by truncating its Schmidt 
	# coefficients whenever the bond dimension > maxD.
	#
	# This is done in one of 3 ways:
	#
	# mode=='RC': make the MPS left-canonical, and then right-canonical
	#             using SVD + truncation.
	#
	# mode=='LC': make the MPS right-canonical, and then left-canonical
	#             using SVD + truncation.
	#
	# mode=='MC': This is the most efficient way. As a pre-calculation, 
	#             locate the range where truncation is needed, and 
	#             locate the minial ranges where left/right canonical 
	#             have to be applied in order to truncate that range. 
	#
	#             This means that maxD *CANNOT* be None.
	#
	#             The output is a an MPS with a *mixed* canonical state
	#             in which there is some central i with undefined 
	#             canonical state, and the tensors to its left are 
	#             left-canonical and to its right are right-canonical:
	#
	#                            < < < < o > > >
	#
	# 
	#
	# Input Parameters:
	# -----------------
	#  maxD, eps --- truncation parameters
	#  mode --- one of the 3 modes 'LC', 'RC', or 'MC' (default)
	#
	#  nr_bulk --- If True, then normalizes all the bulk tensors it 
	#              modifies and absorbs the overall normalization in 
	#              the left-most tensor (i=0). Currently not compatible 
	#              with 'LC' mode.  Default: False
	#
	#

	def reduceD(self, maxD=None, eps=None, mode='MC', nr_bulk=False):

		#
		# Make sure that there's something to reduce. We need at least
		# 3 sites
		#

		if self.N<3:
			return

		if mode=='MC':
			
			if maxD is None:
				print("Error in reduceD(): maxD=None is incompatible with "\
					"mode='MC'")
				exit(1)
				
			
			#
			# We perform compression by Left-canonical, followed by 
			# Right-canonical + trunc. But all of this is done on a *minimal*
			# number of sites.
			#
			
			#
			# first, find the sites [iD0,iD1] where we should truncate the
			# right-side bond.
			#
			# A bond needs truncation if either:
			# (1) It is larger than maxD
			# (2) It is larger than DL*d of the left tensor to which it connects
			# (3) It is larger than DR*d of the right tensor to which it connects
			#
			
			iD0=None
			for i in range(self.N-1):
				if self.A[i].shape[2]>min(maxD, \
					self.A[i].shape[0]*self.A[i].shape[1], \
					self.A[i+1].shape[2]*self.A[i+1].shape[1]):
						
					iD0 = i
					break

			#
			# If there's no site that needs truncation, we quit.
			#
			if iD0 is None:
				return 0

			for i in range(self.N-2, -1, -1):
				if self.A[i].shape[2]> min(maxD, \
					self.A[i].shape[0]*self.A[i].shape[1], \
					self.A[i+1].shape[2]*self.A[i+1].shape[1]):
					
					iD1 = i
					break
				
		
			#
			# Find the i0<=iD0 where Left-canonical should start from 
			# (the right most left-canonical site).
			#
			
			for i in range(iD0+1):
				if self.Corder[i] != 'L':
					break
			i0 = i
					
			#
			# Perform left-canonicalization in [i0, iD1] (the right boundary
			# is the the minimal one we need because of truncation)
			#
			
			self.left_canonical_QR(i0,iD1)

			#
			# Now find where the right-canonicalization should start from
			# (the right-most site of the range)
			#
			for i in range(self.N-1, iD1-1, -1):
				if self.Corder[i] != 'R':
					break
					
			i1 = i
			
			#
			# Perform right-canonicalization + reduction on the range 
			# [iD0+1, i1]. Note that this will also reduce the bond iD0.
			#
					
			truncation_error = self.right_canonical(maxD, eps, i0=iD0+1, i1=i1, \
				nr_bulk=nr_bulk)
		
		elif mode=='RC':
			self.left_canonical_QR()
			truncation_error = self.right_canonical(maxD, eps, nr_bulk=nr_bulk)
			
		else:
			if nr_bulk:
				print("Error in reduceD(): nr_bulk=True is incompatible with "\
					"mode='LC'")
				exit(1)

			
			self.right_canonical()
			self.left_canonical(maxD, eps)
			
			# TODO: add truncation error in the LC case
			truncation_error = None

		return truncation_error







	#
	# --------------------------  reduceDiter  ---------------------------
	#
	def reduceDiter(self, maxD, nr_bulk=False, max_iter=10, err=1e-6):
		
		"""

	Performs a compression of the MPS bond dimension using an iterative
	optimization that does not invlove SVD. For very large tensors this
	method can be much faster than the regular SVD method (done reduceD)
	and use much less memory, but it might be a bit less accurate. 
	Additionally, at small bond dims, SVD will be much faster.
	
	Given a maximal bond dimension maxD, the function first locates a 
	minimal segment [i0,i1] where truncation should be performed. Then
	it sweeps left/right on that segement in a DMRG-like manner using
	only the QR decomposition. The iteration stops once the maximal number
	of iterations has reached or the accuracy thereshold obtained.
	
	The outcome is mixed-canonical MPS with bond dimension <= maxD. 
	It is left/right canonical up the middle of the truncation range. 
	More precisely, if i_mid is the middle of the region [i0, i1] then 
	the output MPS is left-canonical in [0,i_mid-1] and right-canonical in 
	[i_mid+1, N-1]
	
	
	Input Parameters:
	-----------------
	
	maxD    --- The maximal bond dimension
	
	nr_bulk --- Whether or not the normalization of the truncated tensors 
	            should be absorbed in A[0], and through that into the
	            mantissa.
	
	max_iter --- maximal number of optimization iterations 
	
	err      --- Accuracy threshold. This is roughly the average 
	             variation in the distance between the truncated and 
	             un-truncated MPSs along a complete optimization sweep.
	             
	             
	Output:
	-------
	None.
	
	


		"""		

		log = False
		
		N = self.N
		
		#
		# Step I:
		# ------------------------------------------------------------------
		#
		# Find the range [iD0, iD1] where truncation should be done
		# and the larger range [i0 [iD0,iD1] i1] such that:
		#
		#  (*) i0<=iD0 is the maximal site for which [0,i0] are 
		#      left-canonical.
		#  (*) i1>=iD1 is the minimal site for which [i1,N-1] are 
		#      right-canonical
		#
		
		
		# Find iD0, i0
		iD0=None
		i0Found = False
		for i in range(N-1):
			
			if not i0Found and self.Corder[i] != 'L':
				i0Found = True
				i0 = i
				
			if self.A[i].shape[2]>min(maxD, self.A[i].shape[0]*self.A[i].shape[1]):
				iD0 = i
				break
				
		if not i0Found:
			i0 = iD0
			

		#
		# If there's no site that needs truncation, we quit.
		#
		if iD0 is None:
			return
		
		# Find iD1, i1
		iD1 = None
		i1Found = False
		for i in range(N-1, iD0, -1):
			
			if not i1Found and self.Corder[i] != 'R':
				i1Found = True
				i1 = i
			
			if self.A[i].shape[0]> min(maxD, self.A[i].shape[2]*self.A[i].shape[1]):
				iD1 = i
				break
		
		if iD1 is None:
			iD1 = iD0 + 1
		

		if not i1Found:
			i1 = iD1



		#
		# Step II:
		# ------------------------------------------------------------------
		#
		# Create This is mpA as an initial guess of the truncated MPS 
		# tensors in the segmemt [i0,i1]. We do that by simply truncating
		# the high indices of the original MPS tensors in that segment.
		#

		NA = i1-i0+1

		mpA = mps(NA)
		
		for i in range(NA):
			B = self.A[i+i0]
			
			sh = B.shape
			DL, d, DR = sh[0], sh[1], sh[2]
			
			DL2, d2, DR2 = min(DL,maxD), d, min(DR,maxD)
			
			A = B[:DL2,:,:DR2]
			
			mpA.set_site(A,i)
			
		#
		# Step II:
		# --------
		#
		# Make mpA right-canonical and update the RBA list, which holds
		# the right-side envs of the <A|B> inner product.
		#
		

		# 
		# Define two lists that would hold the left/right envs of 
		# the <mpA|mpB> contraction
		#
		LBA = [None]*NA
		RBA = [None]*NA

		# The initial right-env is the identity because the original 
		# MPS is right canonical up to i1.
		DR = mpA.A[NA-1].shape[2]
		CRBA = eye(DR)
		
		for i in range(NA-1,0,-1):
			
			#
			# Make A right-canonical using the RQ decomp'
			#
			A = mpA.A[i]
			sh = A.shape
			M = A.reshape([sh[0], sh[1]*sh[2]])
			
			# Perform RQ by QR + transpose
			Q,R = qr(M.T)
			Q,R = Q.T,R.T
			
			Chi = Q.shape[0]
			
			Q = Q.reshape([Chi, sh[1], sh[2]])

			mpA.set_site(Q, i, Corder='R')
			mpA.set_site(tensordot(mpA.A[i-1],R, axes=([2],[0])),i-1)

			CRBA = updateCRight(CRBA, self.A[i0+i], mpA.A[i], conjB=True)
			
			RBA[i] = CRBA
			

		#
		# Step III:     The Left <---> Right optimization sweeps
		# ------------------------------------------------------------------
		#
		
		S = 0
		S2 = 0
		ell = 0

		if log:
			print(f"\n\n ====== reduceDiter: Entering sweeps in range [{i0}...{i1}] =====\n")
		
		exit_loop=False
		for k in range(max_iter):
			
			
			#
			# The S, S2 variables hold the sum of the distance and 
			# distance^2 between |A> and |B> along an optimization sweep.
			# We use S,S2 to calculate the variation in the distance, 
			# which is used as a stopping criteria
			#
			
				
			
			# =============================================
			# Left => Right sweep
			# =============================================
					
			calc_dist = True
			dist = 0
			
			DL = mpA.A[0].shape[0]
			CL = eye(DL)
			
			if log:
				print(f"[*] LEFT=>RIGHT SWEEP (round {k})")
			
			for i in range(NA-1):
				
				B = self.A[i+i0]
				
				CR = RBA[i+1]
				
				A = tensordot(B, CR, axes=([2],[0]))
				A = tensordot(CL, A, axes=([0],[0]))
				
				nr2 = norm(A)**2
				
				S += nr2
				S2 += nr2**2
				ell += 1
				
				#
				# Make A left-canonical
				#
				
				sh = A.shape
				M = A.reshape([sh[0]*sh[1], sh[2]])
				Q,R = qr(M)
				
				Chi = Q.shape[1]
				
				Q = Q.reshape([sh[0], sh[1], Chi])
				

				mpA.set_site(Q, i, Corder='L')
				mpA.set_site(tensordot(R,mpA.A[i+1], axes=([1],[0])),i+1)
				
				CL = updateCLeft(CL, B, Q, conjB=True)
				
				LBA[i] = CL

			# =============================================
			# Right => Left sweep
			# =============================================


			DR = mpA.A[NA-1].shape[2]
			CR = eye(DR)
			
			if log:
				print(f"[*] LEFT<=RIGHT SWEEP  (round {k})")
				
			for i in range(NA-1, 0, -1):
				
				B = self.A[i+i0]
				
				CL = LBA[i-1]
				
				A = tensordot(B, CR, axes=([2],[0]))
				A = tensordot(CL, A, axes=([0],[0]))
				
				#
				# Up to an additive constant, ||A^2|| is proportional to the 
				# distance || |A> - |B> ||^2
				#
				nr2 = norm(A)**2
				
				
				S += nr2
				S2 += nr2**2
				ell += 1
											
				#
				# Make A right-canonical
				#
				
				sh = A.shape
				M = A.reshape([sh[0], sh[1]*sh[2]])
				
				# perform RQ by QR + transpose
				Q,R = qr(M.T)
				Q,R = Q.T,R.T
				
				Chi = Q.shape[0]
				Q = Q.reshape([Chi, sh[1], sh[2]])
							
				
				mpA.set_site(Q, i, Corder='R')
				mpA.set_site(tensordot(mpA.A[i-1], R, axes=([2],[0])),i-1)
					
				
				CR = updateCRight(CR, B, Q, conjB=True)
				
				RBA[i] = CR
				
				if i==NA//2 and k>0:
				
					if log:
						print(f"\nGot to mid (i={i}): S2={S2:.6g}  S={S:.6g}")
					
					S2 = S2/ell
					S = S/ell
			
					Delta = abs((S2-S**2)/S**2)
			
					
					if log:
						print(f"Normalized: S2={S2:.6g}  S={S:.6g}  Delta={Delta:.6g}")
		
					if Delta<err or k==max_iter-1:
						if log:
							print("\n       * * *  BREAKING  * * *\n")
						exit_loop = True
						break
						
					S2 = 0
					S = 0
					ell = 0
			
			if exit_loop:
				break
					
					
				


		if log:
			print(f"=> reduceDiter done with {k+1} rounds. "\
				f"Delta={Delta:.6g}\n\n")

		#
		# Step IV:     Merge mpA into mpB and exit
		# ------------------------------------------------------------------
		#


		#
		# Now replace the tensors of mpB in the [i0,i1] segment with those
		# of mpA
		#

		self.A[i0:(i1+1)] = mpA.A
		self.Corder[i0:(i1+1)] = mpA.Corder
			
		#
		# If nr_bulk = True, then we move the overall normalization
		# (which is found in A[i0] since we are right-canonical in the 
		# [i0,i1] segment) to A[0] and from there to the mantissa
		# via the update_A0_norm() function.
		#		
		if nr_bulk:
			if i0>0:
				A0 = self.A[i0]
				nr = norm(A0)
				self.set_site(A0/nr, i0)
				
				A0 = self.A[0]
				self.set_site(A0*nr, 0, mpA.Corder[0])
				
			self.update_A0_norm()
		
		return






























	#
	# ---------------------------- PMPS_to_MPS ----------------------------
	#

	def PMPS_to_MPS(self):

		if self.mtype != 'PMPS':
			print("bmpslib.PMPS_to_MPS error: can only be used on PMPS "\
				f"(and here mtype='{self.mtype}')")
			exit(1)

		N = self.N
		
		mp = mps(N)
		
		for i in range(N):
			#
			# Seperate the purifing leg from the physical leg in A[i]
			#
			sh = self.A[i].shape
			A =  self.A[i].reshape([sh[0], sh[1]//self.Ps[i], self.Ps[i], sh[2]])

			# Now A has the form [DL, d, P, DR], where P is the purifying leg

			# Contract A[i] with its complex conjugate along the purifying leg
			B = tensordot(A, conj(A), axes=([2],[2]))

			#
			# Now B is of the form [DL, d, DR, DL*, d*, DR*].
			#
			# Transpose it to [DL,DL*; d,d*; DR,DR*] and fuse the ket-bra
			# legs to make it a ket-bra MPS
			#

			B = B.transpose([0,3,1,4,2,5])
			sh = B.shape
			B = B.reshape([sh[0]*sh[1], sh[2]*sh[3], sh[4]*sh[5]])

			mp.set_site(B, i)

		mp.nr_mantissa = abs(self.nr_mantissa**2)
		mp.nr_exp = 2*self.nr_exp

		return mp





############################ CLASS PEPS #################################
#                                                                      #
#                                                                      #
# Main variables:                                                      #
# ----------------                                                     #
#                                                                      #
# The PEPS is given as an MxN matrix with M rows and N columns         #
#                                                                      #
# M - No. of rows                                                      #                                                                     #
# N - No. of columns                                                   #
#                                                                      #
# A[][] - A double list of the PEPS matrices. Each matrix is an array  #
#         of the shape [d, D_left, D_right, D_up, D_down]              #
#         d - physical leg                                             #
#         D_left/right/up/down - virtual legs                          #
#                                                                      #
#         A[i][j] is the tensor of the i'th row and the j'th column.   #
#                                                                      #
#                                                                      #
# Notice: Also at the boundaries every A should have exactly the  5    #
#         indices (d,Dleft,Dright,Dup,Ddown). It is expected that the  #
#         "un-needed" legs will be of dimension 1. For example, at the #
#         top-left corner A should be of the form                      #
#                         [d,1,Dright,1,Ddown]                         #
#                                                                      #
########################################################################


class peps:

	#
	# --- init
	#

	def __init__(self,M,N):

		self.M = M
		self.N = N
		self.A = [[None]*N for i in range(M)]

	#--------------------------   set_site   ---------------------------
	#
	#  Sets the tensor of a site in the PEPS at row i, column j.
	#
	#  We expect A to be a tensor of the form:
	#                A[d, Dleft, Dright, Dup, Ddown]
	#

	def set_site(self, mat, i,j):

			self.A[i][j] = mat.copy()


	#-------------------------    peps_shape   ----------------------------
	#
	# Returns a string with the dimensions of the matrices that make
	# up the PEPS
	#
	def peps_shape(self):
		peps_shape = ""

		for i in range(self.M):
			for j in range(self.N):
				if (self.A[i][j] is None):
					peps_shape = peps_shape +  " A_{}{}(---)".format(i,j)
				else:
					peps_shape = peps_shape +  " A_{}{}({} {} {} {}) ".\
					format(i,j,self.A[i][j].shape[1],self.A[i][j].shape[2],\
					self.A[i][j].shape[3],self.A[i][j].shape[4])

			peps_shape = peps_shape + "\n"

		return peps_shape




	#-------------------------    calc_line_MPO   ----------------------------
	#
	# Gets a row number or a column number and return its corresponding MPO
	#
	# Parameters:
	# -----------
	# ty  -  Type of the MPO. Could be either 'row-MPS' or 'column-MPS'
	# i   -  The row/column number. Runs between 0 -> (M-1) or (N-1)
	#

	def calc_line_MPO(self, ty, i):

		if ty=='row-MPS':
			#
			# Create an MPO out of ROW i
			#
			n = self.N
			bmpo = mpo(n)

			for k in range(n):
				A0 = self.A[i][k]

				Dleft = A0.shape[1]
				Dright = A0.shape[2]
				Dup = A0.shape[3]
				Ddown = A0.shape[4]

				# contract the physical legs
				# Recall that in MPS the legs are: [d, Dleft, Dright, Dup, Ddown]
				#
				A = tensordot(A0,conj(A0), axes=([0],[0]))
				#
				# resulting tensor is [Dleft,Dright,Dup,Ddown ; Dleft',Dright',Dup',Ddown']
				#                        0      1     2   3        4     5      6    7

				# Recall the MPO site is of the form [dup,ddown,Dleft, Dright]

				A = A.transpose([2,6,3,7,0,4,1,5])
				# Now its of the form [Dup, Dup', Ddown, Ddown',Dleft, Dleft',
				#                       Dright,Dright']


				A = A.reshape([Dup*Dup, Ddown*Ddown, Dleft*Dleft, Dright*Dright])

				bmpo.set_site(A, k)

		else:

			#
			# Create an MPO out of COLUMN i
			#
			n = self.M

			bmpo = mpo(n)

			for k in range(n):
				A0 = self.A[k][i]

				Dleft = A0.shape[1]
				Dright = A0.shape[2]
				Dup = A0.shape[3]
				Ddown = A0.shape[4]


				# contract the physical legs
				A = tensordot(A0,conj(A0), axes=([0],[0]))
				#
				# resulting tensor is [Dleft,Dright,Dup,Ddown ; Dleft',Dright',Dup',Ddown']
				#                        0      1     2   3        4     5    6    7


				# Recall the MPO site is of the form [dup,ddown,Dleft, Dright]
				# but here since its a vertical MPO, then:
				#
				#            up<->left  and   down<->right.
				#

				# So we need to transpose it to Dleft, Dleft', Dright,Dright',
				# Dup,Dup', Ddown, Ddown'

				A = A.transpose([0,4,1,5,2,6,3,7])

				A = A.reshape([Dleft*Dleft, Dright*Dright, Dup*Dup, Ddown*Ddown])

				bmpo.set_site(A, k)



		return bmpo


	#-------------------------    calc_bMPS   ----------------------------
	#
	# Calculates the left/right/up/down boudary MPS
	#
	#

	def calc_bMPS(self, ty):

		if ty=='U':
			MPO = self.calc_line_MPO('row-MPS',0)

		elif ty=='D':
			MPO = self.calc_line_MPO('row-MPS',self.M-1)

		elif ty=='L':
			MPO = self.calc_line_MPO('column-MPS',0)
		elif ty=='R':
			MPO = self.calc_line_MPO('column-MPS',self.N-1)

		bmps = mps(MPO.N)

		for i in range(MPO.N):
			A0 = MPO.A[i]
			# A0 has the shape [dup, ddown, Dleft, Dright]

			if ty=='U' or ty=='L':
				d = A0.shape[1]      # ddown

			else:
				# ty=='D' or ty=='R'
				d = A0.shape[0]      # dup

			Dleft  = A0.shape[2]  # Dleft
			Dright = A0.shape[3]  # Dright

			A = A0.reshape([d,Dleft,Dright])

			A = A.transpose([1,0,2]) # make it [Dleft,d,Dright]

			bmps.set_site(A, i)

		return bmps






















############################ CLASS MPO #################################
#                                                                      #
#                                                                      #
# Main variables:                                                      #
# ----------------                                                     #
#                                                                      #
#                                                                      #
#                                                                      #
# N - No. of sites                                                     #
#                                                                      #
# Structure of the A's:  [D-Left, D-up, D-down, Dright]                #
#                                                                      #
#                                                                      #
########################################################################


class mpo:

	#
	# --- init
	#

	def __init__(self,N):

		self.N = N

		self.A = [None]*N

	#-------------------------    mpo_shape   ----------------------------
	#
	# Returns a string with the dimensions of the matrices that make
	# up the MPO
	#
	def mpo_shape(self):
		mpo_shape = ''

		for i in range(self.N):
			mpo_shape = mpo_shape + ' A_{}({} {} {} {}) '.\
				format(i,self.A[i].shape[0],self.A[i].shape[1], \
					self.A[i].shape[2],self.A[i].shape[3])

		return mpo_shape

	#--------------------------   set_site   ---------------------------
	#
	#  Sets the tensor of a site in the MPS
	#
	#  We expect an array of complex numbers of the form
	#  M[DL, d-up, d-down, DR]
	#
	#  The DL should match the DR of the previous (i.e., the last)
	#  entry in the MPO
	#
	#

	def set_site(self, mat, i):

			self.A[i] = mat.copy()


#
# **********************************************************************
#
#                         F U N C T I O N S
#
# **********************************************************************
#



def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman(f):
    return f/10**fexp(f)




#
# ------------------------- trim_mps  ---------------------------
#

def trim_mps(mp):

	"""

	Trims the left most and right most legs of an MPS.

	In an MPS object all tensors are of the form [D_L, d, D_R] --- even
	the left-most and right-most tensors.  In these tensors, usually
	D_L=1 (left most) or D_R=1 (right-most).

	What trim_mp does it to:
	(*) reshape the left-most tensor:   [1, d, D_R] ==> [d, D_R]
	(*) reshape the right-most tensor:  [D_L, d, 1] ==> [D_L, d]

	"""


	sh = mp.A[0].shape
	
	if len(sh) !=3:
		print("Error in trim_mps: A[0] does not have 3 legs")
		exit(1)
		
	if sh[0] !=1:
		print("Error in trim_mps: A[0]'s D_L != 1")
		exit(1)

	mp.A[0] = mp.A[0].reshape([sh[1], sh[2]])


	sh = mp.A[-1].shape
	
	if len(sh) !=3:
		print("Error in trim_mps: A[-1] does not have 3 legs")
		exit(1)
		
	if sh[2] !=1:
		print("Error in trim_mps: A[-1]'s D_R != 1")
		exit(1)


	mp.A[-1] = mp.A[-1].reshape([sh[0], sh[1]])




#
# ------------------------- untrim_mps  ---------------------------
#

def untrim_mps(mp):

	"""

	Untrims an MPS --- reverse the action of the trim_mps function.

	In a trimmed MPS, the left-most and right-most tensors are of the
	form [d, D_R] and [D_L, d]. The untrim_mps function takes them to 
	the usual form [1,d, D_R] and [D_L, d, 1]


	"""


	sh = mp.A[0].shape
	
	if len(sh) !=2:
		print("Error in untrim_mps: A[0] does not have 2 legs")
		exit(1)
		
	mp.A[0] = mp.A[0].reshape([1, sh[0], sh[1]])


	sh = mp.A[-1].shape
	
	if len(sh) !=2:
		print("Error in untrim_mps: A[-1] does not have 2 legs")
		exit(1)
		

	mp.A[-1] = mp.A[-1].reshape([sh[0], sh[1], 1])



#
# --------------------------   applyMPO   ---------------------------
#  

def applyMPO(M, op, j0=0, cont_leg='U', overwrite=False):
	
	"""
	
	Apply an MPO op to an MPS M. If If OverWrite=True then result is
  written to M itself.

	Input Parameters:
	------------------
	
	M         --- The MPS object
	op        --- The MPO object
  j0        --- An optional parameter specifying the initial left-most 
                site on M where op is to be applied
                
  cont_leg  --- which leg of the MPO to contract - could be 'U' (up)
                 or 'D' (Down).
             
  overwrite --- whether or not to write the new MPS into the old MPS
                object
                
  Output:
  -------
  The new MPS

	
	"""

	if overwrite:
		newM = M
	else:
		newM = M.copy()

	for j in range(j0, j0+op.N):

		#
		# Recall that the MPO legs order is  [D_left, d_up, d_down, D_right]
		# while the MPS element legs are     [D_left, d, Dright]
		#

		if cont_leg=='U':
			#
			# Contract with the upper MPO leg
			#
			newA = tensordot(M.A[j], op.A[j-j0], axes=([1],[1]))
		else:
			#
			# Contract with the lower MPO leg
			#
			newA = tensordot(M.A[j], op.A[j-j0], axes=([1],[2]))
			
		#
		#                     0    1     2     3    4
		# newA has the legs [mpDL, mpDR, opDL, d, opDR]
		#
		# transpose it to: [(mpDL, opDL), d, (mpDR, opDR)]
		
		newA = newA.transpose([0,2,3,1,4])
		
		#
		# Fuse (mpDL, opDL) and (mpDR, opDR)
		#
		
		sh = newA.shape
		newA = newA.reshape( [ sh[0]*sh[1], sh[2], sh[3]*sh[4] ])

		newM.set_site(newA, j)


	return newM


# ---------------------------------------------------------
#  enlargePEPS
#
#  Add trivial rows and columns that surround the original PEPS.
#  This is useful when calculating properties on the boundary on the
#  original PEPS. If these sites are now in the bulk then it might be
#  easier to do the calculations.
#

def enlargePEPS(p):

	M = p.M
	N = p.N

	newp = peps(M+2, N+2)

	trivialA = array([1])
	trivialA = trivialA.reshape([1,1,1,1,1])

	for i in range(N+2):
		newp.A[0][i] = trivialA.copy()
		newp.A[M+1][i] = trivialA.copy()

	for i in range(M+2):
		newp.A[i][0] = trivialA.copy()
		newp.A[i][N+1] = trivialA.copy()

	for i in range(M):
		for j in range(N):
			A = p.A[i][j]
			newp.A[i+1][j+1] = A.copy()


	return newp


#
# ------------------------  updateCOLeft  -----------------------
#  

def updateCOLeft(C, A, op, B):
	"""
	
	Contract a 3-legs tensor C with an MPO tensor and two MPS tensors
	(bottom and top)
	
	This is used for contracting a "sandwitch" of a PEPS row and 
	two boundary MPSs from top and bottom, which can be used to calculate
	local envs.
	
	
                    (B MPS)
	    +-- C-B     -----+-----         B  = [B-left, B-down, B-right]
	    |                |
	    |
	    |                |
	C = +-- C-op     ----+---- op (MPO) op = [op-left, op-up, op-down, op-right]
	    |                |
	    |
	    |                |
	    +-- C-A     -----+------        A  = [A-Left, A-up, A-Right]
	                   (A MPS)
	
	  C = [C-A, C-op, C-B]
	  
	  
	Input Parameters:
	------------------
	C  --- The 3 legs operator:   [C-A, C-op, C-B]
	
	A  --- The bottom MPS tensor: [A-left, A-up, A-right]
	
	op --- The MPO tensor:        [op-left, op-up, op-down, op-right]
	
	B  --- The top MPS tensor     [B-left, B-down, B-right]
	
	
	Output:
	-------
	
	The updated C, which is the contraction of (C, A, op, B)
	
	"""


	if C is None:
		#
		# When C=None we are on the left most side. So we create COLeft
		# from scratch
		#

		# A is  [1, A-up,   A-right]
		# Op is [1, op-up,  op-down, op-right]
		# B is  [1, B-down, B-right]
		
		# contract (A, op) along A-up, op-down
		C1 = tensordot(A[0,:,:], op[0,:,:,:], axes=([0],[1]))

		# C legs are: [A-r-ght, op-up, op-right]

		# contract (C1, B) along op-up, B-down
		C1 = tensordot(C1, B[0,:,:], axes=([1],[0]))

		# result is: [A-right, op-right, B-right]

		return C1


	# C is given as  [C-A, C-op, C-B]
	# A is given as  [A-left, A-up, A-right]
	# Op is given as [op-left, op-up, op-down, op-right]
	# B is given as  [B-left, B-down, B-right]

	# contract (C,A) along (C-A, A-left)
	
	C1 = tensordot(C, A, axes=([0], [0]))
	
	# C1 form: [op-right, B-right, A-up, A-right]
	
	# contract (C1, op) along (op-right, A-up)--(op-left, op-down)
	
	C1 = tensordot(C1, op, axes=([0,2], [0,2]))
	
	# C1 form: [B-right, A-right, op-up, op-right]
	
	# contract (C1, B) along (B-right, op-up) -- (B-left, B-down)
	
	C1 = tensordot(C1, B, axes=([0, 2], [0,1]))
	
	# C1 form: [A-right, op-right, B-right]


	return C1





#
# ------------------------  updateCORight  -----------------------
#  

def updateCORight(C, A, op, B):
	"""
	
	Contract a 3-legs tensor C with an MPO tensor and two MPS tensors
	(bottom and top)
	
	This is used for contracting a "sandwitch" of a PEPS row and 
	two boundary MPSs from top and bottom, which can be used to calculate
	local envs.
	
	
    (B MPS)
  ----+----            ---+     
	    |                   |
	                        |
	    |                   |
	----+---- op         ---+  C
	    |                   |
	                        |
      |                   |
  ----+-----           ---+ 
    (A MPS)
    
  B  = [B-left, B-down, B-right]
  op = [op-left, op-up, op-down, op-right]
  A  = [A-Left, A-up, A-Right]
	 C = [C-A, C-op, C-B]
	  

	  
	Input Parameters:
	------------------
	C  --- The 3 legs operator:   [C-A, C-op, C-B]
	
	A  --- The bottom MPS tensor: [A-left, A-up, A-right]
	
	op --- The MPO tensor:        [op-left, op-up, op-down, op-right]
	
	B  --- The top MPS tensor     [B-left, B-down, B-right]
	
	
	Output:
	-------
	
	The updated C, which is the contraction of (C, A, op, B)
	
	"""


	if C is None:
		#
		# When C=None we are on the left most side. So we create COLeft
		# from scratch
		#

		# A is  [A-left, A-up, 1]
		# Op is [op-left, op-up,  op-down, 1]
		# B is  [B-left, B-down, 1]
		
		# contract (A, op) along A-up, op-down
		C1 = tensordot(A[:,:,0], op[:,:,:,0], axes=([1],[2]))

		# C legs are: [A-left, op-left, op-up]

		# contract (B, C1) along op-up, B-down
		C1 = tensordot(C1, B[:,:,0], axes=([2],[1]))

		# C legs are: [A-left, op-left, B-left]

		return C1


	# C is given as  [C-A, C-op, C-B]
	# A is given as  [A-left, A-up, A-right]
	# Op is given as [op-left, op-up, op-down, op-right]
	# B is given as  [B-left, B-down, B-right]

	# contract (C,A) along (C-A, A-right)
	
	C1 = tensordot(C, A, axes=([0], [2]))
	
	# C1 form: [C-op, C-B, A-left, A-up]
	
	# contract (C1, op) along (C-op, A-up)--(op-right, op-down)
	
	C1 = tensordot(C1, op, axes=([0,3], [3,2]))
	
	# C1 form: [C-B,A-left,op-left,op-up]
	
	# contract (C1, B) along (C-B, op-up)--(B-right,B-down)
	
	C1 = tensordot(C1, B, axes=([0, 3], [2,1]))
	
	# C1 form: [A-left, op-left, B-left]


	return C1





# ---------------------------------------------------------
#  updateCLeft
#
#  Update the contraction of two MPSs from left to right.
#
#  If C is not empty then its a 2 legs tensor:
#  [Da,Db]
#

def updateCLeft(C, A, B, conjB=False):


	if C is None:
		#
		# When C=None we are on the left most side. So we create CLeft
		# from scratch
		#

		# A is  [1, dup, D2a]
		# B is  [1, ddown, D2b]

		if conjB:
			C1 = tensordot(A[0,:,:], conj(B[0,:,:]), axes=([0],[0]))
		else:
			C1 = tensordot(A[0,:,:], B[0,:,:], axes=([0],[0]))

		# C1 legs are: [Da2, Db2]


		return C1


	# C is given as  [D1a, D1b]
	# A is given as  [D1a, dup, D2a]
	# B is given as  [D1b, ddown, D2b]

	C1 = tensordot(C,A, axes=([0],[0]))
	# Now C1 is of the form: [D1b,dup,Da2]

	if conjB:
		C1 = tensordot(C1, conj(B), axes=([0,1], [0,1]))
	else:
		C1 = tensordot(C1, B, axes=([0,1], [0,1]))

	# C1: [D2a, D2b]

	return C1






# ---------------------------------------------------------
#  updateCRight
#
#  Update the contraction of two MPSs from right to left.
#
#  If C is not empty then its a 2 legs tensor:
#  [Da,Db]
#

def updateCRight(C, A, B, conjB=False):


	if C is None:
		#
		# When C=None we are on the right-most side. So we create CRight
		# from scratch by contracting the right-most ket/bra tensors along
		# the physical leg (mid leg)
		#

		# A is  [D1a, dup, 1]
		# B is  [D1b, ddown, 1]

		if conjB:
			C = tensordot(A[:,:,0], conj(B[:,:,0]), axes=([1],[1]))
		else:
			C = tensordot(A[:,:,0], B[:,:,0], axes=([1],[1]))

		# Final C form: [D1a, D1b]

		return C


	# C is given as  [D2a, D2b]
	# A is given as  [D1a, dup, D2a]
	# B is given as  [D1b, ddown, D2b]

	# first, contract A(D2a) with C(D2a)
	C = tensordot(A, C, axes=([2],[0]))
	
	# Now C is of the form: [D1a, dup, D2b]

	if conjB:
		C = tensordot(C, conj(B), axes=([1,2], [1,2]))
	else:
		C = tensordot(C, B, axes=([1,2], [1,2]))

	# Final C form: [D1a, D1b]

	return C



















# ---------------------------------------------------------
#  mps_inner_product - return the inner product <A|B>
#

def mps_inner_product(A,B,conjB=False):

	leftC = None

	for i in range(A.N):

		leftC = updateCLeft(leftC,A.A[i],B.A[i], conjB)

	if conjB:
		return leftC[0,0]*A.overall_factor()*conj(B.overall_factor())
	else:
		return leftC[0,0]*A.overall_factor()*B.overall_factor()

# ---------------------------------------------------------
#  mps_sandwitch - return the expression <A|O|B>
#  for MPSs A,B and an MPO O
#

def mps_sandwitch(A,Op,B,conjB=False):

	leftCO = None

	for i in range(A.N):

		if conjB:
			leftCO = updateCOLeft(leftCO,A.A[i],Op.A[i],conj(B.A[i]))
		else:
			leftCO = updateCOLeft(leftCO,A.A[i],Op.A[i],B.A[i])


	if conjB:
		return leftCO[0,0,0]*A.overall_factor()*conj(B.overall_factor())
	else:
		return leftCO[0,0,0]*A.overall_factor()*B.overall_factor()



# ---------------------------------------------------------
#
# calculate_2RDM_from_a_list - get 2 boundary MPSs (one from above and
# one from below, together with a list of PEPS matrices that
# go between them and outputs the list of 2-local RDM from the bulk
# of the line.
#

def calculate_2RDM_from_a_line(bmpsU, bmpsD, A):

	#
	# first step: calculate the MPO made from the A list.
	#

	N = len(A)

	oplist=[]
	for i in range(N):
		A0 = A[i]

		Dleft = A0.shape[1]
		Dright = A0.shape[2]
		Dup = A0.shape[3]
		Ddown = A0.shape[4]

		# contract the physical legs
		# Recall that in PEPS the legs are: [d, Dleft, Dright, Dup, Ddown]
		#
		Aop = tensordot(A0,conj(A0), axes=([0],[0]))
		#
		# resulting tensor is [Dleft,Dright,Dup,Ddown ; Dleft',Dright',Dup',Ddown']
		#                        0      1     2   3        4     5      6    7

		# Recall the MPO site is of the form [dup,ddown,Dleft, Dright]

		Aop = Aop.transpose([2,6,3,7,0,4,1,5])
		# Now its of the form [Dup, Dup', Ddown, Ddown',Dleft, Dleft',
		#                       Dright,Dright']

		Aop = Aop.reshape([Dup*Dup, Ddown*Ddown, Dleft*Dleft, Dright*Dright])

		oplist.append(Aop)


	rhoL = []

	#
	# Now go over all sites in the bulk and calculate the (j1, j1+1) RDM.
	#
	# To do that we need two tensors which are the
	# contraction of the 3 layers:
	#
	# 1. CLO - contraction from 0 to j1-1
	# 2. CRO - contraction from N-1 to j1+2
	#
	# Once these are found, rho can be calculated from them + the
	# bmpsU, bmpsD tensors of j1,j1+1 as well as the PEPS tensors at
	# j1,j1+1
	#


	CLO=None
	for j1 in range(1,N-2):
		CLO = updateCOLeft(CLO, bmpsU.A[j1-1], oplist[j1-1], bmpsD.A[j1-1])
		# CLO legs: Da1, Do1, Db1


		CRO=None
		for j in range(N-1, j1+1, -1):
			CRO = updateCORight(CRO, bmpsU.A[j], oplist[j], bmpsD.A[j])

		# CRO legs: Da3, Do3, Db3

		AupL = bmpsU.A[j1]    # legs: Da1, ddown1, Da2
		AupR = bmpsU.A[j1+1]  # legs: Da2, ddown2, Da3

		AdownL = bmpsD.A[j1]   # legs: Db1, dup1, Db2
		AdownR = bmpsD.A[j1+1] # legs: Db2, dup2, Db3

		CLO1 = tensordot(CLO, AupL, axes=([0], [0]))  # legs: Do1,Db1,ddown1,Da2
		CLO1 = tensordot(CLO1, AdownL, axes=([1],[0])) # legs: Do1,ddown1, Da2, dup1, Db2
		CLO1 = CLO1.transpose([2,0,4,3,1]) # legs: Da2, Do1, Db2, dup1, ddown1
										 #        0    1    2      3     4

		Ai = A[j1]     # legs: di, Dileft, Diright, Diup, Didown

		di      = Ai.shape[0]
		Dileft  = Ai.shape[1]
		Diright = Ai.shape[2]
		Diup    = Ai.shape[3]
		Didown  = Ai.shape[4]

		AiAi = tensordot(Ai,conj(Ai),0)
		AiAi = AiAi.transpose([0,5,1,6,2,7,3,8,4,9])
		AiAi = AiAi.reshape([di,di,Dileft**2, Diright**2, Diup**2, Didown**2])

		CLO1 = tensordot(CLO1, AiAi, axes=([1,4,3],[2,4,5]))
		# CLO legs: Da2, Db2, di, c-di, Diright


		CRO = tensordot(CRO, AupR, axes=([0],[2])) # legs: Do3, Db3, Da2, ddown2
		CRO = tensordot(CRO, AdownR, axes=([1],[2])) # legs: Do3, Da2, ddown2, Db2, dup2
		CRO = CRO.transpose([1,0,3,4,2])  # legs: Da2, Do3, Db2, dup2, ddown2

		Aj = A[j1+1]  # legs: dj, Djleft, Djright, Djup, Djdown

		dj      = Aj.shape[0]
		Djleft  = Aj.shape[1]
		Djright = Aj.shape[2]
		Djup    = Aj.shape[3]
		Djdown  = Aj.shape[4]

		AjAj = tensordot(Aj,conj(Aj),0)
		AjAj = AjAj.transpose([0,5,1,6,2,7,3,8,4,9])
		AjAj = AjAj.reshape([dj,dj,Djleft**2, Djright**2, Djup**2, Djdown**2])
		#                     0  1    2         3           4        5

		CRO = tensordot(CRO, AjAj, axes=([1,4,3],[3,4,5]))
		# CRO legs: Da2,Db2, dj,c-dj, Djleft


		rho = tensordot(CLO1, CRO, axes=([0,1,4], [0,1,4]))
		# rho legs: di,c-di),dj,c-dj)

		rho = rho/trace(trace(rho))

		rhoL.append(rho)


	return rhoL



# ---------------------------------------------------------
#
#  Given a M\times N PEPS and a truncation bond dimension Dp, this function
#  uses the boundary-MPS method to calculate the 1-body RDM of every
#  node in the PEPS. The output is a list of all such RDMs, starting
#  from the horizontal RDMS:
#    (0,0), (0,1), (0,2), ... , (0,N-1),
#    (1,0), (1,1), ...
#
#
#  Every 1-body RDM is a tensor of the form:
#
#                   rho_{alpha_1,beta_1}
#
#  where the alpha_1,alpha_2 correspond to the ket and beta_1,beta_2
#  correspond to the bra (i.e, the complex-conjugated part).
#
#  Every rho is normalized to have a trace=1.
#
#

def calculate_PEPS_1RDM(p0, Dp):



	rhoL = []

	#
	# We pass to an enlarged PEPS, by adding trivial rows at the
	# top/bottom and trivial columns at left/right. This way the
	# bulk RDMS of the enlarged PEPS
	#
	p = enlargePEPS(p0)

	#
	# =======  Calculate HORIZONTAL 2-body RDMs ==========
	#


	#
	# Go over all rows and for each row calculate the boundary-MPS
	# above it (bmpsU) and the one below it (bmpsD)
	#

	bmpsU = p.calc_bMPS('U')
	bmpsU.reduceD(Dp)

	for i1 in range(1,p.M-1):

		bmpsD = p.calc_bMPS('D')
		bmpsD.reduceD(Dp)

		for i in range(p.M-2,i1,-1):
			op = p.calc_line_MPO('row-MPS',i)
			bmpsD = applyMPO(op,bmpsD,cont_leg='D')
			bmpsD.reduceD(Dp)


		AList = []
		for j in range(p.N):
			AList.append(p.A[i1][j])

		rhoL = rhoL + calculate_2RDM_from_a_line(bmpsU, bmpsD, AList)

		#
		# Updating the bmpsU
		#
		op = p.calc_line_MPO('row-MPS',i1)
		bmpsU = applyMPO(op,bmpsU,cont_leg='U')
		bmpsU.reduceD(Dp)



	#
	# =======  Calculate VERTICAL 2-body RDMs ==========
	#


	#
	# Go over all columns from left to right and for each column
	# calculate the boundary-MPS from its left (bmpsL) and from
	# its right (bmpsR)
	#

	bmpsL = p.calc_bMPS('L')
	bmpsL.reduceD(Dp)

	for j1 in range(1,p.N-1):


		bmpsR = p.calc_bMPS('R')
		bmpsR.reduceD(Dp)

		for j in range(p.N-2,j1,-1):
			op = p.calc_line_MPO('column-MPS',j)
			bmpsR = applyMPO(op,bmpsR,cont_leg='D')
			bmpsR.reduceD(Dp)


		AList = []
		for i in range(p.M):
			A = p.A[i][j1]
			#
			# transpose A's indices so that the left/right indices
			# are moved to up/down to match the requirements of the
			# calculate_2RDM_from_a_line function requirements.
			#

			A = A.transpose([0,3,4,1,2])

			AList.append(A)

		rhoL = rhoL + calculate_2RDM_from_a_line(bmpsL, bmpsR, AList)

		#
		# Updating the bmpsL
		#
		op = p.calc_line_MPO('column-MPS',j1)
		bmpsL = applyMPO(op,bmpsL,cont_leg='U')
		bmpsL.reduceD(Dp)





	return rhoL





# ---------------------------------------------------------
#
#  Given a PEPS and a truncation bond dimension Dp, this function
#  uses the boundary-MPS method to calculate the 2-body RDM of every
#  link in the PEPS. The output is a list of all such RDMs, starting
#  from the horizontal RDMS:
#    [(0,0), (0,1)] , [(0,1),(0,2)], ... , [(0,N-2),(0,N-1)],
#    [(1,0), (1,1)] , [(1,1),(1,2)], ... , [(1,N-2),(1,N-1)],
#    ...
#
#  And then the vertical RDMS:
#
#    [(0,0), (1,0)], [(1,0), (2,0)], ... , [(M-2,0),(M-1,0)],
#    [(0,1), (1,1)], [(1,1), (2,1)], ... , [(M-2,1),(M-1,1)],
#    ...
#
#  Notice that there are (M-1)N horizontal RDMs and M(N-1) vertical RDMs.
#
#  Every 2-body RDM is a tensor of the form:
#
#    rho_{alpha_1,beta_1; alpha_2,beta_2}
#       := <alpha_1|<alpha_2| rho |beta_1>|beta_2>
#
#  where the alpha_1,beta_1 correspond to the ket,bra of the first
#  particle and alpha_2,beta_2 for the second.
#
#  Every rho is normalized to have a trace=1.
#

def calculate_PEPS_2RDM(p0, Dp, log=False):


	#
	# Initialize the list of 2-local RDMs
	#

	rhoL = []

	#
	# We first create a larger PEPS, by padding the original PEPS with
	# trivial rows at the top/bottom and trivial columns at left/right.
	#
	# This gives an easier bookeeping: the RDMs of the original PEPS on
	# the boundary becomes RDMS of the new PEPS in the bulk. So we don't
	# need to worry about edge cases.
	#

	p = enlargePEPS(p0)

	#
	# ==================================
	# Calculate the horizontal RDMS
	# ==================================
	#


	bmps_list = []

	bmpsD = p.calc_bMPS('D')
	bmpsD.reduceD(Dp)
	bmps_list = [bmpsD]

	if log:
		print("\n")
		print("Calculating horizontal 2-body RDMs")
		print("----------------------------------\n")

	if log:
		print("Going Up => Down")
		print("-----------------\n")

	for i in range(p.M-2,1,-1):


		if log:
			print("=> At row ", i)

		op = p.calc_line_MPO('row-MPS',i)
		bmpsD1 = applyMPO(op,bmpsD,cont_leg='D')
		bmpsD1.reduceD(Dp)
		bmps_list.insert(0, bmpsD1)

		bmpsD = bmpsD1

	bmpsU = p.calc_bMPS('U')
	bmpsU.reduceD(Dp)


	if log:
		print("\n")
		print("Going Down => Up")
		print("-----------------\n")

	for i1 in range(1,p.M-1):

		if log:
			print("=> At row ", i1)

		bmpsD = bmps_list[i1-1]

		AList = []
		for j in range(p.N):
			AList.append(p.A[i1][j])

		rhoL = rhoL + calculate_2RDM_from_a_line(bmpsU, bmpsD, AList)

		#
		# Updating the bmpsU
		#
		op = p.calc_line_MPO('row-MPS',i1)
		bmpsU = applyMPO(op,bmpsU,cont_leg='U')
		bmpsU.reduceD(Dp)

	#
	# ==================================
	# Now calculate the veritcal RDMS
	# ==================================
	#

	if log:
		print("\n")
		print("Calculating vertical 2-body RDMs")
		print("--------------------------------\n")

	if log:
		print("Going Right => Left")
		print("-------------------\n")

	bmpsR = p.calc_bMPS('R')
	bmpsR.reduceD(Dp)
	bmps_list = [bmpsR]

	for j in range(p.N-2,1,-1):

		if log:
			print("=> At column ", j)

		op = p.calc_line_MPO('column-MPS',j)
		bmpsR1 = applyMPO(op,bmpsR,cont_leg='D')
		bmpsR1.reduceD(Dp)
		bmps_list.insert(0, bmpsR1)

		bmpsR = bmpsR1

	if log:
		print("\n")
		print("Going Left => Right")
		print("--------------------\n")

	bmpsL = p.calc_bMPS('L')
	bmpsL.reduceD(Dp)

	for j1 in range(1,p.N-1):

		if log:
			print("=> At column ", j1)

		bmpsR = bmps_list[j1-1]

		AList = []
		for i in range(p.M):

			A = p.A[i][j1]
			#
			# transpose A's indices so that the left/right indices
			# are moved to up/down to match the requirements of the
			# calculate_2RDM_from_a_line function requirements.
			#

			A = A.transpose([0,3,4,1,2])

			AList.append(A)

		rhoL = rhoL + calculate_2RDM_from_a_line(bmpsL, bmpsR, AList)

		#
		# Updating the bmpsL
		#
		op = p.calc_line_MPO('column-MPS',j1)
		bmpsL = applyMPO(op,bmpsL,cont_leg='U')
		bmpsL.reduceD(Dp)




	return rhoL




# ----------------------   add_two_MPSs   ------------------------

def add_two_MPSs(mpsA, alpha, mpsB, beta):
	
	"""
	
	Adds two MPSs. Given mpsA, mpsB and two coefficients alpha, beta, 
	create an MPS which is the sum: 
	
	               mpsC := alpha*mpsA + beta*mpsB
	               
	The two MPSs must have the same length and the same physical bond 
	dimension at every site
	
	The bond dimension of mpsC at every site i is
	[DLa + DLb, d, DRa + DRb]
	
	The tensor of mpsC at site i is made of two blocks that correspond
	to the tensors of A,B. The alpha,beta coefficients only multiply
	the i=0 tensors. See https://arxiv.org/abs/1008.3477 Sec 4.3 for 
	full details.
	
	Input Parameters:  
	------------------
	
	mpsA,mpsB --- MPS objects representing mpsA, mpsB
	alpha,beta --- scalars to multiply mpsA and mpsB
	
	Output:
	----------
	
	An MPS object representing  mpsC := alpha*mpsA + beta*mpsB
	
	
	"""
	
	if mpsA.N != mpsB.N:
		print("* * * Error in bmpslib.add_two_MPSs !!! * * *")
		print(f"The two MPSs have different lengths: mpsA.N={mpsA.N} " \
			f" while mpsB.N={mpsB.N}")
			
		exit(1)
		
	N = mpsA.N
	
	mp = mps(N)
	
	for i in range(N):
		
		if mpsA.A[i].shape[1] != mpsB.A[i].shape[1]:
			print("* * * Error in bmpslib.add_two_MPSs !!! * * *")
			print(f"Different physical dimension at site {i}: " \
				f" dA = {mpsA.A[i].shape[1]} while  dB={mpsB.A[i].shape[1]}")
				
			exit(1)
		
		DLa, DRa = mpsA.A[i].shape[0], mpsA.A[i].shape[2]
		DLb, DRb = mpsB.A[i].shape[0], mpsB.A[i].shape[2]
		Dmid = mpsA.A[i].shape[1]
		
		
		
		#
		# Find the dtype of the new tensor we create, and store it in dt
		#
		a = mpsA.A[i][0,0,0] + mpsB.A[i][0,0,0]
		dt = a.dtype
		
		if i==0:
			Asum = zeros( [1, Dmid, DRa + DRb], dtype=dt)
			Asum[:, :, :DRa] = alpha*mpsA.A[i]
			Asum[:, :, DRa:] = beta*mpsB.A[i]
			
		elif i==N-1:
			Asum = zeros( [DLa+DLb, Dmid, 1], dtype=dt)
			Asum[:DLa, :, :] = mpsA.A[i]
			Asum[DLa:, :, :] = mpsB.A[i]
			
		else:
			Asum = zeros( [DLa+DLb, Dmid, DRa+DRb], dtype=dt)
			Asum[:DLa, :, :DRa] = mpsA.A[i]
			Asum[DLa:, :, DRa:] = mpsB.A[i]

		mp.set_site(Asum, i)
		
	return mp
		
		
def _assert_int(x:int|float) -> int:
	int_version = int(x)	
	assert int_version==x, f"{x} must be an integer!"
	return int_version
	

def _perf_svd(m:np.ndarray, svd_emthod:str="rsvd", check_result:bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:		
	if svd_emthod=="svd":
		u, s, vh = svd(m, full_matrices=False)
	elif svd_emthod=="rsvd":
		eps_or_k = 1e-5
		u, s, vh = rsvd(m, eps_or_k)
	else:
		raise ValueError(f"Not an expected case `svd_emthod=={svd_emthod!r}`")	

	if check_result:
		m1 = (u * s) @ vh
		diff = np.sum(abs(m - m1))
	return u, s, vh



def _svd_test(
	n:int=int(20)
):
	from time import perf_counter
	m = np.random.rand(n, n) + 1j*np.random.rand(n, n)

	for svd_emthod in ["svd", "rsvd"]:

		print("") 
		print(svd_emthod)

		t1 = perf_counter()
		u, s, vh = _perf_svd(m, svd_emthod)
		t2 = perf_counter()

		m1 = (u * s) @ vh
		diff = abs(m - m1)
		print(f"time={t2-t1}")
		# print(diff)
		print(f"diff={np.sum(diff)}")

	print("Done.")


if __name__ == "__main__":
	_svd_test()
	print("Done")