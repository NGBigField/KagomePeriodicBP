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
#----------------------------------------------------------
#


import numpy as np
import scipy as sp
import scipy.io as sio

from scipy.sparse.linalg import eigs

from numpy.linalg import norm, svd, qr
from scipy.linalg import rq

from numpy import zeros, ones, array, tensordot, sqrt, diag, dot, \
	reshape, transpose, conj, eye, trace, floor, log10
from sys import exit
from copy import deepcopy




EPSILON = 0.000001



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

	def __init__(self,N):

		self.N = N
		self.A : list[np.ndarray] = [None]*N 
		
		self.Corder = [None]*N  # Canonical order of each element
		
		self.nr_mantissa = 1.0
		self.nr_exp = 0
		
		
	#
	# ------------ update_A0
	#
	
	def update_A0_norm(self):
		
		nr_A0 = norm(self.A[0])
		self.set_site(self.A[0]/nr_A0, 0)
		
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
		

	#--------------------------   maxD   ---------------------------
	#
	# Returns the max bond dimension of the MPS
	#
	
	def maxD(self):
		D_list = [self.A[0].shape[0]] \
			+ [self.A[i].shape[2] for i in range(self.N)]
			
		return max(D_list)
		

		
	#--------------------------   copy   ---------------------------
	#
	# Returns a copy of the mps class. The copy has a new tensor-list
	# but it *does not* copy the tensors themselves.
	#
	#
	
	def copy(self, full:bool=False):
		
		new_mps = mps(self.N)
		
		new_mps.A = self.A.copy()
		new_mps.Corder = self.Corder.copy()

		if full:
			new_mps.nr_exp      = self.nr_exp
			new_mps.nr_mantissa = self.nr_mantissa
		
		return new_mps
		
	#--------------------------   reverse   ---------------------------
	#
	# Returns an MPS with the order of the tensors reversed: the first
	# tensor becomes the last etc.
	#
	#
	
	def reverse(self):
		
		new_mps = mps(self.N)

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
	
	def l2_distance(self, other:"mps") -> float:
		"""l2_distance(A, B) -> float:
		Where A is `self` and B is `other`:
		Compute || <A|A> - <B|B> ||^2_2

		Args:
			self (MPS)
			othher (MPS)
		"""
		# Compute:
		conjB = True
		overlap = mps_inner_product(self, other, conjB)
		distance2 = 2 - 2*overlap.real
		# Validate:
		error_msg = f"L2 Distance should always be a real & positive value. Instead got {distance2}"
		assert np.imag(distance2)==0, error_msg
		if distance2<0:  # If it's a negative value.. it better be very close to zero
			assert abs(distance2)<EPSILON , error_msg
			return 0.0
		# return:
		return distance2

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
					
				mpshape = mpshape + ' {}A_{}({} {} {}) '.\
					format(order_str,i,self.A[i].shape[0],self.A[i].shape[1], \
					self.A[i].shape[2])

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

	def left_canonical(self, maxD=None, eps=None):
		if self.N<2:
			return


		for i in range(self.N-1):

			D1 = self.A[i].shape[0]
			d  = self.A[i].shape[1]
			D2 = self.A[i].shape[2]

			M = self.A[i].reshape(D1*d, D2)

			U,S,V = svd(M, full_matrices=False)


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

	def right_canonical(self, maxD=None, eps=None, i0=None,i1=None, \
		nr_bulk = False):
			
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

			if D1>maxD:
				
				#
				# Use SVD for truncation
				#
				
				U,S,V = svd(M, full_matrices=False)
				
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
			
		#
		# if nr_bulk=True, we absorb the overall norm factor in 
		# the left-most tensor (i=0)
		#
		
		if nr_bulk:
			self.set_site(self.A[0]*overall_norm, 0)
			self.update_A0_norm()

		return


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
			
			iD0=None
			for i in range(self.N-1):
				if self.A[i].shape[2]>maxD:
					iD0 = i
					break

			#
			# If there's no site that needs truncation, we quit.
			#
			if iD0 is None:
				return

			for i in range(self.N-2, -1, -1):
				if self.A[i].shape[2]>maxD:
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
					
			self.right_canonical(maxD, eps, i0=iD0+1, i1=i1, \
				nr_bulk=nr_bulk)
		
		elif mode=='RC':
			self.left_canonical_QR()
			self.right_canonical(maxD, eps, nr_bulk=nr_bulk)
			
		else:
			if nr_bulk:
				print("Error in reduceD(): nr_bulk=True is incompatible with "\
					"mode='LC'")
				exit(1)

			self.right_canonical()
			self.left_canonical(maxD, eps)
			
		

		return




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
# Structure of the A's:  [dup,ddown,Dleft,Dright]                      #
#                                                                      #
#    dup and ddown are the physical entries and must be of the same    #
#    dimensio. dup is the input index and ddown is the output          #
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
			mpo_shape = mpo_shape + ' A_{}({} {}; {} {}) '.\
				format(i,self.A[i].shape[0],self.A[i].shape[1], \
					self.A[i].shape[2],self.A[i].shape[3])

		return mpo_shape

	#--------------------------   set_site   ---------------------------
	#
	#  Sets the tensor of a site in the MPS
	#
	#  We expect an array of complex numbers of the form
	#  M[D1, dup, ddown, D2]
	#
	#  The D1 should match the D2 of the previous (i.e., the last)
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
	if f==0:
		return 0
	elif np.isinf(f):
		raise FloatingPointError(f"f={f}")
	else:
		return int(floor(log10(abs(f)))) 

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



# ----------------------------------------------------------------------
#  applyMPO
#
#  Apply the MPO op to the MPS M. If OverWrite=True then result is
#  written to M itself. Default: OverWrite=False
#
#  Parameters:
#  op       - The MPO
#  M        - the MPS
#  i1       - An optional parameter specifying the initial location on M
#             where op start acting.
#  cont_leg - which leg of the MPO to contract - could be 'U' (up)
#             or 'D' (Down).
#
#

def applyMPO(op, M, i1=0, cont_leg='U'):


	newM = mps(M.N)

	for i in range(i1, i1+op.N):

		MD1 = M.A[i].shape[0]
		Md  = M.A[i].shape[1]
		MD2 = M.A[i].shape[2]

		d_up   = op.A[i-i1].shape[0]
		d_down = op.A[i-i1].shape[1]
		opD1   = op.A[i-i1].shape[2]
		opD2   = op.A[i-i1].shape[3]

		#
		# Recall that the MPO legs order is [d_up, d_down, D_left, D_right]
		# while the MPS element legs are     [D_left, d, Dright]

		if cont_leg=='U':
			newA = tensordot(M.A[i], op.A[i-i1], axes=([1],[0]))
			# newA has the legs [MD1, MD2, d_down, opD1, opD2]
			#                     0    1     2       3    4

			# transpose it to: [(MD1, opD1), d_down, (MD2, opD2)]

			newA = newA.transpose([0,3,2,1,4])
			newA = newA.reshape( [MD1*opD1, d_down, MD2*opD2])
		else:
			#
			# So cont_leg = 'D'
			#

			newA = tensordot(M.A[i], op.A[i-i1], axes=([1],[1]))
			# newA has the legs [MD1, MD2, d_up, opD1, opD2]
			#                     0    1     2       3    4

			# transpose it to: [(MD1, opD1), d_up, (MD2, opD2)]

			newA = newA.transpose([0,3,2,1,4])
			newA = newA.reshape( [MD1*opD1, d_up, MD2*opD2])

		newM.set_site(newA, i)


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



# ---------------------------------------------------------
#  updateCOLeft
#
#  Update the contraction of an MPO with 2 MPSs from left to right.
#
#  If C is not empty then its a 3 legs tensor:
#  [Da,Do,Db]
#

def updateCOLeft(C, A, Op, B):


	if C is None:
		#
		# When C=None we are on the left most side. So we create COLeft
		# from scratch
		#

		# A is  [1, dup, D2a]
		# Op is [up, ddown, 1, D2o]
		# B is  [1, ddown, D2b]

		C1 = tensordot(A[0,:,:], Op[:,:,0,:], axes=([0],[0]))

		# C legs are: [D2a, ddown, D2o]

		C1 = tensordot(C1, B[0,:,:], axes=([1],[0]))

		# result is: [D2a,D2o,D2b]

		return C1


	# C is given as  [D1a, D1o, D1b]
	# A is given as  [D1a, dup, D2a]
	# Op is given as [dup, down, D1o, D2o]
	# B is given as  [D1b, ddown, D2b]

	C1 = tensordot(C,Op, axes=([1],[2]))
	# C1: [D1a,D1b,dup, ddown, D2o]

	C1 = tensordot(C1, A, axes=([0,2],[0,1]))
	# C1: [D1b, ddown, D2o, D2a]

	C1 = tensordot(C1, B, axes=([0,1], [0,1]))

	# C1: [D2o, D2a, D2b]

	C1 = C1.transpose([1,0,2])


	return C1





# ---------------------------------------------------------
#  updateCORight
#
#  Update the contraction of an MPO and two MPSs from right to left
#
#  If C is not empty then its a 3 legs tensor:
#  [Da,Do,Db]
#

def updateCORight(C, A, Op, B):


	if C is None:
		#
		# When C=None we are on the left most side. So we create COLeft
		# from scratch
		#

		# A is  [D1a, dup, 1]
		# Op is [dup, ddown, D1o, 1]
		# B is  [D1b, ddown, 1]

		C1 = tensordot(A[:,:,0], Op[:,:,:,0], axes=([1],[0]))

		# C1 legs are: [D1a, ddown, D1o]

		C1 = tensordot(C1, B[:,:,0], axes=([1],[1]))

		# result is: [D1a,D1o,D1b]

		return C1


	# C is given as  [D2a, D2o, D2b]
	# A is given as  [D1a, dup, D2a]
	# Op is given as [dup, ddown, D1o, D2o]
	# B is given as  [D1b, ddown, D2b]

	C1 = tensordot(C,Op, axes=([1],[3]))
	# C1: [D2a,D2b,dup, ddown, D1o]

	C1 = tensordot(C1, A, axes=([0,2],[2,1]))
	# C1: [D2b, ddown, D1o, D1a]

	C1 = tensordot(C1, B, axes=([0,1], [2,1]))

	# C1: [D1o, D1a, D1b]

	C1 = C1.transpose([1,0,2])


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

def mps_inner_product(A, B, conjB=False):

	leftC = None

	for i in range(A.N):

		leftC = updateCLeft(leftC, A.A[i], B.A[i], conjB)

	return leftC[0,0]

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

	return leftCO[0,0,0]



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
		
		
	
	
	




