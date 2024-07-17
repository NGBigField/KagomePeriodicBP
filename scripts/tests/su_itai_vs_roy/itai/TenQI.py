########################################################################
#                                                                      
#   TenQI - Basic framework for doing QI using tensors                 
#
# History:
# --------
#                                                                      
#  28-Feb-2022: Add partial trace option to op_trace
#
########################################################################

"""

   In the TenQI framework, an operator over n qubits is 
   described by a tensor with 2n legs:

   A_{i_0,j_0; i_1, j_1; ..., i_{n-1},j_{n-1} }
	= <i_0,i_1, ..., i_{n-1}| A | j_0, j_1, ... j_{n-1}>
	
   A = \sum_{i,j} A_{i_0,j_0; ... i_{n-1},j_{n-1}} 
			   |i_0Xj_0|\otimes ... \otimes |i_{n-1} X j_{n-1}|
   
   In other words, the (2k,2k+1) indices describe the k'th qubit.
   
   We refer to this representation as the "TENSOR REPRESENTATION"

   
"""



import numpy as np

from numpy.linalg import norm as npnorm
from numpy import sqrt, conj, tensordot, array, eye

#
# The following are some common 1-qubit matrices that are often used
#

sigma_X  = array([[0,1],[1,0]])
sigma_Y  = 1j*array([[0,-1],[1,0]])
sigma_Z  = array([[1,0],[0,-1]])
ID1      = array([[1,0],[0,1]])
ketbra00 = array([[1,0],[0,0]])
ketbra11 = array([[0,0],[0,1]])
ketbra01 = array([[0,1],[0,0]])


#
#-------------------   op_to_mat   ------------------------------
#
#

def op_to_mat(op):
	
	"""
		maps an n-qubits operator in the tensor representation
		to a 2^n\times 2^n matrix
		
		Input Parameters:
		------------------
		op - the operator in the tensor representation
		
	"""
	
	#
	# get the number of qubits (n) and the total Hilbert 
	# space dimension (N)
	#
	
	n = len(op.shape)//2
	N = 2**n
	
	#
	# First permute the indices [i0,j0; i1,j1, ... ]
	# to [i0,i1,..., j0, j1,...]
	#       
	
	perm_list = list(range(0,2*n,2)) + list(range(1,2*n,2))
	mat = op.transpose(perm_list)
	
	
	#
	# Now fuse the first n indices and the last indices together, 
	# creating a matrix.
	#
	
	mat = mat.reshape([N,N])
	
	return mat
	
	
#
#-------------------   mat_to_op   ------------------------------
#
#

def mat_to_op(mat):
	"""
		maps a 2^n\times 2^n matrix to a n-qubits operator in the 
		tensor representation.
		
		Input Parameters:
		------------------
		mat - the matrix
		
	"""
	
	#
	# Get the Hilbert space dimension and the total number of qubits
	#
	N = mat.shape[0]
	n = int(np.log(N)/np.log(2))
	
	#
	# Spread the indices of the different qubits, so that we get
	# a tensor in the representation [i0, i1, ...; j0, j1, ...]
	#
	op = mat.reshape(2*n*[2])
	  
	#
	# Permute the indices to the tensor representation:
	#
	# [i0, i1, ...; j0, j1, ...] ==> [i0,j1; i1,j1; ...]
	#
	
	listA=list(range(0,n))
	listB=list(range(n,2*n))
	perm_list = [j for i in zip(listA, listB) for j in i]
	   
	op = op.transpose(perm_list)
	
	
	return op
	
 





#
#-------------------   op_dagger   ------------------------------
#
#

def op_dagger(op):
	"""
	
		Calculates the Hermitian dagger of a n-qubits operator in the
		tensor representation:
		
		A^\dagger[i0,j0; i1,j1, ...] := (A[j0,i0; j1,i1; ...])^*
	
		
		Input Parameters:
		------------------
		op - the operator
		
	"""
	
	n = len(op.shape)//2  # no. of qubits
	
	#
	# permute the i and j indices:
	# [0,1;2,3;4,5;...] -> [1,0;3,2;5,4;...]
	#
	list_even=list(range(0,2*n,2))
	list_odd=list(range(1,2*n,2))
	perm_list = [j for i in zip(list_odd, list_even) for j in i]
	
	return conj(op.transpose(perm_list))


#
#-------------------   op_norm   ------------------------------
#
#

def op_norm(op, ntype='op'):
	"""
		Calculates one of several norms of an n-qubit operator. 
		
		Input Parameters:
		------------------
		op - An n-qubits operator in a tensor representation
		
		ntype - the type of norm to calculate:
		
				'op' - Operator norm (L_\infity)
				'tr' - Trace norm (L_1)
				'fr' - Frobenius norm (L_2)
				
				
		returns: the norm.
		
	"""
	

	if ntype=='op' or ntype=='tr':
		
		A = op_to_mat(op)
		
		#
		# Calculate the singular values of A, which 
		# are the square of the eigenvalues of A^\dagger A.
		#
		
		A2 = conj(A.T)@A
		eigs = sqrt(abs(np.linalg.eigvalsh(A2)))
		
		if ntype=='op':
			return eigs[-1]  # maximal singular value
			
		else:
			return sum(eigs)  # sum of singular values
			
	else:
		return np.linalg.norm(op)  # Frobenius norm
		
		
 
#
#-------------------   op_trace   ------------------------------
#
#

 

def op_trace(op, qubits=None):
	
	"""
		Returns the trace of an n-qubits operator in the tensorial 
		representation.
		
		Alterntatively, if a list of qubits is specified, performs 
		a partial trace and returns the resulting operator
	
		Input Parameters:
		------------------
		op     --- An n-qubits operator in a tensor representation
		qubits --- a possible list of qubits on which to perform
				   the partial trace. 
	
	"""
	
	if qubits is None:
		return np.trace(op_to_mat(op))
	
	#
	# Make sure qubits are given in descending order
	#	
	qubits.sort(reverse=True)
	
	op1 = op
	for i in qubits:
		op1 = np.trace(op1, axis1=i*2, axis2=i*2+1)
		
	return op1
	
	
 
 

#
#-------------------   op_times_op   ------------------------------
#
#

def op_times_op(A, B, f=0):
	
	"""
		
		Multiplies an n-qubits operator A by an m-qubits operator B:
		
					   C = A*B
		
		When n != m, the parameter f determines the alignment of the 
		two operators:
		
		(*) When n>m, the contraction is:
		
		  \sum_{j_f,...j_{f+m-1} A[i0,j0; ... i_{n-1},j_{n-1}] * 
								 B[j_f,k_f; ... j_{f+m-1},k_{f+m-1}]
		


		(*) When n<m, the contraction is:
		
		  \sum_{j_f,...j_{f+m-1} A[i_f,j_f; ... i_{f+n-1},j_{f+n-1}] * 
								 B[j_0,k_0; ... j_{m-1},k_{m-1}]
		
		
		A detailed description of the combinatorics can be found in 
		the PDF file op_times_op.pdf
	
	"""
	
	n = len(A.shape)//2
	m = len(B.shape)//2
	
	
	if n==m:
		
		#
		# When n=m it is much faster to turn the tensors to matrices 
		# and use a simple matrix multiplication.
		#
		
		A1 = op_to_mat(A)
		B1 = op_to_mat(B)
		
		C1 = A1@B1
		
		C = mat_to_op(C1)
		
		return C

	
	elif n>m:
		
		#
		#  --------- The n > m case ---------
		#
		
		A_legs = list(range(f*2+1,2*(f+m)+1,2))
		B_legs = list(range(0,2*m,2))
		
		#
		# Multiply the operators by contracting their tensors
		#
		
		C = tensordot(A,B, axes=(A_legs,B_legs))
		
		"""        
		
		A[i_0,j_0;...;i_f,j_f,...,i_{n-1};j_{n-1}] * B[j_f,k_f;...;j_{f+m-1},k_{f+m-1}]
		
										|
										|
										V                           2(n-2)-m
																	 |
			0  1       2f  2f+1         2f+m     2f+m+1  2f+m+2      V      2(n-1)-m  2(n-1)
		C[i_0,j_0,...;i_f;i_{f+1},...;i_{f+m-1};i_{f+m},j_{f+m};...;i_n,j_n;k_f,...,  f_{f+m-1}]
		
		"""
		
		
		#
		# Create the permutation that brings back C to the proper
		# tensor representation
		#
		
		perm_list=list(range(0,2*f))

		
		listA=list(range(2*f,2*f+m,1))
		listB=list(range(2*n-m,2*n,1))
		
		perm_list = perm_list + [j for i in zip(listA, listB) for j in i]
		
		perm_list = perm_list + list(range(2*f+m, 2*n-m,1))

		
	else:        
		
		#
		#  --------- The n < m case ---------
		#
		
		
		A_legs = list(range(1,2*n+1,2))
		B_legs = list(range(f*2,2*(f+n),2))
		
		#
		# Multiply the operators by contracting their tensors
		#
		
		C = tensordot(A,B, axes=(A_legs,B_legs))
		
		"""
		A[i_f,j_f,...,i_n,j_n] * B[j_0,k_0;...j_{f-1},k_{f-1}; i_f,j_f,...,j_{f+n-1},k_{f+n-1},...]
		
		"""
		
		#
		# Create the permutation that brings back C to the proper
		# tensor representation
		#

		perm_list = list(range(n,n+2*f,1))
		
		listA=list(range(0,n,1))
		listB=list(range(n+2*f,2*(n+f),1))
		
		perm_list = perm_list + [j for i in zip(listA, listB) for j in i]
		
		perm_list = perm_list+list(range(2*(n+f), 2*m,1))
		
		
		
	C = C.transpose(perm_list)
	
	
	return C
   
