#----------------------------------------------------------
#
#               blockBP-example.py
#             =======================
#
#  This program uses the blockBP algorithm to contract 
#  a PEPS in the shape of the example given in the document describing
#  blockBP, in Sec.2.2
#
#  It creates a random PEPS with the topology of the TN in the example,
#  and calculates the <psi|psi> doubel-edge TN from it. Then it gives it
#  to the blockBP and uses the converged messages to calculate calculate
#  the expectation value <psi|Z_0|psi>.
#
#  It compares it to the expectation values from ncon. 
#
#  Empirically, the larger the bond dimension, the better is the
#  approximation (intuitively, this is because of the monogamy of 
#  entanglement)
#
# History:
# ---------
#
# 21-Sep-2021 --- Initial version.
#
# 17-Oct-2021 --- Updated the example. Removed the squre PEPS staff.
#                 to concentrate on the example from Sec. 2.2
#            
#                 Add a calculation of <psi|Z_0|psi> and compare
#                 it to the result of ncon.
#
# 8-Jun-2022 --- Minor typos correction and added comments to the 
#                source code.
#
#----------------------------------------------------------
#

import numpy as np

from numpy.linalg import norm, svd, qr

from numpy import zeros, ones, array, tensordot, sqrt, diag, dot, \
	reshape, transpose, conj, eye, trace, pi
	
	
from ItaysModules import bmpslib
from ItaysModules import bubblecon
from ItaysModules import blockbp
from ItaysModules import TenQI 

from ncon import ncon



#
# ------------------------- create_example  ---------------------------
#

def create_example_TN():
	
	"""
	 We create a random PEPS according to the example in Sec. 2.2 in the 
	 blockBP document, which has 12 spins.
	 
	 The physical legs are then contracted together in the <psi|psi> TN 
	 to create a double edge TN, on which we define the parameters that
	 are needed for blockBP contraction.
	 
	 Output:
	 ===========
	 1. The tensors of the original PEPS (single-edges)
	 
	 2. The blockBP parameters: 
	 
	   T_list, edges_list, pos_list, blocks_v_list, sedges_dict, 
	   sedges_list, blocks_con_list
	
	"""
	
	d = 2    # Dimension of physical leg
	
	D = 4    # bond of the PEPS. D^2 is the bond dimension of the 
	         # <psi|psi> double edge TN that is sent to the blockBP 
	         # function.
	         
	D2 = D*D # Bond dimension of the double-edge model
	         
	
	#              T0     T1      T2    T3     T4      T5     T6      T7
	pos_list = [ (3,9), (6,9), (4,7), (2,6), (6,7), (11,7), (10,5), (13,5),
	#   T8     T9     T10    T11
		(6,4), (4,2), (7,2), (5,1)]
	
	#                T0        T1       T2       T3          T4
	edges_list = [ [1,2], [1,3,19,6], [5,3], [2,5,4,14], [19,4,7,8,20], 
	#   T5           T6        T7          T8               9
	[6,7,10], [8,10,9,11], [9,12], [11,20,15,18,17], [16,14,15],
	#  10            11
	[13,17,12], [16,18,13]]
	
	n = len(edges_list)
	
	# The list of the original PEPS tensors. First index in each tensor
	# is the physical leg.
	T_PEPS_list = []

	#
	# The list of the <psi|psi> TN tensors, which is sent to blockBP
	#
	T_list = []

	
	for eL in edges_list:
		k = len(eL)
		sh = [d] + [D]*k
		
		T = np.random.normal(size=sh) + 1j*np.random.normal(size=sh)
		T = T/norm(T)
		
		T_PEPS_list.append(T)
		
		#
		# Contract T with T^* and fuse the corresponding virtual legs 
		#
		
		perm=[]
		for i in range(k):
			perm = perm + [i,i+k]
	
		T2 = tensordot(T, conj(T), axes=([0],[0]))
		T2 = T2.transpose(perm)
		T2 = T2.reshape([D2]*k)
		
		T_list.append(T2)
		
		
		
		
	#                   B0            B1         B2
	blocks_v_list = [ [0,1,2,3,4], [5,6,7], [8,9,10,11] ]
	
	sedges_dict = { 'A':[6,7,8], 'B':[11,12], 'C':[14,20]}
	
	#                        B0                  B1
	sedges_list = [ [('A', 1), ('C', -1)], [('A',-1), ('B',-1)], 
		[('B',1), ('C',1)]]
		
	bcon0 = [ ([('e',14),('e',20),('v',3),('v',4),('v',2),('v',0),('v',1)],\
		pi/2), \
		( [('e',6),('e',7),('e',8),('v',1),('v',4),('v',0),('v',2),('v',3)],pi)]
		
	bcon1=[ ([('e',11),('e',12),('v',7),('v',6),('v',5)],3*pi/2), \
		([('e',6),('e',7),('e',8),('v',5),('v',6),('v',7)],0)]
		
	bcon2=[ ([('e',14),('e',20),('v',9),('v',11),('v',8),('v',10)],3*pi/2), \
		([('e',11),('e',12),('v',10),('v',8),('v',11),('v',9)],pi)]
		
	blocks_con_list = [bcon0, bcon1, bcon2]
	
	
	
	return T_PEPS_list, T_list, edges_list, pos_list, blocks_v_list, \
		sedges_dict, sedges_list, blocks_con_list


#
#
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
#                             M  A  I  N
#
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
#


def main():
	
	print("hello")
	
	np.random.seed(2)
		
	#
	# The truncation bond in the blockBP iteration
	#
	
	Dp=16
	
	
	T_PEPS_list, T_list, edges_list, pos_list, blocks_v_list, \
		sedges_dict, sedges_list, blocks_con_list = create_example_TN()
	
	
	#	
	# ================================================================
	# Use blockBP to calculate the expectation value of Z_0
	# the Z Pauli on qubit 0.
	# ================================================================
	#
	
	print("Running blockBP...")
	m_list, PP_blob = blockbp.blockbp(T_list, edges_list, pos_list, \
		blocks_v_list, sedges_dict, sedges_list, blocks_con_list, \
		D_trunc=Dp, max_iter=10, delta=1e-4)

	eps = PP_blob['final-error']
	iter_no = PP_blob['final-iter']
	print(f"... done with {iter_no} iterations and BP messages error={eps}")

	print("\n\n\n")


	BTN_list = PP_blob['BTN_list']
	
	#
	# qubit 0 is the first qubit in BTN0, so we now contract BTN0
	#
	
	(T1_list, edges1_list, angles1_list, bcon) = BTN_list[0]
	
	#
	# As a pre-processsing step, we replace all the labels of 
	# the edges from strings to integers (ncon doesn't like 
	# strings)
	#
	
	max_e=1000 # we know there's no edge with such large integer
	for elist in edges1_list:
		for i in range(len(elist)):
			e = elist[i]
			if type(e)==str:
				# its of the form 'm<v>', so we extract v and replace it
				# with v+max_e
				v = int(e[1:]) + max_e
				elist[i]=v
	
	#
	# To calculate <Z> we calculate <psi|Z|psi>/<psi|psi>.
	# 
	# (*) For the denominator, we simply contract the BTN.
	# (*) For the enumerator, we replace the double edge Tensor 
	#     (T_0^*) T_0 with the contraction of (T_0)^* Z T_0
	#
				
	enumerator = ncon(T1_list, edges1_list)
	
	T0 = T_PEPS_list[0]
	Z = array([[1,0],[0,-1]])
	
	D = T0.shape[1]
	D2 = D*D
	
	newT0 = tensordot(Z, T0, axes=([0],[0]))
	T = tensordot(newT0, conj(T0), axes=([0],[0]))
	T = T.transpose([0,2,1,3])
	T = T.reshape([D2,D2])
	
	T1_list[0] = T
	denominator = ncon(T1_list, edges1_list)
	
	blockBP_estimate = denominator/enumerator
	
	print(f"(*) blockBP: \t{blockBP_estimate}")
	print("\n")


	#	
	# ================================================================
	# Use ncon to calculate the expectation value of Z_0
	# the Z Pauli on qubit 0.
	# ================================================================
	#

	#
	# calculate <psi|psi>
	#
	denominator = ncon(T_list, edges_list)
	
	#
	# Calculate <psi|Z_0|psi>
	#
	T_list[0] = T
	enumerator = ncon(T_list, edges_list)
	
	ncon_estimate = enumerator/denominator
	
	print(f"(*) ncon: \t{ncon_estimate}")


if __name__ == "__main__":
	main()
