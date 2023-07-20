#!/usr/bin/python3

########################################################################
#
#  Example(s) of using the bubblecon library
#  =========================================
#


import numpy as np
import scipy as sp
from ncon import ncon

from scipy.linalg import sqrtm, polar, expm

from numpy.linalg import norm
from numpy import sqrt, conj, tensordot, array, eye, zeros, ones, pi

import bmpslib
import bubblecon
import TenQI
import pickle


#
# ----------------------- test_with_ncon -------------------------------
#


def test_open_TN_with_ncon():
	
	"""
	
	Test the bubblecon routine by running it over a small open tensor 
	network and compare the resultant MPS to the tensor one gets from ncon 
	
	The TN is made of 7 tensors and has the following form:
	
	         |  |  |
	       --1--2--3--
	         |  |  |
	          --0--4--
	            |  |
	          --6--5--
	            |  |
	          
	Exact describption of the geometry is given in the PDF file 
	in the bubblecon<n>.pdf file in folder.
	          
	"""


	np.random.seed(1)

	D0 = 4  # bond dimension
	
	#
	# All 7 tensors have 4 legs each, and located on a square grid
	# (so that legs angles are 0,pi/2, pi, 3*pi/2
	#
	
	T_list = []
	for i in range(7):
		T = np.random.normal(size=[D0,D0,D0,D0])
		T_list.append(T/norm(T))
		
	#                 T0            T1           T2         T3
	edges_list = [ [-1,2,7,8], [-2,-3,-4,1], [-5,1,3,2], [-6,-7,3,4],\
	#   T4           T5             T6
	[-8,7,4,5], [-9,-10,5,6], [-11,-12,6,8]]
	
	# the possible angles
	U, D, L, R = pi/2, 3*pi/2, pi, 0
	
	angles_list = [ [L,U,R,D], [D,L,U,R], [U,L,R,D], [U,R,L,D],\
	[R,L,U,D], [R,D,U,L], [D,L,R,U]]
	

	#
	# =========== First, contract using bubblecon ================
	#

	swallow_order=[0,2,1,3,4,5,6]
	bubble_angle=0.1
	
	Dp = None     # Truncation bond; None=infinite
	delta = None  # Schmidt value truncation threshold; None=0
	
	print("Comparing the bubblecon contraction to ncon contraction")
	print(f"for an open TN with {len(T_list)} nodes and bond dimension {D0}")
	print("========================================================")
	
	print("\n\n1. Contracting via bubblecon")
	mp1 = bubblecon.bubblecon(T_list, edges_list, angles_list, bubble_angle,\
			swallow_order, D_trunc=Dp, eps=delta, opt='high')
	
	print("  => Done.")
	#
	# ================ Next, contract using ncon ================
	#
	
	print("\n\n2. Contracting via ncon:")
	A = ncon(T_list, edges_list)
	print("  => Done. ")
	
	print("\n\n3. Turning ncon result into an MPS")
	#
	# Turn the resultant tensor into an MPS
	# 
	mp2 = bubblecon.tensor_to_MPS_ID(A)
	print("  => Done. ")
		
	a = bmpslib.mps_inner_product(mp1, mp1)
	b = bmpslib.mps_inner_product(mp2, mp2)
	c = bmpslib.mps_inner_product(mp1, mp2)
	
	print("\n\n4. Checking the overlap of mp1 (bubblecon) and mp2 (ncon)")
	print("\n")
	print("<mp1|mp1>={} \t <mp2|mp2>={} \t <mp1|mp2>={} ".format(a,b,c))
	
	print("\n\n")




#
# -------------------- test_closed_TN_with_ncon ------------------------
#


def test_closed_TN_with_ncon():
	
	"""
	
	Test the bubblecon routine by running it over a small closed tensor 
	made of 5 tensors and compare the resultant number to the result of
	ncon.
	
	The TN has the following form:
	
	              o
	             / \
	            /   \
	           o-----o 
	           |     |
	           |     |
	           o-----o
	
	          
	Exact describption of the geometry is given in the PDF file 
	TN-examples.pdf in this folder.
	          
	"""

	

	np.random.seed(1)

	D = 4  # bond dimension
		
	T_list = []
	
	T0 = np.random.normal(size=[D,D])
	T_list.append(T0/norm(T0))
	
	T1 = np.random.normal(size=[D,D,D])
	T_list.append(T1/norm(T0))
	
	T2 = np.random.normal(size=[D,D,D])
	T_list.append(T2/norm(T0))
	
	T3 = np.random.normal(size=[D,D])
	T_list.append(T3/norm(T0))
	
	T4 = np.random.normal(size=[D,D])
	T_list.append(T4/norm(T0))
	
	print("Comparing the bubblecon contraction to ncon contraction")
	print(f"for a closed TN with {len(T_list)} nodes and bond dimension {D}")
	print("========================================================")
	

	#                 T0     T1       T2     T3      T4
	edges_list = [ [1,6], [1,2,3], [2,4,5], [3,4], [5,6]]
	
	# the possible angles
	U, D, L, R = pi/2, 3*pi/2, pi,0
	#                T0        T1           T2            T3           T4
	angles_list = [[U,R], [D,R,pi/4], [L,3*pi/4,D], [5*pi/4,7*pi/4], [U,L]]
	

	#
	# =========== First, contract using bubblecon ================
	#

	swallow_order=[4,0,2,1,3]
	
	bubble_angle=3*pi/4
	
	Dp = None   # Do not truncate
	delta = None  # Do not truncate
	
	print("\n\n1. Contracting via bubblecon")

	A = bubblecon.bubblecon(T_list, edges_list, angles_list, bubble_angle,\
			swallow_order, D_trunc=Dp, eps=delta, opt='high')
	print("  ==> Done")

	#
	# ================ Next, contract using ncon ================
	#
	
	print("\n\n2. Contracting via ncon:")
	B = ncon(T_list, edges_list)
	print("  ==> Done")
	

	print("\n\n")
	print("3. Contraction results: \n")
	print("   bubblecon={}   ncon={}    Relative error={}".format(\
		A,B,abs( (A-B)/A)))
		
	print("\n\n")



#
# ----------------------- test_with_bmps -------------------------------
#


def test_with_bmps():
	
	"""
	
	Here we compare the bubblecon RDM to the ones that are obtained 
	by the tradional bmps method with very high truncation bond (D=128)
	which was calculated before.
	
	We first turn a given PEPS into a legless TN, and then run the 
	bubblecon on the upper rows and the lower rows. This give us two MPS, 
	which are then contracted together to give the RDM in the middle.
	
	"""
	

	d = 2 # physical bond dimension
	D = 4 # Virtual bond dimension
	
	Dp = 32
	
	ii,jj = 1,1  # Calculate the RDM (ii,jj)--(ii,jj+1)
	
#	PEPS_fname = 'AFH-PEPS.pkl'
	PEPS_fname = 'random-PEPS-seed2.pkl'
	PEPS_fname = 'random-snake-PEPS-seed2.pkl'

	rhoL_fname = 'rhoH-rhoV-Dp128-AFH-full.pkl'
	rhoL_fname = 'rhoH-rhoV-Dp128-snake-full.pkl'
#	rhoL_fname = 'rhoH-rhoV-Dp128-full.pkl'
	
	outfile=open(PEPS_fname,'rb')
	rpeps = pickle.load(outfile)
	outfile.close()
	
	M,N = rpeps.M, rpeps.N # Lattice size
	
	#
	# Create the TN from the PEPS <bra|ket>
	#
	
	T_list = []
	
	for i in range(M):
		for j in range(N):
			T1 = rpeps.A[i][j]
			
			#
			# The legs of T are: [d, D_left, D_right, D_up, D_down]
			#
			
			(d, D_left, D_right, D_up, D_down) = T1.shape
			T = tensordot(T1, conj(T1), axes=([0],[0]))
			T = T.transpose([0,4,1,5,2,6,3,7])
			
			Tdims = []
			if j !=0:
				Tdims.append(D_left*D_left)
				
			if j !=N-1:
				Tdims.append(D_right*D_right)
				
			if i !=0:
				Tdims.append(D_up*D_up)
				
			if i !=M-1:
				Tdims.append(D_down*D_down)
				
			T = T.reshape(Tdims)
			
			
			T_list.append(T)
			
	#
	# Now create the edges list: 
	# (*) we first list all horizontal legs, up->down, left->right
	# (*) then all vertical legs up->down, left->right
	#
	
	q = 5*M*N+10
	edges_list = []
	angles_list = []

	for i in range(M):
		for j in range(N):
			
			eL, eR, eU, eD = -1,-1,-1,-1
			
			e_list = []
			ang_list = []
			
			if j != 0:
				eL = i*(N-1) + j
				e_list.append(eL)
				ang_list.append(pi)
			
			if j != N-1:
				eR = i*(N-1) + j + 1
				e_list.append(eR)
				ang_list.append(0)
			
			if i != 0:
				eU = (i-1)*N + j + q
				e_list.append(eU)
				ang_list.append(pi/2)
			
			if i != M-1:
				eD = i*N + j + q
				e_list.append(eD)
				ang_list.append(3*pi/2)
				
			edges_list.append(e_list)
			angles_list.append(ang_list)
			
	#
	# We first calculate the upper bmps
	#
		
	swallow_order = list(range(N))
	for i in range(1,ii):
		for j in range(0,N,2):
			swallow_order.append(i*N + j)
		for j in range(1,N,2):
			swallow_order.append(i*N + j)
	
	mpsU = bubblecon.bubblecon(T_list, edges_list, angles_list, 3*pi/2,\
		swallow_order, D_trunc=Dp, D_trunc2=Dp*D*D, eps=None)
	
	#
	# The legs of mpsU goes from right to left, so we change it to left->right
	#
	
	n=mpsU.N
	newA = [None]*n
	for i in range(n):
		A = mpsU.A[n-1-i]
		newA[i] = A.transpose([2, 1, 0])
	mpsU.A = newA
	
	# Remove the trivial edges on the ends
	
	n=mpsU.N
	[DL, d, DR] = mpsU.A[0].shape
	mpsU.A[0] = mpsU.A[0].reshape(d, DR)
	
	[DL, d, DR] = mpsU.A[n-1].shape
	mpsU.A[n-1] = mpsU.A[n-1].reshape(DL,d)
	
	#
	# Calculate the lower bmps
	#
	
	swallow_order = list(range((M-1)*N, M*N))
	
	for i in range(M-2, ii, -1):
		for j in range(0,N,2):
			swallow_order.append(i*N + j)
		for j in range(1,N,2):
			swallow_order.append(i*N + j)

	mpsD = bubblecon.bubblecon(T_list, edges_list, angles_list, pi/2,\
		swallow_order, D_trunc=Dp, D_trunc2=Dp*D*D, eps=None)
	
	# Remove the trivial edges on the ends
	n=mpsD.N
	[DL, d, DR] = mpsD.A[0].shape
	mpsD.A[0] = mpsD.A[0].reshape(d, DR)
	
	[DL, d, DR] = mpsD.A[n-1].shape
	mpsD.A[n-1] = mpsD.A[n-1].reshape(DL,d)
		
	#
	# Now use ncon to calculate CLO and CRO
	#
	
	# 1. CLO
	
	T1_list = [mpsU.A[0], T_list[ii*N], mpsD.A[0]]
	#               mpU      mid     mpD
	if jj>1:
		edges1_list = [[1,3], [4,1,2], [2,5] ]
	else:
		edges1_list = [[1,-1], [-2,1,2], [2,-3] ]
		
	
	for j in range(1, jj):
		T1_list = T1_list + [mpsU.A[j], T_list[ii*N+j], mpsD.A[j]]
		if j<jj-1:
			edges1_list = edges1_list + [ [j*5-2, j*5+1, j*5+3], \
				[j*5-1, j*5+4, j*5+1, j*5+2], [j*5, j*5+2, j*5+5]]
		else:
			edges1_list = edges1_list + [ [j*5-2, j*5+1, -1], \
				[j*5-1, -2, j*5+1, j*5+2], [j*5, j*5+2, -3]]
				
	CLO = ncon(T1_list, edges1_list)
	print("Resultant CLO shape: ", CLO.shape)
			
			
	# 2. CRO
	n=mpsU.N
	
	T1_list = [mpsU.A[n-1], T_list[ii*N+n-1], mpsD.A[n-1]]
	#               mpU      mid     mpD
	if jj<n-3:
		edges1_list = [[3,1], [4,1,2], [5,2] ]
	else:
		edges1_list = [[-1,1], [-2,1,2], [-3,2] ]
		
	print("initial: ", edges1_list)
	for j1 in range(n-2, jj+1,-1):
		T1_list = T1_list + [mpsU.A[j1], T_list[ii*N+j1], mpsD.A[j1]]
		j = n-1-j1
		if j1>jj+2:
			e_list =  [ [j*5+3, j*5+1, j*5-2], [j*5+4, j*5-1, j*5+1, j*5+2],
				[j*5+5, j*5+2, j*5]]
		else:
			e_list =  [ [-1, j*5+1, j*5-2], [-2, j*5-1, j*5+1, j*5+2],
				[-3, j*5+2, j*5]]
				
		print("j={} j1={} e-list={}".format(j,j1, e_list))
		edges1_list = edges1_list + e_list
				
	CRO = ncon(T1_list, edges1_list)
	print("Resultant CRO shape: ", CRO.shape)
			
			
	#
	# Now use CLO, CRO together with the mid A's of mpsU, mpsD to 
	# find the 2-local RDM
	#
	
	T1_list = [CLO, mpsU.A[jj], mpsD.A[jj]]
	edges1_list = [ [1,-2,2], [1,-4,-1], [2,-5,-3]]
	CLO = ncon(T1_list, edges1_list)
	
	T1_list = [CRO, mpsU.A[jj+1], mpsD.A[jj+1]]
	edges1_list = [ [1,-2,2], [-1,-4, 1], [-3,-5,2]]
	CRO = ncon(T1_list, edges1_list)
		
	D2 = D*D
	m1 = tensordot(rpeps.A[ii][jj], conj(rpeps.A[ii][jj]), 0)	
	m1 = m1.transpose([0,5,1,6,2,7,3,8,4,9])
	m1 = m1.reshape([2,2,D2,D2,D2,D2])
	
	m2 = tensordot(rpeps.A[ii][jj+1], conj(rpeps.A[ii][jj+1]), 0)	
	m2 = m2.transpose([0,5,1,6,2,7,3,8,4,9])
	m2 = m2.reshape([2,2,D2,D2,D2,D2])
	
	T1_list = [CLO, m1, m2, CRO]
	edges1_list = [ [1,2,3,4,5], [-1,-2,2,6,4,5], [-3,-4, 6,9,7,8], [1,9,3,7,8]]
	
	rho1 = ncon(T1_list, edges1_list, [2,4,5,1,6,3,7,8,9])

	tr = np.einsum('iijj',rho1)
	rho1 = rho1/tr

	print("rho-bubble (Dp={}): ".format(Dp))
	print("---------------\n")
	print(rho1.real)

	#
	#
	# XXXXXXXXXXXXXXXXXXXXX NOW COMPARE TO BMPS CALC XXXXXXXXXXXXXXXXXXXX
	#
	#
	
	outfile=open(rhoL_fname,'rb')
	rho_list128 = pickle.load(outfile)
	outfile.close()
	
	rho2 = rho_list128[ii*(N-1) + jj]              # (i,j)--(i,j+1)
	
		
	
	print("\n\n")
	print("rho-bmps: (Dp=128)")
	print("---------------\n")
	print(rho2.real)

	print("|| rho_bub - rho_bmps(128)|| = ", TenQI.op_norm(rho1-rho2, 'tr'))



def main():

	print("\n\n")
	print("Comparing bubblecon to ncon")
	print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
	print("\n\n")

	np.random.seed(1)
	
	
#	test_open_TN_with_ncon()
	
	test_closed_TN_with_ncon()
	
#	test_with_bmps()


main()
