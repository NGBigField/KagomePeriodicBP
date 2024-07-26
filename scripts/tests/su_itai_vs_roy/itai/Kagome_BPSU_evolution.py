#!/usr/bin/python3


########################################################################
#
#               Kagome-BPSU-evolution.py 
#            =========================================
#
#  Performs a BP simple-update (BPSU) imaginary/real time evolution of a 
#  2-local Hamiltonian on an periodic Kagome lattice
#
#  History:
#  --------
#
#  6-Jul-2024: Initial version
#
#
########################################################################



from pathlib import Path
import sys

## Import itai package:
this_folder = Path(__file__).parent.__str__()
if this_folder not in sys.path:
	sys.path.append(this_folder)


#
# Control the number of threads.
#
from os import environ
N_THREADS = '4'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import scipy

import os

import pickle
import time

from numpy.linalg import norm, svd, qr

from numpy import zeros, ones, array, tensordot, sqrt, diag, dot, \
	reshape, transpose, conj, eye, trace, pi, exp, sinh, log

from scipy.linalg import expm

from TenQI import *

from qbp import qbp, calc_e_dict

from BPSU import  BP_gauging,  merge_SU_weights, \
	apply_2local_gate, local_2RDMs

from ITE import rho_ij

from typing import TypedDict

class _OutputDictType(TypedDict):
	path : str
	data : list[np.ndarray]

#
# -------------------   is_external_e   -----------------------------
#
def is_external_e(e):
	"""
	
	Check if an edge is an external edge (i.e., going out of the hexagon).
	
	External edges have labels like 'DL-UR-4', 'D-U-1', ... 
	Internal edges have labels of the form '<i>-<j>' like '2-12', '27-29', ...
	
	"""
	
	c = e[0]
	
	return c.isalpha()
	



#
# -------------------   random_T_list   -----------------------------
#
def random_T_list(D, d, e_list, bc='periodic'):
	
	"""
	
	Create a list of random tensors of a given bond dimension D and 
	a physical bond dimension d. The TN is either periodic, or is open. 
	If it open, then boundary edes get a bond dimension 1.
	
	The random tensors are drawn from a normal complex distribution, and
	are normalized to have their L_2 norm = 1.
	
	Input Parameters:
	-------------------
	D --- PEPS logical bond dimension
	
	d --- PEPS physical bond dimension
	
	e_list --- The e_list of the TN
	
	bc --- boundary condition --- either 'periodic' or 'open'
	
	Output:
	--------
	
	T_list --- The TN list of random tensors
	
	
	"""
	
	N = len(e_list)
	T_list = []
	
	for es in e_list:
		k = len(es)
		Ds = [d]
		for e in es:
			
			if bc=='periodic':
				Ds.append(D)
				
			else:
				#
				# So we're on open b.c. --- so boundary edges get D=1
				#
				
				if is_external_e(e):
					#
					# Boundary edge
					#
					Ds.append(1)
				else:
					#
					# Then we're on internal edge
					#
					Ds.append(D)

		T = np.random.normal(size=Ds) + 1j*np.random.normal(size=Ds)
		T = T/norm(T)
		
		T_list.append(T)
		
	return T_list
	


#
# -----------------------  Heisenberg_H  -------------------------
#
def Heisenberg_H(ij1, ij2, params=None):

	"""

	Calculate the Heisenberg interaction between two spins that sit
	at vertices ij1 and ij2. 
	
	params is an optional parameters dictionary.

	"""

	Z_field = params['Z_field']

	
	J = 0.25 # conversion between Pauli matrices to Spin 1/2 operators

	h = J*(tensordot(sigma_X, sigma_X, 0) + tensordot(sigma_Y, sigma_Y, 0) \
		+ tensordot(sigma_Z, sigma_Z, 0) ) + \
	 0.5*Z_field*((-1)**ij1*tensordot(sigma_X, ID1, 0) \
		+ (-1)**ij2*tensordot(ID1, sigma_X, 0))


	return h





#
# -----------------------  avE_SU  -------------------------
#
def avE_SU(TN_params, H_params, update_params, H):

	"""

	Calculates the average energy of the Hamiltonian H using 
	the simple update (SU) environment

	<H> := \sum_e <h_e> / (No. of spins)

	To get the local simple-update (SU) env, qBP is first run. Using
	the converged BP messages, the Vidal gauge is calculated, from which
	the local SU envs are defined.


	Input Parameters:
	-----------------

	TN_params     --- Dictionary holding the TN parameters.
	                  Entries are: T_ket_list, e_list, e_dict, Nx
	
	H_params      --- Dictionary holding optional Hamiltonian parameters
	
	update_params --- Dictionary holding qBP params. Entries are
	                  BP_iter, BP_err, BP_damp

	H             --- A local Hamiltonian function of the form H(i,j)


	Output:
	-------

	The average energy per spin.


	"""

	Vidal_T_list = TN_params['Vidal_T_list']
	w_dict = TN_params['w_dict']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']


	#
	# Calculate all the 2-local RDMs using the SU approximation.
	# We then use these RDMs to calculate the energy
	#
	
	rdms_dict = local_2RDMs(Vidal_T_list, e_list,  e_dict, w_dict)
	
	#
	# Now loop over all edges, and calculate the energy of each edge
	# and add it all up
	#
	
	E = 0 # set the total energy to 0
	N = 0

	N = len(Vidal_T_list)

	edges_num = 0
	for e in e_dict.keys():
		
		if H_params['b.c.'] == 'open' and is_external_e(e):
			continue
		
		rho12 = rdms_dict[e]
		
		ij1,leg1, ij2,leg2 = e_dict[e]
		
		h = H(ij1, ij2, H_params)

		e = tensordot(rho12, h, axes=([0,1,2,3], [0,1,2,3]))

		E = E+e.real
		edges_num += 1

	avE_per_site = E/N

	return avE_per_site


#
# -----------------------  subE  -------------------------
#
def subE(TN_params, H_params, update_params, H, es):


	Vidal_T_list = TN_params['Vidal_T_list']
	w_dict = TN_params['w_dict']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']


	#
	# Calculate all the 2-local RDMs using the SU approximation.
	# We then use these RDMs to calculate the energy
	#
	
	rdms_dict = local_2RDMs(Vidal_T_list, e_list,  e_dict, w_dict)
	
	#
	# Now loop over all edges, and calculate the energy of each edge
	# and add it all up
	#
	
	E = 0 # set the total energy to 0

	edges_num = 0
	for e in es:
		
		rho12 = rdms_dict[e]
		
		ij1,leg1, ij2,leg2 = e_dict[e]
		
		h = H(ij1, ij2, H_params)

		e = tensordot(rho12, h, axes=([0,1,2,3], [0,1,2,3]))

		E = E+e.real

	return E




#
# -----------------------  apply_dt_step  -------------------------
#
def apply_dt_step(TN_params, m_list, H_params, update_params, H, Dmax, dt, \
	mode):

	"""

	Applies a dt imaginary/real step. The step can be one of 6 modes:
	
	blue, red, orange, grey, green, magneta
	
	This corresponds to a coloring of the edges. The lattice can be viewed
	as made from triangles
	
	                   A
	                  / \
	                 /   \
	                B-----C
	                
	 (*) The blue/red edges are the 0,1 edges of A
	 (*) The orange/grey are the 0,1 edges of B
	 (*) The green/magneta are the 0,1 edges of C
	 


	The envs are calculating using the MPS/MPO machinary of bmpslib
	(i.e., no bubblecon).

	Input Parameters:
	-----------------

	TN_params     --- Dictionary holding the TN parameters.
	                  Entries are: T_ket_list, e_list, e_dict, Nx
	                  
	m_list        --- A list of the BP messages from last round to be used
	                  as a initial messages. If set to None, a fresh 
	                  initial messages are used.
	
	H_params      --- Dictionary holding optional Hamiltonian parameters
	
	update_params --- Dictionary holding qBP params. Entries are
	                  BP_iter, BP_err, BP_damp

	H          --- The Hamiltonian

	Dmax       --- Maximal bond dimension (truncation bond dimension)

	dt         --- The dt of the imaginary time evolution. To do
	               real time evolution use 1j*dt

	mode       --- One of the possible 6 modes, 'blue', 'red', ...


	Output:
	-------

	The TN_params dictionary with the updated T_ket_list 
	+ final BP messages (to be used for next round)

	"""

	log = False



	Vidal_T_list = TN_params['Vidal_T_list']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']
	w_dict = TN_params['w_dict']
	edges_colors_dict = TN_params['edges_colors_dict']
	

	color_edges = edges_colors_dict[mode]
	
	#
	# ================================================
	# Updating the coloredges
	# ================================================
	#

	if log:
		print(f"=> Updating {mode} edges...")
		

	for e in color_edges:
		
		if (H_params['b.c.'] == 'open') and is_external_e(e):
			#
			# If we're on open b.c., we do not update the external edges
			#
			continue
		
		(i1, leg1, i2,leg2) = e_dict[e]
	
		h = H(i1, i2, H_params)

		h_mat = op_to_mat(h)
		U_mat = expm(dt*h_mat)
		U = mat_to_op(U_mat)

		Vidal_T_list, w_dict = apply_2local_gate(Vidal_T_list, e_list, \
			e_dict, w_dict, U, e, Dmax=Dmax)
				

	if log:
		print("=> Done.\n")

		
	TN_params['Vidal_T_list'] = Vidal_T_list
	TN_params['w_dict'] = w_dict

	return TN_params, m_list




#
# -----------------------  second_order_TZ  -------------------------
#
def second_order_TZ(TN_params, H_params, update_params, H, Dmax, dt):

	"""

	Applies a 2nd order Trotter-Suzuki time evolution e^{dt H}

	dt can be negative and/or imaginary.

	The 2nd order Trotter-Suzuki formula is the symmetric formula from
	arXiv:2210.15817 (formula Eq.4 for S_2(t) )

	U_e0(dt/2) U_o0(dt/2) U_e1(dt/2) U_o1(dt) U_e1(dt/2) U_o0(dt/2) U_e0(dt/2)


	"""
	
	RECYCLE_M = False  # Whether or not to recycle the BP messages.
                     # This can facilitate the convergence, but should
                     # only be used once the bond-dims of the PEPS are
                     # no longer changing.
                     
	INITIAL_M = 'U'
	
	#
	# Initially, start from the uniform messages. Then we re-use them
	# with the different iterations
	#
	
	m_list = INITIAL_M
	
	colors_list = ['blue', 'red', 'orange', 'grey', 'green', 'magneta']
	rev_list = colors_list[::-1]
	
	total_2nd_TZ_colors = colors_list + rev_list[1:]


	for c in total_2nd_TZ_colors:
		#
		# Do the 2nd order weights
		#
		if c=='magneta':
			w = -1.0
		else:
			w = -0.5

		TN_params, m_list = apply_dt_step(TN_params, m_list, H_params, \
			update_params, H, Dmax, w*dt, c)
	
		if not RECYCLE_M:
			m_list = INITIAL_M
	
	return TN_params


#
# -----------------------  refresh_gauge  ----------------------------
#	

def refresh_gauge(TN_params, update_params):

	log = True

	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']

	Vidal_T_list = TN_params['Vidal_T_list']
	w_dict = TN_params['w_dict']
	
	if w_dict is None:
		merged_T_list = TN_params['merged_T_list']
	else:
		merged_T_list = merge_SU_weights(Vidal_T_list, e_dict, w_dict)
		

	BP_iter = update_params['BP-iter']
	BP_err  = update_params['BP-err']
	BP_damp = update_params['BP-damp']
	
	m_list = 'U'
	
	if log:
		print("\n\n")
		print(" * * *  Running qBP to update the Vidal gauge...   * * *\n\n")

	m_list, err, iter_no = qbp(merged_T_list, e_list, e_dict, initial_m=m_list, \
		max_iter=BP_iter, delta=BP_err, \
		damping=BP_damp, permute_order=False)

	if log:
		print(f"... done with {iter_no} iterations and BP messages error={err:.6g}")
		print("\n")

	Vidal_T_list, w_dict = BP_gauging(merged_T_list, e_dict, m_list)
	
	TN_params['merged_T_list'] = merged_T_list
	TN_params['Vidal_T_list'] = Vidal_T_list
	TN_params['w_dict'] = w_dict
	
	return TN_params
	


      ##############################################################
      #                                                            #
      #                       M  A  I  N                           #
      #                                                            #
      ##############################################################


def main(
	d:int=2,
	D:int=3,
	N_Kagome:int=2,  # Kagome size parameter
	lattice_fname:str|None=None,
	save_at:str=""
) -> _OutputDictType:

	np.random.seed(5)

	if lattice_fname is None:
		lattice_fname = f'Kagome-Lattice-n{N_Kagome}.pkl'

	#
	# -----------------------  MODEL PARAMETERS  -------------------------
	#

	H = Heisenberg_H     # Which Hamiltonian function to use


	H_params={}
	H_params['b.c.'] = 'periodic'

	#
	# ----------------  Contraction/qBP parameters  ------------------
	#

	
	BP_iter = 50
	BP_err  = 1e-5
	BP_damp = 0.05


	#
	# ------------   (imaginary) Time Evolution Parameters   -------------
	#

	#
	# Initial/final external field (in order not to get trapped in a
	# local minima)
	#
	Z0 = 1e-1
	Z1 = 1e-10

	#
	# Imaginary time steps (g = e^{-H\cdot dt})
	#

	dt_list = [0.1]*50 + [5e-2]*100 + [2e-2]*100 + [1e-2]*200 \
		+ [5e-3]*400 + [1e-3]*400 + [5e-4]*400
#	dt_list = [1e-2]*100 + [5e-3]*100 + [1e-3]*300  + [5e-4]*300 


	#
	# File name holding the initial configuration. If set to 'None'
	# then starting from an random product state
	#

#	initial_fname = 'initial-BPSU-Kagome-PEPS.pkl'
	initial_fname = '|0>'
#	initial_fname = None

	#
	# File holding the lattice structure
	#

	ENERGY_INTERVAL = 5  # How often to check the energy

	SAVE_INTERVAL   = 5  # How often to save results

	#
	# File name to save results (after each energy measurement)
	#
	fname = save_at+f"BPSU-Kagome-PEPS-n{N_Kagome}-D{D}.pkl"


	#
	# ====================     START OF ITE     =========================
	#

	#
	# Get the lattice structure
	#

	fin = open(lattice_fname, 'rb')
	e_list, edges_colors_dict = pickle.load(fin)
	fin.close()
	
	N = len(e_list)  # Number of particles
		
	e_dict =  calc_e_dict(e_list)
	
	#
	# Set the initial tensors: either from a file, or random, or |00...0>
	#
	
	if initial_fname is None:
		print("\n\n")
		print(f"[*] Using a random TN initialization\n\n")
		merged_T_list = random_T_list(D=D, d=d, e_list=e_list, bc='open')
			
	elif initial_fname=='|0>':
		print("\n\n")
		print("Preparing initial state |0000...0>\n\n")
		
		merged_T_list = []
		T0 = array([1.0,0])
		for i in range(N):
			merged_T_list.append(T0.reshape([2,1,1,1,1]))
			
	else:
		print("\n\n")
		print(f"[*] Loading initial TN from file '{initial_fname}'\n\n")
		fin = open(initial_fname, 'rb')
		merged_T_list = pickle.load(fin)
		fin.close()


	steps = len(dt_list)
	total_T = sum(dt_list)

	print("\n\n")
	print(f"Starting Simple-Update + BP ITE for T={total_T:.6g}\n")


	TN_params={}
	TN_params['merged_T_list'] = merged_T_list
	TN_params['N_Kagome'] = N_Kagome
	TN_params['e_list'] = e_list
	TN_params['e_dict'] = e_dict
	TN_params['edges_colors_dict'] = edges_colors_dict
	
	#
	# Vital_T_list, w_dict are lists of tensors that describe the TN 
	# in the Vidal gauge. They are obtained from merged_T_list via
	# the BP gauge fixing.
	#
	
	TN_params['w_dict'] = None
	TN_params['Vidal_T_list'] = None


	update_params = {}
	update_params['BP-iter'] = BP_iter
	update_params['BP-err'] = BP_err
	update_params['BP-damp'] = BP_damp
	
	#
	# Update the Vidal gauge representation: w_dict, Vidal_T_list
	#
	TN_params = refresh_gauge(TN_params, update_params)

	#
	# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
	# X                                                              X
	# X            S T A R T   O F   I T E   L O O P                 X
	# X                                                              X
	# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
	#
	
	T = 0.0
	for st, dt in enumerate(dt_list):

		print("\n")
		print(f"[*] Running step {st}, t={T:.6g} ...\n")

		# 
		# Set a declining external field, designed to avoid local minima
		# 
		Z_field = Z0*(Z1/Z0)**(T/total_T)

		H_params['Z_field'] = Z_field

		time0 = time.time()
		TN_params = second_order_TZ(TN_params, H_params, update_params, \
			H, D, dt)
		time1 = time.time()
		delta_time = time1-time0
		print(f"  => done in {delta_time:.3g} secs.")
			
		T += dt

		#
		# After ENERGY_INTERVAL steps, we refresh the Vidal gauge
		# and calculate the approximate energy
		#
		if st % ENERGY_INTERVAL == 0 and st>0:

			TN_params = refresh_gauge(TN_params, update_params)
			
			SU_E = avE_SU(TN_params, H_params, update_params, H)
			
			print()
			print(f"Step {st}: t={T:.6g} dt={dt:.6g} SU-E={SU_E:.6g} Z: {Z_field:.6g}")
			print()
			
			if False:
				es = ['14-16', '14-27', '16-27', '27-28', '27-29', '28-29']
				sE = subE(TN_params, H_params, update_params, H, es)/3
				
				print("Sub E: ", sE)

		if st % SAVE_INTERVAL == 0 and st>0:
			print(f"\n   => Saving to file '{fname}' ...\n")
			fout = open(fname, 'wb')
			data = TN_params['merged_T_list']
			pickle.dump(data, fout)
			fout.close()


	print("\n\n\n")
	print("                      * * *   D O N E   * * * \n")

	output_dict = dict(
		path=fname,
		data=data
	)

	return output_dict


if __name__ == "__main__":

	main()
