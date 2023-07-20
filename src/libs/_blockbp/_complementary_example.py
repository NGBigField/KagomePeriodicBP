########################################################################
#
#   Module: helprt_functions_example.py:  
#   ======================================
#
#   Usage example for helper_functions 
#
#
#  LOG:
#  ====
#
#  31-Dec-2021:  Itai  Cosmetic reformatting so that it fits better 
#                      with the rest of the code:
#                      (*) Spaces -> Tabs
#                      (*) Added "#" headers for each function
#                      (*) Moved the import matplotlib to within the
#                          plot_network function (the reset does not
#                          depends on that library)
#
#                Also: Moved the create_example() function from 
#                      helper_functions.py  to the 
#                      helper_functions_example.py module
#                      
#


"""
Module: 
"""
import numpy as np
from helper_functions import create_sedges_info, create_blocks_con_list,\
	plot_network



#
# -----------------------  create_example  ------------------------
#
def create_example(ex_type='blockBP1', size=(4, 4), blocks=None):
	"""
	Creates TN example.

	Parameters:
	------------
		ex_type:  Example type. 'blockBP1' for example in 'blockBP1.pdf'; 
		          'grid' for PEPS TN.
		size:     Tuple containing the size of the PEPS, in case type is 'grid'.
		blocks:   Blocks description, list of blocks in the form 
		          [start_x, start_y, size_x, size_y].
				      Assume that the description matches the size.

	Returns:
	--------
		pos_list:       Tensors positions on the plane. List of (x, y) tuples.
		edges_list:     Edges of the TN using ncon convention.
		blocks_v_list:  List of tensor in each block.
		T_list:         List of tensor IDs.
	"""
	# If blocks not defined, use whole network as one block
	if blocks is None:
		blocks = [[0, 0, size[0], size[1]]]

	pos_list = []
	edges_list = [[] for _ in range(size[0] * size[1])]
	blocks_v_list = [[] for _ in range(len(blocks))]

	if ex_type == 'blockBP1':

		# Create example from blockBP1.pdf
		pos_list = [(3, 9), (6, 9), (4, 7), (2, 6), (6, 7), (11, 7), (10, 5), (13, 5), (6, 4), (4, 2), (7, 2), (5, 1)]
		edges_list = [[1, 2], [1, 3, 19, 6], [5, 3], [2, 5, 4, 14], [19, 4, 7, 8, 20], [6, 7, 10],
					  [8, 10, 9, 11], [9, 12], [11, 20, 15, 18, 17], [16, 14, 15], [13, 17, 12], [16, 18, 13]]
		blocks_v_list = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11]]

	if ex_type == 'obj_edges':
		# Create example from blockBP1.pdf with obj edges
		pos_list = [(3, 9), (6, 9), (4, 7), (2, 6), (6, 7), (11, 7), (10, 5), (13, 5), (6, 4), (4, 2), (7, 2), (5, 1)]
		edges_list = [['1', '2'], ['1', '3', '19', '6'], ['5', '3'], ['2', '5', '4', '14'], ['19', '4', '7', '8', '20'],
					  ['6', '7', '10'], ['8', '10', '9', '11'], ['9', '12'], ['11', '20', '15', '18', '17'],
					  ['16', '14', '15'], ['13', '17', '12'], ['16', '18', '13']]
		blocks_v_list = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11]]

	if ex_type == 'grid':

		# Initialize grid positions
		grid = np.zeros(size)
		pos_list = [(i, j) for i, row in enumerate(grid) for j, _ in enumerate(row)]
		edge = 1

		# Add 4-neighbours to each node
		for i, pos1 in enumerate(pos_list):
			for j, pos2 in enumerate(pos_list[i + 1:]):
				if np.linalg.norm(np.asarray(pos1) - np.asarray(pos2), ord=1) == 1:
					edges_list[i].append(edge)
					edges_list[j + i + 1].append(edge)
					edge += 1

		# Define blocks
		for idx_block, block in enumerate(blocks):
			for idx_tensor, pos in enumerate(pos_list):
				if pos[0] >= block[0] and pos[1] >= block[1]:
					if pos[1] <= block[3] + block[1] - 1 and pos[0] <= block[2] + block[0] - 1:
						blocks_v_list[idx_block].append(idx_tensor)

	T_list = np.arange(len(pos_list))

	return T_list, pos_list, edges_list, blocks_v_list



#
# -----------------------  create_examples  ------------------------
#
def make_examples():
	"""
	BlockBP1 Example
	"""

	T_list, pos_list, edges_list, blocks_v_list = create_example(ex_type="blockBP1")

	# Use functions from helper_functions
	sedges_dict, sedges_list = create_sedges_info(edges_list, pos_list, blocks_v_list)
	blocks_con_list = create_blocks_con_list(edges_list, blocks_v_list, sedges_dict, pos_list)

	# Print results
	print("\n----------- Block BP1 Example: -----------\n")
	print("pos_list: ", pos_list)
	print("edges_list: ", edges_list)
	print("blocks_list: ", blocks_v_list)
	print("\nsedges_dict: ", sedges_dict)
	print("sedges_list: ", sedges_list)
	print("blocks_con_list: ", blocks_con_list)

	plot_network(T_list, pos_list, edges_list, blocks_v_list, eps=1e-1)

	"""
	Obj edges exammple
	"""

	T_list, pos_list, edges_list, blocks_v_list = create_example(ex_type="obj_edges")

	# Use functions from helper_functions
	sedges_dict, sedges_list = create_sedges_info(edges_list, pos_list, \
												  blocks_v_list)
	blocks_con_list = create_blocks_con_list(edges_list, blocks_v_list, \
											 sedges_dict, pos_list)

	# Print results
	print("\n----------- Obj Edges Example: -----------\n")
	print("pos_list: ", pos_list)
	print("edges_list: ", edges_list)
	print("blocks_list: ", blocks_v_list)
	print("\nsedges_dict: ", sedges_dict)
	print("sedges_list: ", sedges_list)
	print("blocks_con_list: ", blocks_con_list)

	plot_network(T_list, pos_list, edges_list, blocks_v_list)


	"""
	Grid Example
	"""

	blocks_grid = [[0, 0, 5, 5], [0, 5, 5, 1], [5, 0, 1, 5], [5, 5, 1, 1]]
	T_list, pos_list, edges_list, blocks_v_list = \
		create_example(ex_type="grid", size=(6, 6), blocks=blocks_grid)

	# Use functions from helper_functions
	sedges_dict, sedges_list = create_sedges_info(edges_list, pos_list, \
		blocks_v_list)

	blocks_con_list = create_blocks_con_list(edges_list, blocks_v_list, \
		sedges_dict, pos_list)

	# Print results
	print("\n----------- Grid Example: -----------\n")
	print("pos_list: ", pos_list)
	print("edges_list: ", edges_list)
	print("blocks_list: ", blocks_v_list)
	print("\nsedges_dict: ", sedges_dict)
	print("sedges_list: ", sedges_list)
	print("blocks_con_list: ", blocks_con_list)

	plot_network(T_list, pos_list, edges_list, blocks_v_list)


if __name__ == "__main__":
	make_examples()
