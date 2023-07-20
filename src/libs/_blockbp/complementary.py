########################################################################
#
#   Module: helprt_functions.py:  Helper functions for blockbp
#   ==========================================================
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
#                Also: moved the create_example() function to the 
#                helper_functions_example.py module
#                      
#


import numpy as np
from collections import Counter

# For type hinting:
from typing import (
    Tuple,
    Dict,
    List,
)

########################################################################
#                                                                      #
#                          Main functions                              #
#                                                                      #
########################################################################


#
# -------------------  create_sedges_info  -----------------------------
#

def create_sedges_info(
    edges_list : List[List[int]], 
    pos_list : List[Tuple[int, ...]], 
    blocks_v_list : List[List[int]]
) -> Tuple[ 
    Dict[str, List[int]],        # sedges_dict
    List[List[Tuple[str, int]]]  # sedges_list
]:
    """
    Derives the super-edges information from TN data.

    Parameters:
    ------------
        edges_list:     Edges of the TN using ncon convention.
        pos_list:       Tensors positions on the plane. List of
                        (x, y) tuples.
        blocks_v_list:  List of tensor in each block.

    Returns:
    --------
        sedges_dict:  Dictionary of super edges. The edges in the super
                      edge are sorted.
        sedges_list:  List of super edges for each block and their
                      relative orientation.
    """

    sedges_dict = {}
    sedges_list = []
    sedges_block_dict = {idx: {} for idx, _ in enumerate(blocks_v_list)}
    outgoing_edges = get_outgoing_edges(edges_list, blocks_v_list)

    #
    # Iterate through outgoing edges from pairs of blocks to find super
    # edges
    #

    for i, edges_1 in enumerate(outgoing_edges):
        for j, edges_2 in enumerate(outgoing_edges[i + 1:]):
            super_edge = np.intersect1d(edges_1, edges_2)
            if len(super_edge) > 0:
                sedges_block_dict[i][f"SE_{i}_{i + j + 1}"] = 0
                sedges_block_dict[i + j + 1][f"SE_{i}_{i + j + 1}"] = 0
                sedges_dict[f"SE_{i}_{i + j + 1}"] = super_edge

    #
    # Sort the super edges and find their orientation
    #
    sedges_dict, sedges_block_dict = sort_sedges_dict(edges_list, \
            pos_list, sedges_dict, sedges_block_dict, blocks_v_list)

    #
    # Convert from Dictionary to List
    #
    for sedge_dict in sedges_block_dict.values():
        sedges_list.append([item for item in sedge_dict.items()])

    return sedges_dict, sedges_list


#
# -------------------  create_blocks_con_list  -------------------------
#

def create_blocks_con_list(edges_list, blocks_v_list, sedges_dict, pos_list):
    """
    Creates contraction order for each block.

    Parameters:
    -----------
        edges_list:     Edges of the TN using ncon convention.
        blocks_v_list:  List of tensor in each block.
        sedges_dict:    Dictionary of super edges. The edges in the super edge are sorted.
        pos_list:       Tensors positions on the plane. List of (x, y) tuples.

    Returns:
    ---------

        blocks_con_list:  Contraction list for each block.
    """

    blocks_con_list = []
    outgoing_edges = get_outgoing_edges(edges_list, blocks_v_list)

    # Iterate through the blocks
    for idx, block in enumerate(blocks_v_list):
        bcon = []
        outgoing = outgoing_edges[idx]

        # For each block iterate through its super edges
        for sedge_label, sedge in sedges_dict.items():
            blocks_in_sedge = [int(s) for s in sedge_label.split("_") if s.isdigit()]

            # Get tensors in outgoing message
            outgoing_tensors = []
            for edge in sedge:
                for tensor in get_tensors_in_edge(edge, edges_list):
                    if tensor in block:
                        outgoing_tensors.append(tensor)

            # For each super edge in a block, create the contraction order for its outgoing MPS message
            if idx in blocks_in_sedge:
                sedges_to_contract = [item for item in outgoing if item not in sedge]
                graph, graph_edges, graph_positions, graph_nodes = create_graph_from_block(
                    edges_list, block, sedges_to_contract, sedges_dict, pos_list)
                con_order = con_order_from_graph(graph, graph_edges, graph_positions, graph_nodes)
                start = con_order[0]
                swangle = get_sw_angle(start[1], pos_list, block, edges_list)
                bcon.append(tuple((con_order, swangle)))

        # Make List of contractions
        blocks_con_list.append(bcon)

    return blocks_con_list




########################################################################
#                                                                      #
#                      Auxiliary functions                             #
#                                                                      #
########################################################################


                  

#
# ------------------------  get_sw_angle  ------------------------------
#

def get_sw_angle(start, pos_list, block, edges_list):
    """
    Returns the swallowing angle for a contraction order.

    Parameters:
    -----------
        start:       First contraction node.
        pos_list:    Tensors positions on the plane. List of (x, y) tuples.
        block:       Block being contracted.
        edges_list:  Edges of the TN using ncon convention.

    Returns:
    --------
        sw_angle:  Swallowing angle.
    """

    center = center_of_mass(block, pos_list)
    start_point, _ = get_mid_point_edge(start, pos_list, edges_list)
    sw_angle = np.arctan2(center[1] - start_point[1], center[0] \
        - start_point[0]) % (2*np.pi)

    return sw_angle


#
# ----------------------  con_order_from_graph  ------------------------
#

def con_order_from_graph(graph, graph_edges, graph_positions, graph_nodes):
    """

    Creates contraction order for a block given its graph representation
    using BFS.

    Parameters:
    -----------
        graph:            Graph representation of a block.
        graph_edges:      Edges list of the graph.
        graph_positions:  Positions of the nodes in the graph.
        graph_nodes:      List of the graph nodes.

    Returns:
    --------
        order:  Contraction order for the block.
    """

    order = []
    edge_nodes = [node for node in graph.keys() if 'e' in node.split("_")]
    queue = []
    start = edge_nodes[0]
    queue.append(start)
    bubble = []

    # Keep visiting nodes until they were all visited
    while queue:

        # Sort Queue using con_order_criteria
        queue.sort(reverse=False, key=lambda x: con_order_criteria(x, \
            bubble, graph_edges, graph_positions, graph_nodes))

        s = queue.pop(0)
        node = s.split("_")
        edge_label = node[1]
        if node[2] == "i":
            edge_label = int(node[1])

        order.append(tuple((node[0], edge_label)))
        bubble.append(s)
        not_bubble = [item for item in graph_nodes if item not in bubble]
        partition_list = [[graph_nodes.index(item) for item in bubble],
                          [graph_nodes.index(item) for item in not_bubble]]

        # Sort outgoing edges for the curent bubble
        sedges_bubble_dict, _ = create_sedges_info(graph_edges, graph_positions, partition_list)

        ordered_bubble_legs = []
        for item in sedges_bubble_dict.values():
            ordered_bubble_legs = item

        queue = []
        for node_bubble in bubble:
            for neighbour in graph[node_bubble]:
                if neighbour not in bubble:
                    idx = graph_nodes.index(neighbour)
                    edges_neighbour = graph_edges[idx]
                    legs_to_swallow = [item for item in edges_neighbour if item in ordered_bubble_legs]
                    indexes = [ordered_bubble_legs.index(item) for item in legs_to_swallow]
                    max_idx = max(indexes)
                    min_idx = min(indexes)
                    num_legs = len(legs_to_swallow)

                    # Make sure swallow is contiguous
                    if max_idx-min_idx+1 == num_legs:
                        queue.append(neighbour)

    return order


#
# ---------------------  get_edge_label_type  ----------------------
#

def get_edge_label_type(edge):
    """
    Return the type of edge label.

    Parameters:
    ------------
        edge:  Edge to consider.

    Returns:
    --------
        type_label:       Edge's label type. "s" for string, "i" for integer.
        graph_edges:      Edges list of the graph.
        graph_positions:  Positions of the nodes in the graph.
        graph_nodes:      List of the graph nodes.
    """

    type_label = "i"
    if isinstance(edge, str):
        type_label = "s"

    return type_label


#
# ---------------------  create_graph_from_block  ----------------------
#

def create_graph_from_block(edges_list, block_tensors, block_edges, sedges_dict, pos_list):
    """
    Creates graph representation for a given block.

    Parameters:
    ------------

        edges_list:     Edges of the TN using ncon convention.
        block_tensors:  Tensors that make the block.
        block_edges:    Super edges of the block.
        sedges_dict:    Dictionary of super edges. The edges in the super
                        edge are sorted.
        pos_list:       Tensors positions on the plane. List of
                        (x, y) tuples.

    Returns:
    --------
        graph:  Graph representation of a block.
    """

    graph_positions = []
    graph = {f"e_{node}_{get_edge_label_type(node)}": [] for node in block_edges}
    graph.update({f"v_{node}_i": [] for node in block_tensors})

    # Iterate through the edges in the block
    for edge in block_edges:

        # Find the super edge that contains edge
        for sedge in sedges_dict.values():
            if edge in sedge:
                index = sedge.index(edge)
                sedge_removed = [f"e_{item}_{get_edge_label_type(item)}" for i, item in enumerate(sedge)\
                    if i == index + 1 or i == index - 1]

                # Add the edge's neighbours (in the super edge) to the graph's
                # adjancency Dictionary
                graph[f"e_{edge}_{get_edge_label_type(edge)}"].extend(sedge_removed)

        # Find the tensors in the block that are connected to the edge and
        # add them to the graph's adjancency Dictionary
        for i, edges in enumerate(edges_list):
            if edge in edges and i in block_tensors:
                graph[f"e_{edge}_{get_edge_label_type(edge)}"].append(f"v_{i}_i")
                graph[f"v_{i}_i"].append(f"e_{edge}_{get_edge_label_type(edge)}")

    # Iterate through the tensors in the block and add connected tesors
    # to the graph's adjancency Dictionary
    for i, tensor1 in enumerate(block_tensors):
        for _, tensor2 in enumerate(block_tensors[i + 1:]):
            common_edge = np.intersect1d(edges_list[tensor1], edges_list[tensor2])
            if len(common_edge) == 1:
                graph[f"v_{tensor1}_i"].append(f"v_{tensor2}_i")
                graph[f"v_{tensor2}_i"].append(f"v_{tensor1}_i")

    graph_nodes = sorted(graph.keys())
    graph_edges = [[] for _ in range(len(graph_nodes))]

    edge = 1
    for i, node in enumerate(graph_nodes):
        neighbours = graph[node]
        for neighbour in neighbours:
            j = graph_nodes.index(neighbour)
            if j > i:
                graph_edges[i].append(edge)
                graph_edges[j].append(edge)
                edge += 1

    for node in graph_nodes:
        labels = node.split("_")
        if labels[0] == "v":
            graph_positions.append(pos_list[int(labels[1])])

        else:
            edge_label = labels[1]
            if labels[2] == "i":
                edge_label = int(labels[1])
            position, _ = get_mid_point_edge(edge_label, pos_list, edges_list)
            graph_positions.append(position)

    return graph, graph_edges, graph_positions, graph_nodes

#
# ------------------------  con_order_criteria  ------------------------
#

def con_order_criteria(node, bubble, graph_edges, graph_positions, graph_nodes):
    """
    Returns the property of a node that acts as the criteria by which the
    graph traversal will occur
    i.e. determines the contraction path.
    In this case it minimizes the bubble width at each step.

    Parameters:
    -----------
        node:             Node in a graph.
        graph:            Graph representation of a block.
        visited:          List of swallowed tensors.
        graph_edges:      Edges list of the graph.
        graph_positions:  Positions of the nodes in the graph.
        graph_nodes:      List of the graph nodes.

    Returns:
    ---------
        criteria:  Criteria by which the graph traversal will occur.
    """

    bubble_new = bubble.copy()
    bubble_new.append(node)
    not_bubble = [item for item in graph_nodes if item not in bubble_new]
    partition_list = [[graph_nodes.index(item) for item in bubble_new],
                      [graph_nodes.index(item) for item in not_bubble]]
    sedges_bubble_dict, _ = create_sedges_info(graph_edges, graph_positions, partition_list)

    legs = []
    for item in sedges_bubble_dict.values():
        legs = item

    criteria = len(legs)

    return criteria


#
# ------------------------  get_outgoing_edges  ------------------------
#

def get_outgoing_edges(edges_list, blocks_v_list):
    """
    Gets outgoing edges for each block.

    Parameters:
    -----------
        edges_list:     Edges of the TN using ncon convention.
        blocks_v_list:  List of tensor in each block.

    Returns:
    --------
        outgoing_edges:  Outgoing edges for each block.
    """

    outgoing_edges = []

    # Iterate through the blocks
    for block in blocks_v_list:
        edges_in_block = []

        # For each block get its edges
        for i in block:
            edges = edges_list[i]
            edges_in_block.extend(edges)

        # Outgoing edges appear only once, as opposed to edges within
        # a block that appear twice
        counter = Counter(edges_in_block)
        counter = Counter({tensor: count for (tensor, count) \
            in counter.items() if count == 1})
        outgoing_edges.append(np.asarray([i for i in counter.keys()]))

    return outgoing_edges


#
# ------------------------  sort_sedges_dict  ------------------------
#

def sort_sedges_dict(edges_list, pos_list, sedges_dict, sedges_block_dict, blocks_v_list):
    """
    Sort the edges in a super edge.

    Parameters:
    ------------
        edges_list:         Edges of the TN using ncon convention.
        pos_list:           Tensors positions on the plane. List of (x, y)
                            tuples.
        sedges_dict:        Dictionary of super edges. The edges in the
                            super edge are unsorted.
        sedges_block_dict:  Dicitionary of dictionaries initialized to zero
                            for every super edge.
        blocks_v_list:      List of tensor in each block.

    Returns:
    --------
        sorted_dict:        sedges_dict after sorting.
        sedges_block_dict:  Dictionary of dictionaries containing the
                            super edges for each block and their relative
                            orientation.
    """

    sorted_dict = {}

    # Iterate through the super edges
    for sedge_label, sedge in sedges_dict.items():
        blocks_in_sedge = [int(s) for s in sedge_label.split("_") if s.isdigit()]
        points_0 = []
        points_1 = []
        #reference_tensors = []
        block_0 = blocks_in_sedge[0]
        block_1 = blocks_in_sedge[1]

        tensors_block_0 = blocks_v_list[block_0]
        tensors_block_1 = blocks_v_list[block_1]

        tensors_sedge_0 = []
        tensors_sedge_1 = []
        # For each super edge, iterate through its edges and determine its
        # start and end tensors
        for edge in sedge:
            tensors_in_edge = get_tensors_in_edge(edge, edges_list)

            for tensor in tensors_in_edge:
                if tensor in tensors_block_0:
                    tensors_sedge_0.append(tensor)
                    points_0.append(pos_list[tensor])
                elif tensor in tensors_block_1:
                    tensors_sedge_1.append(tensor)
                    points_1.append(pos_list[tensor])

        # From the last start and end tensors, determine the start and end
        # reference positions

        center_0 = center_of_mass(tensors_block_0, pos_list)
        center_1 = center_of_mass(tensors_block_0, pos_list)

        tensors_sedge_0.sort(reverse=True, key=lambda x: dist(pos_list[x], center_1, norm=2))
        tensors_sedge_1.sort(reverse=True, key=lambda x: dist(pos_list[x], center_0, norm=2))
        ref_point_0 = pos_list[tensors_sedge_0[0]]
        ref_point_1 = pos_list[tensors_sedge_1[0]]

        # Determine the angles between the start and end points to the
        # start and end reference points respectively
        angles_start = np.asarray(
            [np.arctan2(point[1] - ref_point_1[1], point[0] \
                - ref_point_1[0]) for point in points_0])
        angles_end = np.asarray(
            [np.arctan2(point[1] - ref_point_0[1], point[0] - \
                ref_point_0[0]) for point in points_1])

        # Check to see if moving angles is needed (in case the sedge angles are around pi)
        move_start = move_angles(angles_start)
        move_end = move_angles(angles_end)

        if move_start:
            angles_start = angles_start % (2 * np.pi)

        if move_end:
            angles_end = angles_end % (2 * np.pi)

        # Sort the angles between the start points and the start reference
        idx_start = np.argsort(angles_start)
        start_sorted = angles_start[idx_start]

        # Iterate through all the angles
        for angle in angles_start[idx_start]:
            repeat_idx_original = np.argwhere(angles_start == angle).flatten()
            repeat_idx_to_change = np.argwhere(start_sorted == angle).flatten()

            # Find repeated angles in the sorted List
            if len(repeat_idx_original) > 1:
                # For the repeated angles, sort using the angles between the end
                # points and end reference

                angles_end_to_sort = angles_end[repeat_idx_original]
                idx_end_sorted = np.flip(np.argsort(angles_end_to_sort))
                repeat_idx_sorted = repeat_idx_original[idx_end_sorted]
                idx_start[repeat_idx_to_change] = repeat_idx_sorted

        # The edges were sorted by their angle to a reference point in one
        # of the blocks, and so their relative direction is known.
        sedges_block_dict[blocks_in_sedge[0]][sedge_label] = 1
        sedges_block_dict[blocks_in_sedge[1]][sedge_label] = -1
        sorted_dict[sedge_label] = sedge[idx_start].tolist()

    return sorted_dict, sedges_block_dict


#
# -----------------------  min_angle_range  ------------------------
#

def min_angle_range(arr):
    """
    Determines how centered around zero a set of angles are.

    Parameters:
    ------------
        arr:  List of angles.

    Returns:
    --------
        angle_range:  The minimum interval of two angles that contains zero.
    """

    neg_arr = [item for item in arr if item <= 0]
    pos_arr = [item for item in arr if item >= 0]

    min_pos, max_neg = None, None

    if pos_arr:
        min_pos = min(pos_arr)

    if neg_arr:
        max_neg = max(neg_arr)

    angle_range = 2 * np.pi
    if min_pos is not None and max_neg is not None:
        angle_range = min_pos - max_neg

    return angle_range

#
# -----------------------  move_angles  ------------------------
#

def move_angles(arr):
    """
    Determines if moving the angles from [-pi, pi] to [0, 2*pi] is needed.

    Parameters:
    ------------
        arr:  List of angles.

    Returns:
    --------
        move:  Boolean variable to move: True <-> moving is needed.
    """

    diff2pi = [(item % (2 * np.pi)) - np.pi for item in arr]

    range_0 = min_angle_range(arr)
    range_pi = min_angle_range(diff2pi)

    move = range_pi < range_0

    return move


#
# -----------------------  get_tensors_in_edge  ------------------------
#

def get_tensors_in_edge(edge, edges_list):
    """
    Returns tensors connected by some edge.

    Parameters:
    -----------
        edge:        Edge id.
        edges_list:  Edges of the TN using ncon convention.

    Returns:
    --------
        tensors:         Tensors connected by the edge.
    """
    tensors = []
    # Get the tensors connected by the edge.
    for i, edges in enumerate(edges_list):
        if edge in edges:
            tensors.append(i)

    return tensors


#
# -----------------------  get_mid_point_edge  ------------------------
#

def get_mid_point_edge(edge, pos_list, edges_list, p=0.5):
    """
    Returns point between two tensors defined by an edge.

    Parameters:
    -----------
        edge:        Edge id.
        pos_list:    Tensors positions on the plane. List of (x, y) tuples.
        edges_list:  Edges of the TN using ncon convention.
        p:           Constant that determines what point on the edge to be returned.
                     Defalut is 0.5 i.e. return mid-point.

    Returns:
    --------
        (x_mid, y_mid):  Point in the edge.
        tensors:         Tensors connected by the edge.
    """

    # Get the tensors connected by the edge.
    tensors = get_tensors_in_edge(edge, edges_list)

    # Get the tensors position and calculate mid-point
    pos1 = pos_list[tensors[0]]
    pos2 = pos_list[tensors[1]]
    x_mid = (p * pos1[0] + (1 - p) * pos2[0])
    y_mid = (p * pos1[1] + (1 - p) * pos2[1])

    return tuple((x_mid, y_mid)), tensors


#
# ------------------------------  dist  --------------------------------
#

def dist(a1, a2, norm=None):
    """
    Returns the distance between two vectors.

    Parameters:
    -----------
        a1:    First vector.
        a2:    Second vector.
        norm:  Norm used to calculate distance.

    Returns:
    --------
        d:  distance between a1 and a2.
    """

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    d = np.linalg.norm(a1 - a2, ord=norm)

    return d


#
# ------------------------  center_of_mass  ----------------------------
#

def center_of_mass(tensors, pos_list):
    """
    Returns the center position of a group of tensors

    Parameters:
    -----------
        tensors:   List of tensors.
        pos_list:  Tensors positions on the plane. List of (x, y) tuples.

    Returns:
    ---------
        center:  Center position of tensors.

    """

    tensors = np.asarray(tensors)

    positions = np.asarray([np.asarray(pos_list[tensor]) \
        for tensor in tensors])

    center = np.sum(positions, axis=0)/tensors.size

    return center


#
# --------------------------  plot_network  ----------------------------
#

def plot_network(T_list, pos_list, edges_list, blocks_v_list, \
    eps=1e-2, s=50, block_factor=1.1, draw_blocks=True):

    """
        Plots the TN.

        Parameters:
            T_list:         List of tensor IDs.
            pos_list:       Tensors positions on the plane. List of (x, y) tuples.
            edges_list:     Edges of the TN using ncon convention.
            blocks_v_list:  List of tensor in each block.
            eps:            Separation for text labels.
            s:              Size of nodes.
            block_factor:   Radius factor for drawing blocks.
    """
    import matplotlib.pyplot as plt

    pos_list = np.asarray(pos_list)
    x = pos_list[:, 0]
    y = pos_list[:, 1]

    fig, ax = plt.subplots()
    fig.suptitle('Tensor Network Plot', fontsize=14, fontweight='bold')

    # Draw all the edges
    edges_drawn = []
    for tensor_edges in edges_list:
        for edge in tensor_edges:
            if edge not in edges_drawn:
                tensors_in_edge = get_tensors_in_edge(edge, edges_list)
                t1, t2 = tensors_in_edge[0], tensors_in_edge[1]
                x1, x2, y1, y2 = x[t1], x[t2], y[t1], y[t2]

                ax.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1)

                ax.text(0.5 * (x1 + x2) + eps, 0.5 * (y1 + y2) + eps, f"{edge}", \
                    fontsize=8, color='k')

                edges_drawn.append(edge)

    # Draw the tensors
    ax.scatter(x, y, s=s, color='tab:blue', zorder=len(edges_drawn)+1)

    # Draw tensors labels
    for i in range(len(x)):
        ax.text(x[i] + eps, y[i] + eps, f"{T_list[i]}", fontsize=8, color='tab:blue')

    if draw_blocks:
        # Draw blocks circles
        for block in blocks_v_list:
            center = center_of_mass(block, pos_list)
            distances = []

            for tensor in block:
                distances.append(dist(center, pos_list[tensor], norm=2))

            max_dist = np.amax(np.asarray(distances))
            circle = plt.Circle(center, radius=max_dist * block_factor, \
                fill=False, edgecolor='tab:orange',
                                linestyle='--', linewidth=1)
            ax.add_patch(circle)

    plt.show()
