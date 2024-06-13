if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

# for math:
import numpy as np

# for plotting:
from _config_reader import ALLOW_VISUALS
if ALLOW_VISUALS:
    import matplotlib.pyplot as plt
from utils import visuals

from lattices._common import Node


## Constants:
edge_color ,alpha ,linewidth = 'dimgray', 0.5, 3
angle_color, angle_linewidth, angle_dis = 'green', 2, 0.5

average = lambda lst: sum(lst) / len(lst)

@visuals.matplotlib_wrapper()
def plot_lattice(
	nodes : list[Node],
	edges : dict[str, tuple[int, int]],
    detailed : bool = True
)-> None:
    
    ## Complete data:
    edges_list = [node.edges for node in nodes]
    pos_list = [node.pos for node in nodes]
    angles_list = [node.angles for node in nodes]


    def _tensor_indices(edge_name:str, assert_connections:bool=True) -> list[int]:
        tensors_indices = [t for t, edges in enumerate(edges_list) if edge_name in edges]
        ## Some double checks:
        if assert_connections:
            edge_nodes = [nodes[index] for index in tensors_indices]
            for node, index in zip(edge_nodes, tensors_indices, strict=True):
                assert node.index == index
            splitted = edge_name.split("-")

            if len(splitted)==2:
                for char in splitted:
                    if not char.isnumeric():
                        return tensors_indices # The name mean nothing in here
                        
                splitted = [int(char) for char in splitted]
                splitted.sort()
                sorted_indices = tensors_indices.copy()
                sorted_indices.sort()
                for from_name, from_mapping in zip(splitted, sorted_indices):
                    assert from_name==from_mapping
        return tensors_indices

    def _edge_positions(edge_name:str) -> tuple[list[int], list[int]]:
        tensors_indices = _tensor_indices(edge_name)
        x_vec = [pos_list[tensor_ind][0] for tensor_ind in tensors_indices]
        y_vec = [pos_list[tensor_ind][1] for tensor_ind in tensors_indices]
        assert len(x_vec)==len(y_vec)==2
        return x_vec, y_vec
        
    # Plot nodes:
    for i, pos in enumerate(pos_list):
        node = nodes[i]
        assert node.pos == pos
        x, y = pos
        color = "red" 
        marker = "o" 
        size1 = 15
        size2 = 30
        plt.scatter(x, y, c="black", s=size2, marker=marker, zorder=3)
        plt.scatter(x, y, c=color, s=size1, marker=marker, zorder=4)
        if detailed:
            text = f" [{node.index}]" 
            plt.text(x, y, text)
            assert i==node.index, "Non consistent indexing"

    # Plot edges:
    for edge_name, tensors_indices in edges.items():
    
        ## Gather info:
        x_vec, y_vec = _edge_positions(edge_name)
        
        ## Define plot function:      
        plt.plot(x_vec, y_vec, color=edge_color, alpha=alpha, linewidth=linewidth, zorder=1 )
        if detailed:
            x = average(x_vec)
            y = average(y_vec)
            plt.text(x, y, f"{edge_name!r}\n", fontdict={'color':'darkorchid', 'size':10 } )
        

    # Plot angles of all edges per node:
    for i_node, (angles, origin, edges_) in enumerate(zip(angles_list, pos_list, edges_list, strict=True)):
        assert len(angles)==len(edges_)
        if not detailed:
            continue
        for i_edge, angle in enumerate(angles):
            dx, dy = angle_dis*np.cos(angle), angle_dis*np.sin(angle)
            x1, y1 = origin
            x2, y2 = x1+dx, y1+dy
            plt.plot([x1, x2], [y1, y2], color=angle_color, alpha=alpha, linewidth=angle_linewidth )
            plt.text(x2, y2, f"{i_edge}", fontdict={'color':'olivedrab', 'size':8 } )	

