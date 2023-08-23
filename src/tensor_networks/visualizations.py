if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )


# for plotting:
import matplotlib.pyplot as plt

# everyone needs numpy:
import numpy as np

# our utilities:
from utils import (
    visuals,
    lists,
    arguments,
    tuples,
)

# type hinting:
from typing import (
    Optional,
    List,
    Any,
    Tuple,
    Dict,
    TypedDict,
    Generator,
)

# For OOP
from dataclasses import dataclass, field

# Common types:
from tensor_networks.node import TensorNode, NodeFunctionality, UnitCellFlavor
from lattices.directions import Direction, check
from _error_types import DirectionError

# for smart iterations:
import itertools


def draw_now():
    visuals.draw_now()



@dataclass
class _OnBoundsPerDim():
    max : bool = False 
    min : bool = False

    def _check_key(self, key:str)->None:
        if key not in ['min', 'max']:
            raise KeyError(f"Not a valid key '{key}'. Accepts only 'max' or 'min'")

    def __setitem__(self, key:str, val:bool)->None:
        self._check_key(key)
        return self.__setattr__(key, val)

    def __getitem__(self, key:str)->bool:
        self._check_key(key)
        return self.__getattribute__(key)
    
    def __repr__(self) -> str:
        s = ""
        for min_max in ['min', 'max']:
            s += f" {self[min_max]:1} "
        return s


@dataclass
class _OnBounds():
    x : _OnBoundsPerDim = field(default_factory=_OnBoundsPerDim)
    y : _OnBoundsPerDim = field(default_factory=_OnBoundsPerDim)

    def _check_key(self, key:str)->None:
        if key not in ['x', 'y']:
            raise KeyError(f"Not a valid key '{key}'. Accepts only 'x' or 'y'")

    def __setitem__(self, key:str, val:_OnBoundsPerDim)->None:
        self._check_key(key)
        return self.__setattr__(key, val)

    def __getitem__(self, key:str)->_OnBoundsPerDim:
        self._check_key(key)
        return self.__getattribute__(key)

    def __repr__(self) -> str:
        s = "   min  max \n"
        for xy in ['x', 'y']:
            s += xy+": ["
            s += str(self[xy])
            s += "]\n"
        return s


def _check_skip_vec(order:str, edge_index:str, on_boundary:str, x_vec:list, y_vec:list)->bool:
    
    if len(x_vec)==len(y_vec)==0:
        return True
    
    if not isinstance(edge_index, str):
        return False
    
    edge_dir = edge_index[0]
    
    if on_boundary=='x' and edge_dir=="R" and order=="low":     return True
    if on_boundary=='x' and edge_dir=="L" and order=="high":    return True
    if on_boundary=='y' and edge_dir=="U" and order=="low":     return True
    if on_boundary=='y' and edge_dir=="D" and order=="high":    return True
    
    return False
    
    
def _outer_directions_vectors(x_vec:List[int], y_vec:List[int], dir:Direction, delta:float)->Tuple[List[float], ...]:     
    x = float(x_vec[0])
    y = float(y_vec[0])
    x_vec : list[float] = [ x, x + dir.unit_vector[0] ]
    y_vec : list[float] = [ y, y + dir.unit_vector[1] ]
        
    return x_vec, y_vec

def _opposite(ob1:_OnBoundsPerDim, ob2:_OnBoundsPerDim)->bool:
    if ob1.max==True and ob2.min==True:
        return True
    elif ob1.min==True and ob2.max==True:
        return True
    else:
        return False

def _get_block_plot_props(block_indices, block_colors, include_blocks:bool):
    if include_blocks and block_indices[0]==block_indices[1]: 
        color = block_colors[block_indices[0]] 
        alpha = 0.5
        linewidth = 4
    else: 
        color = 'black'
        alpha = 0.7
        linewidth = 1
    return color ,alpha ,linewidth

def _neighbors_pos(this_pos:tuple[int, ...], dir:Direction, delta:float)->tuple[float, ...]:
    if   dir is Direction.Left:  return (this_pos[0]-delta , this_pos[1]+0.0  )
    elif dir is Direction.Right: return (this_pos[0]+delta , this_pos[1]+0.0  )
    elif dir is Direction.Up:    return (this_pos[0]+0.0   , this_pos[1]+delta)
    elif dir is Direction.Down:  return (this_pos[0]+0.0   , this_pos[1]-delta)
    raise DirectionError(f"Not a valid option dir={dir}")     



def _check_on_boundaries(
    tensor_indices:tuple[int, int], 
    network_bounds:List[Tuple[int, ...]], 
    pos_list:List[Tuple[int, int]],
    edge_name:str,
    nodes:List[TensorNode],
    delta:float
)->Direction|None:

    assert isinstance(tensor_indices, tuple)
    if len(tensor_indices)==1:
        raise ValueError("")
    elif len(tensor_indices)==2:
        if tensor_indices[0]==tensor_indices[1]:
            node = nodes[tensor_indices[0]]
            edge_index = node.edges.index(edge_name)
            return node.directions[edge_index]
    else:
        raise ValueError("")
    return None

def _derive_smallest_distance(pos_list:List[Tuple[int, int]]) -> float:
    dummy = pos_list[0]
    min_ : float = np.inf
    for axis_ind in range(len(dummy)):  # x and y
        for pos1, pos2 in itertools.permutations(pos_list, r=2):
            pos_projection1 = pos1[axis_ind]
            pos_projection2 = pos2[axis_ind]
            distance = abs(pos_projection2 - pos_projection1)
            if distance == 0:
                continue
            min_ = min(distance, min_)
    return min_

def _derive_boundary(pos_list:List[Tuple[int, int]])->List[Tuple[int, ...]]:
    bounds : List = []
    dummy = pos_list[0]
    for axis_ind in range(len(dummy)):  # x and y
        max_ = -np.inf
        min_ = +np.inf
        for pos in pos_list:
            pos_projection = pos[axis_ind]
            max_ = max(pos_projection, max_)
            min_ = min(pos_projection, min_)
        bounds.append(( min_, max_ ))
    return bounds


def plot_contraction_order(positions:List[Tuple[int,...]], con_order:List[int])->None:
    not_all_the_way = 0.85
    for (from_, to_), color in zip(itertools.pairwise(con_order), visuals.color_gradient(len(positions)) ):
        # if from_<0 or to_<0:
        #     continue  # just a marker for something
        x1, y1 = positions[from_]
        x2, y2 = positions[to_]
        plt.arrow(
            x1, y1, (x2-x1)*not_all_the_way, (y2-y1)*not_all_the_way, 
            width=0.20,
            color=color,
            zorder=0
        )

def plot_contraction_nodes(positions:List[Tuple[int,...]], con_order:List[int])->None:
    area = 20
    for ind in con_order:
        x, y = positions[ind]        
        plt.scatter(x, y, s=400, c="cyan", zorder=4, alpha=0.5)


@visuals.matplotlib_wrapper()
def plot_network(
	nodes : List[TensorNode],
	edges : Dict[str, Tuple[int, int]],
    detailed : bool = True
)-> None:
    
    ## Constants:
    edge_color ,alpha ,linewidth = 'dimgray', 0.5, 3
    angle_color, angle_linewidth, angle_dis = 'green', 2, 0.5
    
    ## Complete data:
    edges_list = [node.edges for node in nodes]
    pos_list = [node.pos for node in nodes]
    angles_list = [node.angles for node in nodes]

    ## Define helper functions:
    average = lambda lst: sum(lst) / len(lst)

    def node_style(node:TensorNode):
        # Marker:
        if node.functionality is NodeFunctionality.CenterCore:
            marker = f"${node.unit_cell_flavor}$"
            size1 = 120
            size2 = 180
        elif node.functionality is NodeFunctionality.AroundCore:
            marker = "H"
            size1 = 60
            size2 = 80
        else:
            marker = "o"
            size1 = 15
            size2 = 30
        name = ""
        # Color:
        match node.unit_cell_flavor:
            case UnitCellFlavor.A:
                color = 'red'
            case UnitCellFlavor.B:
                color = 'green'
            case UnitCellFlavor.C:
                color = 'blue'
            case UnitCellFlavor.NoneLattice:
                name = f"{node.name}"
                if node.functionality is NodeFunctionality.Message:
                    color = "orange"
                else:
                    color = "yellow"
        return color, marker, size1, size2, name

    def _tensor_indices(edge_name:str, assert_connections:bool=False) -> List[int]:
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

    def _edge_positions(edge_name:str) -> Tuple[List[int], List[int]]:
        tensors_indices = _tensor_indices(edge_name)
        x_vec = [pos_list[tensor_ind][0] for tensor_ind in tensors_indices]
        y_vec = [pos_list[tensor_ind][1] for tensor_ind in tensors_indices]
        return x_vec, y_vec

    def _edge_dim(edge_name:str)->int:
        tensors_indices = _tensor_indices(edge_name)
        if len(tensors_indices)==1:
            node = nodes[tensors_indices[0]]
            edge_ind = node.edges.index(edge_name)
            return node.dims[edge_ind]

        node1 = nodes[tensors_indices[0]]
        node2 = nodes[tensors_indices[1]]
        edge_ind1 = node1.edges.index(edge_name)
        edge_ind2 = node2.edges.index(edge_name)
        dim1 = node1.dims[edge_ind1]
        dim2 = node2.dims[edge_ind2]
        assert dim1==dim2, f"Dimensions don't agree on edge '{edge_name}'"
        dir1 = node1.directions[edge_ind1]
        dir2 = node2.directions[edge_ind2]
        if NodeFunctionality.Message in [node1.functionality, node2.functionality]:
            pass
        else:
            assert check.is_equal(dir1, dir2.opposite()), f"Legs of connection in a lattice must be of opposite directions"
        return dim1

        
    # Plot nodes:
    for i, pos in enumerate(pos_list):
        node = nodes[i]
        assert node.pos == pos
        x, y = pos
        color, marker, size1, size2, name = node_style(node)
        plt.scatter(x, y, c="black", s=size2, marker=marker, zorder=3)
        plt.scatter(x, y, c=color, s=size1, marker=marker, zorder=4)
        if detailed:
            text = f" [{node.index}]" + f" {name}" 
            plt.text(x, y, text)
            assert i==node.index, "Non consistent indexing"

    ## Collect basic data:
    network_bounds = _derive_boundary(pos_list)
    smallest_distance = _derive_smallest_distance(pos_list)

    # Plot edges:
    for edge_name, tensors_indices in edges.items():
    
        ## Gather info:
        x_vec, y_vec = _edge_positions(edge_name)
        edge_dim = _edge_dim(edge_name)      
        
        ## Define plot function:      
        def plot_and_text(x_vec, y_vec, on_boundary):
            plt.plot(x_vec, y_vec, color=edge_color, alpha=alpha, linewidth=linewidth, zorder=1 )
            if detailed:
                if on_boundary is None:
                    x = average(x_vec)
                    y = average(y_vec)
                else: 
                    x = x_vec[1]
                    y = y_vec[1]
                plt.text(x, y, f"{edge_name!r}\n", fontdict={'color':'darkorchid', 'size':10 } )
                plt.text(x, y, f"\n{edge_dim}", fontdict={'color':'crimson', 'size':10 } )
            
        ## Plot this edge:
        on_boundary = _check_on_boundaries(tensors_indices, network_bounds=network_bounds, pos_list=pos_list, edge_name=edge_name, nodes=nodes, delta=smallest_distance)
        if on_boundary is not None:
            x_vec, y_vec = _outer_directions_vectors(x_vec, y_vec, on_boundary, smallest_distance)            

        plot_and_text(x_vec, y_vec, on_boundary)

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

