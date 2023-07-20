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
from tensor_networks.node import Node, NodeFunctionality
from lattices.directions import DirectionsError, Directions

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
    
    
def _outer_directions_vectors(x_vec:List[int], y_vec:List[int], dir:str|Directions, delta:float)->Tuple[List[float], ...]:     
    if isinstance(dir, str) and not isinstance(dir, Directions):
        if dir=='x':
            low, high  = min(x_vec), max(x_vec)
            x_vec1 = [low,  low-delta]
            x_vec2 = [high, high+delta]
            y_vec1 = y_vec
            y_vec2 = y_vec
        elif dir=="y":
            low, high  = min(y_vec), max(y_vec)
            x_vec1 = x_vec
            x_vec2 = x_vec
            y_vec1 = [low,  low-delta]
            y_vec2 = [high, high+delta]    
        else: 
            raise ValueError(f"Not a valid option dir={dir}")
        
        if len(x_vec1)==1: x_vec1.append(x_vec1[0])
        if len(x_vec2)==1: x_vec2.append(x_vec2[0])
        if len(y_vec1)==1: y_vec1.append(y_vec1[0])
        if len(y_vec2)==1: y_vec2.append(y_vec2[0])
    
    elif isinstance(dir, Directions):
        x = float(x_vec[0])
        y = float(y_vec[0])
        x_vec1 : list[float]
        y_vec1 : list[float]
        x_vec2 : list[float] = []
        y_vec2 : list[float] = []
        if dir is Directions.Left:
            x_vec1 = [x, x-delta]
            y_vec1 = [y, y]
        elif dir is Directions.Right:
            x_vec1 = [x, x+delta]
            y_vec1 = [y, y]
        elif dir is Directions.Up:
            x_vec1 = [x, x]
            y_vec1 = [y, y+delta]
        elif dir is Directions.Down:
            x_vec1 = [x, x]
            y_vec1 = [y, y-delta]
        else: 
            raise ValueError(f"Not a valid option dir={dir}")        
    
    else: 
        raise TypeError(f"Not a supported type for dir of type <{type(dir)}>")
    
    return x_vec1, x_vec2, y_vec1, y_vec2

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

def _neighbors_pos(this_pos:tuple[int, ...], dir:Directions, delta:float)->tuple[float, ...]:
    if   dir is Directions.Left:  return (this_pos[0]-delta , this_pos[1]+0.0  )
    elif dir is Directions.Right: return (this_pos[0]+delta , this_pos[1]+0.0  )
    elif dir is Directions.Up:    return (this_pos[0]+0.0   , this_pos[1]+delta)
    elif dir is Directions.Down:  return (this_pos[0]+0.0   , this_pos[1]-delta)
    raise DirectionsError(f"Not a valid option dir={dir}")     



def _check_periodic_boundary_connected_tensors(
    tensor_indices:tuple[int, int], 
    network_bounds:List[Tuple[int, ...]], 
    pos_list:List[Tuple[int, int]],
    edge_name:str,
    nodes:List[Node],
    delta:float
)->str|Directions|None:

    assert isinstance(tensor_indices, tuple)
    if len(tensor_indices)==1:
        assert isinstance(edge_name, str)
        if edge_name[0] in ["D", "U"]:
            return 'y'
        elif edge_name[0] in ["L", "R"]:
            return 'x'
        else:
            raise ValueError("Not supported")
        
    elif len(tensor_indices)==2:
    
        t1 = tensor_indices[0]
        t2 = tensor_indices[1]
        
        ## if an edge of an unconnected node:
        if t1==t2:
            node = nodes[t1]
            # Definitely a boundary edge
            edge_ind = node.edges.index(edge_name)
            edge_dir = node.directions[edge_ind]
            return edge_dir
        
        ## if an inner edge:
        these_nodes = [nodes[ind] for ind in tensor_indices]
        edges_indices = [node.edges.index(edge_name) for node in these_nodes]
        edges_dirs = [node.directions[edge_ind] for node, edge_ind in zip(these_nodes, edges_indices)]
        these_poses = [node.pos for node in these_nodes]
        try:
            neighbors_poses = [_neighbors_pos(this_pos, dir, delta) for this_pos, dir in zip(these_poses, edges_dirs)]
            if tuples.equal(these_poses[0], neighbors_poses[1]) and tuples.equal(these_poses[1], neighbors_poses[0]):
                return None
        except DirectionsError:
            return None
            
        t1_on_bounds = _OnBounds()
        t2_on_bounds = _OnBounds()
        tensors_on_bounds : List[_OnBounds] = [t1_on_bounds, t2_on_bounds]

        positions = [pos_list[t] for t in [t1, t2]]
        for i_tensor, (pos, tensor_ind) in enumerate(zip(positions, [t1, t2])):
            for i_axis, axis in [(0, 'x'), (1, 'y')] :
                pos_projection = pos[i_axis]
                for boundary_pos, min_max in zip(network_bounds[i_axis], ['min', 'max']):
                    if pos_projection == boundary_pos:
                        tensors_on_bounds[i_tensor][axis][min_max] = True

        # Tensors are connected through the periodic boundary if 
        # for one of the dimensions, one tensor is at max while the other
        # is at min
        for axis in ['x', 'y']:
            if _opposite( t1_on_bounds[axis], t2_on_bounds[axis]):
                return axis
            
    else: 
        raise ValueError(f"Not a possible input tensor_indices={tensor_indices}")
    
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
    not_all_the_way = 0.80
    for (from_, to_), color in zip(itertools.pairwise(con_order), visuals.color_gradient(len(positions)) ):
        x1, y1 = positions[from_]
        x2, y2 = positions[to_]
        plt.arrow(
            x1, y1, (x2-x1)*not_all_the_way, (y2-y1)*not_all_the_way, 
            width=0.04,
            color=color
        )


@visuals.matplotlib_wrapper()
def plot_network(
	nodes : List[Node],
	edges : Dict[str, Tuple[int, int]],
    detailed : bool = True
)-> None:
    
    ## Constants:
    edge_color ,alpha ,linewidth = 'gray', 0.5, 3
    angle_color, angle_linewidth, angle_dis = 'green', 2, 0.2
    
    ## Complete data:
    edges_list = [node.edges for node in nodes]
    pos_list = [node.pos for node in nodes]
    angles_list = [node.angles for node in nodes]

    ## Define helper functions:
    average = lambda lst: sum(lst) / len(lst)
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
        assert Directions.is_equal(dir1, Directions.opposite_direction(dir2)), f"Legs of connection in a lattice must be of opposite directions"
        return dim1

        
    # Plot nodes:
    for i, pos in enumerate(pos_list):
        node = nodes[i]
        assert node.pos == pos
        x, y = pos
        if node.functionality is NodeFunctionality.Core:
            node_color = 'blue'
        else:
            node_color = 'red'
        plt.scatter(x, y, c=node_color)
        text = f"{node.name}"
        if detailed:
            text += f" [{node.index}]"
        plt.text(x, y, text)

    ## Collect basic data:
    network_bounds = _derive_boundary(pos_list)
    smallest_distance = _derive_smallest_distance(pos_list)

    # Plot edges:
    for edge_name, tensors_indices in edges.items():
    
        ## Gather info:
        x_vec, y_vec = _edge_positions(edge_name)
        edge_dim = _edge_dim(edge_name)      
        on_boundary = _check_periodic_boundary_connected_tensors(tensors_indices, network_bounds=network_bounds, pos_list=pos_list, edge_name=edge_name, nodes=nodes, delta=smallest_distance)
        
        ## Define plot function:      
        def plot_and_text(x_vec, y_vec):
            plt.plot(x_vec, y_vec, color=edge_color, alpha=alpha, linewidth=linewidth )
            if detailed:
                plt.text(
                    average(x_vec), average(y_vec), f"{edge_name}:\n{edge_dim}",
                    fontdict={'color':'darkorchid', 'size':10 }
                )
            
        ## Plot this edge:
        if on_boundary is not None:
            x_vec1, x_vec2, y_vec1, y_vec2 = _outer_directions_vectors(x_vec, y_vec, on_boundary, smallest_distance)            
            for order, x_vec, y_vec in zip(['low', 'high'], [x_vec1, x_vec2], [y_vec1, y_vec2]):
                if _check_skip_vec(order, edge_name, on_boundary, x_vec, y_vec):
                    continue
                plot_and_text(x_vec, y_vec)
        else:
            plot_and_text(x_vec, y_vec)

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
            plt.text(x2, y2, f"{i_edge}", fontdict={'color':'olivedrab', 'size':10 } )	

