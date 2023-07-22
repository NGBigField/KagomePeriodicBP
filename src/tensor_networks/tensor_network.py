if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )
    from project_paths import add_src, add_base, add_scripts
    add_src()
    add_base()
    add_scripts()


## Get config:
from _config_reader import DEBUG_MODE

from lattices.directions import Direction
from tensor_networks.node import Node
from lattices.edges import edges_dict_from_edges_list
from tensor_networks.unit_cell import UnitCell

from lattices.kagome import KagomeLattice
import lattices.triangle as triangle_lattice
from _types import EdgeIndicator, PosScalarType

from enums import NodeFunctionality
from utils import assertions, lists, tuples

import numpy as np
from typing import NamedTuple, Generator
from copy import deepcopy

import itertools

from _error_types import TensorNetworkError, LatticeError

class _ReplaceHere(): ... 


class TensorDims(NamedTuple):
    virtual  : int
    physical : int


def _is_open_edge(edge:tuple[int, int])->bool:
    if edge[0]==edge[1]:    return True
    else:                   return False


class KagomeTensorNetwork():

    # ================================================= #
    #|                Basic Attributes                 |#
    # ================================================= #
    def __init__(
        self, 
        lattice : KagomeLattice,
        unit_cell : UnitCell,
        d : int,
        D : int
    ) -> None:
        # Save data:
        self.lattice : KagomeLattice = lattice
        self.messages : dict[Direction, Node] = {}
        self.unit_cell = unit_cell
        self.d : int = d
        self.D : int = D

    # ================================================= #
    #|                   Add messages                  |#
    # ================================================= #


    # ================================================= #
    #|                Network Structure                |#
    # ================================================= #
    @property
    def nodes(self)->list[Node]:
        
        # init lists and iterators:
        unit_cell_tensors = itertools.cycle(self.unit_cell.all())
        
        # Prepare info:
        center_triangle_index = triangle_lattice.center_vertex_index(self.lattice.N)
        center_nodes : list[Node] = []
        
        # Create all nodes:
        nodes : list[Node] = []
        for (lattice_node, triangle), (tensor, cell_type) in zip(self.lattice.nodes_and_triangles(), unit_cell_tensors):            
            # Which type of node:
            functionality = NodeFunctionality.Core if triangle.index == center_triangle_index else NodeFunctionality.Padding

            # Add:
            network_node = Node(
                index=lattice_node.index,
                tensor=tensor,
                is_ket=True,
                pos=lattice_node.pos,
                edges=lattice_node.edges,
                directions=lattice_node.directions,
                functionality=functionality,
                core_cell_type=cell_type,
                boundaries=lattice_node.boundaries,
                name=f"{cell_type}"
            )
            nodes.append(network_node)

            if functionality is NodeFunctionality.Core:
                center_nodes.append(network_node)

        # Nearest-neighbors of the center triangles are also part of the core:
        for node in center_nodes:
            for edge in node.edges:
                neighbor_index = self.lattice.get_neighbor(node, edge).index
                neihgbor = nodes[neighbor_index]
                neihgbor.functionality = NodeFunctionality.Core
                    
        return nodes


    @property
    def edges(self)->dict[str, tuple[int, int]]:
        edges = self.lattice.edges
        return edges

    # ================================================= #
    #|              Basic Derived Properties           |#
    # ================================================= #
    @property
    def tesnors(self)->list[np.ndarray]:
        return [v.tensor for v in self.nodes]
    
    @property
    def kets(self)->list[bool]:
        return [v.is_ket for v in self.nodes]
    
    @property
    def edges_list(self)->list[list[EdgeIndicator]]:
        return [v.edges for v in self.nodes]

    @property
    def positions(self)->list[tuple[PosScalarType,...]]:
        return [v.pos for v in self.nodes]

    @property
    def angles(self)->list[list[float]]:
        return [v.angles for v in self.nodes]

    @property
    def size(self)->int:
        return len(self.nodes)

    # ================================================= #
    #|               instance methods                  |#
    # ================================================= #

    def to_dict(self)->dict:
        d = dict()
        d["tensors"], d["is_ket"] = self.tesnors
        d["edgs"] = self.edges_list
        d["edgs_dict"] = self.edges
        d["angles"] = self.angles
        d["positions"] = self.positions
        return d

    def nodes_on_boundary(self, edge_side:Direction)->list[Node]: 
        return [t for t in self.nodes if edge_side in t.boundaries ] 

    def nodes_connected_to_edge(self, edge:str)->list[Node]:
        indices = self.edges[edge]
        indices = list(set(indices))  # If the tensor apears twice, get only 1 copy of its index.
        return [ self.nodes[node_ind] for node_ind in indices ]

    def _find_neighbor_by_edge(self, node:Node, edge:EdgeIndicator)->Node:
        nodes = self.nodes_connected_to_edge(edge)
        assert len(nodes)==2, f"Only tensor '{node.index}' is connected to edge '{edge}' in direction '{str(dir)}'  "
        if nodes[0] == node:
            return nodes[1]
        elif nodes[1] == node:
            return nodes[0]
        else:
            raise TensorNetworkError(f"Couldn't find a neighbor due to a bug with the tensors connected to the same edge '{edge}'")
        
    def find_neighbor(self, node:Node, dir:Direction|EdgeIndicator|None)->Node:
        # Get edge of this node:
        if isinstance(dir, Direction):
            edge = node.edge_in_dir(dir)
        elif isinstance(dir, EdgeIndicator):
            edge = dir
        elif dir is None:
            edge = lists.random_item( node.edges )
        else:
            raise TypeError(f"Not a valid type of variable `dir`. Given type '{type(dir)}'")
        
        # Find neihbor connected with this edge
        return self._find_neighbor_by_edge(node, edge=edge)
    
    def are_neigbors(self, n1:Node, n2:Node) -> bool:
        assert n1 is not n2, f"Nodes must be different"
        for edge in n1.edges:
            if edge in n2.edges:
                return True
        return False    

    def _pop_node(self, ind:int)->Node:
        # Update indices of all nodes places above this index:
        for i in range(ind+1, len(self.nodes)):
            self.nodes[i].index -= 1
        # Update edges dict:
        for edge, (i1, i2) in self.edges.items():
            if i1 is not _ReplaceHere and i1>ind:  i1-=1
            if i2 is not _ReplaceHere and i2>ind:  i2-=1
            self.edges[edge] = (i1, i2)  # type: ignore
        # pop from list:
        return self.nodes.pop(ind)

    #TODO Check normalize
    def contract_nodes(self, n1:Node|int, n2:Node|int)->None:
        ## Also accept indices as inputs:
        if isinstance(n1, int) and isinstance(n2, int):
            n1 = self.nodes[n1]
            n2 = self.nodes[n2]
        assert isinstance(n1, Node) and isinstance(n2, Node)
        ## Collect basic data:
        i1, i2 = n1.index, n2.index
        if DEBUG_MODE:
            assert self.nodes[i1] is n1
            assert self.nodes[i2] is n2
        ## Get contracted tensor data:
        contraction_edge, edges, directions, functionality, pos, name, on_boundary = _derive_node_data_from_contracted_nodes_and_fix_neighbors(self, n1, n2)
        ## Remove edge from dict and mark edges that point to these nodes:
        ia, ib = self.edges.pop(contraction_edge)
        if DEBUG_MODE:
            assert ia != ib
            assert ( ia == i1 and ib == i2 ) or ( ia == i2 and ib == i1 )
        for edge, (ia, ib) in self.edges.items():
            if ia in [i1, i2]: ia=_ReplaceHere
            if ib in [i1, i2]: ib=_ReplaceHere
            self.edges[edge] = (ia, ib)  # type: ignore  # a temporary place-holder
        ## Remove nodes from list while updating indices in both self.edges and self.nodes:
        n1_ = self._pop_node(n1.index)    
        n2_ = self._pop_node(n2.index)            
        if DEBUG_MODE:
            assert n1_ is n1
            assert n2_ is n2
        ## Derive index of edge for each node:
        ie1 = n1.edges.index(contraction_edge)
        ie2 = n2.edges.index(contraction_edge)

        ## Compute contracted tensor:
        if n1.is_ket and n2.is_ket:
            tensor = np.tensordot(n1.physical_tensor, n2.physical_tensor, axes=(ie1+1, ie2+1))
            is_ket = True
        elif not n1.is_ket and not n2.is_ket:
            tensor = np.tensordot(n1.fused_tensor, n2.fused_tensor, axes=(ie1, ie2))
            is_ket = False
        else:
            raise TensorNetworkError("Not an implemented option yet.")
        ## Add node:
        new_node = Node(
            tensor=tensor,
            is_ket=is_ket,
            functionality=functionality,
            edges=edges,
            directions=directions,
            pos=pos,
            index=len(self.nodes),  # Two old nodes are already deleted from list, new will be appended to the end.
            name=name,
            boundaries=on_boundary
        )
        self.nodes.append(new_node)
        ## Fix edges in edges-dict that point to either n1 or n2:
        for edge, (ia, ib) in self.edges.items():
            if ia is _ReplaceHere: ia = new_node.index
            if ib is _ReplaceHere: ib = new_node.index
            self.edges[edge] = (ia, ib)
        ## Check:
        if DEBUG_MODE: self.validate()
       

    def plot(self, detailed:bool=True)->None:
        from tensor_networks.visualizations import plot_network
        nodes = self.nodes
        edges = self.edges
        plot_network(nodes=nodes, edges=edges, detailed=detailed)
        

    def copy(self)->"KagomeTensorNetwork":
        cls = type(self)
        nodes = [node.copy() for node in self.nodes]
        edges = deepcopy(self.edges)
        t_dims = self.tensor_dims
        new = cls(nodes=nodes, edges=edges, _copied=True, tensor_dims=t_dims)
        new.original_lattice_dims = self.original_lattice_dims
        if DEBUG_MODE: new.validate()
        return new
    
    def sub_tn(self, indices)->"KagomeTensorNetwork":
        cls = type(self)
        
        ## the nodes in the sub-system must be indexed again:
        nodes : list[Node] = []
        for new_index, old_index in enumerate(indices):
            # Basic info:
            old_node = self.nodes[old_index]
            # Create new node:
            new_node = Node(
                is_ket=old_node.is_ket,   
                tensor=old_node.tensor.copy(),
                functionality=old_node.functionality,
                edges=deepcopy(old_node.edges),  # These are the old edges names
                pos=deepcopy(old_node.pos),
                directions=deepcopy(old_node.directions),
                index=new_index,
                name=old_node.name, 
                boundaries=deepcopy(old_node.on_boundary)
            )
            # add to list:
            nodes.append(new_node)
        
        ## The edges of the sub-system are the same, but the node indices must change:
        edges_list = [node.edges for node in nodes ]
        edges = edges_dict_from_edges_list(edges_list)
        
        ## Fill in boundary information:
        for node in nodes:
            for edge in node.edges:
                pair = edges[edge]
                if pair[0]==pair[1]:
                    side = node.directions[node.edges.index(edge)]
                    assert isinstance(side, Direction)
                    node.boundaries.append(side)
        
        ## Copy additional straight-forward info:
        t_dims = self.tensor_dims
        new = cls(nodes=nodes, edges=edges, _copied=True, tensor_dims=t_dims)
        min_x, max_x, min_y, max_y = new.boundaries()
        new.original_lattice_dims = ( int(max_y-min_y+1), int(max_x-min_x+1) )
        return new
        
    def replace(self, node:Node):
        # validate:
        ind = node.index
        if DEBUG_MODE:
            new_node = node  # for easier reading
            old_node = self.nodes[ind]
            _error_msg = f"Trying to replace node {old_node.name!r} with node {node.name!r}, but nodes don't match!"
            old_shape = old_node.fused_tensor.shape
            new_shape = new_node.fused_tensor.shape
            assert len(old_shape)==len(new_shape), _error_msg
            for i, (old_edge, old_direction) in enumerate(zip(old_node.edges, old_node.directions)):
                old_dim, new_dim = old_shape[i], new_shape[i]
                new_edge, new_direction = new_node.edges[i], new_node.directions[i]
                assert new_edge==old_edge, _error_msg
                assert new_direction==old_direction, _error_msg
                assert new_dim==old_dim, _error_msg
        # replace:
        self.nodes[ind] = node


    def add(self, node:Node):
        # Add node:
        assert node.index == len(self.nodes)
        self.nodes.append(node)
        new_node_index = node.index

        # Add edges:
        for new_edge in node.edges:   # type: ignore            
            if new_edge in self.edges:
                connected_nodes = self.edges[new_edge]
                assert connected_nodes[0]==connected_nodes[1], f"The edge must be an open edge, otherwise a new connection is impossible."
                self.edges[new_edge] = (connected_nodes[0], new_node_index)
            else:
                self.edges[new_edge] = (new_node_index, new_node_index)

    def get_tensor_in_pos(self, pos:tuple[PosScalarType, ...]) -> Node:
        tensors_in_position = [node for node in self.nodes if node.pos == pos]
        if len(tensors_in_position)==0:
            raise TensorNetworkError(f"Couldn't find any tensor in position {pos}")
        elif len(tensors_in_position)>1:
            raise TensorNetworkError(f"Found more than 1 tensor in position {pos}")
        return tensors_in_position[0]
    
    def validate(self)->None:
        # Generic error message:
        _error_message = f"Failed validation of tensor-network."
        # Check each node individually:
        for node in self.nodes:
            # Self validation:
            node.validate()
            # Validation with neighbors in lattice:
            for dir in node.directions:
                edge_name = node.edge_in_dir(dir)
                if _is_open_edge(self.edges[edge_name]):
                    continue
                neighbor = self.find_neighbor(node, dir)
                # Is neighbors pointing back to us?
                _neigbor_error_message = _error_message+f"\nnode '{node.name}' is not the neighbor of '{neighbor.name}' on edge '{edge_name}'"
                opposite_dir = dir.opposite()
                assert self._find_neighbor_by_edge(neighbor, edge_name) == node, _neigbor_error_message
                assert self.find_neighbor(neighbor, opposite_dir) == node, _neigbor_error_message
                assert neighbor.edge_in_dir(opposite_dir) == edge_name
        # Check edges:
        for edge_name, node_indices in self.edges.items():
            if node_indices[0] == node_indices[1]:  # happens in outer legs (going outside of the lattice without any connection)
                continue
            nodes = [self.nodes[ind] for ind in node_indices ]
            try:
                indices_of_this_edge = [node.edges.index(edge_name) for node in nodes]
            except:
                raise KeyError(_error_message)
            assert nodes[0].dims[indices_of_this_edge[0]] == nodes[1].dims[indices_of_this_edge[1]] , _error_message
            dir1 = nodes[0].directions[indices_of_this_edge[0]]
            dir2 = nodes[1].directions[indices_of_this_edge[1]].opposite()
            assert dir1 == dir2, _error_message
            for node, node_index in zip(nodes, node_indices, strict=True):
                assert node.index == node_index , _error_message
        # Check lattice dimensions:
        min_x, max_x, min_y, max_y = self.boundaries()
        #TODO Size check
        
    def boundaries(self)->tuple[int, ...]:
        min_x, max_x = lists.min_max([node.pos[0] for node in self.nodes])
        min_y, max_y = lists.min_max([node.pos[1] for node in self.nodes])
        return min_x, max_x, min_y, max_y
    
    def get_corner_nodes(self)->list[Node]:
        min_x, max_x, min_y, max_y = self.boundaries()
        corner_nodes = []
        for x, y in itertools.product([min_x, max_x], [min_y, max_y]):
            try:
                node = self.get_tensor_in_pos((x,y))
            except TensorNetworkError:
                continue
            else:
                corner_nodes.append(node)
        return corner_nodes
    
    def get_core_nodes(self)->list[Node]:
        return [n for n in self.nodes if n.functionality is NodeFunctionality.Core]

    def normalize_tensors(self)->None:
        for node in self.nodes:
            node.normalize()


    
    


def get_common_edge_legs(
    n1:Node, n2:Node
)->tuple[
    int,             # leg1 index
    int              # leg2 index
]:
    edge = get_common_edge(n1, n2)
    i1 = n1.edges.index(edge)
    i2 = n2.edges.index(edge)
    return i1, i2


def get_common_edge(n1:Node, n2:Node)->EdgeIndicator:
    for e1 in n1.edges:
        for e2 in n2.edges:
            if e1==e2:
                return e1
    raise ValueError(f"Nodes '{n1}' and '{n2}' don't have a common edge.")


def _derive_node_data_from_contracted_nodes_and_fix_neighbors(tn:KagomeTensorNetwork, n1:Node, n2:Node):
    contraction_edge = get_common_edge(n1, n2)
    if n1.functionality is n2.functionality:
        functionality = n1.functionality
    else: 
        functionality = NodeFunctionality.Undefined
    pos = n2.pos # pos = tuples.mean_itemwise(n1.pos, n2.pos)  # A more sensefull option, but unneeded in our case
    edges = []
    directions = []      #TODO: Directions are sometimes wrong in this case!
    for n in [n1, n2]:
        for e in n.edges:
            if e!=contraction_edge:
                neigbor = tn._find_neighbor_by_edge(n, e)
                angle = tuples.angle(pos, neigbor.pos) 
                try:                        
                    dir = Direction.from_angle(angle)
                except ValueError:
                    dir = angle
                    # Also fix neighbours direction:
                    neighbor_dir = Directions.opposite_direction(dir)                    
                    edge_ind = neigbor.edges.index(e)
                    tn.nodes[neigbor.index].directions[edge_ind] = neighbor_dir
                edges.append(e)        
                directions.append(dir)        
    name = n1.name+"+"+n2.name
    on_boundary = list(set(n1.boundaries+n2.boundaries))

    return contraction_edge, edges, directions, functionality, pos, name, on_boundary



if __name__ == "__main__":
    from scripts.build_tn import main_test
    main_test()