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

# Directions:
from lattices.directions import BlockSide, LatticeDirection, Direction
from lattices.directions import check

from tensor_networks.node import TensorNode
from lattices.edges import edges_dict_from_edges_list, same_dicts
from tensor_networks.unit_cell import UnitCell

from lattices.kagome import KagomeLattice, Node, UpperTriangle
import lattices.triangle as triangle_lattice
from _types import EdgeIndicatorType, PosScalarType, EdgesDictType

from enums import NodeFunctionality
from utils import assertions, lists, tuples, numerics, indices, decorators, dicts

import numpy as np
from typing import NamedTuple
from copy import deepcopy

# For efficient functions:
import itertools
import functools

from _error_types import TensorNetworkError, LatticeError, DirectionError, NetworkConnectionError
from containers import MessageDictType, Message, TNSizesAndDimensions

# Other supporting algo:
from algo.mps import initial_message

# For OOP:
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, TypeAlias




class _ReplaceHere(): ... 


class TensorDims(NamedTuple):
    virtual  : int
    physical : int


def _is_open_edge(edge:tuple[int, int])->bool:
    if edge[0]==edge[1]:    return True
    else:                   return False

functools.cache
def _derive_message_indices(N:int, direction:BlockSide)->list[int]:
    # Get info:
    num_lattice_nodes = triangle_lattice.total_vertices(N)*3
    message_size = 2*N - 1
    direction_index_in_order = indices.index_of_first_appearance( BlockSide.all_in_counter_clockwise_order(), direction )
    # Compute:
    res = np.arange(message_size) + num_lattice_nodes + message_size*direction_index_in_order
    return res.tolist()


class _AbstractTensorNetwork(ABC):
    # ================================================= #
    #|   Implementation Specific Methods\Properties    |#
    # ================================================= #
    nodes : list[TensorNode] = None

    @abstractmethod
    def copy(self)->"_AbstractTensorNetwork": ...

    # ================================================= #
    #|              Basic Derived Properties           |#
    # ================================================= #
    @property
    def size(self)->list[np.ndarray]:
        return len(self.nodes)
    
    @property
    def tensors(self)->list[np.ndarray]:
        return [v.tensor for v in self.nodes]
    
    @property
    def kets(self)->list[bool]:
        return [v.is_ket for v in self.nodes]
    
    @property
    def edges_list(self)->list[list[EdgeIndicatorType]]:
        return [v.edges for v in self.nodes]
    
    @property
    def edges_dict(self)->EdgesDictType:
        return edges_dict_from_edges_list(self.edges_list)

    @property
    def positions(self)->list[tuple[PosScalarType,...]]:
        return [v.pos for v in self.nodes]

    @property
    def angles(self)->list[list[float]]:
        return [v.angles for v in self.nodes]

    # ================================================= #
    #|            Instance Copy\Creation               |#
    # ================================================= #
    def sub_tn(self, indices:list[int])->"_AbstractTensorNetwork":
        return _derive_sub_tn(self, indices)

    # ================================================= #
    #|                Basic Structure                  |#
    # ================================================= #
    def validate(self)->None:
        _validate_tn(self)

    def to_dict(self)->dict[str, Any]:
        d = dict()
        d["tensors"] = self.tensors
        d["is_ket"] = self.kets
        d["edgs"] = self.edges_list
        d["edgs_dict"] = self.edges_dict
        d["angles"] = self.angles
        d["positions"] = self.positions
        return d

    # ================================================= #
    #|                    Visuals                      |#
    # ================================================= #
    def plot(self, detailed:bool=True)->None:
        from tensor_networks.visualizations import plot_network
        nodes = self.nodes
        edges = self.edges_dict
        plot_network(nodes=nodes, edges=edges, detailed=detailed)

    # ================================================= #
    #|                   Get Nodes                     |#
    # ================================================= #
    def get_node_in_pos(self, pos:tuple[PosScalarType, ...]) -> TensorNode:
        tensors_in_position = [node for node in self.nodes if node.pos == pos]
        if len(tensors_in_position)==0:
            raise TensorNetworkError(f"Couldn't find any tensor in position {pos}")
        elif len(tensors_in_position)>1:
            raise TensorNetworkError(f"Found more than 1 tensor in position {pos}")
        return tensors_in_position[0]
    
    def get_nodes_by_functionality(self, functions:list[NodeFunctionality])->list[TensorNode]:
        return [n for n in self.nodes if n.functionality in functions]
    
    def get_nodes_on_boundary(self, side:BlockSide)->list[TensorNode]: 
        return [t for t in self.nodes if side in t.boundaries ] 

    def get_core_nodes(self)->list[TensorNode]:
        return self.get_nodes_by_functionality([NodeFunctionality.Core, NodeFunctionality.CenterUnitCell])
    
    def get_center_unit_cell_nodes(self)->list[TensorNode]:
        return self.get_nodes_by_functionality([NodeFunctionality.CenterUnitCell])
    
    # ================================================= #
    #|                   Neighbors                     |#
    # ================================================= #
    
    def nodes_connected_to_edge(self, edge:str)->list[TensorNode]:
        indices = self.edges_dict[edge]
        indices = list(set(indices))  # If the tensor apears twice, get only 1 copy of its index.
        return [ self.nodes[node_ind] for node_ind in indices ]

    def _find_neighbor_by_edge(self, node:TensorNode, edge:EdgeIndicatorType)->TensorNode:
        nodes = self.nodes_connected_to_edge(edge)
        assert len(nodes)==2, f"Only tensor '{node.index}' is connected to edge '{edge}' in direction '{str(dir)}'  "
        if nodes[0] is node:
            return nodes[1]
        elif nodes[1] is node:
            return nodes[0]
        else:
            raise TensorNetworkError(f"Couldn't find a neighbor due to a bug with the tensors connected to the same edge '{edge}'")
        
    def find_neighbor(self, node:TensorNode, dir_or_edge:Direction|EdgeIndicatorType|None)->TensorNode:
        # Get edge of this node:
        if isinstance(dir_or_edge, Direction):
            edge = node.edge_in_dir(dir_or_edge)
        elif isinstance(dir_or_edge, EdgeIndicatorType):
            edge = dir_or_edge
        elif dir_or_edge is None:
            edge = lists.random_item( node.edges )
        else:
            raise TypeError(f"Not a valid type of variable `dir`. Given type '{type(dir_or_edge)}'")
        
        # Find neihbor connected with this edge
        return self._find_neighbor_by_edge(node, edge=edge)
    
    def are_neigbors(self, n1:TensorNode, n2:TensorNode) -> bool:
        assert n1 is not n2, f"Nodes must be different"
        for edge in n1.edges:
            if edge in n2.edges:
                return True
        return False    


class KagomeTensorNetwork(_AbstractTensorNetwork):

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
        self.messages : MessageDictType = {}
        self.unit_cell : UnitCell = unit_cell
        self.dimensions : TNSizesAndDimensions = TNSizesAndDimensions(
            virtual_dim=D,
            physical_dim=d,
            core_size=2,
            big_lattice_size=lattice.N
        )

    # ================================================= #
    #|       Mandatory Implementations of ABC          |#
    # ================================================= #
    @functools.cached_property
    def nodes(self)->list[TensorNode]:
        return _derive_nodes_kagome_tn(self)

    def copy(self, with_messages:bool=True)->"KagomeTensorNetwork":
        new = KagomeTensorNetwork(
            lattice=self.lattice,
            unit_cell=self.unit_cell.copy(),
            d=self.dimensions.physical_dim,
            D=self.dimensions.virtual_dim,
        )
        if with_messages:
            new.messages = self.messages

        if DEBUG_MODE: 
            new.validate()
        return new

    # ================================================= #
    #|                 Cache Control                   |#
    # ================================================= #
    def clear_cache(self)->None:
        # nodes
        try:    
            del self.nodes
        except AttributeError:  
            pass

    # ================================================= #
    #|                    messages                     |#
    # ================================================= #
    def connect_messages(self, messages:MessageDictType) -> None:   
        # Fuse:
        for block_side, message in messages.items():
            self.messages[block_side] = message
        # Clear cache so that nodes are derived again
        self.clear_cache()

    # add sub method:
    def connect_random_messages(self) -> None:
        D = self.dimensions.virtual_dim
        message_length = self.num_message_connections
        messages = { 
            edge_side : Message(
                mps=initial_message(D=D, N=message_length), 
                order_direction=edge_side.orthogonal_counterclockwise_lattice_direction() 
            ) 
            for edge_side in BlockSide.all_in_counter_clockwise_order()  \
        }
        self.connect_messages(messages)

    def message_indices(self, direction:BlockSide)->list[int]:
        return _derive_message_indices(self.lattice.N, direction)

    # ================================================= #
    #|              Geometry Functions                 |#
    # ================================================= #
    @property
    def num_message_connections(self)->int:
        return self.lattice.num_message_connections
    
    def num_boundary_nodes(self, boundary:BlockSide)->int:
        return self.lattice.num_boundary_nodes(boundary)
    
    def positions_min_max(self)->tuple[int, ...]:
        min_x, max_x = lists.min_max([node.pos[0] for node in self.nodes])
        min_y, max_y = lists.min_max([node.pos[1] for node in self.nodes])
        return min_x, max_x, min_y, max_y
    
    def get_center_triangle(self)->UpperTriangle:
        triangle = self.lattice.get_center_triangle()
        if DEBUG_MODE:
            unit_cell_indices = [node.index for node in self.get_center_unit_cell_nodes()]
            assert len(unit_cell_indices)==3
            for node in triangle.all_nodes():
                assert node.index in unit_cell_indices
        return triangle



class ArbitraryTensorNetwork(_AbstractTensorNetwork):
    def __init__(self, nodes:list[TensorNode]) -> None:
        self.nodes : list[TensorNode] = nodes
        
    # ================================================= #
    #|       Mandatory Implementations of ABC          |#
    # ================================================= #
    def copy(self)->"ArbitraryTensorNetwork":
        return ArbitraryTensorNetwork(nodes=self.nodes)


    # ================================================= #
    #|                Instance method                  |#
    # ================================================= #
    def _pop_node(self, ind:int)->TensorNode:
        #TODO should be here?

        # Update indices of all nodes places above this index:
        for i in range(ind+1, len(self.nodes)):
            self.nodes[i].index -= 1
        # Update edges dict:
        for edge, (i1, i2) in self.edges_dict.items():
            if i1 is not _ReplaceHere and i1>ind:  i1-=1
            if i2 is not _ReplaceHere and i2>ind:  i2-=1
            self.edges_dict[edge] = (i1, i2)  # type: ignore
        # pop from list:
        return self.nodes.pop(ind)

    def contract_nodes(self, n1:TensorNode|int, n2:TensorNode|int)->None:
        #TODO should be here?

        ## Also accept indices as inputs:
        if isinstance(n1, int) and isinstance(n2, int):
            n1 = self.nodes[n1]
            n2 = self.nodes[n2]
        assert isinstance(n1, TensorNode) and isinstance(n2, TensorNode)
        ## Collect basic data:
        i1, i2 = n1.index, n2.index
        if DEBUG_MODE:
            assert self.nodes[i1] is n1
            assert self.nodes[i2] is n2
        ## Get contracted tensor data:
        contraction_edge, edges, directions, functionality, pos, name, on_boundary = _derive_node_data_from_contracted_nodes_and_fix_neighbors(self, n1, n2)
        ## Remove edge from dict and mark edges that point to these nodes:
        ia, ib = self.edges_dict.pop(contraction_edge)
        if DEBUG_MODE:
            assert ia != ib
            assert ( ia == i1 and ib == i2 ) or ( ia == i2 and ib == i1 )
        for edge, (ia, ib) in self.edges_dict.items():
            if ia in [i1, i2]: ia=_ReplaceHere
            if ib in [i1, i2]: ib=_ReplaceHere
            self.edges_dict[edge] = (ia, ib)  # type: ignore  # a temporary place-holder
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
        new_node = TensorNode(
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
        for edge, (ia, ib) in self.edges_dict.items():
            if ia is _ReplaceHere: ia = new_node.index
            if ib is _ReplaceHere: ib = new_node.index
            self.edges_dict[edge] = (ia, ib)
        ## Check:
        if DEBUG_MODE: self.validate()

    def replace(self, node:TensorNode):
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

    def add(self, node:TensorNode):
        # Add node:
        assert node.index == len(self.nodes)
        self.nodes.append(node)
        new_node_index = node.index

        # Add edges:
        for new_edge in node.edges:   # type: ignore            
            if new_edge in self.edges_dict:
                connected_nodes = self.edges_dict[new_edge]
                assert connected_nodes[0]==connected_nodes[1], f"The edge must be an open edge, otherwise a new connection is impossible."
                self.edges_dict[new_edge] = (connected_nodes[0], new_node_index)
            else:
                self.edges_dict[new_edge] = (new_node_index, new_node_index)
    


def _derive_message_node_position(nodes_on_boundary:list[Node], edge:EdgeIndicatorType, boundary_delta:tuple[float,...]):
    node = next(node for node in nodes_on_boundary if edge in node.edges)
    direction = node.directions[ node.edges.index(edge) ]
    delta = tuples.add(boundary_delta, direction.unit_vector)
    # delta = (numerics.furthest_absolute_integer(delta[0]), numerics.furthest_absolute_integer(delta[1]))
    return tuples.add(node.pos, delta)     


def _message_nodes(
    tn : KagomeTensorNetwork,
    message : Message,
    boundary_side : BlockSide,
    first_node_index : int
) -> list[TensorNode]:

    # Unpack Inputs:
    mps = message.mps
    mps_order_dir = message.order_direction

    ## Check data:
    if DEBUG_MODE:
        assert check.is_orthogonal(boundary_side, mps_order_dir), f"MPS must be oethogonal in its own ordering direction to the lattice"

    # Where is the message located compared to the lattice:
    boundary_delta = boundary_side.unit_vector

    ## Derive the common directions defining this incoming message:
    msg_dir_to_lattice = boundary_side.opposite()
    
    # Get all tensors on edge:
    nodes_on_boundary = tn.lattice.sorted_boundary_nodes(boundary_side)
    edges_on_boundary = tn.lattice.sorted_boundary_edges(boundary_side)
    assert len(mps.A) == len(edges_on_boundary) == mps.N

    ## derive shared properties of all message tensors:
    message_edge_names = [ "M-"+str(boundary_side)+f"-{i}" for i in range(mps.N-1) ]
    edges_per_message_tensor = [(message_edge_names[0],)] + list( itertools.pairwise(message_edge_names) ) + [(message_edge_names[-1],)]
    message_positions = [_derive_message_node_position(nodes_on_boundary, edge, boundary_delta) for edge in edges_on_boundary]

    ## Prepare output::
    index = 0
    res = []    
    
    ## Connect each message-tensor and its corresponding lattice node:
    for (is_first, is_last, mps_tensor), edge_to_lattice, m_edges, m_pos in \
        zip( lists.iterate_with_edge_indicators(mps.A), edges_on_boundary, edges_per_message_tensor, message_positions, strict=True ):

        ## Include mantissa+exponent info:
        # mps_tensor *= mantissa*10**exp

        ## Change tensor shape according to it's position on the lattice
        shape = mps_tensor.shape
        assert len(shape)==3, f"All message tensors are 3 dimensional when they are created. This tensor has {len(shape)} dimensions!"
        if is_first:
            new_tensor = mps_tensor.reshape([shape[1], shape[2]])
            directions = [msg_dir_to_lattice, mps_order_dir]
            edges = [ edge_to_lattice, m_edges[0] ]
        elif is_last:
            new_tensor = mps_tensor.reshape([shape[0], shape[1]])
            directions = [mps_order_dir.opposite(), msg_dir_to_lattice]
            edges = [ m_edges[0], edge_to_lattice ]
        else:
            new_tensor = 1.0*mps_tensor
            directions = [mps_order_dir.opposite(), msg_dir_to_lattice,  mps_order_dir]
            assert len(m_edges)==2
            edges = [ m_edges[0], edge_to_lattice , m_edges[1] ]


        ## Add new node to Tensor-Network:
        new_node = TensorNode(
            is_ket=False,
            tensor=new_tensor,
            edges=edges,
            directions=directions, # type: ignore
            pos=m_pos,  # type: ignore
            index=first_node_index+index,
            name=f"m{boundary_side}{index}",
            functionality=NodeFunctionality.Message
        )

        res.append(new_node)
        index += 1

    return res


def  _derive_nodes_kagome_tn(tn:KagomeTensorNetwork)->list[TensorNode]:
    # init lists and iterators:
    unit_cell_tensors = itertools.cycle(tn.unit_cell.all())
    
    # Prepare info:
    center_triangle_index = triangle_lattice.center_vertex_index(tn.lattice.N)
    center_nodes : list[TensorNode] = []
    
    # Create all nodes:
    nodes : list[TensorNode] = []
    for (lattice_node, triangle), (tensor, cell_type) in zip(tn.lattice.nodes_and_triangles(), unit_cell_tensors):            
        # Which type of node:
        functionality = NodeFunctionality.CenterUnitCell if triangle.index == center_triangle_index else NodeFunctionality.Padding

        # Add:
        network_node = TensorNode(
            index=lattice_node.index,
            tensor=tensor,
            is_ket=True,
            pos=lattice_node.pos,
            edges=lattice_node.edges,
            directions=lattice_node.directions,
            functionality=functionality,
            core_cell_flavor=cell_type,
            boundaries=lattice_node.boundaries,
            name=f"{cell_type}"
        )
        nodes.append(network_node)

        if functionality is NodeFunctionality.CenterUnitCell:
            center_nodes.append(network_node)

    # Nearest-neighbors of the center triangles are part of the core:
    for node in center_nodes:
        for edge in node.edges:
            neighbor_index = tn.lattice.get_neighbor(node, edge).index
            neihgbor = nodes[neighbor_index]
            if neihgbor in center_nodes:
                continue
            neihgbor.functionality = NodeFunctionality.Core

    ## Add messages:
    if len(tn.messages)>0:
        crnt_node_index = tn.lattice.size
        for side in BlockSide.all_in_counter_clockwise_order():
            if side in tn.messages:
                message = tn.messages[side]
                message_nodes = _message_nodes(tn, message, side, crnt_node_index)
                nodes.extend(message_nodes)
                crnt_node_index += len(message_nodes)

    return nodes


def _derive_sub_tn(tn:_AbstractTensorNetwork, indices:list[int])->ArbitraryTensorNetwork:
    
    ## the nodes in the sub-system must be indexed again:
    nodes : list[TensorNode] = []
    for new_index, old_index in enumerate(indices):
        # Basic info:
        old_node = tn.nodes[old_index]
        # Create new node:
        new_node = TensorNode(
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
                assert isinstance(side, LatticeDirection)
                node.boundaries.append(side)
    
    ## Copy additional straight-forward info:
    t_dims = tn.tensor_dims
    new = ArbitraryTensorNetwork(nodes=nodes, edges=edges, _copied=True, tensor_dims=t_dims)
    min_x, max_x, min_y, max_y = new.positions_min_max()
    new.original_lattice_dims = ( int(max_y-min_y+1), int(max_x-min_x+1) )
    return new

def _derive_edges_kagome_tn(self:KagomeTensorNetwork)->dict[str, tuple[int, int]]:
    ## Lattice edges:
    edges = self.lattice.edges

    if 'M-D-0' in edges:  # Already added messages
        return edges

    ## Message edges:
    if len(self.messages)>0:
        crnt_index = self.lattice.size
        for side, message in self.messages.items():
            side_edges = self.lattice.sorted_boundary_edges(side)
            for edge in side_edges:
                connected_nodes = edges[edge]
                assert connected_nodes[0]==connected_nodes[1], f"Outer edge should have been with only 1 node"
                edges[edge] = (connected_nodes[0], crnt_index) 
                crnt_index+=1

        num_message_inner_edges = self.lattice.num_message_connections - 1
        crnt_index = self.lattice.size
        for side in self.messages.keys():
            for i in range(num_message_inner_edges):
                edge = f"M-{side}-{i}"
                edges[edge] = (crnt_index, crnt_index+1)
                crnt_index += 1
            crnt_index += 1

    return edges


def _validate_tn(tn:_AbstractTensorNetwork):
    # Generic error message:
    _error_message = f"Failed validation of tensor-network."
    # Provoke cached properties:
    all_nodes = tn.nodes
    all_edges = tn.edges_dict
    # Check each node individually:
    for node in all_nodes:
        # Self validation:
        node.validate()
        # Validation with neighbors in lattice:
        for dir in node.directions:
            edge_name = node.edge_in_dir(dir)
            if _is_open_edge(all_edges[edge_name]):
                continue
            neighbor = tn.find_neighbor(node, dir)
            # Is neighbors pointing back to us?
            _neigbor_error_message = _error_message+f"\nnode '{node.name}' is not the neighbor of '{neighbor.name}' on edge '{edge_name}'"
            opposite_dir = dir.opposite()
            neighnors_dir = neighbor.directions[neighbor.edges.index(edge_name)]
            assert check.is_opposite(neighnors_dir, dir)
            assert tn._find_neighbor_by_edge(neighbor, edge_name) is node, _neigbor_error_message
            assert tn.find_neighbor(neighbor, neighnors_dir) is node, _neigbor_error_message
            # Messages struggle with this check:
            if not isinstance(dir, LatticeDirection) and isinstance(neighnors_dir, BlockSide):
                assert neighbor.edge_in_dir(opposite_dir) == edge_name

    # Check edges:
    for edge_name, node_indices in all_edges.items():
        if node_indices[0] == node_indices[1]:  # happens in outer legs (going outside of the lattice without any connection)
            continue
        nodes = [all_nodes[ind] for ind in node_indices ]
        try:
            indices_of_this_edge = [node.edges.index(edge_name) for node in nodes]
        except:
            raise KeyError(_error_message)
        assert nodes[0].dims[indices_of_this_edge[0]] == nodes[1].dims[indices_of_this_edge[1]] , _error_message
        dir1 = nodes[0].directions[indices_of_this_edge[0]]
        dir2 = nodes[1].directions[indices_of_this_edge[1]]
        assert check.is_opposite(dir1, dir2)
        for node, node_index in zip(nodes, node_indices, strict=True):
            assert node.index == node_index , _error_message


def get_common_edge_legs(
    n1:TensorNode, n2:TensorNode
)->tuple[
    int,             # leg1 index
    int              # leg2 index
]:
    edge = get_common_edge(n1, n2)
    i1 = n1.edges.index(edge)
    i2 = n2.edges.index(edge)
    return i1, i2


def get_common_edge(n1:TensorNode, n2:TensorNode)->EdgeIndicatorType:
    for e1 in n1.edges:
        for e2 in n2.edges:
            if e1==e2:
                return e1
    raise ValueError(f"Nodes '{n1}' and '{n2}' don't have a common edge.")


def _derive_node_data_from_contracted_nodes_and_fix_neighbors(tn:KagomeTensorNetwork, n1:TensorNode, n2:TensorNode):
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