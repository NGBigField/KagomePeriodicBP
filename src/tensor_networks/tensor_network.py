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
from lattices.directions import check, create

# Other lattice structure:
from tensor_networks.node import TensorNode
from lattices.edges import edges_dict_from_edges_list, same_dicts
from tensor_networks.unit_cell import UnitCell
from lattices.kagome import KagomeLattice, Node, UpperTriangle
import lattices.triangle as triangle_lattice

from _types import EdgeIndicatorType, PosScalarType, EdgesDictType

from enums import NodeFunctionality, UpdateMode
from utils import assertions, lists, tuples, numerics, indices, strings

import numpy as np
from typing import NamedTuple
from copy import deepcopy

# For efficient functions:
import itertools
import functools
import operator

from _error_types import TensorNetworkError, LatticeError, DirectionError, NetworkConnectionError
from containers import MessageDictType, Message, TNSizesAndDimensions, MPSOrientation

# Other supporting algo:
from algo.mps import initial_message

# For OOP:
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Self



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


class BaseTensorNetwork(ABC):
    # ================================================= #
    #|   Implementation Specific Methods\Properties    |#
    # ================================================= #
    nodes : list[TensorNode] = None

    @abstractmethod
    def copy(self)->Self: ...

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
    def edges_names(self)->set[EdgeIndicatorType]:
        return lists.deep_unique(self.edges_list)

    @property
    def positions(self)->list[tuple[PosScalarType,...]]:
        return [v.pos for v in self.nodes]

    @property
    def angles(self)->list[list[float]]:
        return [v.angles for v in self.nodes]

    # ================================================= #
    #|            Instance Copy\Creation               |#
    # ================================================= #
    def sub_tn(self, indices:list[int])->"BaseTensorNetwork":
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
    
    def get_nodes_by_functionality(self, functions:list[NodeFunctionality]|NodeFunctionality)->list[TensorNode]:
        if not isinstance(functions, list):
            functions = [functions]
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
        
    def all_neighbors(self, node:TensorNode)->list[TensorNode]:
        res = []
        for edge in node.edges:
            try:
                res.append(self.find_neighbor(node, edge))
            except TensorNetworkError:
                pass
        return res

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
    
    def are_neighbors(self, n1:TensorNode, n2:TensorNode) -> bool:
        assert n1 is not n2, f"Nodes must be different"
        for edge in n1.edges:
            if edge in n2.edges:
                return True
        return False    

    # ================================================= #
    #|                     Edges                       |#
    # ================================================= #    
    def unique_edge_name(self, suggestion:str)->str:
        crnt_suggestion = suggestion
        existing_edges = self.edges_names
        while suggestion in existing_edges:
            crnt_suggestion += strings.random(1)
        return crnt_suggestion


class KagomeTN(BaseTensorNetwork):

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

    def copy(self, with_messages:bool=True)->Self:
        new = KagomeTN(
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
                orientation=MPSOrientation.standard(edge_side.opposite())
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



class ArbitraryTN(BaseTensorNetwork):
    def __init__(self, nodes:list[TensorNode], copy=True) -> None:
        if copy:
            nodes = _copy_nodes_and_fix_indices(nodes)
        self.nodes = nodes
        
    # ================================================= #
    #|       Mandatory Implementations of ABC          |#
    # ================================================= #
    def copy(self)->Self:
        return ArbitraryTN(nodes=self.nodes)


    # ================================================= #
    #|                Instance method                  |#
    # ================================================= #
    def contract_nodes(self, n1:TensorNode|int, n2:TensorNode|int)->TensorNode:
        return _contract_nodes(self, n1, n2)

    def replace(self, node:TensorNode):
        #TODO Check if used
        print("replace is used")

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

    def add_node(self, node:TensorNode):
        # Add node:
        assert node.index == len(self.nodes)
        self.nodes.append(node)
    
    def pop_node(self, ind:int)->TensorNode:
        # Update indices of all nodes places above this index:
        for i in range(ind+1, len(self.nodes)):
            self.nodes[i].index -= 1
        # pop from list:
        return self.nodes.pop(ind)
    
    def qr_decomp(self, node:TensorNode, edges1:list[EdgeIndicatorType], edges2:list[EdgeIndicatorType])->tuple[TensorNode, TensorNode]:
        """Perform QR decomposition on a tensor, while preserving the relation to other tensors in the network via the inputs edges1/2

        Args:
            node (TensorNode): the node
            edges1 (list[EdgeIndicatorType]): the edges that will be connected to Q
            edges2 (list[EdgeIndicatorType]): the edges that will be connected to R

        Returns:
            tuple[TensorNode, TensorNode]: the Q and R tensors
        """
        return _qr_decomposition(self, node, edges1, edges2)

class _FrozenSpecificNetwork(BaseTensorNetwork):
    def __init__(self, nodes:list[TensorNode], copy=True) -> None:
        if copy:
            nodes = _copy_nodes_and_fix_indices(nodes)
        self._nodes = nodes

    @classmethod
    def from_arbitrary_tn(cls, tn:ArbitraryTN, **kwargs) -> "_FrozenSpecificNetwork":        
        new = cls(tn.nodes, copy=False, **kwargs)
        del tn
        return new
    
    def to_arbitrary_tn(self)->ArbitraryTN:
        return ArbitraryTN(self.nodes, copy=False)
    
    # ================================================= #
    #|       Mandatory Implementations of ABC          |#
    # ================================================= #
    def copy(self, **kwargs)->Self:
        return type(self)(nodes=self.nodes, copy=True, **kwargs)
    
    # ================================================= #
    #|        Protecting nodes from change             |#
    # ================================================= #
    @property
    def nodes(self)->list[TensorNode]:
        return self._nodes
    

class CoreTN(_FrozenSpecificNetwork):
    all_mode_sides = [BlockSide.U, BlockSide.DL, BlockSide.DR]

    def __init__(self, nodes: list[TensorNode], copy=True) -> None:
        super().__init__(nodes, copy)

    # ================================================= #
    #|        Core nodes and their relations           |#
    # ================================================= #
    def mode_core_nodes(self, mode:UpdateMode)->list[TensorNode]:
        ## find center nodes of the 
        center_node = next((node for node in self.center_nodes if mode.is_matches_flavor(node.unit_cell_flavor)))
        first_neighbors = self.all_neighbors(center_node)
        return first_neighbors + [center_node]
        
    @property
    def center_nodes(self)->list[TensorNode]:
        return [self.nodes[i] for i in [2, 4, 5]]
    
    @property
    def env_nodes(self)->list[TensorNode]:
        return [self.nodes[i] for i in range(9, self.size)]
    
    def env_nodes_on_side(self, side:BlockSide)->list[TensorNode]:
        res = []
        for node in self.get_nodes_on_boundary(side):
            for neighbor in self.all_neighbors(node):
                if neighbor.functionality is NodeFunctionality.Environment:
                    res.append(neighbor)
        return res

    # ================================================= #
    #|             Structure and Geometry              |#
    # ================================================= #
    pass


class ModeTN(_FrozenSpecificNetwork):

    def __init__(self, nodes: list[TensorNode], mode:UpdateMode, copy=True) -> None:
        super().__init__(nodes, copy)
        self.mode : UpdateMode = mode
    
    @classmethod 
    def from_arbitrary_tn(cls, tn: ArbitraryTN, mode:UpdateMode) -> "ModeTN":
        return super().from_arbitrary_tn(tn, mode=mode)
    
    def copy(self) -> Self:
        return super().copy(mode=self.mode)
        
    # ================================================= #
    #|        Core nodes and their relations           |#
    # ================================================= #

    @functools.cached_property
    def center_node(self)->TensorNode:
        return next((node for node in self.nodes if self.mode.is_matches_flavor(node.unit_cell_flavor)))

    def nodes_on_side(self, side:BlockSide)->list[TensorNode]:
        assert side in self.major_directions
        return [self.find_neighbor(self.center_node, direction) for direction in side.matching_lattice_directions()]



    # ================================================= #
    #|             Structure and Geometry              |#
    # ================================================= #
    @property
    def major_directions(self)->list[BlockSide]:
        match self.mode:
            case UpdateMode.A:  return [BlockSide.U , BlockSide.D ]
            case UpdateMode.B:  return [BlockSide.UR, BlockSide.DL]
            case UpdateMode.C:  return [BlockSide.UL, BlockSide.DR]



def _copy_nodes_and_fix_indices(nodes:list[TensorNode])->list[TensorNode]:
    new_nodes = []
    for i, node in enumerate(nodes):
        new_node = node.copy()
        new_node.index = i
        new_nodes.append(new_node)
    return new_nodes


def _derive_message_node_position(nodes_on_boundary:list[Node], edge:EdgeIndicatorType, boundary_delta:tuple[float,...]):
    node = next(node for node in nodes_on_boundary if edge in node.edges)
    direction = node.directions[ node.edges.index(edge) ]
    delta = tuples.add(boundary_delta, direction.unit_vector)
    # delta = (numerics.furthest_absolute_integer(delta[0]), numerics.furthest_absolute_integer(delta[1]))
    return tuples.add(node.pos, delta)     


def _message_nodes(
    tn : KagomeTN,
    message : Message,
    boundary_side : BlockSide,
    first_node_index : int
) -> list[TensorNode]:

    # Unpack Inputs:
    mps = message.mps    
    mps_order_dir = message.orientation.ordered

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


def  _derive_nodes_kagome_tn(tn:KagomeTN)->list[TensorNode]:
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
            unit_cell_flavor=cell_type,
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



def _fuse_double_legs(n1:TensorNode, n2:TensorNode)->None:
    """Fuse the common edges of nodes n1 and n2 which indices 
    """
    ## derive basic data for algo:
    common_edges : list[str] = [edge for edge in n1.edges if edge in n2.edges]
    new_edge_name = "+".join(common_edges)

    ## Fuse legs:
    for node in [n1, n2]:
        # Find double legs:
        indices_to_fuse = [node.edges.index(edge) for edge in common_edges]
        # Fuse:
        node.fuse_legs(indices_to_fuse, new_edge_name)
    


def _derive_sub_tn(tn:BaseTensorNetwork, indices:list[int])->ArbitraryTN:
    
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
    new = ArbitraryTN(nodes=nodes, edges=edges, _copied=True, tensor_dims=t_dims)
    min_x, max_x, min_y, max_y = new.positions_min_max()
    new.original_lattice_dims = ( int(max_y-min_y+1), int(max_x-min_x+1) )
    return new

def _derive_edges_kagome_tn(self:KagomeTN)->dict[str, tuple[int, int]]:
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


def _validate_tn(tn:BaseTensorNetwork):
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
    for edge in n1.edges:
        if edge in n2.edges:
            return edge
    raise NetworkConnectionError(f"Nodes '{n1}' and '{n2}' don't have a common edge.")


def _derive_node_data_from_contracted_nodes_and_fix_neighbors(tn:KagomeTN, n1:TensorNode, n2:TensorNode):
    contraction_edge = get_common_edge(n1, n2)
    if n1.functionality is n2.functionality:
        functionality = n1.functionality
    else: 
        functionality = NodeFunctionality.Undefined
    pos = n2.pos # pos = tuples.mean_itemwise(n1.pos, n2.pos)  # A more sensefull option, but unneeded in our case
    edges = []
    directions = []      #TODO: Directions are sometimes wrong in this case!
    for node in [n1, n2]:
        for edge in node.edges:
            if edge==contraction_edge:
                continue
            neigbor = tn._find_neighbor_by_edge(node, edge)
            angle = tuples.angle(pos, neigbor.pos) 
            try:                        
                dir = LatticeDirection.from_angle(angle, eps=0.2)
                neighbor_dir = LatticeDirection.opposite(dir)                    
            except DirectionError:
                dir = Direction("adhoc", angle=angle)
                neighbor_dir = Direction("adhoc", angle=numerics.force_between_0_and_2pi(angle+np.pi))
            # Also fix neighbors direction:                
            edge_ind = neigbor.edges.index(edge)
            tn.nodes[neigbor.index].directions[edge_ind] = neighbor_dir
            edges.append(edge)        
            directions.append(dir)        
    name = n1.name+"+"+n2.name
    on_boundary = n1.boundaries.union(n2.boundaries) 

    return contraction_edge, edges, directions, functionality, pos, name, on_boundary


def _qr_decomposition(tn:ArbitraryTN, node:TensorNode, edges1:list[EdgeIndicatorType], edges2:list[EdgeIndicatorType])->tuple[TensorNode, TensorNode]:

    ## Some checks:
    if DEBUG_MODE:
        assert not node.is_ket, "we don't yet support ket tensors"
        assert set(edges1).isdisjoint(set(edges2)), "edges must be strictly different"
        assert len(edges1)+len(edges2)==len(node.directions)

    ## Derive indices and dimensions in consistent order of legs:
    indices1, indices2 = [], []
    directions1, directions2 = [], []
    dims1, dims2 = [], []
    edges_in_order1, edges_in_order2 = [], []
    for i, (direction, edge, dim) in enumerate(node.legs()):
        if edge in edges1:
            dims1.append(dim)
            directions1.append(direction)
            indices1.append(i)
            edges_in_order1.append(edge)

        elif edge in edges2:
            dims2.append(dim)
            directions2.append(direction)
            indices2.append(i)
            edges_in_order2.append(edge)

        else:
            raise ValueError("Not an expected case")

    ## Turn tensor into a matrix:
    *_ , dim1 = itertools.accumulate(dims1, func=operator.mul)
    *_ , dim2 = itertools.accumulate(dims2, func=operator.mul)
    node.permute(indices1+indices2)
    m = node.tensor.reshape((dim1, dim2))

    ## Perform QR-Decomposition:
    q, r = np.linalg.qr(m, mode='reduced')

    ## spread dimensions of q-r matrices into tensors that match original dimensions and connections:
    k = q.shape[-1]
    assert k == r.shape[0]
    t1 = q.reshape(dims1+[k])
    t2 = r.reshape([k]+dims2)

    ## Derive the properties of the new nodes:
    pos1 = tuples.add(node.pos, tuples.multiply(create.mean_direction(directions1).unit_vector, 0.6))
    pos2 = tuples.add(node.pos, tuples.multiply(create.mean_direction(directions2).unit_vector, 0.6))
    qr_edge_name = tn.unique_edge_name("qr_edge")
    qr_direction1 = create.direction_from_positions(pos1, pos2)

    ## Create the two new tensors
    n1 = TensorNode(
        index=tn.size-1,
        name="Q",
        tensor=t1,
        is_ket=False,
        pos=pos1,
        edges=edges_in_order1+[qr_edge_name],
        directions=directions1+[qr_direction1]
    )
    n2 = TensorNode(
        index=tn.size,
        name="R",
        tensor=t2,
        is_ket=False,
        pos=pos2,
        edges=[qr_edge_name]+edges_in_order2,
        directions=[qr_direction1.opposite()]+directions2
    )
    if DEBUG_MODE:
        n1.validate()
        n2.validate()

    ## Insert the new nodes instead of the old node
    tn.pop_node(node.index)
    tn.add_node(n1)
    tn.add_node(n2)

    if DEBUG_MODE:
        tn.validate()

    return n1, n2


def _contract_nodes(tn:ArbitraryTN, n1:TensorNode|int, n2:TensorNode|int)->TensorNode:

    ## Also accept indices as inputs:
    if isinstance(n1, int) and isinstance(n2, int):
        n1 = tn.nodes[n1]
        n2 = tn.nodes[n2]
    assert isinstance(n1, TensorNode) and isinstance(n2, TensorNode)
    ## Collect basic data:
    i1, i2 = n1.index, n2.index
    if DEBUG_MODE:
        assert tn.nodes[i1] is n1, f"Node in index {i1} should match {n1.index}"
        assert tn.nodes[i2] is n2, f"Node in index {i2} should match {n2.index}"
    ## Get contracted tensor data:
    contraction_edge, edges, directions, functionality, pos, name, on_boundary = _derive_node_data_from_contracted_nodes_and_fix_neighbors(tn, n1, n2)
    ## Remove nodes from list while updating indices in both self.edges and self.nodes:
    n1_ = tn.pop_node(n1.index)    
    n2_ = tn.pop_node(n2.index)            
    if DEBUG_MODE:
        assert n1_ is n1
        assert n2_ is n2
    ## Derive index of edge for each node:
    ie1 = n1.edges.index(contraction_edge)
    ie2 = n2.edges.index(contraction_edge)

    ## Open tensors (with physical legs) behave differently than fused tensors.
    # Always use fused-tensors. If not already fuse, cause them to be fused.
    is_ket = False
    tensor = np.tensordot(n1.fused_tensor, n2.fused_tensor, axes=(ie1, ie2))
        
    ## Add node:
    new_node = TensorNode(
        tensor=tensor,
        is_ket=is_ket,
        functionality=functionality,
        edges=edges,
        directions=directions,
        pos=pos,
        index=len(tn.nodes),  # Two old nodes are already deleted from list, new will be appended to the end.
        name=name,
        boundaries=on_boundary
    )
    tn.nodes.append(new_node)

    ## Fuse double edges:
    seen_neighbors : set[int] = set()
    for edge in new_node.edges:
        neighbor = tn._find_neighbor_by_edge(new_node, edge)
        ## if we've already seen this neighbor, then this is a double edge
        if neighbor.index in seen_neighbors:
            _fuse_double_legs(new_node, neighbor)
        seen_neighbors.add(neighbor.index)

    ## Check:
    if DEBUG_MODE: tn.validate()
    return new_node


if __name__ == "__main__":
    from scripts.contraction_test import main_test
    main_test()