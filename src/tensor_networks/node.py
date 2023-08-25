
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

import numpy as np
from numpy import ndarray as np_ndarray


# Config:
from _config_reader import DEBUG_MODE

# Use some of our utilities:
from utils import lists, strings, tuples, assertions

# for type namings:
from _types import EdgeIndicatorType, PosScalarType
from _error_types import NetworkConnectionError

# For OOP Style:
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Tuple, Generator

# For TN methods and types:
from tensor_networks.operations import fuse_tensor_to_itself
from lattices.directions import LatticeDirection, BlockSide, Direction, check
from enums import NodeFunctionality, UnitCellFlavor

# For smart iterations:
import itertools
import operator

@dataclass
class TensorNode():
    index : int
    name : str 
    tensor : np.ndarray
    is_ket : bool
    pos : Tuple[PosScalarType, ...]
    edges : list[EdgeIndicatorType]
    directions : list[LatticeDirection] 
    functionality : NodeFunctionality = field(default=NodeFunctionality.Undefined) 
    unit_cell_flavor : UnitCellFlavor = field(default=UnitCellFlavor.NoneLattice) 
    boundaries : set[BlockSide] = field(default_factory=set) 

    def __hash__(self)->int:
        return hash((self.name, self.pos, self.functionality, self.unit_cell_flavor))

    @property
    def physical_tensor(self) -> np_ndarray:
        if not self.is_ket:
            raise ValueError("Not a ket tensor")
        return self.tensor

    @property
    def fused_tensor(self) -> np_ndarray:
        if self.is_ket:
            return fuse_tensor_to_itself(self.tensor)
        else:
            return self.tensor

    @property
    def angles(self) -> list[float]:
        return [direction.angle for direction in self.directions ]

    @property
    def dims(self) -> Tuple[int]:
        if not self.is_ket:
            return self.tensor.shape
        connectable_dims = self.tensor.shape[1:]        
        return tuples.power(connectable_dims, 2)
    
    @property
    def norm(self) -> np.float64:
        return np.linalg.norm(self.tensor)
    
    def legs(self) -> Generator[tuple[LatticeDirection, EdgeIndicatorType, int], None, None]:
        for direction, edge, dim in zip(self.directions, self.edges, self.dims, strict=True):
            yield direction, edge, dim

    def normalize(self) -> None:
        self.tensor = self.tensor / self.norm
    
    def copy(self) -> "TensorNode":
        new = TensorNode.empty()
        for f in fields(self):
            val = getattr(self, f.name)
            if hasattr(val, "copy"):
                val = val.copy()
            else:
                try:
                    val = deepcopy(val)
                except:
                    pass
            setattr(new, f.name, val)
        return new

    def edge_in_dir(self, dir:Direction)->EdgeIndicatorType:
        assert isinstance(dir, Direction), f"Not an expected type '{type(dir)}'"
        try:
            index = self.directions.index(dir)
        except Exception as e:
            raise NetworkConnectionError(f"Direction {dir!r} is not in directions of nodes: {[dir.name for dir in self.directions]}")
        
        if DEBUG_MODE:
            assert len([_dir for _dir in self.directions if _dir is dir])==1, "Multiple legs with the same direction"

        return self.edges[index]
    
    def permute(self, axes:list[int])->None:
        """Like numpy.transpose but permutes "physical_tensor", "fused_tensor", "edges" & "direction" all together
        """
        if self.is_ket:
            physical_indices = [0]+[i+1 for i in axes]
            self.tensor = self.tensor.transpose(physical_indices)
        else:
            self.tensor = self.tensor.transpose(axes)
        self.edges = lists.rearrange(self.edges, axes)
        self.directions = lists.rearrange(self.directions, axes)      

    def validate(self)->None:
        # Generic validation error message:
        _failed_validation_error_msg = f"Node at index {self.index} failed its validation."
        #
        assert self.fused_tensor.shape==self.dims , _failed_validation_error_msg
        assert len(self.fused_tensor.shape)==len(self.edges)==len(self.directions)==len(self.dims)==len(self.angles) , _failed_validation_error_msg
        # check all directions are different:
        assert len(self.directions)==len(set(self.directions)), _failed_validation_error_msg+f"\nNot all directions are different: {self.directions}" 
         
    def plot(self)->None:
        ## Some special imports:
        from matplotlib import pyplot as plt
        from utils import visuals
                
        plt.figure()
        if self.functionality is NodeFunctionality.CenterCore:
            node_color = 'blue'
        else:
            node_color = 'red'
        plt.scatter(0, 0, c=node_color)
        text = f"{self.name} [{self.index}]"
        plt.text(0, 0, text)
                
        for edge_index, (edge_direction, edge_name, edge_dim)  in enumerate(self.legs()):
            vector = edge_direction.unit_vector
            x, y = vector[0], vector[1]
            plt.plot([0, x], [0, y], color="black", alpha=0.8, linewidth=1.5 )
            plt.text(x/2, y/2, f"{edge_name}:\n{edge_dim} [{edge_index}]")
        
        visuals.draw_now()

    @classmethod
    def empty(cls)->"TensorNode":
        return TensorNode(
            is_ket=False,
            tensor=np.zeros((2,2)),
            functionality=NodeFunctionality.Undefined,
            edges=[],
            directions=[],
            pos=(0,0),
            index=-1,
            name="",
            boundaries=[]
        )  
        
    def is_data_equal(self, other)->bool:
        assert isinstance(other, TensorNode), "Must be same Node type for comparison"
        if self.physical_tensor is None and other.physical_tensor is not None:
            return False
        if self.physical_tensor is not None and other.physical_tensor is None:
            return False
        if self.physical_tensor is not None and other.physical_tensor is not None \
            and not np.all( np.equal(self.physical_tensor, other.physical_tensor) ):
            return False
        if not np.all( np.equal(self.fused_tensor, other.fused_tensor) ):
            return False
        for dir1, dir2 in zip(self.directions, other.directions, strict=True):
            if dir1 != dir2:
                return False
        return True
    
    def turn_into_bracket(self)->None:
        self.tensor = self.fused_tensor
        self.is_ket = True
    
    def fuse_legs(self, indices_to_fuse:list[int], new_edge_name:str=strings.random(10))->None:
        ## Check:
        directions = [self.directions[i] for i in indices_to_fuse]
        try:
            assert check.all_same(directions), f"Only supports leg fusion when legs are in the same direction"
        except Exception as e:
            check.is_equal(directions[0], directions[1])
            pass

        ## Collect some data:
        old_dimes = self.dims
        indices_to_keep = [i for i, _ in enumerate(old_dimes) if i not in indices_to_fuse]
        num_fused_legs = len(indices_to_fuse)

        ## Derive the new size of the tensor after fusion:
        *_, fused_dim = itertools.accumulate([old_dimes[i] for i in indices_to_fuse], operator.mul)
        new_dims = [old_dimes[i] for i in indices_to_keep] + [fused_dim]

        ## Change indices of tensor object so that the dimensions to fuse are last:
        self.permute(indices_to_keep+indices_to_fuse)

        ## Fuse legs:
        # fuse tensor with numpy.reshape():
        if self.is_ket:
            d = self.tensor.shape[0]
            new_dims = [d]+[validated_int_square_root(dim) for dim in new_dims]
        self.tensor = self.tensor.reshape(new_dims)
        # Deal with the rest of the data:
        for _ in range(num_fused_legs-1):
            self.directions.pop()
            self.edges.pop()
        self.edges[-1] = new_edge_name


    def __repr__(self) -> str:
        positions = lists.convert_whole_numbers_to_int(list(self.pos))
        return f"Node '{self.name}' at index [{self.index}] on site {tuple(positions)}"

    
def validated_int_square_root(a:int)->int:
    return assertions.integer(np.sqrt(a))


def two_nodes_ordered_by_relative_direction(n1:TensorNode, n2:TensorNode, direction:Direction)->tuple[TensorNode, TensorNode]:
    opposite = direction.opposite()
    if n1.edge_in_dir(direction) == n2.edge_in_dir(opposite):
        return n1, n2
    elif n2.edge_in_dir(direction) == n1.edge_in_dir(opposite):
        return n2, n1
    else:
        raise NetworkConnectionError(f"Nodes are not connected in given direction {direction.name!r}")

