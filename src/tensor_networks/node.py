
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

## Get config:
from utils.config import DEBUG_MODE, VERBOSE_MODE


import numpy as np
from numpy import ndarray as np_ndarray

from typing import NamedTuple, TypeAlias, Tuple, Generator

# Use some of our utilities:
from utils import lists

# For OOP Style:
from copy import deepcopy
from dataclasses import dataclass, field, fields

# For lattice methods and types:
from tensor_networks.operations import fuse_tensor
from enums import Directions, NodeFunctionality

_EdgeIndicator : TypeAlias = str
_PosScalarType : TypeAlias = int


def _angle(direction:Directions|float)->float:
    if isinstance(direction, Directions):
        return direction.angle
    elif isinstance(direction, float):
        return direction
    else:
        raise TypeError(f"Not an expected type '{type(direction)}'")


@dataclass
class Node():
    is_ket : bool
    tensor : np.ndarray
    functionality : NodeFunctionality
    edges : list[_EdgeIndicator]
    directions : list[Directions|float]  # if Direction enum then less expansive than floats
    pos : Tuple[_PosScalarType,...]
    index : int
    name : str 
    on_boundary : list[Directions] = field(default_factory=list) 


    @property
    def physical_tensor(self) -> np_ndarray:
        if not self.is_ket:
            raise ValueError("Not a ket tensor")
        return self.tensor
    
    @property
    def fused_tensor(self) -> np_ndarray:
        if self.is_ket:
            return fuse_tensor(self.tensor)
        else:
            return self.tensor

    @property
    def angles(self) -> list[float]:
        return [_angle(direction) for direction in self.directions ]

    @property
    def dims(self) -> Tuple[int]:
        return self.fused_tensor.shape
    
    @property
    def norm(self) -> np.float64:
        return np.linalg.norm(self.tensor)
    
    def normalize(self) -> None:
        self.tensor = self.tensor / self.norm
    
    def copy(self) -> "Node":
        new = Node.empty()
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

    def edge_in_dir(self, dir:Directions|float)->_EdgeIndicator:
        if isinstance(dir, Directions):
            index = self.directions.index(dir)
        elif isinstance(dir, float):
            index = lists.index_by_approx_value(self.directions, dir)
        else:
            raise TypeError(f"Not an expected type '{type(dir)}'")
        return self.edges[index]
    
    def permute(self, axes:list[int]):
        """Like numpy.transpose but permutes "physical_tensor", "fused_tensor", "edges" & "direction" all together
        """
        if self.is_ket:
            physical_indices = [0]+[i+1 for i in axes]
            self.tensor = self.tensor.transpose(physical_indices)
        else:
            self.tensor = self.tensor.transpose(axes)
        self.edges = lists.rearrange(self.edges, axes)
        self.directions = lists.rearrange(self.directions, axes)      
      
    def all_legs_data_gen(self)->Generator[tuple[str, int, Directions|float, int], None, None]:
        for edge_index, (edge_name, edge_dim, edge_direction) in enumerate(zip(self.edges, self.dims, self.directions, strict=True)):
            yield edge_name, edge_dim, edge_direction, edge_index

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
        from enums.directions import unit_vector_from_angle
        from utils import visuals
                
        plt.figure()
        if self.functionality is NodeFunctionality.Core:
            node_color = 'blue'
        else:
            node_color = 'red'
        plt.scatter(0, 0, c=node_color)
        text = f"{self.name} [{self.index}]"
        plt.text(0, 0, text)
                
        for edge_name, edge_dim, edge_direction, edge_index in self.all_legs_data_gen():
            if isinstance(edge_direction, Directions):
                vector = edge_direction.unit_vector()
            else:
                vector = unit_vector_from_angle(edge_direction)                            
            x, y = vector[0], vector[1]
            plt.plot([0, x], [0, y], color="black", alpha=0.8, linewidth=1.5 )
            plt.text(x/2, y/2, f"{edge_name}:\n{edge_dim} [{edge_index}]")
        
        visuals.draw_now()

    @classmethod
    def empty(cls)->"Node":
        return Node(
            is_ket=False,
            tensor=np.zeros((2,2)),
            functionality=NodeFunctionality.Undefined,
            edges=[],
            directions=[],
            pos=(0,0),
            index=-1,
            name="",
            on_boundary=[]
        )
        
    def __eq__(self, other)->bool:
        assert isinstance(other, Node), "Must be same Node type for comparison"
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


    def __repr__(self) -> str:
        positions = lists.convert_whole_numbers_to_int(list(self.pos))
        return f"Node '{self.name}' at index [{self.index}] on site {tuple(positions)}"

    
