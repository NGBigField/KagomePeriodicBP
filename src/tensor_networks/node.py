
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

import numpy as np
from numpy import ndarray as np_ndarray

from typing import Tuple, Generator

# Use some of our utilities:
from utils import lists

# for type namings:
from _types import EdgeIndicatorType, PosScalarType
from _error_types import NetworkConnectionError

# For OOP Style:
from copy import deepcopy
from dataclasses import dataclass, field, fields

# For TN methods and types:
from tensor_networks.operations import fuse_tensor_to_itself
from lattices.directions import LatticeDirection, BlockSide, Direction
from enums import NodeFunctionality, UnitCellFlavor



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
    core_cell_flavor : UnitCellFlavor = field(default=UnitCellFlavor.NoneLattice) 
    boundaries : list[BlockSide] = field(default_factory=list) 


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
        return self.fused_tensor.shape
    
    
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
        from lattices.directions import unit_vector_from_angle
        from utils import visuals
                
        plt.figure()
        if self.functionality is NodeFunctionality.CenterUnitCell:
            node_color = 'blue'
        else:
            node_color = 'red'
        plt.scatter(0, 0, c=node_color)
        text = f"{self.name} [{self.index}]"
        plt.text(0, 0, text)
                
        for edge_index, (edge_direction, edge_name, edge_dim)  in enumerate(self.legs()):
            vector = edge_direction.unit_vector()
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
        
        
    def __eq__(self, other)->bool:
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


    def __repr__(self) -> str:
        positions = lists.convert_whole_numbers_to_int(list(self.pos))
        return f"Node '{self.name}' at index [{self.index}] on site {tuple(positions)}"

    
