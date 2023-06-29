

# ============================================================================ #
#                                  Imports                                     #
# ============================================================================ #

# for allowing forward referencing of type "Self" in alternative constructors:
from __future__ import annotations


# For type hinting:
from typing import (
    TypeAlias,
    TypeVar,
    Type,
    Tuple,
    NamedTuple,
    List,
    Dict,
    Any,
    Union,
    Optional,
    Generator,
    ClassVar,
    Iterator,
    Callable,
    Final,
)

# For matrices and tensors:
import numpy as np

# For OOP:
from abc import ABC, abstractmethod  # for implementing abstract classes:
import attr  # Get with ```pip install attrs```  NOT with ```pip install attr```
from enum import Enum, auto  # for enums:

# some of our utilities that are needed here:

from utils import (
    errors,
    strings,
    assertions,
    indices,      
    visuals, 
    arguments,
    errors,
    lists,
)



# For plotting:
import matplotlib.pyplot as plt

# for common errors:
from tensor_networks.errors import(
    ContractionDimensionsTooBigError,
    NoNetworkImplementation,
)

# ============================================================================ #
#                                Constants                                     #
# ============================================================================ #
INCLUDE_OTHER_IDENTIFIERS_IN_REPR : Final[bool] = True



# ============================================================================ #
#                              helper functions                                #
# ============================================================================ #


# ============================================================================ #
#                                Helper Types                                  #
# ============================================================================ #
_TensorDateType : TypeAlias = np.ndarray

# ============================================================================ #
#                               Generic Classes                                #
# ============================================================================ #


# ============================================================================ #
#                                    Legs                                      #
# ============================================================================ #

_leg_copyable_attributes = ["name", "tag", "connected", "_other_identifiers"]

@attr.s()
class Leg:
    
    # ----------------- Nested class ----------------- #
    class Tag(Enum):
        Undefined   = auto()
        Physical    = auto()
        Virtual     = auto()
        Other       = auto() 

        def __repr__(self) -> str:
            s = self.name
            return s

    # ----------------- Leg Creation ----------------- #
    ind                 : int       = attr.ib()  # index of leg in the tensor
    dim                 : int       = attr.ib()  # dimension of the leg
    tensor              : Tensor    = attr.ib(repr=False)  # Reference to tensor that holds this leg
    tag                 : Tag       = attr.ib(default=Tag.Undefined, validator=attr.validators.instance_of(Tag), repr=True) 
    name                : str       = attr.ib(default="", validator=attr.validators.instance_of(str), repr=True)
    connected           : bool      = attr.ib(default=False, validator=attr.validators.instance_of(bool), repr=True, init=False)  # Not part of __init__
    _other_identifiers  : dict      = attr.ib(factory=dict, init=False, repr=INCLUDE_OTHER_IDENTIFIERS_IN_REPR)

    @dim.validator
    def _check_dim(self, attribute, value):
        assertions.integer(value)
        assert value>0, f"Dimension={value} but must be greater than 0"
    @ind.validator
    def _check_ind(self, attribute, value):
        assertions.index(value)
    @tensor.validator
    def _check_tensor(self, attribute, value):
        assert isinstance(value, Tensor)
        assert value.shape[self.ind] == self.dim  # Check that dimensions match
        

    # ----------------- Override common operations ----------------- #    

    def __mul__(self, other:Leg) -> Tensor:  # called in the multiplication operator self * other
        return Leg.contract(self, other)

    def __or__(self, other:Leg) -> Leg:  # called in the statement self | other
        return Leg.fuse([self, other])

    def __add__(self, other:Leg) -> None:  # called in the statement self + other
        if not hasattr(self.tensor, 'network'):
            raise NoNetworkImplementation()
        tn = self.tensor.network
        tn += (self, other)


    # ----------------- Methods ----------------- #
    def copy_attribs(self, source:Leg, except_list:List[str]=[]) -> None:
        for attrib_name in _leg_copyable_attributes:
            if attrib_name not in except_list:
                value = source.__getattribute__(attrib_name)
                self.__setattr__(attrib_name, value)

    
    def copy(self, tensor) -> Leg:
        new = Leg(
            tensor  = tensor,
            ind     = self.ind, 
            dim     = self.dim
        )
        new.copy_attribs(source=self)
        return new 

    def braket_contraction(self) -> Tensor:
        return bracket_contraction(leg=self, op=None)
        
    # ----------------- Static Methods ----------------- #

    @staticmethod
    def contract(legs1:Union[Leg, List[Leg]], legs2:Union[Leg, List[Leg]]) -> Tensor :
        """contract Performs leg contraction between two tensors in a network.

        Replaces two tensors with one contracted tensor (output).

        Args:
            leg1 (Leg): The leg with which to contract tensor1
            leg2 (Leg): The leg with which to contract tensor2

        Returns:
            Tensor: Contraction of tensor1 and tensor2
        """
        ## Check input:
        legs1, legs2 = _check_contraction_input(legs1, legs2)

        # Init lists:
        remaining_legs : List[List[Leg]] = []  # list of all legs that remain after contraction
        contraction_indices : List[List[int]] = [] # list of indices of legs that are used for contraction
        tensors : Tuple[Tensor, Tensor] = (legs1[0].tensor, legs2[0].tensor)

        # Derive remaining Legs and contraction indices:
        for i, legs in enumerate([legs1, legs2]):
            remaining_legs.append( [leg for leg in tensors[i].legs if leg not in legs ] )
            contraction_indices.append( [leg.ind for leg in tensors[i].legs if leg in legs ] )
        
        ## Contract using `numpy.tensordot`
        try:
            new_data = np.tensordot(tensors[0].data, tensors[1].data, axes=(contraction_indices[0], contraction_indices[1]))
        except MemoryError as e:
            raise ContractionDimensionsTooBigError(f"The contraction of tensor1: "+
            f"{tensors[0].shape} and tensor2: {tensors[1].shape} was to resource-wasteful for the computer") 
            
            
        new_tensor = Tensor(new_data)
        # Inherit attributes:
        if tensors[0].name == tensors[1].name:
            new_tensor.name = tensors[0].name
        if tensors[0].pos == tensors[1].pos:
            new_tensor.pos = tensors[0].pos

        # Relate all previous legs to this tensor
        all_remaining_legs : List[Leg] = remaining_legs[0]+remaining_legs[1]
        assert len(all_remaining_legs) == new_tensor.num_legs
        for i, old_leg in enumerate(all_remaining_legs):
            old_leg._other_identifiers["contraction_index"] = i
            new_tensor.legs[i].copy_attribs(old_leg)

        return new_tensor

    @staticmethod
    def fuse(legs : List[Leg], fused_leg_first:bool=True) -> Leg :
        # Check input:
        tensor = legs[0].tensor
        for leg in legs:
            assert isinstance(leg, Leg)
            assert leg.tensor is tensor, f"Legs must be of same tensor."        
        
        # Pre-allocate info dims and containers:
        fused_leg_dim = 1
        other_legs_dims : List[int] = []

        class Indices(NamedTuple):
            to_fuse : List[int]
            to_keep : List[int]
        indices = Indices(to_fuse=list(), to_keep=list())

        # Derive info
        for index, leg in enumerate(tensor.legs):
            assert index == leg.ind
            if leg in legs:
                fused_leg_dim *= leg.dim
                indices.to_fuse.append(leg.ind)
            else:
                other_legs_dims.append(leg.dim)
                indices.to_keep.append(leg.ind)

        # Choose indices Order 
        if fused_leg_first:
            all_legs_dims = [fused_leg_dim] + other_legs_dims
            indices_order = indices.to_fuse + indices.to_keep
        else:
            all_legs_dims = other_legs_dims + [fused_leg_dim]
            indices_order = indices.to_keep + indices.to_fuse

        # reshape tensor data with numpy:
        tensor.data = tensor.data.transpose(indices_order)

        # Fuse legs with numpy.reshape():
        tensor.data = tensor.data.reshape(all_legs_dims)

        # remove the fused legs
        tensor.legs = [leg for leg in tensor.legs if leg not in legs]

        # create the new leg
        new_leg = Leg(ind=0, dim=fused_leg_dim, tensor=tensor)
        # Inherit leg attributes for the fused legs:
        example_of_an_old_fused_leg = legs[0]
        except_list = [
            attrib_name for attrib_name in _leg_copyable_attributes 
            if not all([leg.__getattribute__(attrib_name)==example_of_an_old_fused_leg.__getattribute__(attrib_name) for leg in legs])
        ]
        new_leg.copy_attribs(source=example_of_an_old_fused_leg, except_list=except_list)
        
        # add the new leg to tensor:
        if fused_leg_first:
            tensor.legs.insert(0, new_leg)
        else:
            tensor.legs.append(new_leg)

        # re-assign legs indices:
        for i, leg in enumerate(tensor.legs):
            leg.ind = i

        return new_leg


        
# ============================================================================ #
#                                   Tensors                                    #
# ============================================================================ #


class Tensor():
    # ----------------- class variables ----------------- #
    _class_ids_gen : ClassVar[Iterator] = indices.indices_gen()
    class Leg(Leg): ...

    # ----------------- Constructors ----------------- #
    def __init__(
        self, 
        data :_TensorDateType, 
        name : str = '', 
        pos  : Optional[Tuple[float, ...]] = None
    ) -> None:

        self.data : _TensorDateType = data
        self.legs : List[Leg] = []
        for ind, dim in enumerate(self.shape):
            self.legs.append( Leg(dim=dim, ind=ind, tensor=self) )
        self.unique_id = next(Tensor._class_ids_gen)
        self.name = name
        self._indicator : Any = None
        self.pos : Tuple[float|None, ...] = arguments.default_value(pos, (None, None))
        self.network : Optional[Any] = None  # Used to affiliate the tensor with some tensor-network object

    @classmethod
    def empty(cls:type, dims:Tuple[int, ...]) -> Tensor:        
        data : _TensorDateType = np.zeros(shape=dims, dtype='complex_')
        return cls(data)

    @classmethod
    def random(cls:type, dims:Tuple[int, ...]) -> Tensor:        
        data : _TensorDateType = np.random.normal(size=dims) + 1j*np.random.normal(size=dims)        
        return cls(data)

    @classmethod
    def copy(
        cls:Type["Tensor"], 
        old:"Tensor", 
        data_func:Callable[[_TensorDateType], _TensorDateType] = lambda data: data
    ) -> "Tensor" :
        # Copy Data:
        new_data = data_func(old.data)
        # Create Tensor object:
        new = cls(new_data)
        # Copy Legs:
        new.legs = [ l.copy(new) for l in old.legs ]
        # Copy other data:
        new.name = old.name
        if hasattr(old, 'pos'):
            new.pos = old.pos
        return new

    # ----------------- Visuals ----------------- #
    @visuals.matplotlib_wrapper()
    def draw(self) -> None:
        # Constants:
        d = 0.05
        # Plot tensor:
        plt.scatter([0], [0], s=300, color='green', zorder=10)
        plt.text( d, 0, s=f"{self.name}", fontdict=dict(size=16), zorder=20)
        # Plot legs:
        angles = np.linspace(0, 2*np.pi, self.num_legs+1)[0:-1]  # Uniform spray of angles
        for leg, angle in zip(self.legs, angles):
            leg_sta = [0, 0]
            leg_end = [np.cos(angle), np.sin(angle)]
            leg_mid = [pos/2 for pos in leg_end]
            leg_far = [1.1*np.cos(angle), 1.1*np.sin(angle)]
            plt.plot(  
                [leg_sta[0], leg_end[0]],
                [leg_sta[1], leg_end[1]],
                color='blue', 
                zorder=5   # Draw beneath other lines
            )
            plt.text( *leg_mid, s=f"{leg.dim}" , fontdict=dict(size=16), zorder=20, color='red')
            plt.text( *leg_far, s=f"{leg.name}", fontdict=dict(size=16), zorder=20, color='black')
        # Change axis appearance:
        plt.axis("off")


    # ----------------- Hash-ability ----------------- #
    def __hash__(self) -> int:
        return hash((self.unique_id))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and other.unique_id == self.unique_id

    # ----------------- Override common operations ----------------- #    
    def __repr__(self) -> str:
        s = f"{type(self).__name__} <{self.unique_id}> with dimensions {self.shape}"
        if self.name != '':
            s += f" '{self.name}'"
        if INCLUDE_OTHER_IDENTIFIERS_IN_REPR:
            s += f"  indicator={self._indicator}"
        return s

    def __str__(self) -> str:
        # define repetitively used variables:
        newline = "\n"
        tab = "    "
        ind_str_len = len(str(self.num_legs))
        # derive the full string representation of tensor:
        s = self.__repr__() + newline  # Call function __repr__
        for i, l in enumerate(self.legs):
            s += tab+f"{i:{ind_str_len}}: "+str(l)+newline
        s += np.array_str(self.data, max_line_width=200, precision=3, suppress_small=True, )
        return s

    def __invert__(self:"Tensor") -> "Tensor":  # called in the inversion operator ~self
        return self.conj()

    # ----------------- Tensor (numpy) indexing ----------------- #    

    # called in operations self[key], self[i:j:k], x in self;
    def __getitem__(self, key: (
        None | slice | ellipsis | tuple[None | slice | ellipsis , ...]
    )) -> _TensorDateType:
        return self.data.__getitem__(key)

    # called in operations self[key] = value and self[i:j:k]=value;
    def __setitem__(
        self, 
        key: (None | slice | ellipsis | tuple[None | slice | ellipsis , ...]), 
        value 
    ) -> None : 
        return self.data.__setitem__(key, value)


    # ----------------- Leg management ----------------- #

    def get_legs_with(self, attribute:str, value:Any, dict_key:Optional[Any]=None) -> List[Leg] : 
        """get_legs_with Return a list of all legs that have the given attribute value.

        Args:
            attribute (str): name of leg instance attribute.
            value (Any): target value for the attribute

        Returns:
            List[Leg]
        """
        if dict_key is None:
            return [leg for leg in self.legs if leg.__getattribute__(attribute) == value]
        else:
            return [
                leg for leg in self.legs 
                    if dict_key in leg.__getattribute__(attribute) and leg.__getattribute__(attribute)[dict_key] == value
                ]
    
    def get_leg_with(self, attribute:str, value:Any, dict_key:Optional[Any]=None) -> Leg :
        """get_leg_with Return the first leg that have the given attribute value.

        Args:
            attribute (str): name of leg instance attribute.
            value (Any): target value for the attribute

        Returns:
            Leg

        Example:
            >>> tensor = LatticeTensor.empty((2,4,4))
            >>> legs = tensor.get_leg_with('tag', Leg.Tag.Physical)
        """
        lis = self.get_legs_with(attribute, value, dict_key=dict_key)
        return lis[0]

    def get_legs_with_identifier(self, key:str, value:Any) -> list[Leg] :
        return self.get_legs_with(attribute="_other_identifiers", value=value, dict_key=key)

    def get_leg_with_identifier(self, key:str, value:Any) -> Leg :
        return self.get_legs_with_identifier(key, value)[0]

    # ----------------- Getters\Setters ----------------- #    
    @property
    def shape(self) -> Tuple[int, ...] :
        return self.data.shape

    @property
    def num_legs(self) -> int:
        return len(self.shape)

    @property
    def ndim(self) -> int: 
        """
        Returns the number of dimensions describing the tensor.
        """
        return self.data.ndim

    @property
    def isscalar(self) -> bool:
        return self.ndim == 0

    # ----------------- Tensor Operations ----------------- #    

    def conj(self:"Tensor") -> "Tensor" :
        return self.copy(self, data_func=np.conj) 

    def fuse_legs_with_similar_identifier(self, key:str)->None:
        # Find all unique identifiers:
        all_identifiers = []        
        for leg in self.legs:
            identifier = leg._other_identifiers[key]
            if identifier not in all_identifiers:
                all_identifiers.append(identifier)
        # Fuse all legs that share identifier:
        for identifier in all_identifiers:
            legs = self.get_legs_with_identifier(key, identifier)
            fused_leg = legs[0] | legs[1]
        

    # ----------------- inner\helper functions ----------------- #    


    # ----------------- Static\Class methods ----------------- #    



# ============================================================================ #
#                               Inner Functions                                #
# ============================================================================ #

def _check_contraction_input(legs1:Union[Leg, List[Leg]], legs2:Union[Leg, List[Leg]]) -> Tuple[List[Leg], List[Leg]]:
    if isinstance(legs1, list) and isinstance(legs2, list): 
        for legs in [legs1, legs2]:
            for leg in legs:
                assert isinstance(leg, Leg)
    # Also accept single leg, but cast to list of legs:
    elif isinstance(legs1, Leg) and isinstance(legs2, Leg):
        legs1 = [legs1]
        legs2 = [legs2]
    else:
        raise TypeError("Both inputs should be either list of legs or single legs")

    for leg1, leg2 in zip(legs1, legs2):            
        assert leg1.dim == leg2.dim
        assert leg1.tensor is not leg2.tensor

    for legs in [legs1, legs2]:
        tensor = legs[0].tensor
        for leg in legs:
            assert leg.tensor is tensor

    return legs1, legs2


# ============================================================================ #
#                              Declared Functions                              #
# ============================================================================ #

def bracket_contraction(leg:Leg, op:Tensor|None) -> Tensor:
    ## Acquire conjugate copy of the tensor, including the leg which initiated the function:
    indicator_key = "braket_contraction"
    leg._other_identifiers[indicator_key] = "leg_for_contraction"
    t1 = leg.tensor
    t2 = ~t1  # Conjugate      
    leg_conj = t2.get_leg_with('_other_identifiers', leg._other_identifiers)

    ## Keep track of all the other legs:
    other_legs_indicators : List[str] = []
    for i, (l1, l2) in enumerate(zip(t1.legs, t2.legs)):
        if l1 is leg:
            assert l2 is leg_conj  # Assert order is maintained
            continue
        indicator = f"other_legs:{i}"
        l1._other_identifiers[indicator_key] = indicator
        l2._other_identifiers[indicator_key] = indicator
        other_legs_indicators.append( indicator )

    ## Leg contraction
    if op is None:
        contracted_tensor = leg * leg_conj 
    elif isinstance(op, Tensor):
        assert op.num_legs == 2, "Operator on a single qubit must have 2 legs"
        op_leg_in, op_leg_out = op.legs[0], op.legs[1]
        op_leg_in._other_identifiers["braket"] = "op_leg_in"
        op_leg_out._other_identifiers["braket"] = "op_leg_out"
        contracted_tensor = leg * op_leg_in
        op_leg_out = contracted_tensor.get_leg_with_identifier("braket", "op_leg_out")
        contracted_tensor = op_leg_out * leg_conj
    else:
        raise TypeError(f"Expected `op` to be of type `Tensor`. Instead got type: {type(op)}")

    ## Fuse respective legs:
    for indicator in other_legs_indicators:
        legs = contracted_tensor.get_legs_with("_other_identifiers", indicator, dict_key=indicator_key)
        assert len(legs)==2
        fused_leg = legs[0] | legs[1]  # leg fusion

    return contracted_tensor


# ============================================================================ #
#                                     tests                                    #
# ============================================================================ #

def _main_example():
    from src.scripts.blockbp_test import main_example
    main_example(is_plot=False)

def _simple_test():
    shape = tuple( [3,5,2] )
    N = 2
    tensors : List[Tensor] = []

    for i in range(N):
        t = Tensor.empty(shape)
        t.legs[0].tag = Leg.Tag.Physical
        t.legs[1].tag = Leg.Tag.Virtual
        t.legs[2].tag = Leg.Tag.Virtual
        t.legs[0].name = 'a'
        t.legs[1].name = 'b'
        t.legs[2].name = 'c'
        tensors.append(t)
        t.name = "A"

        t.draw()

        legs1 = t.get_legs_with('tag', Leg.Tag.Virtual)
        
    
    print(tensors)


if __name__ == "__main__":
    # _simple_test()
    _main_example()
    print("Done.")