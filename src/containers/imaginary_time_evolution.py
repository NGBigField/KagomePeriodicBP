# For allowing 'Self' import
import numpy as np
from physics.hamiltonians import zero
from dataclasses import dataclass, field, fields
from utils import strings, saveload, assertions, lists
from utils.arguments import Stats
from copy import deepcopy

# For type hinting:
from typing import Generator, NamedTuple, Callable, TypeVar, Generic, Iterable, Any, TypeAlias, Union, List, Final

# Other containers and enums:
from containers._meta import _ConfigClass
from containers.density_matrices import MatrixMetrics 
from enums.imaginary_time_evolution import UpdateMode
from enums.tensor_networks import UnitCellFlavor
from containers.belief_propagation import BPStats
from lattices.directions import LatticeDirection


# For smart iterations:
import itertools

_HamilInputType = TypeVar("_HamilInputType")
_HamilInputRuleType : TypeAlias = Callable[[Any], _HamilInputType]
NUM_EDGES_PER_MODE : int = 6

def _Identity_function(x):
    return x

A = UnitCellFlavor.A
B = UnitCellFlavor.B
C = UnitCellFlavor.C

_NEXT_IN_ABC_ORDER = {
    A : B,
    B : C,
    C : A
}


def _optional_arguments_for_hamiltonian_function(args:_HamilInputType, rule:_HamilInputRuleType, **kwargs)->_HamilInputType|None:
    if isinstance(args, str) and args=="delta_t" and "delta_t" in kwargs:
        args = kwargs["delta_t"]
    if rule is not None:
        args = rule(args)
    return args


@dataclass
class HamiltonianFuncAndInputs(Generic[_HamilInputType]):
    func: Callable[[_HamilInputType], np.ndarray] 
    args: _HamilInputType
    args_rule: _HamilInputRuleType     

    def __repr__(self) -> str:
        return f"Hamiltonian function {self.func.__name__!r} with arguments: {self.args}"

    @staticmethod
    def default()->"HamiltonianFuncAndInputs":
        return HamiltonianFuncAndInputs(func=zero, args=None, args_rule=None)  # type:ignore

    @staticmethod
    def standard(self_or_tuple)->"HamiltonianFuncAndInputs":
        if isinstance(self_or_tuple, HamiltonianFuncAndInputs):
            pass
        elif isinstance(self_or_tuple, tuple):
            assert len(self_or_tuple)==3
            self_or_tuple = HamiltonianFuncAndInputs(func=self_or_tuple[0], args=self_or_tuple[1], args_rule=self_or_tuple[2])
        else:
            raise TypeError("Not a supported type")
        
        assert callable(self_or_tuple.func)
        return self_or_tuple
    

    def call(self, **kwargs)->np.ndarray:
        ## Unpack:
        func = self.func
        args = self.args
        rule = self.args_rule

        # choose input for hamiltonian function
        args = _optional_arguments_for_hamiltonian_function(args, rule, **kwargs) 

        # call:
        if args is None:
            return func()  # type:ignore
        elif isinstance(args, Iterable):
            return func(*args)
        else:
            return func(args)
            

class UpdateEdge(NamedTuple): 
    first : UnitCellFlavor
    second : UnitCellFlavor

    def is_in_core(self)->bool:  
        """Checking if the edge is part of the 3 edges in-core, is the same as checking if the first-second items are ordered in the 'ABCA' order
        Returns:
            bool
        """
        return self.second is _NEXT_IN_ABC_ORDER[self.first]
    
    def first_to_second_direction(self) -> LatticeDirection:
        if self.first == A:
                if   self.second == B:    return LatticeDirection.DL 
                elif self.second == C:    return LatticeDirection.UL 
                        
        elif self.first == B:
                if   self.second == A:    return LatticeDirection.DL
                elif self.second == C:    return LatticeDirection.R

        elif self.first == C:
                if   self.second == A:    return LatticeDirection.UL
                elif self.second == B:    return LatticeDirection.R

        raise ValueError("Not an expected case")

    
    def __repr__(self) -> str:
        return UpdateEdge.to_str(self)
    
    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def all_options()->Generator["UpdateEdge", None, None]:
        flavors = [A, B, C]
        for a, b in itertools.permutations(flavors, 2):
            yield UpdateEdge(a, b)

    @staticmethod
    def all_in_random_order(num_edges:int=NUM_EDGES_PER_MODE)->Generator["UpdateEdge", None, None]:
        random_order = lists.shuffle(list(UpdateEdge.all_options()))
        if num_edges != NUM_EDGES_PER_MODE:
            random_order = lists.repeat_list(random_order, num_items=num_edges)
        return (mode for mode in random_order)

    @staticmethod
    def to_str(edge_tuple: Union["UpdateEdge", tuple[str, str], str])->str:
        if isinstance(edge_tuple, UpdateEdge):
            return f"({edge_tuple.first}, {edge_tuple.second})"
        elif isinstance(edge_tuple, tuple):
            return f"({edge_tuple[0]}, {edge_tuple[1]})"
        elif isinstance(edge_tuple, str):
            return edge_tuple
        else:
            raise TypeError("Not an expected type")


DEFAULT_TIME_STEPS = lambda: [0.02]*5 + [0.01]*5 + [0.001]*100 + [1e-4]*100 + [1e-5]*100 + [1e-6]*100 + [1e-7]*100 + \
    [1e-8]*100 + [1e-9]*100 + [1e-10]*100 + [1e-11]*100 + [1e-12]*100 + [1e-13]*100 + [1e-15]*200

@dataclass
class IterativeProcessConfig(_ConfigClass):
    # Belief-Propagation flags:
    use_bp : bool = True  # Controls if we use block-belief-propagation or not
    start_segment_with_new_bp_message : bool = True
    bp_not_converged_raises_error : bool = False
    bp_every_edge : bool = True
    # Control numbers:
    num_total_errors_threshold : int = 10    
    num_errors_per_delta_t_threshold : int = 2    
    segment_error_cause_state_revert : bool = False   #TODO Implement when `True`
    keep_harder_bp_config_between_segments : bool = False
    randomly_rotate_unit_cell_between_segments : bool = True
    # Measure expectation and energies:
    num_mode_repetitions_per_segment : int = 5  # number of modes between each measurement of energy
    num_edge_repetitions_per_mode : int = 6  # number of edges before new segment
    change_config_for_measurements_func : Callable = _Identity_function

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class ITEConfig(_ConfigClass):
    # hamiltonian:
    _interaction_hamiltonian : HamiltonianFuncAndInputs = field(default_factory=HamiltonianFuncAndInputs.default)
    # ITE time steps:
    _time_steps : list[float] = field(default_factory=DEFAULT_TIME_STEPS)
    # Control flags:
    random_edge_order : bool = True
    random_mode_order : bool = True
    always_use_lowest_energy_state : bool = False
    check_converges : bool = False  # If several steps didn't improve the lowest energy, go to next delta_t
    # After update flags:
    normalize_tensors_after_update : bool = True
    force_hermitian_tensors_after_update : bool = True
    # Optimization params:
    add_gaussian_noise_fraction : float|None = None
    # Hamiltonian commutation constraints:
    symmetric_second_order_trotterization : bool = True

    @property
    def reference_ground_energy(self)->float|None:  
        """Ground truth energy, if known
        """
        func = self._interaction_hamiltonian.func
        if hasattr(func, "reference"):
            return func.reference

    def __repr__(self) -> str:
        obj = self
        s = f"{obj.__class__.__name__}:"
        for field in fields(obj):
            s += "\n"
            value = getattr(obj, field.name)
            if isinstance(value, np.ndarray):
                value_str = f"ndarray of shape {value.shape}"
            elif isinstance(value, list) and field.name == "_time_steps":
                value_str = _time_steps_str(value)
            else:
                value_str = str(value)
            s += f"    {field.name}: {value_str}"
        return s
    
    def __post_init__(self)->None:
        self._interaction_hamiltonian = HamiltonianFuncAndInputs.standard(self._interaction_hamiltonian)
        self._time_steps = _fix_and_rearrange_time_steps(self._time_steps)

    ## ================== Getters and Setters: ================== ##        
    @property
    def interaction_hamiltonian(self)->HamiltonianFuncAndInputs:
        return self._interaction_hamiltonian
    @interaction_hamiltonian.setter
    def interaction_hamiltonian(self, value:HamiltonianFuncAndInputs)->None:
        self._interaction_hamiltonian = HamiltonianFuncAndInputs.standard(value)

    @property
    def time_steps(self)->list[float]:
        return self._time_steps
    @time_steps.setter
    def time_steps(self, value:list[float]|list[list[float]])->None:
        self._time_steps = _fix_and_rearrange_time_steps(value)
    
    

class ITEPerModeStats(Stats):
    bp_stats : list[BPStats] 
    env_metrics : list[MatrixMetrics]

    def __post_init__(self):
        self.bp_stats = list()
        self.env_metrics = list()  # Avoid python's infamous immutable lists problem

        
class ITESegmentStats(Stats):
    ite_per_mode_stats : list[ITEPerModeStats] = None # type: ignore
    modes_order : list[UpdateMode] = None  # type: ignore
    delta_t : float = None  # type: ignore
    mean_energy : float = None  # type: ignore
    had_to_revert : bool = False

    def __post_init__(self):
        self.ite_per_mode_stats = list()  # Avoid python's infamous immutable lists problem


EPSILON = 1e-5
def _close_to(x:float, y:float) -> bool:
    relation = abs(x-y)/x
    return relation < EPSILON

def _formatted_delta_t_str(delta_t:float) -> str:
    ## Try exponent notation:
    for e in range(1, 20):
        for sign in [-1, +1]:
            val = 10**(sign*e)
            if _close_to(val, delta_t):
                return f"{val}"

def _time_steps_str(time_steps:list[float])->str:
    s = ""
    last = None
    counter = 0
    for dt in time_steps:
        if dt==last:
            counter+=1
        else:
            if last is not None:
                s += "["+_formatted_delta_t_str(last)+"]*"+f"{counter} + "
            last = dt
            counter = 1  

    assert last is not None  
    s += "["+_formatted_delta_t_str(last)+"]*"+f"{counter}"
    return s



def _fix_and_rearrange_time_steps(times:list[float]|list[list[float]])->list[float]:
    if isinstance(times, list):
        return lists.join_sub_lists(times)  # type: ignore
    else:
        raise TypeError(f"Not an expected type of argument `times`. It has type {type(times)}")
    

UpdateEdgesOrder : TypeAlias = list[tuple[UpdateEdge, float]]  # contains update-edge and delta_t per edge