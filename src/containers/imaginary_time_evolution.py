# For allowing 'Self' import
import numpy as np
from physics.hamiltonians import zero
from dataclasses import dataclass, field, fields
from utils import strings, saveload, assertions, lists
from utils.arguments import Stats
from copy import deepcopy

# For type hinting:
from typing import Generator, NamedTuple, Callable, TypeVar, Generic, Iterable, Any
_T = TypeVar("_T")

# Other containers and enums:
from enums.imaginary_time_evolution import UpdateMode
from enums.tensor_networks import UnitCellFlavor
from containers.belief_propagation import BPStats
from containers.density_matrices import MatrixMetrics 
from tensor_networks import UnitCell
from _error_types import ITEError

# For smart iterations:
import itertools


DEFAULT_ITE_TRACKER_MEMORY_LENGTH : int = 5


_NEXT_IN_ABC_ORDER = {
    UnitCellFlavor.A : UnitCellFlavor.B,
    UnitCellFlavor.B : UnitCellFlavor.C,
    UnitCellFlavor.C : UnitCellFlavor.A
}



class _LimitedLengthList(Generic[_T]):
    __slots__ = ("length", "_list")

    def __init__(self, length:int) -> None:
        self.length : int = length
        self._list : list[_T] = []

    def pop_oldest(self)->_T:
        return self.pop(0)
    
    def pop(self, index:int|None=None)->_T:
        if index is None:
            return self._list.pop()
        else:
            return self._list.pop(index)

    def append(self, item:_T)->None:
        self._list.append(item)
        if len(self._list)>self.length:
            self.pop_oldest()

    def __setitem__(self, key, value)->None:
        return self._list.__setitem__(key, value)
    
    def __getitem__(self, key)->_T:
        return self._list.__getitem__(key)
    
    def __delitem__(self, key)->_T:
        return self._list.__delitem__(key)



class HamiltonianFuncAndInputs(NamedTuple):
    func: Callable[[_T], np.ndarray] 
    args: _T|tuple[_T]|None

    def __repr__(self) -> str:
        return f"Hamiltonian function {self.func.__name__!r} with arguments: {self.args}"

    @staticmethod
    def default()->"HamiltonianFuncAndInputs":
        return HamiltonianFuncAndInputs(func=zero, args=None)

    @staticmethod
    def standard(self_or_tuple)->"HamiltonianFuncAndInputs":
        assert len(self_or_tuple)==2
        assert callable(self_or_tuple[0])
        if isinstance(self_or_tuple, HamiltonianFuncAndInputs):
            assert callable(self_or_tuple.func)
            return self_or_tuple
        
        if isinstance(self_or_tuple, tuple):
            return HamiltonianFuncAndInputs(func=self_or_tuple[0], args=self_or_tuple[1])

        raise ValueError(f"Not a valid input for class {HamiltonianFuncAndInputs.__name__!r}") 

    def call(self)->np.ndarray:
        assert len(self)==2
        func   = self.func
        args   = self.args

        # call:
        if args is None:
            return func()
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
    
    def __str__(self) -> str:
        return f"({self.first}, {self.second})"
    
    @property
    def as_strings(self)->tuple[str, str]:
        return (self.first.name, self.second.name)

    @staticmethod
    def all_options()->Generator["UpdateEdge", None, None]:
        flavors = [UnitCellFlavor.A, UnitCellFlavor.B, UnitCellFlavor.C]
        for a, b in itertools.permutations(flavors, 2):
            yield UpdateEdge(a, b)

    @staticmethod
    def all_in_random_order()->Generator["UpdateEdge", None, None]:
        random_order = lists.shuffle(list(UpdateEdge.all_options()))
        return (mode for mode in random_order)



SUB_FOLDER = "ite_trackers"
DEFAULT_TIME_STEPS = lambda:  [0.2]*5 + [0.1]*30 + [0.01]*10 + [0.001]*10 + \
    [0.01]*100 + [0.001]*100 + [1e-4]*100 + [1e-5]*100 + [1e-6]*100 + [1e-7]*100 + \
    [1e-8]*100 + [1e-9]*100 + [1e-10]*100 + [1e-11]*100 + [1e-12]*100 + [1e-13]*100 + [1e-15]*200

# def DEFAULT_TIME_STEPS()->list[float]:
#     dts = []
#     for exp in [-1, -2, -3, -4]:
#         for mantissa in [5, 2, 1]:
#             if exp==-1 and mantissa in [5, 2]:
#                 continue
#             elif exp==-1 and mantissa==1:
#                 repeats = 20
#             else:
#                 repeats = 50
#             dts += [mantissa*10**exp]*repeats
#     return dts



@dataclass
class ITEConfig():
    # hamiltonian:
    interaction_hamiltonian : HamiltonianFuncAndInputs = field(default_factory=HamiltonianFuncAndInputs.default)
    # ITE time steps:
    time_steps : list[float] = field(default_factory=DEFAULT_TIME_STEPS)
    num_mode_repetitions_per_segment : int = 1        
    # File:
    backup_file_name : str = "ite_backup"+strings.time_stamp()+" "+strings.random(6)
    # Control flags:
    random_mode_order : bool = True
    start_segment_with_new_bp_message : bool = True
    check_converges : bool = False  # If several steps didn't improve the lowest energy, go to next delta_t
    segment_error_cause_state_revert : bool = False
    # Control numbers:
    num_errors_threshold : int = 10    
    # Belief-Propagation on full tn:
    bp_not_converged_raises_error : bool = True
    bp_every_edge : bool = True


    @property
    def reference_ground_energy(self)->float|None:  
        """Ground truth energy, if known
        """
        func = self.interaction_hamiltonian.func
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
            elif isinstance(value, list) and field.name == "time_steps":
                value_str = _time_steps_str(value)
            else:
                value_str = str(value)
            s += f"    {field.name}: {value_str}"
        return s
    
    def __post_init__(self)->None:
        self.interaction_hamiltonian = HamiltonianFuncAndInputs.standard(self.interaction_hamiltonian)
    

class ITEPerModeStats(Stats):
    bp_stats : BPStats = BPStats()
    env_metrics : list[MatrixMetrics] = field(default_factory=list)
    

class ITESegmentStats(Stats):
    ite_per_mode_stats : list[ITEPerModeStats] = None # type: ignore
    modes_order : list[UpdateMode] = None  # type: ignore
    delta_t : float = None  # type: ignore
    mean_energy : float = None

    def __post_init__(self):
        self.ite_per_mode_stats = list()  # Avoid python's infamous immutable lists problem



                        
_TrackerStepOutputs = tuple[
    float,              # delta_t
    complex|None,       # energy    
    ITESegmentStats,    # env_hermicity      
    dict,               # expectation-value
    UnitCell,           # core
    dict                # messages
]  


class ITEProgressTracker():

    def __init__(self, unit_cell:UnitCell, messages:dict|None, config:Any, mem_length:int=DEFAULT_ITE_TRACKER_MEMORY_LENGTH):
        # From input:
        self.last_unit_cell : UnitCell = unit_cell.copy()
        self.last_messages : dict = deepcopy(messages)  #type: ignore
        self.config = config
        # Fresh variables:
        self.last_iter : int = 0
        self.error_counters : dict[type, int] = {}
        self.file_name : str = "ite-tracker_"+strings.time_stamp()+" "+strings.random(5)
        # Lists memory:
        self.delta_ts           : _LimitedLengthList[float]             = _LimitedLengthList[float](mem_length) 
        self.energies           : _LimitedLengthList[complex|None]      = _LimitedLengthList[complex|None](mem_length)     
        self.expectation_values : _LimitedLengthList[dict]              = _LimitedLengthList[dict](mem_length) 
        self.unit_cells         : _LimitedLengthList[UnitCell]          = _LimitedLengthList[UnitCell](mem_length) 
        self.messages           : _LimitedLengthList[dict]              = _LimitedLengthList[dict](mem_length) 
        self.stats              : _LimitedLengthList[ITESegmentStats]   = _LimitedLengthList[ITESegmentStats](mem_length)            
    
    def log_segment(self, delta_t:float, unit_cell:UnitCell, messages:dict, expectation_values:dict, energy:complex, stats:ITESegmentStats )->None:
        # Get a solid copy
        _unit_cell = unit_cell.copy()
        messages = deepcopy(messages)
        expectation_values = deepcopy(expectation_values)
        ## Lists:
        self.delta_ts.append(delta_t)
        self.energies.append(energy)
        self.unit_cells.append(_unit_cell)
        self.messages.append(messages)
        self.expectation_values.append(expectation_values)
        self.stats.append(stats)
        ## Up_to_date memory:
        self.last_unit_cell = _unit_cell
        self.last_messages = messages
        self.last_iter += 1
        ## Save to file
        self.save()
    
    def log_error(self, error:Exception)->int:
        """Log the error and return the number of errors so-far

        Args:
            error (Exception)

        Returns:
            int: number of errors logged.
        """
        # Log:
        e_type = type(error)
        if e_type in self.error_counters:
            self.error_counters[e_type] += 1
        else:
            self.error_counters[e_type] = 1
        # Sum:
        sum_ = 0
        for error_type, count in self.error_counters.items():
            sum_ += count
        return sum_
        
    def revert_back(self, num_iter:int=1)->_TrackerStepOutputs:
        # Check
        num_iter = assertions.integer(num_iter)
        assert num_iter >= 1
        if self.last_iter < num_iter:
            raise ITEError(f"There are no more saved results to revert to.")
        # lists:
        for _ in range(num_iter):
            delta_t = self.delta_ts.pop()
            energy = self.energies.pop()
            core = self.unit_cells.pop()
            messages = self.messages.pop()
            expectation_values = self.expectation_values.pop()
            step_stats = self.stats.pop()
        # Up to date memory:
        self.last_unit_cell = core  #type: ignore
        self.last_messages = messages  #type: ignore
        self.last_iter -= num_iter
        # Return:
        return delta_t, energy, step_stats, expectation_values, core, messages  #type: ignore


    @staticmethod
    def load(file_name)->"ITEProgressTracker":
        ite_tracker = saveload.load(file_name, sub_folder=SUB_FOLDER, if_exist=True)        
        return ite_tracker

    def save(self)->None:
        return saveload.save(self, self.file_name, sub_folder=SUB_FOLDER)        

    @property
    def full_path(self) -> str:
        return saveload._fullpath(name=self.file_name, sub_folder=SUB_FOLDER)
        


def _time_steps_str(time_steps:list[float])->str:
    s = ""
    last = None
    counter = 0
    for dt in time_steps:
        if dt==last:
            counter+=1
        else:
            if last is not None:
                s += "["+f"{last}"+"]*"+f"{counter} + "
            last = dt
            counter = 1    
    s += "["+f"{last}"+"]*"+f"{counter}"
    return s