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
from containers.results import MeasurementsOnUnitCell
from enums.imaginary_time_evolution import UpdateMode
from enums.tensor_networks import UnitCellFlavor
from containers.belief_propagation import BPStats
from tensor_networks import UnitCell
from _error_types import ITEError


# For smart iterations:
import itertools

_T = TypeVar("_T")


DEFAULT_ITE_TRACKER_MEMORY_LENGTH : int = 10
_HamilInputType = TypeVar("_HamilInputType")
_HamilInputRuleType : TypeAlias = Callable[[Any], _HamilInputType]
NUM_EDGES_PER_MODE : int = 6

def _Identity_function(x):
    return x


_NEXT_IN_ABC_ORDER = {
    UnitCellFlavor.A : UnitCellFlavor.B,
    UnitCellFlavor.B : UnitCellFlavor.C,
    UnitCellFlavor.C : UnitCellFlavor.A
}



class _LimitedLengthList(List[_T]):
    __slots__ = ("length_limit", "_list")

    def __init__(self, length:int) -> None:
        self.length_limit : int = length
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
        if len(self._list)>self.length_limit:
            self.pop_oldest()

    def __setitem__(self, key, value) -> None:
        return self._list.__setitem__(key, value)
    
    def __getitem__(self, key) -> _T:
        return self._list.__getitem__(key)
    
    def __delitem__(self, key) -> None:
        return self._list.__delitem__(key)

    def __len__(self)->int:
        return len(self._list)


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
    
    def __repr__(self) -> str:
        return UpdateEdge.to_str(self)
    
    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def all_options()->Generator["UpdateEdge", None, None]:
        flavors = [UnitCellFlavor.A, UnitCellFlavor.B, UnitCellFlavor.C]
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


SUB_FOLDER : Final[str] = "ite_trackers"
DEFAULT_TIME_STEPS = lambda: [0.02]*5 + [0.01]*5 + [0.001]*100 + [1e-4]*100 + [1e-5]*100 + [1e-6]*100 + [1e-7]*100 + \
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
class IterativeProcessConfig(_ConfigClass):
    # Belief-Propagation flags:
    start_segment_with_new_bp_message : bool = True
    bp_not_converged_raises_error : bool = False
    bp_every_edge : bool = True
    # Control numbers:
    use_bp : bool = True  # Controls if we use block-belief-propagation or not
    num_total_errors_threshold : int = 20    
    num_errors_per_delta_t_threshold : int = 5    
    segment_error_cause_state_revert : bool = False    
    keep_harder_bp_config_between_segments : bool = False
    randomly_rotate_unit_cell_between_segments : bool = True
    # Measure expectation and energies:
    num_mode_repetitions_per_segment : int = 1  # number of modes between each measurement of energy
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
    symmetric_product_formula : bool = True

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



                        
_TrackerStepOutputs : TypeAlias = tuple[
    float,              # delta_t
    complex|None,       # energy    
    ITESegmentStats,    # env_hermicity      
    dict,               # expectation-value
    UnitCell,           # core
    dict                # messages
]  


class ITEProgressTracker():

    def __init__(self, unit_cell:UnitCell, messages:dict|None, config:Any, mem_length:int=DEFAULT_ITE_TRACKER_MEMORY_LENGTH, filename:str|None=None):
        # From input:
        self.last_unit_cell : UnitCell = unit_cell.copy()
        self.last_messages : dict = deepcopy(messages)  #type: ignore
        self.config = config
        # Fresh variables:
        self.last_iter : int = 0
        self.error_counters : dict[type, int] = {}
        self.file_name : str = filename if filename is not None else "ite-tracker_"+strings.time_stamp()+" "+strings.random(5)
        # Lists memory:
        self.delta_ts           : _LimitedLengthList[float]             = _LimitedLengthList[float](mem_length) 
        self.energies           : _LimitedLengthList[float]             = _LimitedLengthList[float](mem_length)     
        self.expectation_values : _LimitedLengthList[dict]              = _LimitedLengthList[dict](mem_length) 
        self.unit_cells         : _LimitedLengthList[UnitCell]          = _LimitedLengthList[UnitCell](mem_length) 
        self.messages           : _LimitedLengthList[dict]              = _LimitedLengthList[dict](mem_length) 
        self.stats              : _LimitedLengthList[ITESegmentStats]   = _LimitedLengthList[ITESegmentStats](mem_length)            

    @property
    def memory_usage(self)->int:
        return saveload.get_size(self.file_name, sub_folder=SUB_FOLDER)        
    
    @property
    def full_path(self) -> str:
        return saveload._fullpath(name=self.file_name, sub_folder=SUB_FOLDER)

    def log_segment(self, delta_t:float, unit_cell:UnitCell, messages:dict|None, measurements:MeasurementsOnUnitCell, stats:ITESegmentStats )->None:
        # Get a solid copy
        _unit_cell = unit_cell.copy()
        messages = deepcopy(messages)
        expectation_values = deepcopy(measurements.expectations)
        ## Lists:
        self.delta_ts.append(delta_t)
        self.energies.append(measurements.mean_energy)
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
        if self.last_iter <= 0:
            raise ITEError("ITE Tracker is empty.")
        if self.last_iter < num_iter:
            num_iter = self.last_iter  # There are no more saved results to revert to 
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
        step_stats.had_to_revert = True
        return delta_t, energy, step_stats, expectation_values, core, messages  #type: ignore


    @staticmethod
    def load(file_name)->"ITEProgressTracker":
        ite_tracker = saveload.load(file_name, sub_folder=SUB_FOLDER, none_if_not_exist=True)        
        return ite_tracker

    def save(self)->str:
        # Save:
        filename : str = self.file_name
        sub_folder : str = SUB_FOLDER
        return saveload.save(self, name=filename, sub_folder=sub_folder)        

    def plot(self, live_plot:bool=False)->None:
        # Specific plotting imports:
        from visualizations.ite import ITEPlots
        from utils import visuals
        from containers import Config

        config : Config = deepcopy(self.config)
        config.visuals.live_plots = [1, 1, 1]

        plots = ITEPlots(active=True, config=config)
        for delta_t, energy, unit_cell, msg, expectations, stats \
            in zip(self.delta_ts, self.energies, self.unit_cells, self.messages, self.expectation_values, self.stats ):

            raise NotImplementedError("Need to fix this call") #TODO fic
            plots.update(energies=energy, segment_stats=stats, delta_t=delta_t, expectations=expectations, unit_cell=unit_cell, _draw_now=live_plot)            
        
        visuals.draw_now()
        print("Done plotting ITE.")

            
        
    def __len__(self)->int:
        assert len(self.delta_ts) == len(self.energies) == len(self.expectation_values) ==  len(self.unit_cells) == len(self.messages) == len(self.stats)             
        return len(self.delta_ts)

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



def _fix_and_rearrange_time_steps(times:list[float]|list[list[float]])->list[float]:
    if isinstance(times, list):
        return lists.join_sub_lists(times)  # type: ignore
    else:
        raise TypeError(f"Not an expected type of argument `times`. It has type {type(times)}")