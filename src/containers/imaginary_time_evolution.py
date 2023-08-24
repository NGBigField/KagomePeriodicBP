import numpy as np
from physics.hamiltonians import HamiltonianFuncType, zero
from dataclasses import dataclass, field, fields
from utils import strings, saveload, assertions
from utils.arguments import Stats
from typing import Generator, Union, Any, TypeAlias, NamedTuple
from copy import deepcopy

# Other containers and enums:
from enums.imaginary_time_evolution import UpdateMode
from enums.tensor_networks import UnitCellFlavor
from containers.belief_propagation import BPStats
from containers.density_matrices import MatrixMetrics 
from _error_types import ITEError

# For smart iterations:
import itertools


KagomeTensorNetwork = None  #TODO fix


_NEXT_IN_ABC_ORDER = {
    UnitCellFlavor.A : UnitCellFlavor.B,
    UnitCellFlavor.B : UnitCellFlavor.C,
    UnitCellFlavor.C : UnitCellFlavor.A
}

class UpdateEdge(NamedTuple): 
    first : UnitCellFlavor
    second : UnitCellFlavor

    def is_in_core(self)->bool:  
        """Checking if the edge is part of the 3 edges in-core, is the same as checking if the first-second items are ordered in the 'ABCA' order
        Returns:
            bool
        """
        return self.second is _NEXT_IN_ABC_ORDER[self.first]
    
    @staticmethod
    def all_options()->Generator["UpdateEdge", None, None]:
        flavors = [UnitCellFlavor.A, UnitCellFlavor.B, UnitCellFlavor.C]
        for a, b in itertools.combinations(flavors, 2):
            yield (a, b)



SUB_FOLDER = "ite_trackers"
# DEFAULT_TIME_STEPS = lambda:  [0.1]*20 + [0.01]*50 + [0.001]*50 + [0.0001]*50
def DEFAULT_TIME_STEPS()->list[float]:
    dts = []
    for exp in [-1, -2, -3, -4]:
        for mantissa in [5, 2, 1]:
            if exp==-1 and mantissa in [5, 2]:
                continue
            elif exp==-1 and mantissa==1:
                repeats = 20
            else:
                repeats = 50
            dts += [mantissa*10**exp]*repeats
    return dts



@dataclass
class ITEConfig():
    # hamiltonian:
    interaction_hamiltonain : np.ndarray = field(default_factory=zero)
    _GT_energy : float|None = None  # Ground truth energy, if known
    # ITE time steps:
    time_steps : list[float] = field(default_factory=DEFAULT_TIME_STEPS)
    num_mode_repetitions_per_segment : int = 1    
    # File:
    backup_file_name : str = "ite_backup"+strings.time_stamp()+" "+strings.random(6)
    # Control flags:
    random_mode_order : bool = True
    start_segment_with_new_bp_message : bool = True
    bp_not_converged_raises_error : bool = True
    check_converges : bool = False  # If sevral steps didn't improve the lowest energy, go to next delta_t
    segment_error_cause_state_revert : bool = False
    # Control numbers:
    num_errors_threshold : int = 10


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
    

class ITEPerModeStats(Stats):
    bp_stats : BPStats = BPStats()
    env_metrics : MatrixMetrics 
    

class ITESegmentStats(Stats):
    ite_per_mode_stats : list[ITEPerModeStats] = None # type: ignore
    modes_order : list[UpdateMode] = None  # type: ignore
    delta_t : float = None  # type: ignore

    def __post_init__(self):
        self.ite_per_mode_stats = list()  # Avoid python's infamous immutable lists problem




_TrackerStepOutputs = tuple[float, complex|None, ITESegmentStats, dict, KagomeTensorNetwork, dict]  # delta_t, energy, env_hermicity, expectation, value, core, messages


class ITEProgressTracker():

    def __init__(self, core:KagomeTensorNetwork, messages:dict|None, config:Any):
        # From input:
        self.last_core : KagomeTensorNetwork = core.copy()
        self.last_messages : dict = deepcopy(messages)  #type: ignore
        self.config = config
        # Fresh variables:
        self.last_iter : int = 0
        self.error_counters : dict[type, int] = {}
        self.file_name : str = "ite-tracker_"+strings.time_stamp()+" "+strings.random(5)
        # Lists memory:
        self.delta_ts : list[float] = []
        self.energies : list[complex|None] = []
        self.expectation_values : list[dict] = []
        self.cores : list[KagomeTensorNetwork] = []
        self.messages : list[dict] = []
        self.stats : list[ITESegmentStats] = []
    
    def log_segment(self, delta_t:float, core:KagomeTensorNetwork, messages:dict, expectation_values:dict, energy:complex, stats:ITESegmentStats )->None:
        # Get a solid copy
        _core = core.copy()
        messages = deepcopy(messages)
        expectation_values = deepcopy(expectation_values)
        ## Lists:
        self.delta_ts.append(delta_t)
        self.energies.append(energy)
        self.cores.append(_core)
        self.messages.append(messages)
        self.expectation_values.append(expectation_values)
        self.stats.append(stats)
        ## Up_to_date memory:
        self.last_core = _core
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
            core = self.cores.pop()
            messages = self.messages.pop()
            exepectation_values = self.expectation_values.pop()
            step_stats = self.stats.pop()
        # Up to date memory:
        self.last_core = core  #type: ignore
        self.last_messages = messages  #type: ignore
        self.last_iter -= num_iter
        # Return:
        return delta_t, energy, step_stats, exepectation_values, core, messages  #type: ignore


    @staticmethod
    def load(file_name)->"ITEProgressTracker":
        ite_tracker = saveload.load(file_name, sub_folder=SUB_FOLDER, if_exist=True)        
        return ite_tracker

    def save(self)->None:
        saveload.save(self, self.file_name, sub_folder=SUB_FOLDER)        

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