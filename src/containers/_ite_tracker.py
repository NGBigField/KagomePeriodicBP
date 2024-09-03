from tensor_networks import UnitCell
from typing import TypeAlias, Final, List, TypeVar, Any
from copy import deepcopy
from utils import strings, assertions, saveload

from containers.results import MeasurementsOnUnitCell
from _error_types import ITEError
from .imaginary_time_evolution import ITESegmentStats


SUB_FOLDER_NAME : Final[str] = "ite_trackers"
_T = TypeVar("_T")
                        
_TrackerStepOutputs : TypeAlias = tuple[
    float,              # delta_t
    complex|None,       # energy    
    ITESegmentStats,    # env_hermicity      
    dict,               # expectation-value
    UnitCell,           # core
    dict                # messages
]  



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


class ITEProgressTracker():
    __slots__ = "config", "last_iter", "fullpath", "delta_ts", "energies", "expectation_values", "unit_cells", "messages", "stats", "error_counters"

    def __init__(self, unit_cell:UnitCell, messages:dict|None, config:Any, filename:str|None=None):
        # Derive:
        mem_length = config.iterative_process.num_total_errors_threshold
        file_name = filename if filename is not None else "ite-tracker_"+strings.time_stamp()+" "+strings.random(5)
        folder_path = config.io.data.subfolder("ite_trackers")
        # From input:
        self.config = config
        # Fresh variables:
        self.last_iter : int = 0
        self.error_counters : dict[type, int] = {}
        self.fullpath : str = folder_path + saveload.PATH_SEP + file_name
        # Lists memory:
        self.delta_ts           : _LimitedLengthList[float]             = _LimitedLengthList[float](mem_length) 
        self.energies           : _LimitedLengthList[float]             = _LimitedLengthList[float](mem_length)     
        self.expectation_values : _LimitedLengthList[dict]              = _LimitedLengthList[dict](mem_length) 
        self.unit_cells         : _LimitedLengthList[UnitCell]          = _LimitedLengthList[UnitCell](mem_length) 
        self.messages           : _LimitedLengthList[dict]              = _LimitedLengthList[dict](mem_length) 
        self.stats              : _LimitedLengthList[ITESegmentStats]   = _LimitedLengthList[ITESegmentStats](mem_length)            

        ## First lists values:
        self.delta_ts.append(None)              #type: ignore
        self.energies.append(None)              #type: ignore
        self.expectation_values.append(None)    #type: ignore
        self.unit_cells.append(unit_cell)       #type: ignore
        self.messages.append(messages)          #type: ignore
        self.stats.append(None)                 #type: ignore

    @property
    def last_unit_cell(self) ->UnitCell:
        return self.unit_cells[-1]
    
    @property
    def last_messages(self) -> dict:
        return self.messages[-1]


    def log_segment(self, delta_t:float, unit_cell:UnitCell, messages:dict|None, measurements:MeasurementsOnUnitCell, stats:ITESegmentStats )->None:
        # Get a solid copy
        _unit_cell = unit_cell.copy()
        messages = deepcopy(messages)
        expectation_values = deepcopy(measurements.expectations)
        ## Lists:
        self.delta_ts.append(delta_t)
        self.energies.append(measurements.mean_energy)
        self.unit_cells.append(_unit_cell)
        self.messages.append(messages)  #type: ignore
        self.expectation_values.append(expectation_values)
        self.stats.append(stats)
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
    def load(filename, folder:str|None=None)->"ITEProgressTracker":
        if folder is None:
            ite_tracker = saveload.load(filename, sub_folder=SUB_FOLDER_NAME, none_if_not_exist=True)        
        else:
            fullpath = folder + saveload.PATH_SEP + filename
            mode = saveload.Modes.Read
            ite_tracker = saveload.save_or_load_with_fullpath(fullpath, 'load')
        return ite_tracker

    def save(self)->str:
        return saveload.save_or_load_with_fullpath(self.fullpath, 'save', self)

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
