from dataclasses import dataclass, fields, field
from containers.contractions import BubbleConGlobalConfig, BubbleconContractionConfig
from containers.belief_propagation import BPConfig
from containers.sizes_and_dimensions import TNDimensions
from containers.imaginary_time_evolution import ITEConfig, IterativeProcessConfig
from containers.visuals import VisualsConfig
from containers.performance import MonitoringSystemConfig
from containers.io import IOConfig
from containers._meta import _ConfigClass
from utils import prints, sizes
from copy import deepcopy

# Control flags:
from _config_reader import DEBUG_MODE, ALLOW_VISUALS


class _StoreConfigClasses:
    # For easier reach of these needed config data classes:
    BPConfig      = BPConfig
    TNDimensions  = TNDimensions
    ITEConfig     = ITEConfig
    IterativeProcessConfig = IterativeProcessConfig
    VisualsConfig = VisualsConfig
    ContractionConfig = BubbleconContractionConfig


@dataclass
class Config(_StoreConfigClasses, _ConfigClass): 
    # The actual stored data:
    bp : BPConfig                               = field(default_factory=BPConfig) 
    ite : ITEConfig                             = field(default_factory=ITEConfig) 
    iterative_process : IterativeProcessConfig  = field(default_factory=IterativeProcessConfig)
    dims : TNDimensions                         = field(default_factory=TNDimensions)
    visuals : VisualsConfig                     = field(default_factory=VisualsConfig)
    contraction : BubbleconContractionConfig    = field(default_factory=BubbleconContractionConfig)  
    monitoring_system : MonitoringSystemConfig  = field(default_factory=MonitoringSystemConfig)  
    io : IOConfig                               = field(default_factory=IOConfig)  

    @staticmethod
    def derive_from_dimensions(D:int)->"Config":
        config = Config()
        config.dims=TNDimensions(virtual_dim=D)
        config.bp=BPConfig(trunc_dim=2*D**2)
        config.contraction=BubbleconContractionConfig(trunc_dim=2*D**2+10)
        # if D>3:
        #     config.bp.trunc_dim = D**2
        #     config.trunc_dim = 2*D**2
        return config

    @property
    def chi(self) -> int:
        return self.contraction.trunc_dim
    @chi.setter
    def chi(self, value) -> None:
        self.contraction.trunc_dim = int(value)
    @property
    def chi_bp(self) -> int:
        return self.bp.trunc_dim
    @chi_bp.setter
    def chi_bp(self, value) -> None:
        self.bp.trunc_dim = int(value)

    def set_parallel(self, value:bool) -> None:
        assert isinstance(self, bool) or value in [0, 1]
        self.bp.parallel_msgs = value
        self.contraction.parallel = value

    def post_creation_fix(self):
        self.__post_init__()
        
    def __post_init__(self)->None:
        ## Call other post inits:
        self.ite.__post_init__()
        self.bp.__post_init__()
        
        ## Check chi
        if self.chi_bp > self.chi:
            prints.print_warning(f" truncation dim of BP is greater than that of the other bubblcon usages.")
        if not ALLOW_VISUALS:
            self.visuals.live_plots = False

        ## combination between visuals and contraction progress bar:
        if not self.visuals.progress_bars.is_active_at('bubblecon'):
            self.contraction.progress_bar = False 
            self.bp.visuals.bubblecon_progress_bar = False

        if not self.visuals.progress_bars.is_active_at('blockBP'):
            self.bp.visuals.main_progress_bar = False
        


    def strengthen(self, _harder_target:bool=True):
        if isinstance(self.bp.max_iterations, int):
            self.bp.max_iterations *= 2
        # self.bp.max_swallowing_dim *= 2
        # self.bp.allowed_retries += 1
        self.chi *= 4
        if _harder_target:
            self.bp.msg_diff_terminate /= 10

    def __repr__(self) -> str:        
        s = f"{self.__class__.__name__}:"
        for field in fields(self):
            s += "\n"
            value = getattr(self, field.name)
            s += f"{field.name}: {value}"
        return s

    def copy(self)->"Config":
        return deepcopy(self)
    
    


