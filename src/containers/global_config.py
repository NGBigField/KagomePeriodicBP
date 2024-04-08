from dataclasses import dataclass, fields
from containers.belief_propagation import BPConfig
from containers.sizes_and_dimensions import TNDimensions
from containers.imaginary_time_evolution import ITEConfig, IterativeProcessConfig
from containers.visuals import VisualsConfig
from containers._meta import _ConfigClass
from utils import size, prints
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


@dataclass
class Config(_StoreConfigClasses, _ConfigClass): 
    # The actual stored data:
    bp : BPConfig 
    ite : ITEConfig 
    iterative_process : IterativeProcessConfig
    dims : TNDimensions
    visuals : VisualsConfig
    trunc_dim : int

    @staticmethod
    def derive_from_physical_dim(D:int)->"Config":
        return Config(
            bp=BPConfig(max_swallowing_dim=D**2),
            ite=ITEConfig(),
            iterative_process=IterativeProcessConfig(),
            dims=TNDimensions(virtual_dim=D),
            visuals=VisualsConfig(),
            trunc_dim=2*D**2+10
        )

    def post_creation_fix(self):
        self.__post_init__()
        
    def __post_init__(self)->None:
        # other post inits:
        self.ite.__post_init__()
        # specials:
        trunc_d_bp = self.bp.max_swallowing_dim
        trunc_d_all_other = self.trunc_dim
        if trunc_d_bp > trunc_d_all_other:
            prints.print_warning(f" truncation dim of BP is greater than that of the other bubblcon usages.")
        if self.trunc_dim == -1:
            self.trunc_dim = self.bp.max_swallowing_dim*2
        if not ALLOW_VISUALS:
            self.visuals.live_plots = False

    def strengthen(self, _harder_target:bool=True):
        if isinstance(self.bp.max_iterations, int):
            self.bp.max_iterations *= 2
        # self.bp.max_swallowing_dim *= 2
        # self.bp.allowed_retries += 1
        self.trunc_dim *= 4
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
    
    


