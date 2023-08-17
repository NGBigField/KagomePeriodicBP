from dataclasses import dataclass, fields
from enums import MessageModel
from containers.beliefe_propagation import BPConfig
from containers.sizes_and_dimensions import TNSizesAndDimensions
from containers.imaginary_time_evolution import ITEConfig
from utils import strings

# Control flags:
from _config_reader import DEBUG_MODE, ALLOW_VISUALS

@dataclass
class Config: 
    bp : BPConfig 
    ite : ITEConfig 
    tn : TNSizesAndDimensions
    bubblecon_trunc_dim : int
    live_plots : bool = False

    @staticmethod
    def from_D(D:int)->"Config":
        return Config(
            bp=BPConfig(max_swallowing_dim=D**2),
            ite=ITEConfig(),
            tn=TNSizesAndDimensions(virtual_dim=D),
            bubblecon_trunc_dim=2*D**2
        )

    def strengthen(self, _harder_target:bool=True):
        if isinstance(self.bp.max_iterations, int):
            self.bp.max_iterations *= 2
        # self.bp.max_swallowing_dim *= 2
        # self.bp.allowed_retries += 1
        self.bubblecon_trunc_dim *= 4
        if _harder_target:
            self.bp.target_msg_diff /= 10

    def __post_init__(self)->None:
        trunc_d_bp = self.bp.max_swallowing_dim
        trunc_d_all_other = self.bubblecon_trunc_dim
        if trunc_d_bp > trunc_d_all_other:
            strings.print_warning(f" truncation dim of BP is greater than that of the other bubblcon usages.")
        if self.bubblecon_trunc_dim == -1:
            self.bubblecon_trunc_dim = self.bp.max_swallowing_dim*2
        if not ALLOW_VISUALS:
            self.live_plots = False


    def __repr__(self) -> str:        
        s = f"{self.__class__.__name__}:"
        for field in fields(self):
            s += "\n"
            value = getattr(self, field.name)
            s += f"{field.name}: {value}"
        return s
    
