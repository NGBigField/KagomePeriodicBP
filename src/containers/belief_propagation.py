from dataclasses import dataclass, field
from enums import MessageModel
from utils.arguments import Stats
from containers._meta import _ConfigClass
from containers.contractions import MPSOrientation
from typing import NamedTuple, TypeAlias, Dict
import numpy as np
from lattices.directions import LatticeDirection
from libs.bmpslib import mps as MPS
from lattices.directions import BlockSide



@dataclass
class BPVisualsConfig(_ConfigClass):
    update_plots_between_steps:bool=False
    main_progress_bar:bool=True
    bubblecon_progress_bar:bool=True

    def __repr__(self) -> str:
        return super().__repr__(level=2)
    
    def set_all_progress_bars(self, value:bool) -> None:
        assert isinstance(value, bool)
        for attr_nmae in ["main_progress_bar", "bubblecon_progress_bar"]:
            self.__setattr__(attr_nmae, value)
    

@dataclass
class BPConfig(_ConfigClass): 
    init_msg: MessageModel = MessageModel.RANDOM_QUANTUM
    max_iterations : int|None = 50   # None is used for unlimited number of iterations
    trunc_dim : int = 9
    msg_diff_terminate : float = 1e-10
    msg_diff_good_enough : float = 1e-5
    msg_diff_squared : bool = True  # True is easier to get to 
    allowed_retries : int = 2
    times_to_deem_failure_when_diff_increases  : int = 3
    parallel_msgs : bool = False
    damping : float|None = None  # The `learning-step` of the messages. 
    # damping=0 will take 100% the new message while damping=1 will keep the old message.
    hermitize_msgs_when_finished : bool = True
    fix_msg_each_step : bool = True
    # Visuals
    visuals : BPVisualsConfig = field(default_factory=BPVisualsConfig)

    
    def __repr__(self) -> str:
        return super().__repr__()
    
    def __post_init__(self) -> None:
        if self.msg_diff_terminate > self.msg_diff_good_enough:
            raise ValueError(f"In bp config, msg_diff_terminate={self.msg_diff_terminate}"+
                             f"< msg_diff_good_enough={self.msg_diff_good_enough}"+
                             f"  This will cause bp to fail at every step."
                            )


@dataclass
class BPStats(Stats):
    iterations      : int   = -1
    attempts        : int   = 1
    final_error     : float = -1.0  
    success         : bool = False
    final_config    : BPConfig = field(default_factory=BPConfig)


class Message(NamedTuple):
    mps : MPS
    orientation : MPSOrientation

    def copy(self)->"Message":
        return Message(
            mps=self.mps.copy(full=True),
            orientation=self.orientation
        )
    

MessageDictType : TypeAlias = Dict[BlockSide, Message]
