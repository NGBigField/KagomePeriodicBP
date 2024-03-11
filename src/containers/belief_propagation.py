from dataclasses import dataclass, field
from enums import MessageModel
from utils.arguments import Stats
from containers._meta import container_repr
from containers.contractions import MPSOrientation
from typing import NamedTuple, TypeAlias
from lattices.directions import LatticeDirection
from libs.bmpslib import mps as MPS
from lattices.directions import BlockSide
from copy import deepcopy


@dataclass
class BPConfig: 
    max_iterations : int|None = 30   # None is used for unlimited number of iterations
    max_swallowing_dim : int = 9
    target_msg_diff : float = 1e-5
    msg_diff_squared : bool = True  # True is easier to get to 
    init_msg: MessageModel = MessageModel.RANDOM_QUANTUM
    allowed_retries : int = 3
    times_to_deem_failure_when_diff_increases  : int = 4
    parallel_msgs : bool = False
    damping : float|None = None  # The `learning-step` of the messages. 
    hermitize_msgs : bool = True
    # damping=0 will take 100% the new message while damping=1 will keep the old message.

    def __repr__(self) -> str:
        return container_repr(self)
    
    def copy(self)->"BPConfig":
        return deepcopy(self)


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
    

class MessageDictType(dict[BlockSide, Message]):
    def mpss(self)->list[MPS]:
        return [msg.mps.A for side, msg in self.items()]