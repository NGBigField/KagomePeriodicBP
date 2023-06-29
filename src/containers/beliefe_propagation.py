from dataclasses import dataclass, field
from enums import MessageModel
from utils.arguments import Stats
from containers._meta import container_repr

@dataclass
class BPConfig: 
    max_iterations : int|None = 30   # None is used for unlimited number of iterations
    max_swallowing_dim : int = 9
    target_msg_diff : float = 1e-5
    msg_diff_squared : bool = True  # True is easier to get to 
    init_msg: MessageModel = MessageModel.RANDOM_QUANTUM
    allowed_retries : int = 2
    hermitize_messages_between_iterations : bool = True
    times_to_deem_failure_when_diff_increases  : int = 3

    def __repr__(self) -> str:
        return container_repr(self)


@dataclass
class BPStats(Stats):
    iterations      : int   = -1
    final_error     : float = -1.0  
    attempts        : int   = 1
    final_config    : BPConfig = field(default_factory=BPConfig)