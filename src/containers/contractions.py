from dataclasses import dataclass
from containers._meta import container_repr

@dataclass
class BubbleConConfig: 
    # trunc_dim_2=None
    # eps=None
    progress_bar=True
    separate_exp=True


@dataclass
class ContractionConfig:
    trunc_dim : int 
    random_snake_pattern : bool = False
    last_con_order_determines_mps_order : bool = False

    def __repr__(self) -> str:
        return container_repr(self)