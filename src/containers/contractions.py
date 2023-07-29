from dataclasses import dataclass
from containers._meta import container_repr

@dataclass
class BubbleConConfig: 
    # trunc_dim_2=None
    # eps=None
    progress_bar=True
    separate_exp=True

