from dataclasses import dataclass
from tensor_networks.node import Node

@dataclass
class UnitCell: 
    A : Node
    B : Node
    C : Node


