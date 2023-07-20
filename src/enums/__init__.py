from typing import TypeAlias

# Enums from within this folder:
# from tensor_networks.directions import Directions
from enums.contractions import ContractionDepth, ReduceToEdgeMethod, ReduceToCoreMethod
from enums.tensor_networks import NodeFunctionality, CoreCellType
from enums.imaginary_time_evolution import UpdateModes

# Used to shorten the path to import these enums:
from lib._blockbp.enums import MessageModel

# Some aliases for better readability:
# Sides : TypeAlias = Directions



        