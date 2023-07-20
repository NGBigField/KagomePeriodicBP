
# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ ## For better user-hints:

# For OOP:
from enum import Enum

# For type hints:
from typing import (
	List, 
	Tuple,
	Any,
	Dict,
  	TypeAlias,
)

# ============================================================================ #
#|                                Types                                       |#
# ============================================================================ #
BlocksConType : TypeAlias = List[Tuple[List[Tuple[str, int]], float]]  # Type alias


# ============================================================================ #
#|                                Classes                                     |#
# ============================================================================ #

class MessageModel(Enum):
	UNIFORM_QUANTUM = 'UQ'  # 1) UQ --- Uniform quantum. The initial messages are  \delta_{\alpha\beta} (i.e., the identity)
	RANDOM_QUANTUM 	= 'RQ'  # 2) RQ --- The initial messages are product states of |v><v|, where |v> is a random unit vector
	UNIFORM_CLASSIC = 'UC'  # 3) UC --- For when the TN represents a classical probability. in this case the initial message is simply the uniform prob.
	RANDOM_CLASSIC 	= 'RC'  # 4) RC --- For the TN represents a classical prob. Then the inital messages are product of random prob dist.



# ============================================================================ #
#|                              main test                                     |#
# ============================================================================ #

def _main_test():
	e1 = MessageModel.RANDOM_CLASSIC
	e2 = MessageModel('UQ')
	print(e2)

if __name__ == "__main__":
	_main_test()
	print("Done.")
