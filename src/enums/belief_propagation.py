from enum import Enum

class MessageModel(Enum):
	UNIFORM_QUANTUM = 'UQ'  # 1) UQ --- Uniform quantum. The initial messages are  \delta_{\alpha\beta} (i.e., the identity)
	RANDOM_QUANTUM 	= 'RQ'  # 2) RQ --- The initial messages are product states of |v><v|, where |v> is a random unit vector
	UNIFORM_CLASSIC = 'UC'  # 3) UC --- For when the TN represents a classical probability. in this case the initial message is simply the uniform prob.
	RANDOM_CLASSIC 	= 'RC'  # 4) RC --- For the TN represents a classical prob. Then the inital messages are product of random prob dist.
