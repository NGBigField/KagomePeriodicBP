from utils.errors import CumulativeError


class BlockBPError(CumulativeError): 
	_msg = 'Error in blockbp!\n'
    
class MPIError(BlockBPError): 
	_msg = 'MPI error: '
