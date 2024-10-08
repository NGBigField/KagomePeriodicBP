from typing import TypeVar, Literal
import numpy as np

# Define a type variable that can be either a list of floats or a numpy array of floats
Signal = TypeVar('Signal', list[float], np.ndarray)



def _initialize_output(input:Signal) -> Signal:
    # Example processing: multiply all elements by 2
    if isinstance(input, list):
        return [0.0 for _ in input]
    elif isinstance(input, np.ndarray):
        return np.zeros(input.shape)
    else:
        raise TypeError("Unsupported data type")


def max_or_min_filter(sig:Signal, size:int, what:Literal["max", "min"]) -> Signal:
    ## Check inputs and prepare outputs:
    out = _initialize_output(sig)
    assert isinstance(size, int), f"Size must be a positive integer. Gut {size!r}."
    assert size>=1, f"Size must be a positive integer. Gut {size!r}."
    # operation:
    match what:
        case "max": op=max
        case "min": op=min
        case _:
            raise ValueError(f"No an expected input. Got 'what'={what!r}.")
    
    ## Perform filter:
    for i, _ in enumerate(sig):
        if i+1<size:
            sample = sig[:(i+1)]
        else:
            sample = sig[(i-size+1):(i+1)]
        out[i] = op(sample)

    return out