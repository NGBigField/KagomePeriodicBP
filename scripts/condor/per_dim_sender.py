import pathlib, sys
if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from sender import main as main_sender
from sender import DEFAULT_VALS
from sys import argv


NUM_EXPECTED_ARGS = 3

def main():

    ## Check function call:
    assert len(argv)==NUM_EXPECTED_ARGS, f"Expected {NUM_EXPECTED_ARGS} arguments. Got {len(argv)}."

    ## Parse args:
    print(f"The {NUM_EXPECTED_ARGS} arguments are:")

    i = 0  # 0
    this_func_name = argv[i]
    print(f"{i}: this_func_name={this_func_name!r}")

    i += 1
    D = int(argv[i])
    print(f"{i}: D={D!r}")

    i += 1  # 2
    N = int(argv[i])
    print(f"{i}: N={N}")

    vals = DEFAULT_VALS
    vals['N'] = [N]
    vals['D'] = [D]

    ## Compute needed memory
    match D:
        case 1:     raise ValueError("No ITE for D=1")
        case 2:     request_memory_gb = 4
        case 3|4:   request_memory_gb = 8 
        case 5:     request_memory_gb = 16 
        case 6:     request_memory_gb = 32
        case 7:     request_memory_gb = 512
        case _:     request_memory_gb = 512

    # request_memory_gb += N
    request_memory_gb = int(request_memory_gb)
    print(f"request_memory_gb: {request_memory_gb}gb")

    return main_sender(
        job_type="ite_afm",
        vals=vals,
        request_memory_gb=request_memory_gb
    )

if __name__=="__main__":
    main()