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
    D = argv[i]
    print(f"{i}: D={D!r}")

    i += 1  # 2
    N = int(argv[i])
    print(f"{i}: N={N}")

    vals = DEFAULT_VALS
    vals['N'] = [N]
    vals['D'] = [D]

    request_memory_gb = N * D 

    return main_sender(
        vals=vals,
        request_memory_gb=request_memory_gb
    )

if __name__=="__main__":
    main()