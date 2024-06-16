if __name__ == "__main__":
    import pathlib, sys 
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent/"src"))

from sys import argv

# Import scripts to use
import _import_scripts

# Import DictWriter class from CSV module
from time import perf_counter, sleep
from csv import DictWriter
from typing import Any, Generator

from src.utils import errors

# Import the possible job types:
from scripts.condor import job_bp
from scripts.condor import job_parallel_timing 
from scripts.condor import job_bp_convergence 
from scripts.condor import job_ite_afm

from scripts.condor.sender import Arguments


# import numpy for random matrices:
import numpy as np


NUM_EXPECTED_ARGS = 10
SAFETY_BUFFER_FRACTION = 0.3  # safety buffer (adjust based on needs)

# A main function to parse inputs:
def main():

    ## Check function call:
    # assert len(argv)==NUM_EXPECTED_ARGS, f"Expected {NUM_EXPECTED_ARGS} arguments. Got {len(argv)}."
    if len(argv)!=NUM_EXPECTED_ARGS: 
        f"Expected {NUM_EXPECTED_ARGS} arguments. Got {len(argv)}."

    ## Parse args:
    print(f"The {len(argv)} arguments are:")

    i = 0  # 0
    this_func_name = argv[i]
    print(f"{i}: this_func_name={this_func_name!r}")

    i += 1  # 1
    output_file = argv[i]
    print(f"{i}: output_file={output_file!r}")

    i += 1  # 2
    job_type = argv[i]
    print(f"{i}: job_type={job_type}")

    i += 1  # 3
    req_mem_gb = int(argv[i])
    print(f"{i}: req_mem_gb={req_mem_gb}")

    i += 1  # 4
    seed = int(argv[i])
    print(f"{i}: seed={seed}")

    i += 1  # 5
    method = int(argv[i])
    print(f"{i}: method={method}")

    i += 1  # 6
    D = int(argv[i])
    print(f"{i}: D={D}")

    i += 1  # 7
    N = int(argv[i])
    print(f"{i}: N={N}")

    i += 1  # 8
    chi = int(argv[i])
    print(f"{i}: chi={chi}")

    i += 1  # 9
    result_keys = _parse_list_of_strings(argv[i])
    print(f"{i}: result_keys={result_keys}")


    ## Force usage of requested Giga-bytes:
    _create_random_array_by_ram(req_mem_gb)

    ## Run:
    results : dict[str, Any]
    t1 = perf_counter()
    try:
        match job_type:
            case "bp":  
                results = job_bp.main(D=D, N=N, method=method)
            case "parallel_timings":             
                results = job_parallel_timing.main(D=D, N=N, method=method)
            case "bp_convergence":
                results = job_bp_convergence.main(D=D, N=N)
            case "ite_afm":
                results = job_ite_afm.main(D=D, N=N, chi_factor=chi, seed=seed)
            case _:
                raise ValueError(f"Not an expected job_type={job_type!r}")
    except Exception as e:
        results = dict(
            e=errors.get_traceback(e)
        )
    t2 = perf_counter()

    print(f"res={results}")
    
    ## check output:
    assert isinstance(results, dict), f"result must be of type `dict`! got {type(results)}"
    results["exec_time"]=t2-t1
    results["seed"]=seed

    ## Prepare result for printing:
    row_to_write = dict()
    for result_key in result_keys:
        try:
            res = results[result_key]
        except KeyError:
            res = None
        row_to_write[result_key] = res

    ## Write result:
    with open( output_file ,'a') as f:
        dict_writer = DictWriter(f, fieldnames=result_keys )
        dict_writer.writerow(row_to_write)

    ## End
    print("Results are written into:")
    print(f"{output_file!r}")




def _clean_item(s:str)->str:
    s = s.replace(" ", "")
    s = s.replace(",", "")
    return s

def _parse_list_of_strings(s:str)->list[str]:
    assert isinstance(s, str)
    assert s[0]=="["
    s = s.removeprefix("[")
    s = s.removesuffix("]")
    items = s.split(",")
    items = [_clean_item(item) for item in items]
    return items


def _create_random_array_by_ram(ram_gb):
    """
    Creates a random NumPy array that utilizes the specified amount of RAM in gigabytes.

    Args:
        ram_gb (float): The amount of RAM to use in gigabytes.

    Returns:
        numpy.ndarray: The randomly generated NumPy array.

    Raises:
        ValueError: If the requested RAM usage exceeds available memory.
    """

    print(f"Writing trash memory of up to {ram_gb}gb")

    # Convert RAM to bytes and consider safety factor (adjust as needed)
    target_bytes = ram_gb * 1024**3 * SAFETY_BUFFER_FRACTION

    # Calculate element count and data type based on target size
    element_size = np.dtype(float).itemsize  # Adjust for desired data type if needed
    max_elements = int(target_bytes // element_size)

    for i, num_elements in enumerate(np.linspace(10, max_elements, 10)):

        num_elements = int(num_elements)
        print(f"    {i+1}/10: Writing trash memory with {num_elements} element")

        # Create the random array
        array_shape = (num_elements,)  # Adjust shape as desired for multidimensional arrays
        array = np.random.random(array_shape)

        ## Do some fake calculations:
        array = np.dot(array*2, array+1)
        array = array + (array/2 - 1)
        sleep(1)
        del array
    
    print("    Done writing trash memory")


if __name__ == "__main__":
    main()
