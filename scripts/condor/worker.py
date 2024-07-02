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
import threading

from src.utils import errors

# Import the possible job types:
from scripts.condor import job_bp
from scripts.condor import job_parallel_timing 
from scripts.condor import job_bp_convergence 
from scripts.condor import job_ite_afm

from scripts.condor.sender import Arguments


# import numpy for random matrices:
import numpy as np

from utils import sizes


NUM_EXPECTED_ARGS = 10
SAFETY_BUFFER_FRACTION = 0.4  # safety buffer (adjust based on needs)

# A main function to parse inputs:
def main():

    print("\n"*3)

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
    chi = float(argv[i])
    print(f"{i}: chi={chi}")

    i += 1  # 9
    result_keys = _parse_list_of_strings(argv[i])
    print(f"{i}: result_keys={result_keys}")


    ## Force usage of requested Giga-bytes:
    _compute_with_random_mat_by_ram(req_mem_gb)
    thread = threading.Thread(target=_auto_timed_compute_with_random_mat_by_ram, args=(req_mem_gb,))
    thread.start()

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

    thread.join()

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


def _auto_timed_compute_with_random_mat_by_ram(ram_gb, sleep_time:float|int=60*5):
    while True:
        sleep(sleep_time)
        _compute_with_random_mat_by_ram(ram_gb)


def _compute_with_random_mat_by_ram(ram_gb, num_growing_sizes:int=3, computation_repetitions:int=3, progress_bar:bool=True, sleep_time:float|int=0.5):
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

    # Calculate element count and data type based on target size
    target_final_size = ram_gb*SAFETY_BUFFER_FRACTION

    if num_growing_sizes>1:
        sizes_gb = np.linspace(1e-9, target_final_size, num_growing_sizes)
    else:
        sizes_gb = [target_final_size]
        progress_bar = False

    if progress_bar:
        from src.utils.prints import ProgressBar
        prog_bar = ProgressBar(expected_end=num_growing_sizes)

    for i, crnt_size in enumerate(sizes_gb):

        if progress_bar:
            prog_bar.next(extra_str=f"size = {crnt_size}[gb]")
            
        sizes.do_computation_by_ram_size(crnt_size, repetitions=computation_repetitions)

        sleep(sleep_time)
        del mat1, mat2, mat3
    
    if progress_bar:
        prog_bar.clear()
    print("    Done writing trash memory")


if __name__ == "__main__":
    main()
