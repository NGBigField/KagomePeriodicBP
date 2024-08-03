import _import_main_and_src

from scripts.condor.main_sender import main as main_sender
from scripts.run_ite import main as run_ite
from utils.dicts import pass_values_from_dict1_to_dict2_on_matching_keys

from sys import argv

RESULT_KEYS = ["seed","D", "N", "chi", "energy", "parallel", "method", "path"]



def job(
    # Values from sender:
    D : int = 2,
    N : int = 2,
    chi_factor : int|float = 1,
    seed : int = -1,
    method : int = 1,
    parallel : int = 0,
    control : int = 0,
    # Default values:
    progress_bar : bool = True,
) -> dict:
    
    # method
    match method:
        case 1: unit_cell_from = "best"
        case 2: unit_cell_from = "tnsu"
        case 3: unit_cell_from = "random"
        case _:
            raise ValueError("Not a valid option")

    # Parallel:
    if parallel==0:     parallel=False
    elif parallel==1:   parallel=True
    else:
        raise ValueError(f"Invalid parallel value {parallel!r}")

    # Progress bar:
    if progress_bar == True:
        progress_bar_in = 'only_main'
    else:
        progress_bar_in = 'all_disabled'

    ## Additional control:
    if control == 0:
        hamiltonian_str = "AFM"
    elif control == 1:
        hamiltonian_str = "AFM-T"

    ## Run:
    energy, unit_cell_file_path = run_ite(
        N=N, D=D, chi_factor=chi_factor, 
        live_plots=False, 
        parallel=parallel, 
        progress_bar=progress_bar_in,
        hamiltonian=hamiltonian_str,
        unit_cell_from=unit_cell_from
    )
    
    # Expected outputs: 
    # ["seed","D", "N", "chi", "energy", "parallel", "path", "method"]
    results = dict(
        D=D,
        N=N,
        chi=chi_factor,
        energy=energy,
        parallel=parallel,
        path=unit_cell_file_path
    )
    
    return results



def _check_given_dimensions() -> dict[str, int]|None:
    NUM_EXPECTED_ARGS = 3

    ## Check function call:
    if len(argv) != NUM_EXPECTED_ARGS:
        None

    ## Parse args:
    print(f"The {NUM_EXPECTED_ARGS} arguments are:")

    i = 0  # 0
    this_func_name = argv[i]
    print(f"    {i}: this_func_name={this_func_name!r}")

    i += 1
    D = int(argv[i])
    print(f"    {i}: D={D!r}")

    i += 1  # 2
    N = int(argv[i])
    print(f"    {i}: N={N}")

    vals = {}
    vals['N'] = [N]
    vals['D'] = [D]

    return vals


def _choose_requested_memory(D:int) -> int:
    ## Compute needed memory
    match D:
        case 1:     raise ValueError("No ITE for D=1")
        case 2:     request_memory_gb = 2
        case 3:     request_memory_gb = 4 
        case 4:     request_memory_gb = 8
        case 5:     request_memory_gb = 8 
        case 6:     request_memory_gb = 16
        case 7:     request_memory_gb = 32
        case _:     request_memory_gb = 64

    return request_memory_gb



def sender(
    request_memory_gb:int = 8
) -> None:
    vals = {}
    vals['D'] = [2, 3]
    vals['N'] = list(range(3, 6))
    vals['chi'] = [1, 2, 3]
    vals['method'] = [1, 3]
    vals['seed'] = list(range(5))
    vals['parallel'] = [0]
    vals['control'] = [0]

    user_input = _check_given_dimensions()
    if user_input is not None:
        pass_values_from_dict1_to_dict2_on_matching_keys(user_input, vals)

    if len(vals['D'])==1:
        request_memory_gb = _choose_requested_memory(vals['D'][0])

    main_sender(
        job_type="ite_afm",
        request_cpus=4,
        request_memory_gb=request_memory_gb,
        vals=vals,
        result_keys=RESULT_KEYS,
        _local_test=False
    )

if __name__ == "__main__":    
    sender()

