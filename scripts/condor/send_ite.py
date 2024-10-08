import _import_main_and_src

from scripts.condor.main_sender import main as main_sender
from scripts.run_ite import main as run_ite
from utils.dicts import pass_values_from_dict1_to_dict2_on_matching_keys

from sys import argv

RESULT_KEYS = ["seed","D", "N", "chi", "energy", "parallel", "method", "path"]


def _parse_given_inputs() -> dict[str, int]|None:
    NUM_EXPECTED_ARGS = {2, 3}

    ## Check function call:
    if len(argv) not in NUM_EXPECTED_ARGS:
        return None

    vals = {}

    ## Parse args:
    print(f"The {len(argv)} arguments are:")

    i = 0  # 0
    this_func_name = argv[i]
    print(f"    {i}: this_func_name={this_func_name!r}")

    i += 1
    D = int(argv[i])
    print(f"    {i}: D={D!r}")
    vals['D'] = [D]

    i += 1  # 2
    if i + 1 <= len(argv): 
        N = int(argv[i])
        print(f"    {i}: N={N}")
        vals['N'] = [N]


    return vals


def _choose_requested_memory(D:int) -> int:
    ## Compute needed memory
    match D:
        case 1:     raise ValueError("No ITE for D=1")
        case 2:     request_memory_gb = 2
        case 3:     request_memory_gb = 2
        case 4:     request_memory_gb = 4
        case 5:     request_memory_gb = 6
        case 6:     request_memory_gb = 8
        case 7:     request_memory_gb = 50
        case _:     request_memory_gb = 64

    return request_memory_gb


def job(
    # Values from sender:
    D : int = 2,
    N : int = 2,
    chi_factor : int|float = 1,
    seed : int = -1,
    method : int = 1,
    parallel : int = 0,
    control : int = 0,
    noise_initial : float = 0,
    noise_per_segment : float = 0,
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
    if progress_bar == True:    progress_bar_in = 'only_main'
    else:                       progress_bar_in = 'all_disabled'

    ## Additional control:
    if abs(control) == 1:    hamiltonian_str = "AFM"
    elif abs(control) == 2:  hamiltonian_str = "AFM-T"
    else:
        raise ValueError(f"Invalid control value {control!r}")
    #
    if control>0:
        messages_init = 'random'
    else: 
        messages_init = 'uniform'

    ## Run:
    energy, unit_cell_file_path = run_ite(
        N=N, D=D, chi_factor=chi_factor, 
        live_plots=False, 
        parallel=parallel, 
        progress_bar=progress_bar_in,
        hamiltonian=hamiltonian_str,
        unit_cell_from=unit_cell_from,
        io='condor',
        messages_init=messages_init,
        noise_initial=noise_initial,
        noise_per_segment=noise_per_segment
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


def sender(
    request_memory_gb:int = 8
) -> None:
    default_vals = {}
    default_vals['D'] = [2, 3]
    default_vals['N'] = list(range(2, 5))
    default_vals['chi'] = [1, 2, 3]
    default_vals['method'] = [1]
    default_vals['seed'] = list(range(1))
    default_vals['parallel'] = [0]
    default_vals['noise_initial'] = [1e-3, 1e-2, 1e-1, 1, 0]
    default_vals['noise_per_segment'] = [1e-3, 1e-2, 1e-1, 0, 1e1]
    default_vals['control'] = [-2, -1, 1, 2]

    user_input = _parse_given_inputs()
    if user_input is not None:
        pass_values_from_dict1_to_dict2_on_matching_keys(user_input, default_vals)

    if len(default_vals['D'])==1:
        D = default_vals['D'][0]
        request_memory_gb = _choose_requested_memory(D)

    main_sender(
        job_type="ite_afm",
        request_cpus=4,
        request_memory_gb=request_memory_gb,
        vals=default_vals,
        result_keys=RESULT_KEYS,
        _local_test=False
    )

if __name__ == "__main__":    
    sender()

