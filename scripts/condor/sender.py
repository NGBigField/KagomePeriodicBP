import pathlib, sys, os
import numpy as np
from copy import deepcopy

# Import DictWriter class from CSV module
from csv import DictWriter

if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# for smart iterations:
from itertools import product

from src import project_paths

results_dir = project_paths.data/"condor"
results_dir = results_dir.__str__()
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


RESULT_KEYS_DICT = dict(
    bp = ["with_bp", 'D', 'N', 'A_X', 'A_Y', 'A_Z', 'B_X', 'B_Y', 'B_Z', 'C_X', 'C_Y', 'C_Z'],
    parallel_timings = ["parallel", 'D', 'N', 'seed', 'bp-step', 'reduction'],
    ite_afm = ["seed","D", "N", "energy", "path"],
    bp_convergence = ['seed', 'D', 'N', 'chi', 'iterations', 'rdm_diff_bp', 'rdm_diff_random', 'z_bp', 'z_random', 'time_bp', 'time_random']
)

## all values:
DEFAULT_VALS = {}
DEFAULT_VALS['N'] = [2, 4, 6] #range(2, 11, 1)
DEFAULT_VALS['D'] = [2, 3]
DEFAULT_VALS['chi'] = [-1]
DEFAULT_VALS['method'] = [1]
DEFAULT_VALS['seed'] = range(1)

Arguments = '$(outfile) $(seed) $(method) $(D) $(N) $(chi) $(job_type) $(result_keys)'


def main(
    job_type="ite_afm",  # "ite_afm" / "bp" / "parallel_timings" / "bp_convergence"
    request_cpus:int=2,
    request_memory_gb:int=4,
    vals:dict=DEFAULT_VALS,
    result_file_name:str|None=None
):

    ## Check inputs and 
    _max_str_per_key = {key:max((len(str(val)) for val in lis)) for key, lis in vals.items()}
    if result_file_name is None:
        result_file_name="results_"+job_type

    ## Get from job type:
    result_keys = RESULT_KEYS_DICT[job_type]

    ## Define paths and names:
    sep = os.sep
    this_folder_path = pathlib.Path(__file__).parent.__str__()
    #
    worker_script_fullpath  = this_folder_path+sep+"worker.py"
    results_fullpath        = results_dir+sep+result_file_name+".csv"
    output_files_prefix     = "kagome-bp-"+job_type
    #
    print(f"script_fullpath={worker_script_fullpath!r}")
    print(f"results_fullpath={results_fullpath!r}")
    print(f"output_files_prefix={output_files_prefix!r}")
    print(f"job_type={job_type!r}")

    ## Define job params:
    job_params : list[dict] = []
    for N, D, method, seed, chi in product(vals['N'], vals['D'], vals['method'], vals['seed'], vals['chi'] ):
        # To strings:
        N = f"{N}"
        D = f"{D}"
        method = f"{method}"
        seed = f"{seed}"
        chi = f"{chi}"

        job_params.append( dict(
            outfile=results_fullpath,
            D=D,
            N=N,
            chi=chi,
            method=method,
            seed=seed,
            job_type=job_type,
            result_keys=_encode_list_as_str(result_keys)
        ))

    ## Print:
    for params in job_params:
        params2print = deepcopy(params)
        params2print.pop("outfile")
        _print_inputs(params2print, _max_str_per_key)

    ## Prepare output file:    
    with open( results_fullpath ,'a') as f:        
        dict_writer = DictWriter(f, fieldnames=result_keys)
        dict_writer.writerow({field:field for field in result_keys})
        f.close()

    ## Call condor:
    print(f"Calling condor with shared inputs:")
    print(f"    output_files_prefix={output_files_prefix}")
    print(f"    request_cpus={request_cpus}")
    print(f"    requestMemory={request_memory_gb}gb")
    print(f"    Arguments={Arguments}")

    import CondorJobSender
    CondorJobSender.send_batch_of_jobs_to_condor(
        worker_script_fullpath,
        output_files_prefix,
        job_params,
        request_cpus=f"{request_cpus}",
        requestMemory=f"{request_memory_gb}gb",
        Arguments=Arguments
    )

    print("Called condor successfully")


def _encode_list_as_str(lis:list)->str:
    assert isinstance(lis, list)
    s = '['
    for item in lis:
        s += f'{item},'
    s = s[:-1]+']'
    return s


def _print_inputs(inputs:dict[str, str], _max_str_per_key:dict[str, int])->None:
    total_string = ""
    for key, value in inputs.items():
        s = f"{value}"
        if key in _max_str_per_key:
            max_length = _max_str_per_key[key]
            crnt_length = len(f"{value}")
            pad_length = max_length-crnt_length
            if pad_length>0:
                s = " "*pad_length + s
        total_string += " " + f"{key!r}: " + s + ","
    total_string = total_string.removesuffix(",")
    print(total_string)
        


if __name__=="__main__":
    main()