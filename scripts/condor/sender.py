import pathlib, sys, os
if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

    
import numpy as np
from copy import deepcopy

# Import DictWriter class from CSV module
from csv import DictWriter


# for smart iterations:
from itertools import product

import string
import random
import time

from src import project_paths


results_dir = project_paths.data/"condor"
results_dir = results_dir.__str__()
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


RAM_MEMORY_IN_2_EXPONENTS = False
LOCAL_TEST = False

RESULT_KEYS_DICT = dict(
    bp = ["with_bp", 'D', 'N', 'A_X', 'A_Y', 'A_Z', 'B_X', 'B_Y', 'B_Z', 'C_X', 'C_Y', 'C_Z'],
    parallel_timings = ["parallel", 'D', 'N', 'seed', 'bp-step', 'reduction'],
    ite_afm = ["seed","D", "N", "chi", "energy", "path"],
    bp_convergence = ['seed', 'D', 'N', 'chi', 'iterations', 'rdm_diff_bp', 'rdm_diff_random', 'z_bp', 'z_random', 'time_bp', 'time_random']
)

## all values:
DEFAULT_VALS = {}
DEFAULT_VALS['D'] = [4, 5]
DEFAULT_VALS['N'] = [2] 
DEFAULT_VALS['chi'] = [1]
DEFAULT_VALS['method'] = [1]
DEFAULT_VALS['seed'] = [0]

Arguments = '$(outfile) $(job_type) $(req_mem_gb) $(seed) $(method) $(D) $(N) $(chi) $(result_keys)'


def main(
    job_type="ite_afm",  # "ite_afm" / "bp" / "parallel_timings" / "bp_convergence"
    request_cpus:int=4,
    request_memory_gb:int=16,
    vals:dict=DEFAULT_VALS,
    result_file_name:str|None=None
):

    ## Check inputs and fix:
    _max_str_per_key = {key:max((len(str(val)) for val in lis)) for key, lis in vals.items()}
    if result_file_name is None:
        result_file_name="results_"+job_type
    # Memory:
    request_memory_gb = _legit_memory_sizes(request_memory_gb)
    request_memory_bytes = 1073741824 * request_memory_gb

    ## Get from job type:
    result_keys = RESULT_KEYS_DICT[job_type]

    ## Define paths and names:
    sep = os.sep
    this_folder_path = pathlib.Path(__file__).parent.__str__()
    #
    worker_script_fullpath = this_folder_path+sep+"worker.py"
    results_fullpath       = results_dir+sep+result_file_name+".csv"
    output_files_prefix    = "kagome-bp-"+job_type+"-"+_time_stamp()
    #
    print(f"script_fullpath={worker_script_fullpath!r}")
    print(f"results_fullpath={results_fullpath!r}")
    print(f"output_files_prefix={output_files_prefix!r}")
    print(f"job_type={job_type!r}")

    ## Define job params:
    job_params_dicts : list[dict] = []
    for N, D, method, seed, chi in product(vals['N'], vals['D'], vals['method'], vals['seed'], vals['chi'] ):
        # To strings:
        N = f"{N}"
        D = f"{D}"
        method = f"{method}"
        seed = f"{seed}"
        chi = f"{chi}"
        req_ram_mem_gb = f"{request_memory_gb}"

        job_params_dicts.append( {
            "outfile"       : results_fullpath,                 # outfile
            "job_type"      : job_type,                         # job_type
            "req_mem_gb"    : req_ram_mem_gb,                   # req_ram_mem_gb
            "seed"          : seed,                             # seed
            "method"        : method,                           # method
            "D"             : D,                                # D
            "N"             : N,                                # N
            "chi"           : chi,                              # chi
            "result_keys"   : _encode_list_as_str(result_keys)  # result_keys
        })

    ## Print:
    for params in job_params_dicts:
        _print_inputs(params, _max_str_per_key)

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
    print(f"    requestMemory={request_memory_bytes}-bytes")
    print(f"    Arguments={Arguments}")


    if LOCAL_TEST:
        import subprocess
        for params_dict in job_params_dicts:
            args = ["python", worker_script_fullpath] + list(params_dict.values())
            subprocess.run(args)

    else:
        import CondorJobSender
        CondorJobSender.send_batch_of_jobs_to_condor(
            worker_script_fullpath,
            output_files_prefix,
            job_params_dicts,
            request_cpus=f"{request_cpus}",
            requestMemory=f"{request_memory_gb}gb",
            # requestMemory=f"{request_memory_bytes}",
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

    exceptions = {"outfile", "result_keys"}

    total_string = ""
    for key, value in inputs.items():
        s = f"{value}"
        if key in exceptions:
            total_string += " " + f"{key!r}: " + f"()" + ","
            continue

        if key in _max_str_per_key:
            max_length = _max_str_per_key[key]
            crnt_length = len(f"{value}")
            pad_length = max_length-crnt_length
            if pad_length>0:
                s = " "*pad_length + s

        total_string += " " + f"{key!r}: " + s + ","

    total_string = total_string.removesuffix(",")
    print(total_string)
        

def _time_stamp():
    t = time.localtime()
    return f"{t.tm_year}.{t.tm_mon:02}.{t.tm_mday:02}_{t.tm_hour:02}.{t.tm_min:02}.{t.tm_sec:02}"

def _random_letters(num:int)->str:
    s = ""
    for _ in range(num):
        s += random.choice(string.ascii_letters)
    return s


def _legit_memory_sizes(request_memory_gb:int) -> int:
    request_memory_gb = int(request_memory_gb)

    if RAM_MEMORY_IN_2_EXPONENTS:
        for x in range(10):
            if request_memory_gb <= 2**x:
                return 2**x
        raise ValueError(f"request_memory_gb={request_memory_gb}. Not a legit value!")
    
    else:
        return request_memory_gb


if __name__=="__main__":
    main()