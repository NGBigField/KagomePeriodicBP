if __name__ == "__main__":
    import _import_main_and_src


import pathlib, sys, os

# Import DictWriter class from CSV module
from csv import DictWriter

# for smart iterations:
from itertools import product

# for cached look-ups
from functools import cache

# for user hinting:
from typing import Literal, TypeAlias

# for time stamps and random strings:
import string
import random
import time

# for paths
import project_paths

sep = os.sep
_results_dir = project_paths.condor_paths['io_dir']/'data'/'condor'
results_dir_str : str = _results_dir.__str__()
if not os.path.exists(results_dir_str):
    os.makedirs(results_dir_str)

RAM_MEMORY_IN_2_EXPONENTS = False

DEFAULT_RESULT_KEYS_DICT = dict(
    parallel_timings = ["parallel", 'D', 'N', 'seed', 'bp_step'],
    ite_afm = ["seed","D", "N", "chi", "energy", "parallel", "method", "path"],
    bp = ["seed", "D", "N", "method", "time", "energy", "z", "fidelity"]
)

## all values:
DEFAULT_VALS = {}
DEFAULT_VALS['D'] = [2]
DEFAULT_VALS['N'] = [2] # list(range(3, 6))
DEFAULT_VALS['chi'] = [1, 1.5, 2]
DEFAULT_VALS['method'] = [1, 3]
DEFAULT_VALS['seed'] = list(range(3))
DEFAULT_VALS['parallel'] = [0]
DEFAULT_VALS['noise_initial'] = [1e-3, 1e-2, 1e-1, 0]
DEFAULT_VALS['noise_per_segment'] = [1e-3, 1e-2, 1e-1, 0, 1e1]
DEFAULT_VALS['control'] = [1, 0]

Arguments = '$(outfile) $(job_type) $(req_mem_gb) $(seed) $(method) $(D) $(N) $(chi) $(parallel) $(control) $(noise_initial) $(noise_per_segment) $(result_keys)'
JobType : TypeAlias = Literal["ite_afm", "bp", "parallel_timings"]


def main(
    job_type:JobType="ite_afm",  
    request_cpus:int=2,
    request_memory_gb:int=8,
    vals:dict=DEFAULT_VALS,
    result_file_name:str|None=None,
    result_keys:list[str]|None=None,
    _local_test = False
):

    ## Check inputs and fix:
    _max_str_per_key = {key:max((len(str(val)) for val in lis)) for key, lis in vals.items()}
    if result_file_name is None:
        result_file_name="results_"+job_type
    # Memory:
    request_memory_gb = _legit_memory_sizes(request_memory_gb)
    # CPUs:
    if "parallel" in vals and (1 in vals["parallel"] or True in vals["parallel"]):
        request_cpus = max(request_cpus, 10)
    # keys:
    if result_keys is None:
        result_keys = DEFAULT_RESULT_KEYS_DICT[job_type]

    ## Define paths and names:
    this_folder_path = pathlib.Path(__file__).parent.__str__()  # This folder
    #
    worker_script_fullpath = this_folder_path+sep+"worker.py"
    results_fullpath       = results_dir_str+sep+result_file_name+".csv"
    output_files_prefix    = "kagome-"+job_type+"-"+_time_stamp()+"-"+_random_letters(3)
    #
    print(f"script_fullpath={worker_script_fullpath!r}")
    print(f"results_fullpath={results_fullpath!r}")
    print(f"output_files_prefix={output_files_prefix!r}")
    print(f"job_type={job_type!r}")

    ## Define job params:
    req_ram_mem_gb = f"{request_memory_gb}"
    job_params_dicts : list[dict] = []
    for N, D, method, seed, chi, parallel, noise_initial, noise_per_segment, control in \
            product(vals['N'], vals['D'], vals['method'], vals['seed'], vals['chi'], vals['parallel'], vals['noise_initial'], vals['noise_per_segment'],  vals['control'] ):
        # To strings:
        N = f"{N}"
        D = f"{D}"
        method = f"{method}"
        seed = f"{seed}"
        chi = f"{chi}"
        parallel = f"{parallel}"
        noise_initial = f"{noise_initial}"
        noise_per_segment = f"{noise_per_segment}"
        control = f"{control}"

        job_params_dicts.append( {
            "outfile"           : results_fullpath,                 # outfile
            "job_type"          : job_type,                         # job_type
            "req_mem_gb"        : req_ram_mem_gb,                   # req_ram_mem_gb
            "seed"              : seed,                             # seed
            "method"            : method,                           # method
            "D"                 : D,                                # D
            "N"                 : N,                                # N
            "chi"               : chi,                              # chi
            "parallel"          : parallel,                         # parallel
            "noise_initial"     : noise_initial,                    # noise_initial
            "noise_per_segment" : noise_per_segment,                # noise_per_segment
            "control"           : control,                          # control
            "result_keys"       : _encode_list_as_str(result_keys)  # result_keys
        })

    ## Print:
    for params in job_params_dicts:
        _print_inputs(params, _max_str_per_key)

    ## Prepare output file:    
    with open( results_fullpath ,'a') as f:        
        dict_writer = DictWriter(f, fieldnames=result_keys)
        dict_writer.writerow({field:field for field in result_keys})
        f.close()

    ## Prepare IO data folder:

    ## Call condor:
    print(f"Calling condor with shared inputs:")
    print(f"    output_files_prefix={output_files_prefix}")
    print(f"    request_cpus={request_cpus}")
    print(f"    requestMemory={request_memory_gb}gb")
    # print(f"    requestMemory={request_memory_bytes}-bytes")
    print(f"    Arguments={Arguments}")

    if _local_test:
        from src import project_paths
        project_paths.add_src()
        from utils.prints import print_warning
        print_warning(f"Running a local test!")
        import subprocess

        for params_dict in job_params_dicts:
            arguments = ["python", worker_script_fullpath]
            for argument_name in _split_argument():
                argument = params_dict[argument_name]
                arguments += [argument]
            subprocess.run(arguments)

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

    expectations_reduce = {"outfile", "result_keys"}
    expectations_omit = {"req_mem_gb"}

    total_string = ""
    for key, value in inputs.items():
        assert key in Arguments, f"key {key!r} is not in input arguments"

        if key in expectations_omit:
            continue

        s = f"{value}"
        if key in expectations_reduce:
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


@cache
def _split_argument() -> list[str]:
    res = []
    splitted = Arguments.split("$")
    for i, s in enumerate(splitted):

        # ignore first string:
        if i==0:
            continue

        _, s = s.split("(")
        s, _ = s.split(")")

        res.append(s)
    return res



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