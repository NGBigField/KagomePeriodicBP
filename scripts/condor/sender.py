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

# result_keys = ["seed", "method", "D", "h", "x", "y", "z", "e", "exec_time"]
result_keys = ["with_bp", 'D', 'N', 'A_X', 'A_Y', 'A_Z', 'B_X', 'B_Y', 'B_Z', 'C_X', 'C_Y', 'C_Z']

## all values:
vals = {}
vals['N'] = range(2, 20, 2)
vals['D'] = [2, 3, 4]
vals['method'] = [0, 1]
vals['seed'] = [0]

def main(
    num_seeds:int=1, 
    job_type="bp",  # "ite_it" / ite_it_all_h / "bp"
    request_cpus:int=4
):

    ## Define paths and names:
    sep = os.sep
    this_folder_path = pathlib.Path(__file__).parent.__str__()
    #
    worker_script_fullpath  = this_folder_path+sep+"worker.py"
    results_fullpath        = results_dir+sep+"results_"+job_type+".csv"
    output_files_prefix     = "kagome-bp-"+job_type
    #
    print(f"script_fullpath={worker_script_fullpath!r}")
    print(f"results_fullpath={results_fullpath!r}")
    print(f"output_files_prefix={output_files_prefix!r}")
    print(f"job_type={job_type!r}")

    ## Define job params:
    job_params : list[dict] = []
    for N, D, method, seed in product(vals['N'], vals['D'], vals['method'], vals['seed']):
        # To strings:
        N = f"{N}"
        D = f"{D}"
        method = f"{method}"
        seed = f"{seed}"

        job_params.append( dict(
            outfile=results_fullpath,
            N=N,
            seed=seed,
            method=method,
            job_type=job_type,
            result_keys=_encode_list_as_str(result_keys)
        ))

    for params in job_params:
        params2print = deepcopy(params)
        params2print.pop("outfile")
        print(params2print)

    ## Prepare output file:    
    with open( results_fullpath ,'a') as f:        
        dict_writer = DictWriter(f, fieldnames=result_keys)
        dict_writer.writerow({field:field for field in result_keys})
        f.close()

    ## Call condor:
    import CondorJobSender
    CondorJobSender.send_batch_of_jobs_to_condor(
        worker_script_fullpath,
        output_files_prefix,
        job_params,
        request_cpus=f"{request_cpus}",
        requestMemory='4gb',
        Arguments='$(outfile) $(seed) $(method) $(h) $(N) $(job_type) $(result_keys)'
    )

    print("Called condor successfully")


def _encode_list_as_str(lis:list)->str:
    assert isinstance(lis, list)
    s = '['
    for item in lis:
        s += f'{item},'
    s = s[:-1]+']'
    return s



if __name__=="__main__":
    main()