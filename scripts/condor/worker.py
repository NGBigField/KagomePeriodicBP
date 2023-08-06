if __name__ == "__main__":
    import pathlib, sys 
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from sys import argv

# Import scripts to use
import _import_scripts

# Import DictWriter class from CSV module
from time import perf_counter
from csv import DictWriter
from typing import Any

# Import the possible job types:
from scripts.condor.job_bp import main as run_job_bp


NUM_EXPECTED_ARGS = 8


# A main function to parse inputs:
def main():

    ## Check function call:
    assert len(argv)==NUM_EXPECTED_ARGS, f"Expected {NUM_EXPECTED_ARGS} arguments. Got {len(argv)}."

    ## Parse args:
    print(f"The {NUM_EXPECTED_ARGS} arguments are:")

    i = 0  # 0
    this_func_name = argv[i]
    print(f"{i}: this_func_name={this_func_name!r}")

    i += 1
    output_file = argv[i]
    print(f"{i}: output_file={output_file!r}")

    i += 1  # 2
    seed = int(argv[i])
    print(f"{i}: seed={seed}")

    i += 1  # 3
    method = int(argv[i])
    print(f"{i}: method={method}")

    i += 1  # 4
    D = int(argv[i])
    print(f"{i}: D={D}")

    i += 1
    N = int(argv[i])
    print(f"{i}: N={N}")

    i += 1  # 6
    job_type = argv[i]
    print(f"{i}: job_type={job_type}")

    i += 1  # 7
    result_keys = _parse_list_of_strings(argv[i])
    print(f"{i}: result_keys={result_keys}")


    ## Run:
    results : dict[str, Any]
    t1 = perf_counter()
    try:
        if job_type=="bp":
            results = run_job_bp(D=D, N=N, method=method)
        else: 
            raise ValueError(f"Not an expected job_type={job_type!r}")
    except Exception as e:
        results = dict(
            e=e
        )
    t2 = perf_counter()

    print(f"res={results}")
    
    ## check output:
    assert isinstance(results, dict), f"result must be of type `dict`! got {type(results)}"
    results["exec_time"]=t2-t1
    results["seed"]=seed

    ## Prepare result for printing:
    row_to_print = dict()
    for result_key in result_keys:
        try:
            res = results[result_key]
        except KeyError:
            res = None
        row_to_print[result_key] = res

    ## Write result:
    with open( output_file ,'a') as f:
        dict_writer = DictWriter(f, fieldnames=result_keys )
        dict_writer.writerow(row_to_print)
        f.close()



def _get_keys_from_sender()->list[str]:
    from scripts.condor.sender import result_keys
    return result_keys

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



if __name__ == "__main__":
    main()

