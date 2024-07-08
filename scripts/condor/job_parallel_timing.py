import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent)
    )
    import _import_src
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )


from scripts.tests.parallel import test_parallel_execution_time, _single_test_parallel_execution_time_single_bp_step


def main(
    D : int = 2,
    N : int = 2,
    parallel : int = 0   
) -> dict:
    
    ## Parse:
    if parallel==0:
        parallel = False
    else:
        parallel = True


    ## Run:
    bp_step_time = _single_test_parallel_execution_time_single_bp_step(D=D, N=N, parallel=parallel)

    ## Collect as a dict:
    # res_keys = ["parallel", 'D', 'N', 'seed', 'bp-step', 'reduction']
    results = {
        "parallel" : parallel,
        'D' : D,
        'N' : N, 
        'bp_step' : bp_step_time
    }

    return results



if __name__ == "__main__":    
    main(h=2.2, method=2)