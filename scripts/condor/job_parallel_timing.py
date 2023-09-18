import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent)
    )
    import _import_src
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )


from scripts.test_parallel import test_parallel_execution_time


def main(
    D : int = 2,
    N : int = 2,
    method : int = 0   
) -> dict:
    
    ## Parse:
    if method==0:
        parallel = False
    else:
        parallel = True


    ## Run:
    bp_time, reduction_time = test_parallel_execution_time(D=D, N=N, parallel=parallel)

    ## Collect as a dict:
    # res_keys = ["parallel", 'D', 'N', 'seed', 'bp-step', 'reduction']
    results = {
        "parallel" : parallel,
        'D' : D,
        'N' : N, 
        'bp-step' : bp_time,
        'reduction' : reduction_time
    }

    return results



if __name__ == "__main__":    
    main(h=2.2, method=2)