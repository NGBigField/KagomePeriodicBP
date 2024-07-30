import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent)
    )
    import _import_src
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )


from scripts.tests.bp import _per_D_N_single_test, ComparisonResultsType
from containers.configs import Config
from time import perf_counter


def _parse_method(method:int) -> str:
    match method:
        case 0: 
            return "exact"
        case 1:
            return "random"
        case 2:
            return "bp"
    raise ValueError("Not an expected case")




def main(
    D : int = 2,
    N : int = 2,
    chi : float = 1, 
    method : int = 1, 
    parallel : int = 0
) -> dict:
    
    config = Config.derive_from_dimensions(D)
    parallel_in = bool(parallel)

    ## Parse inputs:
    method_in = _parse_method(method)
    config.chi *= chi
    # config.chi_bp *= chi

    ## Run:
    t1 = perf_counter()
    res : ComparisonResultsType = _per_D_N_single_test(D, N, method_in, config, parallel_in)
    t2 = perf_counter()
    time = t2-t1
    
    # Expected outputs: 
    # ["seed", "D", "N", "method", "time", "energy", "z", "fidelity"]

    results = dict(
        D=D ,
        N=N ,
        method=method_in ,
        time=time ,
        energy=res.energy ,
        z=res.z ,
        fidelity=res.fidelity       
    )
    
    return results


if __name__ == "__main__":    
    main(h=2.2, method=2)

