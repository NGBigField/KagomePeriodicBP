import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent)
    )
    import _import_src
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )


from scripts.test_bp import test_bp_convergence_steps_single_run


def main(
    D : int = 2,
    N : int = 2
) -> dict:
    
    ## Run:
    iterations, chi, time_random, time_bp, z_none, z_bp, diff_random, diff_bp = test_bp_convergence_steps_single_run(N=N, D=D, parallel_bp=True)
    
    # Expected outputs: 
    # ['seed', 'D', 'N', 'chi', 'iterations', 'rdm_diff_bp', 'rdm_diff_random', 'z_bp', 'z_random', 'time_bp', 'time_random']
    results = dict(
        D=D,
        N=N,
        chi=chi,
        iterations=iterations,
        rdm_diff_random=diff_random,
        rdm_diff_bp=diff_bp,
        z_random=z_none,
        z_bp=z_bp,
        time_bp=time_bp,
        time_random=time_random
    )
    
    return results


if __name__ == "__main__":    
    main(h=2.2, method=2)

