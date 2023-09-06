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
    mean_energy, num_tensors, chi = test_bp_convergence_steps_single_run(N=N, D=D)
    
    # Expected outputs: 
    # ['seed', 'D', 'N', 'chi', 'iterations', 'num_tensors', 'energy', 'exec_time']
    results = dict(
        D=D,
        N=N,
        chi=chi,
        iterations=0,
        num_tensors=num_tensors,
        energy=mean_energy
    )
    
    return results


if __name__ == "__main__":    
    main(h=2.2, method=2)

