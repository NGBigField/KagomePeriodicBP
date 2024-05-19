import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent)
    )
    import _import_src
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )


from utils import strings
from scripts.run_ite import main as run_ite


def main(
    D : int = 2,
    N : int = 2,
    chi_factor : int = 1,
    seed : int = -1
) -> dict:
    

    results_filename = f"AFM_D={D}_N={N}"
    if seed == -1:
        results_filename += "_"+strings.time_stamp()+"_"+strings.random(3)
    else:
        results_filename += f"_{seed}"

    ## Run:
    energy, unit_cell_file_path = run_ite(N=N, D=D, chi_factor=chi_factor, live_plots=False, parallel=False, results_filename=results_filename)
    
    # Expected outputs: 
    # ["seed","D", "N", "energy", "path"]
    results = dict(
        D=D,
        N=N,
        chi=chi_factor,
        energy=energy,
        path=unit_cell_file_path
    )
    
    return results


if __name__ == "__main__":    
    main()

