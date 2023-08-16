import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn, UnitCell

# Measure core data
from algo.core_measurements import measure_xyz_expectation_values_with_tn


# usefull utils:
from utils import visuals, saveload, csvs
from matplotlib import pyplot as plt

# For tests:
from time import perf_counter


def contract_to_core_test(
    d = 2,
    D = 2,
    max_N = 12
):

    ## Graphs:
    time_plot = visuals.AppendablePlot()
    error_plot = visuals.AppendablePlot()

    time_plot.axis.set_xlabel("N")
    time_plot.axis.set_ylabel("time [sec]")
    time_plot.axis.set_title("Execution Time Comparison")

    error_plot.axis.set_xlabel("N")
    error_plot.axis.set_ylabel("results-error")
    error_plot.axis.set_title("Same Results check")

    
    ## Load or randomize unit_cell
    unit_cell= UnitCell.load(f"random_D={D}")
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)
        unit_cell.save(f"random_D={D}")

    for N in range(2, max_N+1):
        times, results = dict(), dict()
        ## contract network:
        tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
        # tn, messages, stats = belief_propagation(tn, None, bp_config)
        tn.connect_random_messages()
        for reduce in [False, True]:
            t1 = perf_counter()
            res = measure_xyz_expectation_values_with_tn(tn, reduce=reduce)
            t2 = perf_counter()
            time = t2-t1
            print("   ")
            print(f"For reduce={reduce!r}")
            print(f"res={res}")  
            print(f"time={time}")
            results[reduce] = res
            times[reduce] = time

        error = abs(results[True]['x'].A - results[False]['x'].A )
        time_plot.append(core=(N, times[True]), reduced=(N, times[False]))
        error_plot.append(error=(N, error))

    print("Done")

    

def main_test():
    contract_to_core_test()


if __name__ == "__main__":
    main_test()