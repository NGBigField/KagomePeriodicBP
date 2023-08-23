import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn, UnitCell

# Types in the code:
from enums import UpdateMode, UnitCellFlavor
from containers import UpdateEdge

# Algos we test here:
from algo.core_measurements import measure_xyz_expectation_values_with_tn
from algo.tn_reduction import reduce_full_kagome_to_core, reduce_core_to_mode, reduce_mode_to_edge_and_env

# useful utils:
from utils import visuals

# For tests:
from time import perf_counter


A = UnitCellFlavor.A
B = UnitCellFlavor.B
C = UnitCellFlavor.C


def contract_to_core_test(
    d = 2,
    D = 2,
    max_N = 8
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


def contract_to_mode_test(
    d = 2,
    D = 2,
    chi = 8,
    N = 2,
):
    ## Load or randomize unit_cell
    unit_cell= UnitCell.load(f"random_D={D}")
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)
        unit_cell.save(f"random_D={D}")
    
    # Full tn:
    full_tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
    full_tn.connect_random_messages()
    res = measure_xyz_expectation_values_with_tn(full_tn, reduce=False)
    print(res['x'])
    # contract network:
    core_tn = reduce_full_kagome_to_core(full_tn, bubblecon_trunc_dim=chi)
    # base results:
    res = measure_xyz_expectation_values_with_tn(core_tn, reduce=False)
    print(res['x'])

    for mode in UpdateMode:
    
        # contract to mode:
        mode_tn = reduce_core_to_mode(core_tn.copy(), bubblecon_trunc_dim=chi, mode=mode)
        res = measure_xyz_expectation_values_with_tn(mode_tn, reduce=False)
        print(res['x'])

    print("Done")


def contract_to_edge_test(
    d = 2,
    D = 2,
    chi = 8,
    N = 4,
):
    ## Load or randomize unit_cell
    unit_cell= UnitCell.load(f"random_D={D}")
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)
        unit_cell.save(f"random_D={D}")

    mode = UpdateMode.A
    edge = UpdateEdge(A, B)
    
    ##Contraction Sequence:
    full_tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
    full_tn.connect_random_messages()
    core_tn = reduce_full_kagome_to_core(full_tn, bubblecon_trunc_dim=chi)
    mode_tn = reduce_core_to_mode(core_tn, mode=mode)
    edge_tn = reduce_mode_to_edge_and_env(mode_tn, edge)
    print("Done")


    
def main_test():
    # contract_to_core_test()
    # contract_to_mode_test()
    contract_to_edge_test()
    


if __name__ == "__main__":
    main_test()