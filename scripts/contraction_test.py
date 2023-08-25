import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn

# Types in the code:
from enums import UpdateMode, UnitCellFlavor
from containers import UpdateEdge
from tensor_networks.unit_cell import UnitCell
from lattices.directions import BlockSide

# Algos we test here:
from algo.measurements import derive_xyz_expectation_values_with_tn, derive_xyz_expectation_values_using_rdm, print_results_table
from algo.tn_reduction import reduce_full_kagome_to_core, reduce_core_to_mode, reduce_mode_to_edge

# useful utils:
from utils import visuals, dicts, saveload

# For testing and showing performance:
from time import perf_counter
from matplotlib import pyplot as plt

# Useful for math:
import numpy as np



A = UnitCellFlavor.A
B = UnitCellFlavor.B
C = UnitCellFlavor.C


def load_or_randomize_unit_cell(d, D)->UnitCell:
    unit_cell= UnitCell.load(f"random_D={D}")
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)
        unit_cell.save(f"random_D={D}")
    return unit_cell


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
            res = derive_xyz_expectation_values_with_tn(tn, reduce=reduce)
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
    res = derive_xyz_expectation_values_with_tn(full_tn, reduce=False)
    print(res['x'])
    # contract network:
    core_tn = reduce_full_kagome_to_core(full_tn, trunc_dim=chi)
    # base results:
    res = derive_xyz_expectation_values_with_tn(core_tn, reduce=False)
    print(res['x'])

    for mode in UpdateMode:
    
        # contract to mode:
        mode_tn = reduce_core_to_mode(core_tn.copy(), bubblecon_trunc_dim=chi, mode=mode)
        res = derive_xyz_expectation_values_with_tn(mode_tn, reduce=False)
        print(res['x'])

    print("Done")


def contract_to_edge_test(
    d = 2,
    D = 2,
    chi = 14,
    N = 3,
    with_bp:bool = True
):
    ## Load or randomize unit_cell
    unit_cell = load_or_randomize_unit_cell(d, D)

    mode = UpdateMode.A
    edge = UpdateEdge(C, B)
    
    ##Contraction Sequence:
    full_tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
    if with_bp:
        messages = saveload.load("last_messages", if_exist=True)
        from algo.belief_propagation import belief_propagation, BPConfig
        bp_config=BPConfig(max_swallowing_dim=chi//2, target_msg_diff=1e-6)
        _, messages, _ = belief_propagation(full_tn, messages, bp_config=bp_config)
        saveload.save(messages, "last_messages")
    else:        
        full_tn.connect_random_messages()

    core_tn = reduce_full_kagome_to_core(full_tn, trunc_dim=chi, direction=BlockSide.U)
    mode_tn = reduce_core_to_mode(core_tn, mode=mode)
    edge_tn = reduce_mode_to_edge(mode_tn, edge, trunc_dim=chi)
    print("Done")

    ## Get measurements in two different methods:
    print("Using TN Contraction:")
    results1 = derive_xyz_expectation_values_with_tn(core_tn, bubblecon_trunc_dim=chi, force_real=False)
    print_results_table(results1)

    print("")

    print("Using RDMs:")
    results2 = derive_xyz_expectation_values_using_rdm(edge_tn, force_real=False)
    print_results_table(results2)

    print("")

    print("Diff:")
    diff = dicts.subtract(results1, results2)
    print_results_table(diff)


    print("")
    print("Done")



def test_all_edges_contraction(
    d = 2,
    D = 3,
    chi = 16,
    N = 4,
    with_bp:bool = False,
    real_results = False
):
    accumulted_diff = 0.0

    ## Load or randomize unit_cell
    unit_cell = load_or_randomize_unit_cell(d, D)

    ## Full tn with messages:
    full_tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
    if with_bp:
        from algo.belief_propagation import belief_propagation, BPConfig
        bp_config=BPConfig(max_swallowing_dim=chi, target_msg_diff=1e-7)
        belief_propagation(full_tn, bp_config=bp_config)
    else:        
        full_tn.connect_random_messages()

    ## Core tn:
    core_tn = reduce_full_kagome_to_core(full_tn, trunc_dim=chi)
    results_base = derive_xyz_expectation_values_with_tn(core_tn, bubblecon_trunc_dim=chi, force_real=real_results)

    ## Mode tn:
    for mode in UpdateMode.all_options():
        mode_tn = reduce_core_to_mode(core_tn, mode=mode)

        ## Edge TN:
        for edge in UpdateEdge.all_options():        
            print(f"mode={mode} ; edge={edge} :")   
            edge_tn = reduce_mode_to_edge(mode_tn, edge, trunc_dim=chi)

            results_rdms = derive_xyz_expectation_values_using_rdm(edge_tn, force_real=real_results)
            diff = dicts.subtract(results_base, results_rdms)
            print_results_table(diff)
            print(" ")

            accumulted_diff += dicts.accumulate_values(diff, function=lambda x: abs(x))

    print(f"accumulted absolute diff = {accumulted_diff}")
    mean_accumulted_diff = accumulted_diff/(3*6*6)
    print(f"mean absolute diff       = {mean_accumulted_diff}")


    print("Done")
    return mean_accumulted_diff


def test_all_edges_contraction_accumulated_diff_with_chi(
    d = 2,
    D = 3,
    N = 4,
    with_bp:bool = True,
    real_results = False ,       
    chis = range(2, 61)
):
    title_str = f"Core-TN Contraction vs RDM from Edge-TN \nD={D}"
    filename_str = f"Core-TN Contraction vs RDM from Edge-TN D={D}"

    plot = visuals.AppendablePlot()
    plot.axis.set_xlabel("chi")
    plot.axis.set_ylabel("diff")
    plot.axis.set_title(title_str)
    plot.axis.set_ylim(bottom=0.0)
    plt.grid("on")
    visuals.draw_now()

    diffs = []
    for chi in chis:
        diff = test_all_edges_contraction(d=d, D=D, chi=chi, N=N, real_results=real_results, with_bp=with_bp)
        plot.append(diff=(chi, diff))
        diffs.append(diff)
        plot.axis.set_ylim(top=max(diffs))

    print("All tests done")
    visuals.save_figure(file_name=filename_str)
    print("Done saving")
    

    
def main_test():
    # contract_to_core_test()
    # contract_to_mode_test()
    # contract_to_edge_test()
    # test_all_edges_contraction()
    test_all_edges_contraction_accumulated_diff_with_chi()
    


if __name__ == "__main__":
    main_test()