import _import_src  ## Needed to import src folders when scripts are called from an outside directory


import project_paths

# Tensor-Networks creation:
from tensor_networks.construction import create_repeated_kagome_tn, UnitCell

# BP:
from algo.belief_propagation import belief_propagation, robust_belief_propagation

# Config and containers:
from containers import BPConfig
from tensor_networks import KagomeTNRepeatedUnitCell, CoreTN, ModeTN, EdgeTN

# config:
from enums import UpdateMode, UnitCellFlavor
from containers import UpdateEdge, Config

# Measure core data
from algo.measurements import derive_xyz_expectation_values_with_tn, calc_unit_cell_expectation_values_from_tn, pauli, measure_energies_and_observables_together
from algo.tn_reduction import reduce_tn
from physics import hamiltonians

# Performance:
from time import perf_counter, sleep

# Numpy for math stuff:
import numpy as np

# useful utils:
from utils import visuals, saveload, csvs, dicts, lists
from utils.visuals import AppendablePlot
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms


from libs import TenQI

## common config:
mode = UpdateMode.A
update_edge = UpdateEdge(UnitCellFlavor.A, UnitCellFlavor.B)
d=2

  

def _get_expectation_values(edge_tn:EdgeTN) -> float:
    rho = edge_tn.rdm
    rho_i = np.trace(rho, axis1=2, axis2=3)
    z = np.trace(rho_i @ pauli.z,)
    z = float(z)
    return z


def _per_D_N_single_test(
    D:int, N:int, method:str, config:Config, parallel:bool,
    **kwargs
) -> tuple[
    EdgeTN, 
    float,  # z
    float  # energy
]:
    
    config = config.copy()

    ## Create kagome TN:
    unit_cell = UnitCell.load_best(D)
    tn = create_repeated_kagome_tn(d, D, N, unit_cell)

    ## Get environment:
    match method:   #["random", "exact","bp"]
        case "random"|"exact":
            tn.connect_uniform_messages()
        case "bp":
            belief_propagation(tn, None, config.bp, update_plots_between_steps=False)
            
    ## Define contraction precision and other config andjustments:
    match method:
        case "exact":
            config.chi = 1e3*D**2
        case "bp":
            config.set_parallel(parallel)

    ## Contract to edge:
    edge_tn = reduce_tn(
        tn, EdgeTN, config.contraction, 
        mode=mode, edge_tuple=update_edge
    )
    ## Get results:
    z = _get_expectation_values(edge_tn)
    h = hamiltonians.heisenberg_afm()
    energy = np.dot(edge_tn.rdm.flatten(),  h.flatten()) 

    return edge_tn, z, energy


def _get_full_converges_run_plots(
    combined_figure:bool, 
    live_plots:bool=True
) -> tuple[AppendablePlot, ...]:
    plot_values = AppendablePlot()
    plt.xlabel("linear system size", fontsize=12)
    plt.ylabel("$\left\langle Z \right\rangle>$", fontsize=12)
    plot_values.axis.set_yscale('log')

    if combined_figure:
        plot_times  = plot_values.get_twin_plot()
    else:
        plot_times  = AppendablePlot()
        plt.xlabel("linear system size", fontsize=12)
    plt.ylabel("Computation time [sec]", fontsize=12)

    p_values_vs_times = AppendablePlot()
    plt.xlabel("Computation time [sec]", fontsize=12)
    plt.ylabel(r"<Z>", fontsize=12)

    if live_plots:
        visuals.draw_now()
    return plot_values, plot_times, p_values_vs_times


def test_bp_convergence_single_run(
    D:int, N:int, method:str, config:Config,
    parallel:bool
)->None:

    _t1 = perf_counter()
    edge_tn, z, energy = _per_D_N_single_test(
        D, N, method, config, parallel
    )
    _t2 = perf_counter()
    t = _t2 - _t1

    return z, t

def test_bp_convergence_all_runs(
    D:int = 3,
    combined_figure:bool = False,
    live_plots:bool|None = None,  # True: live, False: at end, None: no plots
    parallel:bool = True,
) -> None:
    
    ##  Params:
    methods_and_Ns:dict[str: list[int]] = dict(
        # exact  = [2, 3, 4],
        random = [2, 3, 4, 5, 6, 7, 8, 9, 10],
        bp     = [2, 3, 4, 5]
    ) 

    ## config:
    config = Config.derive_from_dimensions(D)
    config.chi_bp = 2*D**2 
    config.chi    = 2*D**2 + 10
    config.bp.msg_diff_terminate = 1e-6

    # exact_config = config.copy()

    ## Prepare plots and csv:
    if live_plots is not None:
        p_values, p_times, p_values_vs_times = _get_full_converges_run_plots(combined_figure, live_plots)
    csv = csvs.CSVManager(["D", "N", "method", "z", "energy" "time"])

    line_styles =   {'bp': "-", 'random': '--', 'exact':":"}
    marker_styles = {'bp': "X", 'random': 'o' , 'exact':"^"}
    def _get_plt_kwargs(method:str)->str:
        return dict(
            linestyle=line_styles[method], 
            marker=marker_styles[ method]
        )


    for method, Ns in methods_and_Ns.items(): 
        print(f"method={method}")
        
        for N in Ns:
            print(f"  N={N}")
            sleep(0.1)

            ## Get results:
            _t1 = perf_counter()
            try:
                edge_tn, z, energy = _per_D_N_single_test(
                    D, N, method, config, parallel
                )
            except Exception:
                break
            _t2 = perf_counter()
            time = _t2 - _t1

            ## plot:
            if live_plots is not None:
                plt_kwargs = _get_plt_kwargs(method)
                p_values.append(plot_kwargs=plt_kwargs ,**{ method : (N, z)}, draw_now_=live_plots)
                p_times.append( plot_kwargs=plt_kwargs ,**{ method : (N, time)}, draw_now_=live_plots)
                p_values_vs_times.append( 
                                plot_kwargs=plt_kwargs ,**{ method : (time, z)}, draw_now_=live_plots
                )

            # csv:
            csv.append([D, N, method, z, energy, time])

    # plot_values.axis.set_ylim(bottom=1e-8*2)
    if live_plots is not None:
        visuals.save_figure(p_values.fig)
    print("Done")    


def plot_bp_convergence_results(
    filename:str = "2024.07.26_22.01.47 YRU"
) -> None:
    
    ## Get Data:
    fullpath = saveload._fullpath(filename+".csv", sub_folder="results")
    table = csvs.TableByKey(fullpath)

    ## Derive params:
    Ds = table.unique_values("D")
    methods = table.unique_values("method")

    ## Prepare plots:
    ax_z_vs_t = plt.subplot(1,1,1)
    ax_z_vs_t.set_xlabel("compute time [sec]")
    ax_z_vs_t.set_ylabel("expectation value <Z>")
    ax_z_vs_t.grid()

    ## Get lists and plot:
    for D in Ds:
        for method in methods:
            t, z = [], []
            data = table.get_matching_table_elements(D=D, method=method)

            print(data)
            N = data['N']
            z = data['z']
            t = data['t']
            
            ax_z_vs_t.plot(t, z, label=method, linewidth=4)
            
    visuals.draw_now()
    ax_z_vs_t.legend()
    print("Done")
    

def _calc_inf_exact_results(D:int) -> dict:
    ## Config:
    method = "exact"
    N = 10
    config = Config.derive_from_dimensions(D)

    ## Compute:
    edge_tn, z, energy = _per_D_N_single_test(D, N, method, config, parallel=False)

    ## Save:
    res = dict(
        edge_tn=edge_tn,
        z=z,
        energy=energy,
        D=D,
        N=N,
    )

    return res
    

ResultsSubFolderName = "results"
def get_inf_exact_results(D:int=3) -> dict:
    filename = f"infinite_exact_results D={D}"
    if saveload.exist(filename, sub_folder=ResultsSubFolderName):
        results = saveload.load(filename, sub_folder=ResultsSubFolderName)
    else:
        results = _calc_inf_exact_results(D)
        saveload.save(results, filename, sub_folder=ResultsSubFolderName)
    return results


def main_test():
    get_inf_exact_results()
    # test_bp_convergence_all_runs()
    # plot_bp_convergence_results()

if __name__ == "__main__":
    main_test()