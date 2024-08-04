import _import_src  ## Needed to import src folders when scripts are called from an outside directory


import project_paths

from typing import NamedTuple

# Tensor-Networks creation:
from tensor_networks.construction import create_repeated_kagome_tn, UnitCell

# BP:
from algo.belief_propagation import belief_propagation, robust_belief_propagation

# Config and containers:
from tensor_networks import EdgeTN
from enums import UpdateMode, UnitCellFlavor
from containers import UpdateEdge, Config

# Measure core data
from algo.measurements import pauli, compute_negativity_of_rdm
from algo.tn_reduction import reduce_tn
from physics import hamiltonians
from physics.metrics import entanglement_entropy

# Performance:
from time import perf_counter, sleep

# Numpy for math stuff:
import numpy as np

# useful utils:
from utils import visuals, saveload, csvs, dicts, lists, strings, files
from utils.visuals import AppendablePlot
from matplotlib import pyplot as plt

## Metrics:
from physics.metrics import fidelity


from libs import TenQI

## common config:
mode = UpdateMode.A
update_edge = UpdateEdge(UnitCellFlavor.A, UnitCellFlavor.B)
d=2

EXACT_CHI = 150
EXACT_N = 7


class CSVRowData(NamedTuple):
    D : int
    N : int
    method : str
    time : float
    energy : float
    z : float
    fidelity : float
    negativity : float
    entanglement_entropy : float

    @classmethod
    def fields(cls) -> list[str]:
        return list(cls._fields)
    
    def row(self) -> list:
        return [self.__getattribute__(field) for field in self._fields]
            


class ComparisonResultsType(NamedTuple):
    edge_tn: EdgeTN
    z : float  
    energy: float  
    fidelity : float
    unit_cell : UnitCell
    negativity : float
    entanglement_entropy : float




def _get_expectation_values(edge_tn:EdgeTN) -> float:
    rho = edge_tn.rdm
    rho_i = np.trace(rho, axis1=2, axis2=3)
    z = np.trace(rho_i @ pauli.z,)
    z = float(z)
    return z


def _per_D_N_single_test(
    D:int, N:int, method:str, config:Config, parallel:bool,
    **kwargs
) -> ComparisonResultsType:
    ## Inputs:
    config = config.copy()

    ## Make sure there is something to compare to:
    this_is_exact_run = 'exact_run' in kwargs and kwargs['exact_run']==True

    if this_is_exact_run:
        unit_cell = UnitCell.load_best(D)
    else:
        exact = get_inf_exact_results(D)
        unit_cell = exact['unit_cell']

    ## Create kagome TN:
    tn = create_repeated_kagome_tn(d, D, N, unit_cell)

    ## Define contraction precision and other config adjustments:
    match method:
        case "exact":
            config.chi = EXACT_CHI
        case "bp":
            config.set_parallel(parallel)

    ## Get environment:
    match method:   #["random", "exact","bp"]
        case "random"|"exact":
            tn.connect_uniform_messages()
        case "bp":
            belief_propagation(tn, None, config.bp)
            

    ## Contract to edge:
    edge_tn = reduce_tn(
        tn, EdgeTN, config.contraction, 
        mode=mode, edge_tuple=update_edge
    )
    ## Get results:
    rdm = edge_tn.rdm
    # expectations:
    z = _get_expectation_values(edge_tn)
    # energy:
    h = hamiltonians.heisenberg_afm()
    energy = np.dot(rdm.flatten(),  h.flatten()) 
    energy = float(energy)
    # negativity:
    negativity = compute_negativity_of_rdm(rdm)
    entanglement_entropy_ = entanglement_entropy(rdm)

    ## Compare with exact results:
    rho_exact = exact['edge_tn'].rdm
    if this_is_exact_run:
        fidelity_ = 0
    else:
        rho_now = TenQI.op_to_mat(edge_tn.rdm)
        rho_exact = TenQI.op_to_mat(rho_exact)
        fidelity_ = 1-fidelity(rho_now, rho_exact)

    return ComparisonResultsType(edge_tn, z, energy, fidelity_, unit_cell, negativity, entanglement_entropy_)


def _get_full_converges_run_plots(
    combined_figure:bool, 
    live_plots:bool=True
) -> tuple[AppendablePlot, ...]:
    
    ## Create figures:
    plot_values = AppendablePlot()
    plt.xlabel("linear system size", fontsize=12)
    plt.ylabel("<Z>", fontsize=12)
    plot_values.axis.set_yscale('log')

    if combined_figure:
        plot_times  = plot_values.get_twin_plot()
    else:
        plot_times  = AppendablePlot()
        plt.xlabel("linear system size", fontsize=12)
    plt.ylabel("Computation time [sec]", fontsize=12)

    p_values_vs_times = AppendablePlot()
    plt.xlabel("Computation time [sec]", fontsize=12)
    plt.ylabel("<Z>", fontsize=12)

    if live_plots:
        visuals.draw_now()

    return plot_values, plot_times, p_values_vs_times


def test_bp_convergence_all_runs(
    D:int|list = 2,
    combined_figure:bool = False,
    parallel:bool = True,
    live_plots:bool|None = None,  # True: live, False: at end, None: no plots
) -> None:
    
    if isinstance(D, list):
        return [
            test_bp_convergence_all_runs(val, combined_figure=combined_figure, parallel=parallel, live_plots=live_plots) 
            for val in D
        ]

    ##  Params:
    methods_and_Ns:dict[str, list[int]] = dict(
        random = list(range(2, 8)),
        bp     = [2, 3, 4]
    ) 

    ## config:
    config = Config.derive_from_dimensions(D)
    config.chi    = 2*D**2 + 4
    config.chi_bp =   D**2 
    config.bp.msg_diff_terminate = 1e-14
    config.bp.max_iterations = 6
    config.bp.visuals.set_all_progress_bars(False)

    # exact_config = config.copy()

    ## Prepare plots and csv:
    if live_plots is not None:
        p_values, p_times, p_values_vs_times = _get_full_converges_run_plots(combined_figure, live_plots)
    csv = csvs.CSVManager(CSVRowData.fields())

    line_styles =   {'bp': "-", 'random': '--', 'exact':":"}
    marker_styles = {'bp': "X", 'random': 'o' , 'exact':"^"}
    def _get_plt_kwargs(method:str)->dict:
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
                res = _per_D_N_single_test(
                    D, N, method, config, parallel
                )
            except Exception as e:
                print(e)
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
            row_data = CSVRowData(D, N, method, time, res.energy, res.z, res.fidelity, res.negativity, res.entanglement_entropy)
            csv.append(row_data.row())

    # plot_values.axis.set_ylim(bottom=1e-8*2)
    if live_plots is not None:
        ## Add reference:
        exact = get_inf_exact_results()
        z = exact['z']
        for i, plot in enumerate([p_values, p_values_vs_times]):
            ax = plot.axis
            ax.axhline(y=z, color='k', linestyle='--', linewidth=2, label="Exact")
            ax.legend()
            visuals.save_figure(plot.fig, file_name=strings.time_stamp()+f" {i+1}")

    print("Done")    



def _plot_from_table_per_D(D:str, methods:set[str], table:csvs.TableByKey) -> None:

    ## Prepare plots:
    # Z vs t plot:
    plt.figure()
    ax_z_vs_t = plt.subplot(1,1,1)
    ax_z_vs_t.set_xlabel("expectation value <Z>")
    ax_z_vs_t.set_ylabel("compute time [sec]")
    ax_z_vs_t.set_xscale('log')

    # Fidelity vs t plot:
    plt.figure()
    ax_fidelity = plt.subplot(1,1,1)
    ax_fidelity.set_xlabel("1 - Fidelity to Exact")
    ax_fidelity.set_ylabel("compute time [sec]")
    ax_fidelity.set_xscale('log')

    ## Get reference:
    exact = get_inf_exact_results(int(D))
    z_ref = exact['z']
    x_range = table.unique_values("time")
    ax_z_vs_t.hlines(y=[0], xmin=min(x_range), xmax=max(x_range), linestyles="--", color='k', label="Exact")
    visuals.draw_now()


    def plot(ax, x, y, method:str) -> None:
        sorted_xy = sorted(( (x_, y_) for x_, y_ in zip(x, y, strict=True)), key=lambda tuple_: tuple_[0])
        x = [tuple_[0] for tuple_ in sorted_xy]
        y = [tuple_[1] for tuple_ in sorted_xy]
        ax.plot(x, y, label=method, linewidth=4)
        visuals.draw_now()


    for method in methods:
        t, z = [], []
        data = table.get_matching_table_elements(D=D, method=method)

        print(data)
        N = data['N']
        z = [f-z_ref for f in data['z']]
        t = data['time']
        f = [1-f for f in data['fidelity']]

        plot(ax_z_vs_t,   z, t, method)
        plot(ax_fidelity, f, t, method)

    ## Pretify
    for ax in [ax_z_vs_t, ax_fidelity]:
        ax.legend(loc="best")
        ax.grid()
    visuals.draw_now()

    print("Done")


def _get_data_from_csvs(filename:str|None) -> csvs.TableByKey:
    if isinstance(filename, str):
        fullpath = saveload.derive_fullpath(filename+".csv", sub_folder="results")
        table = csvs.TableByKey(fullpath)
    elif filename is None:
        folder_fullpath = saveload.DATA_FOLDER + saveload.PATH_SEP + "results"
        table = csvs.TableByKey()
        for fullpath in files.get_all_files_fullpath_in_folder(folder_fullpath):

            try:
                crnt_table = csvs.TableByKey(fullpath)
            except UnicodeDecodeError:
                continue

            crnt_table.verify()

            try:
                table += crnt_table
            except KeyError:
                continue

            table.verify()
            

    table.verify()
    return table



def plot_bp_convergence_results(
    filename:str|None = None
) -> None:
    
    ## Get Data:
    table = _get_data_from_csvs(filename)

    ## Derive params:
    Ds = table.unique_values("D")
    methods = table.unique_values("method")

    ## Get lists and plot:
    for D in Ds:
        print(f"D={D}")
        _plot_from_table_per_D(D, methods, table)

    ## Finish:
    print("Done printing for all Ds")
    

def _calc_inf_exact_results(D:int) -> dict:
    ## Config:
    method = "exact"
    N = EXACT_N
    config = Config.derive_from_dimensions(D)

    ## Compute:
    res = _per_D_N_single_test(D, N, method, config, parallel=False, exact_run=True)

    ## Prepare output:
    res = res._asdict()
    res['D'] = D
    res['N'] = N
    res['chi'] = EXACT_CHI

    return res
    

ResultsSubFolderName = "results"
def get_inf_exact_results(D:int|list[int]=2) -> dict:

    if isinstance(D, list):
        return [get_inf_exact_results(val) for val in D]

    filename = f"infinite_exact_results D={D}"
    if saveload.exist(filename, sub_folder=ResultsSubFolderName):
        results = saveload.load(filename, sub_folder=ResultsSubFolderName)
    else:
        results = _calc_inf_exact_results(D)
        saveload.save(results, filename, sub_folder=ResultsSubFolderName)
    return results


def main_test():
    # results = get_inf_exact_results([2, 3]); print(results)
    test_bp_convergence_all_runs([3])
    # plot_bp_convergence_results()


if __name__ == "__main__":
    main_test()