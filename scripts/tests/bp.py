import _import_src  ## Needed to import src folders when scripts are called from an outside directory
import project_paths

from typing import NamedTuple, overload, Literal

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
from utils import visuals, saveload, csvs, dicts, lists, strings, files, errors
from utils.visuals import AppendablePlot

## Metrics:
from physics.metrics import fidelity

## Lattice soze:
from lattices import kagome

## plots:
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt


from libs import TenQI


## common config:
mode = UpdateMode.A
update_edge = UpdateEdge(UnitCellFlavor.A, UnitCellFlavor.B)
d=2

EXACT_CHI = 200
EXACT_N = 10


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
    if this_is_exact_run:
        fidelity_ = 0
    else:
        rho_exact = exact['edge_tn'].rdm
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
    D:int|list[int] = 3,
    combined_figure:bool = False,
    parallel:bool = True,
    live_plots:bool|None = None,  # True: live, False: at end, None: no plots
) -> None:
    
    if isinstance(D, list):
        _ = [
            test_bp_convergence_all_runs(val, combined_figure=combined_figure, parallel=parallel, live_plots=live_plots) 
            for val in D
        ]
        return None

    ##  Params:
    methods_and_Ns:dict[str, list[int]] = dict(
        random = list(range(2, 7)),
        bp     = [2, 3, 4]
    ) 

    ## config:
    config = Config.derive_from_dimensions(D)
    config.chi    = 3*D**2 + 10
    config.chi_bp = 2*D**2 + 10
    config.bp.msg_diff_terminate = 1e-14
    config.bp.max_iterations = 50
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
                errors.print_traceback(e)
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
        exact = get_inf_exact_results(D)
        z = exact['z']
        for i, plot in enumerate([p_values, p_values_vs_times]):
            ax = plot.axis
            ax.axhline(y=z, color='k', linestyle='--', linewidth=2, label="Exact")
            ax.legend()
            visuals.save_figure(plot.fig, file_name=strings.time_stamp()+f" {i+1}")

    print("Done")    


def _new_axes() -> Axes:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return ax


def _plot_from_table_per_D_per_x_y(D:int, table:csvs.TableByKey, x:str, y:str, what:Literal['true', 'error'], yscale:Literal['linear', 'log']="linear") -> Axes:
    ## Prepare plots:
    ax = _new_axes()

    ## Get data:
    data = table.get_matching_table_elements(D=D)    
    ref_y = 0

    def plot(method:Literal['random', 'bp', 'exact']) -> list[Line2D]:
        ## init output:
        lines = []

        ## Style:
        color = _method_color(method)
        marker, marker_size = _method_marker(method)

        ## Get data:
        data = table.get_matching_table_elements(D=D, method=method)            
        x_vals = _x_values(data)
        if what == 'true':
            y_vals = [y for y in data[y]] 
            if method == 'exact':
                ref_line = plt.hlines([ref_y], xmin=min(x_values), xmax=max(x_values), label="reference", linestyles="--", color="k")
                lines.append(ref_line)
                return lines
        elif what == 'error':
            y_vals = [abs(y-ref_y) for y in data[y]] 
            if method == 'exact':
                return []
            
        if method == "bp":
            label = "blockBP"
        elif method == "random":
            label = "Random"

        lines += visuals.plot_with_spread(x_vals=x_vals, y_vals=y_vals, axes=ax, also_plot_max_min_dots=False, 
                                         linewidth=5, marker=marker, color=color, markersize=marker_size, label=label,
                                         disable_spread=True)

        return lines
    
    def _method_marker(method:str) -> tuple[str, int]:
        match method:
            case 'bp':      return ("*", 8)
            case 'random':  return (".", 8)
            case 'exact':   return ("" , 0)
            case _:
                raise ValueError("")

    def _method_color(method:str) -> str:
        match method:
            case 'bp':      return "tab:blue"
            case 'random':  return "tab:orange"
            case 'exact':   return "k"
            case _:
                raise ValueError("")

    def _x_values(data) -> list[float]|list[int]:
        if x == "N":
            x_vals = [kagome.num_nodes_by_lattice_size(N) for N in data[x]]
        else:
            x_vals = data[x]
        return x_vals

    def _print_reference_data(dict) -> None:
        print("Reference:")
        print(f"    D={D}")
        print(f"    N={dict['N']}")
        print(f"    chi={dict['chi']}")
        print(f"    num_tensors={kagome.num_nodes_by_lattice_size(dict['N'])}")


    ## get reference:
    x_values = _x_values(data)
    ref = get_inf_exact_results(D)
    ref_y = ref[y]    
    _print_reference_data(ref)
            
    ## Plot:
    for method in ['random', 'bp', 'exact']:
        lines = plot(method)


    ## Pretty:
    ax.legend(loc="best")
    ax.grid()
    #

    ## Labels:
    if x == "N":
        xlabel = "#tensors"
    else:
        xlabel = x    

    if what == 'true':
        ylabel = y
    elif what == 'error':
        ylabel = "abs "+y+" error"

    if y == "fidelity" and what == "error":
        ylabel = "1-Fidelity"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #
    # ax.set_xscale('linear')
    # ax.set_yscale('log')
    ax.set_yscale(yscale)

    ax.figure.tight_layout()

    visuals.draw_now()
    print("")

    return ax


def _plot_from_table_per_D(D:int, table:csvs.TableByKey) -> None:

    for x, y in [
        # ('time', 'z'), 
        # ('N', 'entanglement_entropy'), 
        # ('N', 'negativity'), 
        ('N', 'fidelity'),
        # ('N', 'energy'),
    ]:
        print(f"x={x!r} ; y={y!r}")
        ax1 =_plot_from_table_per_D_per_x_y(D, table, x, y, 'true')
        # ax2 =_plot_from_table_per_D_per_x_y(D, table, x, y, 'error')
        ax3 =_plot_from_table_per_D_per_x_y(D, table, x, y, 'error', yscale='log')
        print("")

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
    filename:str|None = None,
    D:list[int]|None=None
) -> None:
    
    ## Get Data:
    table = _get_data_from_csvs(filename)
    print(table)

    if D is None:
        Ds = table.unique_values('D')
    elif isinstance(D, list):
        assert isinstance(D[0], int)
        Ds = D


    ## Get lists and plot:
    for D_ in Ds:
        print(f"D={D_}")
        _plot_from_table_per_D(D_, table)

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
@overload
def get_inf_exact_results(D:int) -> dict: ...
@overload
def get_inf_exact_results(D:list[int]) -> list[dict]: ...
def get_inf_exact_results(D:int|list[int]=2) -> dict|list[dict]:

    if isinstance(D, list):
        return [get_inf_exact_results(val) for val in D]

    if isinstance(D, float):
        assert int(D)==D
        D = int(D)

    filename = f"infinite_exact_results D={D}"
    if saveload.exist(filename, sub_folder=ResultsSubFolderName):
        results = saveload.load(filename, sub_folder=ResultsSubFolderName)
    else:
        results = _calc_inf_exact_results(D)
        saveload.save(results, filename, sub_folder=ResultsSubFolderName)
    return results


def main_test():
    # results = get_inf_exact_results([2, 3]); print(results)
    # test_bp_convergence_all_runs([2, 3])
    plot_bp_convergence_results()

    print("Done.")

if __name__ == "__main__":
    main_test()