import _import_src  ## Needed to import src folders when scripts are called from an outside directory


import project_paths

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn, UnitCell

# BP:
from algo.belief_propagation import belief_propagation, robust_belief_propagation

# Config and containers:
from containers import BPConfig
from tensor_networks import KagomeTNRepeatedUnitCell, CoreTN, ModeTN, EdgeTN

# update config
from enums import UpdateMode, UnitCellFlavor
from containers import UpdateEdge

# Measure core data
from algo.measurements import derive_xyz_expectation_values_with_tn, calc_unit_cell_expectation_values_from_tn, pauli, measure_energies_and_observables_together
from algo.tn_reduction import reduce_tn
from physics import hamiltonians

from time import perf_counter

# Numpy for math stuff:
import numpy as np

# useful utils:
from utils import visuals, saveload, csvs, dicts
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms


from libs import TenQI

d=2

def _standard_row_from_results(D:int, N:int, results:list[UnitCell])->None:
    row = [D, N]
    for tensor_key in ['A', 'B', 'C']:
        for per_expectation, observable_name in zip( results, ['x', 'y', 'z'], strict=True):
            per_expectation_per_tensor = per_expectation[tensor_key]
            row.append(per_expectation_per_tensor)
    return row

def load_results(
    D = 3
):
    all_results = saveload.load(f"all_results_D={D}", sub_folder="results")
    csv = csvs.CSVManager(['D', 'N', 'A_X', 'A_Y', 'A_Z', 'B_X', 'B_Y', 'B_Z', 'C_X', 'C_Y', 'C_Z'], name=f"Results_D={D}")
    # ['N', 'D', 'A_X', 'A_Y', 'A_Z', 'B_X', 'B_Y', 'B_Z', 'C_X', 'C_Y', 'C_Z']
    for N, results in all_results:
        row = _standard_row_from_results(D, N, results)
        csv.append(row)
    print("Done")

def bp_single_call(
    D : int = 3,
    N : int = 2,
    with_bp : bool = True        
):
    ## Config:
    bubblecon_trunc_dim = 4*D**2
    bp_config = BPConfig(
        max_swallowing_dim=D**2,
        max_iterations=50,
        msg_diff_terminate=1e-5
    )

    ## Tensor-Network
    unit_cell= UnitCell.load(f"random_D={D}")
    tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
    tn.validate()

    ## Find or guess messages:
    if with_bp:
        tn, messages, stats = belief_propagation(tn, messages=None, bp_config=bp_config)
    else:
        tn.connect_random_messages()
    
    return calc_unit_cell_expectation_values_from_tn(tn, operators=[pauli.x, pauli.y, pauli.z], bubblecon_trunc_dim=bubblecon_trunc_dim, force_real=True, reduce=False)


def growing_tn_bp_test2(
    D = 2,
    min_N = 2,
    max_N = 4,
    live_plot = False,
    with_bp = True
):
    ## Config:
    bubblecon_trunc_dim = 2*D**2
    bp_config = BPConfig(
        max_swallowing_dim=D**2,
        max_iterations=50,
        msg_diff_terminate=1e-5
    )
    
    ## Load or randomize unit_cell
    unit_cell= UnitCell.load(f"random_D={D}")
    if unit_cell is None:
        unit_cell = UnitCell.random(d=d, D=D)
        unit_cell.save(f"random_D={D}")


    # if with_bp:
    #     csv_name =  f"bp_res_D={D}_with-BP"
    # else:
    #     csv_name =  f"bp_res_D={D}_random-messages"
    # csv = csvs.CSVManager(['D', 'N', 'A_X', 'A_Y', 'A_Z', 'B_X', 'B_Y', 'B_Z', 'C_X', 'C_Y', 'C_Z'], name=csv_name)

    ## Growing N networks:
    all_results = []
    for N in range(min_N, max_N+1, 1):
        print(" ")
        print(f"N={N:2}: ")
        tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
        tn.validate()

        ## Find or guess messages:
        if with_bp:
            tn, messages, stats = belief_propagation(tn, messages=None, bp_config=bp_config)
        else:
            tn.connect_random_messages()
        
        results = calc_unit_cell_expectation_values_from_tn(tn, operators=[pauli.x, pauli.y, pauli.z], bubblecon_trunc_dim=bubblecon_trunc_dim, force_real=True, reduce=False)
        zs = results[2]
        print(zs)
        a = zs['A']
        b = zs['B']
        c = zs['C']

        all_results.append((N, results))
        saveload.save(all_results, f"all_results_D={D}", sub_folder="results")

        row = _standard_row_from_results(D, N, results)
        # csv.append(row)
        
    ## End:
    print("Done")
                

def growing_tn_bp_test(
    D = 2,
    bp_N = 2,
    min_N = 2,
    max_N = 13
):
    ## Config:
    bp_config = BPConfig(
        max_swallowing_dim=8,
        msg_diff_terminate=1e-7
    )
    unit_cell = UnitCell.random(d=d, D=D)

    ## small network:
    small_tn = create_kagome_tn(d=d, D=D, N=bp_N, unit_cell=unit_cell)
    small_tn, messages, stats = belief_propagation(small_tn, messages=None, bp_config=bp_config)
    small_res = derive_xyz_expectation_values_with_tn(small_tn, reduce=False)
    print(" ")
    print("Base values")
    print(small_res)

    ## Big network
    distances = []
    Ns = []
    for N in range(min_N, max_N+1):
        big_tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
        big_tn.connect_random_messages()
        big_res = derive_xyz_expectation_values_with_tn(big_tn, reduce=False)
        print(" ")
        print(f"N={N:2}: ")
        print(big_res)
        distance = 0.0
        for op in ['x', 'y', 'x']:
            for key in UnitCell.all_keys():
                r_small = small_res[op].__getattribute__(key)
                r_big = big_res[op].__getattribute__(key) 
                distance += abs(r_small - r_big)
        print(f"    Distance={distance}")
        distances.append(distance)
        Ns.append(N)

    ## Plot:
    visuals.draw_now()
    for linear_or_log in ["linear", "log"]:
        plt.figure()
        plt.plot(Ns, distances)
        plt.xlabel("N")
        plt.ylabel("L1 Error in expectation values")
        plt.title(f"Error Convergence to BP on block of size N={bp_N}")
        plt.yscale(linear_or_log)
        plt.grid()

    visuals.draw_now()
    print("Done")
                


def test_single_bp_vs_growing_TN(
    Ds = [2, 3, 4],
    bp_N = 4
):
    
    fig1 = plt.figure()
    plt.yscale("log")
    plt.xlabel("# tensors in lattice")
    plt.ylabel("energy error")
    # plt.title(f"Error between Block-BP on {bp_num_tensors} tensors and random environment tensors")
    plt.grid("on")
    plt.legend()

    visuals.draw_now()


    
    csv_fullpath = project_paths.data/"condor"/"results_bp_convergence.csv"
    table = csvs.read_csv_table(csv_fullpath)
    # markers = ["x", "+", "*"]


    for i, D in enumerate(Ds):
        marker_style = "*"
        
        chi = 2*D**2 + 40
        bp_chi = D**2 + 20
        unit_cell = UnitCell.load(f"best_heisenberg_AFM_D{D}")
        hamiltonian = hamiltonians.heisenberg_afm()

        bp_config = BPConfig(
            max_iterations=60,
            max_swallowing_dim=bp_chi,
            msg_diff_terminate=1e-7
        )

        tn = create_kagome_tn(d, D, bp_N, unit_cell)
        bp_num_tensors = tn.size

        _, _ = robust_belief_propagation(tn, None, bp_config)

        core = reduce_tn(tn, CoreTN, chi)
        _, _, mean_energy = measure_energies_and_observables_together(core, hamiltonian, trunc_dim=chi)

        print(f"mean_energy={mean_energy}") 
        print(f"num_tensors={bp_num_tensors}") 
        print(f"chi={chi}") 


        delta_energies = []
        times = []
        num_tensors = []

        Ns = list(set(table["N"]))
        Ns.sort()
        for N in Ns:
            matching_rows = csvs.get_matching_table_element(table, N=N, D=D)

            mean, std = dicts.statistics_along_key(matching_rows, "energy")
            d_e = abs(mean-mean_energy)
            delta_energies.append(d_e)

            mean, std = dicts.statistics_along_key(matching_rows, "exec_time")
            times.append(mean)

            n_t = matching_rows[0]["num_tensors"]
            num_tensors.append(n_t)

            
        
        assert len(num_tensors)==len(times)==len(delta_energies)

        p, *_ = plt.plot(num_tensors, delta_energies, label=f"D={D}", linewidth=3, marker=marker_style)
        

    
    plt.legend()
    visuals.draw_now()
    visuals.save_figure()

    print("Done")

    return mean_energy, num_tensors, chi



def test_bp_convergence_steps(
    D = 2,
    N_vals   = [2, 3, 4, 5, 6, 7, 8, 9, 10],
    chi_vals = [2, 4, 8, 16]
):
    
    unit_cell = UnitCell.random(d, D)

    plot = visuals.AppendablePlot()
    plt.xlabel("N")
    plt.ylabel("#Iterations")
    plt.title(f"#Iteration for Block-BP convergence")
    plt.grid("on")
    plt.legend()

    visuals.draw_now()
    
    for chi in chi_vals:

        ## Config:
        bp_config = BPConfig(
            max_swallowing_dim=chi,
            msg_diff_terminate=1e-5
        )


        for N in N_vals:
            tn = create_kagome_tn(d, D, N, unit_cell)
            messages, stats = belief_propagation(tn, None, bp_config)

            x = N
            y = stats.iterations

            dict_ = {f"chi={chi}" : (x, y)}
            plot.append(**dict_)


    print("Done")
                


def test_bp_convergence_steps_single_run(
    N:int=2,
    D:int=2,
    parallel_bp:bool=True,
    ref_rdm : np.ndarray|None = None
):
    
    chi = 2*D**2 + 10
    update_mode = UpdateMode.A
    update_edge = UpdateEdge(UnitCellFlavor.A, UnitCellFlavor.B)
    bp_chi = D**2 

    def _get_rho_i(tn:KagomeTNRepeatedUnitCell)->np.ndarray:
        edge_tn = reduce_tn(tn , EdgeTN, trunc_dim=chi, mode=update_mode, edge_tuple=update_edge)
        rdm = edge_tn.rdm
        return np.trace(rdm, axis1=2, axis2=3)

    unit_cell = UnitCell.load(f"best_heisenberg_AFM_D{D}")


    if ref_rdm is None:
        N_inf = 10
        fname = f"ref_rdm_D{D}_N{N_inf}"
        
        if saveload.exist(fname):
            ref_rdm = saveload.load(fname)
        else:
            tn_inf = create_kagome_tn(d, D, N_inf, unit_cell)
            tn_inf.connect_random_messages()
            ref_rdm = _get_rho_i(tn_inf)
            saveload.save(ref_rdm, fname)


    tn = create_kagome_tn(d, D, N, unit_cell)
    tn.connect_random_messages()

    t1 = perf_counter()
    rho_no_bp = _get_rho_i(tn)
    t2 = perf_counter()
    time_none = t2-t1


    diff_random = TenQI.op_norm(rho_no_bp - ref_rdm, ntype='tr')
    time_bp = None
    diff_bp = None
    z_bp    = None

    if N>4:
        t1 = perf_counter()
        bp_config = BPConfig(max_iterations=50, max_swallowing_dim=bp_chi, msg_diff_terminate=1e-7, parallel_msgs=parallel_bp)
        _, stats = belief_propagation(tn, None, config=bp_config)
        t2 = perf_counter()
        time_bp = t2-t1
            
        if not parallel_bp:
            time_bp /= 4

        t1 = perf_counter()
        rho_with_bp = _get_rho_i(tn)
        t2 = perf_counter()
        time_bp += t2-t1

    
        diff_bp     = TenQI.op_norm(rho_with_bp - ref_rdm, ntype='tr')

        z_random = np.trace(pauli.z @ rho_no_bp)
        z_bp     = np.trace(pauli.z @ rho_with_bp)
        z_random = np.real(z_random)
        z_bp     = np.real(z_bp)

    iterations = stats.iterations

    return iterations, chi, time_none, time_bp, z_random, z_bp, diff_random, diff_bp


def test_bp_convergence_all_runs(
    parallel_bp=False
):

    N_inf = 14
    update_mode = UpdateMode.A
    update_edge = UpdateEdge(UnitCellFlavor.A, UnitCellFlavor.B)

    def _get_rho_i(tn:KagomeTNRepeatedUnitCell, chi:int)->np.ndarray:
        edge_tn = reduce_tn(tn , EdgeTN, trunc_dim=chi, mode=update_mode, edge_tuple=update_edge)
        rdm = edge_tn.rdm
        return np.trace(rdm, axis1=2, axis2=3)


    plot_values = visuals.AppendablePlot()
    plt.xlabel("linear system size", fontsize=12)
    plt.ylabel("Trace distance between $\\rho_{A}$ and $\\rho_{A\infty}$", fontsize=12)
    plot_values.axis.set_yscale('log')


    plot_times  = visuals.AppendablePlot()
    plt.xlabel("linear system size", fontsize=12)
    plt.ylabel("Computation time [sec]", fontsize=12)

    visuals.draw_now()


    colors_D = {2: 'blue', 3: 'red', 4: 'green'}
    line_styles = {'bp': "-", 'random': '--'}
    marker_styles = {'bp': "X", 'random': 'o'}

    Ds = [3, 4]

    def _legend_key(D:int, data:str)->str:
        ## diff bp:
        if len(Ds)==1:
            legend_key = ""
        else:
            legend_key = f"D={D} " 

        if data=="bp":
            legend_key += "BlockBP"
        elif data=="none":
            legend_key += "random env"
        return legend_key


    for D in Ds:

        chi = 2*D**2 + 10

        chi_inf = 2*D**2 + 40 
        unit_cell = UnitCell.load(f"best_heisenberg_AFM_D{D}")
        tn = create_kagome_tn(d, D, N_inf, unit_cell)
        tn.connect_random_messages()

        fname = f"ref_rdm_D{D}_N{N_inf}"
        if saveload.exist(fname):
            ref_rdm = saveload.load(fname)
        else:
            ref_rdm = _get_rho_i(tn, chi_inf)
            saveload.save(ref_rdm, fname)


        for N in range(2, 9):
            iterations, chi, time_none, time_bp, z_none, z_bp, diff_none, diff_bp = test_bp_convergence_steps_single_run(D=D, N=N, parallel_bp=parallel_bp, ref_rdm=ref_rdm)


            dict_ = { _legend_key(D, 'bp')  : (N, diff_bp) }
            plt_kwargs = dict(color=colors_D[D], linestyle=line_styles['bp'], marker=marker_styles['bp'])
            plot_values.append(plt_kwargs=plt_kwargs ,**dict_)

            ## diff random:
            dict_ = { _legend_key(D, 'none') : (N, diff_none) }
            plt_kwargs = dict(color=colors_D[D], linestyle=line_styles['random'], marker=marker_styles['random'])
            plot_values.append(plt_kwargs=plt_kwargs ,**dict_)

            ## Time bp:
            dict_ = { _legend_key(D, 'bp')  : (N, time_bp) }
            plt_kwargs = dict(color=colors_D[D], linestyle=line_styles['bp'], marker=marker_styles['bp'])
            plot_times.append(plt_kwargs=plt_kwargs ,**dict_)

            ## Time random:
            dict_ = { _legend_key(D, 'none') : (N, time_none) }
            plt_kwargs = dict(color=colors_D[D], linestyle=line_styles['random'], marker=marker_styles['random'])
            plot_times.append(plt_kwargs=plt_kwargs ,**dict_)

            plot_values._update()
            plot_times._update()

    visuals.draw_now()

    for plot in [plot_values, plot_times]:
        points = [[0.17, 0.06], [0.90, 0.98]]
        box = mtransforms.Bbox(points)
        plot.fig.set_size_inches((4.5, 8))
        plot.axis.set_position(box)



    plot_values.axis.set_ylim(bottom=1e-8*2)
    
    print("Done")    

    

def main_test():
    # test_single_bp_vs_growing_TN()
    # growing_tn_bp_test()
    # growing_tn_bp_test2()
    # load_results()
    # test_bp_convergence_steps_single_run()
    test_bp_convergence_all_runs()


if __name__ == "__main__":
    main_test()