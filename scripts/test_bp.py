import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn, UnitCell

# BP:
from algo.belief_propagation import belief_propagation

# Config and containers:
from containers import BPConfig

# Measure core data
from algo.measurements import derive_xyz_expectation_values_with_tn, calc_unit_cell_expectation_values_from_tn, pauli

# Numpy for math stuff:
import numpy as np

# usefull utils:
from utils import visuals, saveload, csvs
from matplotlib import pyplot as plt


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
    d : int = 2,
    D : int = 3,
    N : int = 2,
    with_bp : bool = True        
):
    ## Config:
    bubblecon_trunc_dim = 4*D**2
    bp_config = BPConfig(
        max_swallowing_dim=D**2,
        max_iterations=50,
        target_msg_diff=1e-5
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
    d = 2,
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
        target_msg_diff=1e-5
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
    d = 2,
    D = 2,
    bp_N = 2,
    min_N = 2,
    max_N = 13
):
    ## Config:
    bp_config = BPConfig(
        max_swallowing_dim=8,
        target_msg_diff=1e-7
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
                


def single_bp_test(
    d = 2,
    D = 5,
    N = 10
):

    ## Config:
    bp_config = BPConfig(
        max_swallowing_dim=18        
    )
    
    ## first network:
    tn = create_kagome_tn(d=d, D=D, N=N)

    ## BP:
    messages, stats = belief_propagation(tn, messages=None, config=bp_config)
    messages, _ = belief_propagation(tn, messages=messages, config=bp_config)

    print(stats)


def main_test():
    single_bp_test()
    # growing_tn_bp_test()
    # growing_tn_bp_test2()
    # load_results()


if __name__ == "__main__":
    main_test()