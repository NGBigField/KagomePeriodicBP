import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn, UnitCell

# BP:
from algo.belief_propagation import belief_propagation

# Config and containers:
from containers import BPConfig

# Measure core data
from algo.core_measurements import measure_xyz_expectation_values_with_tn, calc_unit_cell_expectation_values, pauli

# Numpy for math stuff:
import numpy as np

# usefull utils:
from utils import visuals, saveload
from matplotlib import pyplot as plt


def growing_tn_bp_test2(
    d = 2,
    D = 3,
    min_N = 2,
    max_N = 14,
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

    ## Figure:
    visuals.draw_now()
    plots = visuals.AppendablePlot()
    plots.axis.set_title("<Z> on sites")
    plots.axis.set_xlabel("N")
    plots.axis.set_ylabel("<Z>")

    ## Growing N networks:
    all_results = []
    for N in range(min_N, max_N+1, 2):
        print(" ")
        print(f"N={N:2}: ")
        tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
        tn.validate()
        tn, messages, stats = belief_propagation(tn, messages=None, bp_config=bp_config)
        results = calc_unit_cell_expectation_values(tn, operators=[pauli.x, pauli.y, pauli.z], bubblecon_trunc_dim=bubblecon_trunc_dim, force_real=True, reduce=False)
        zs = results[2]
        print(zs)
        a = zs.__getattribute__('A')
        b = zs.__getattribute__('B')
        c = zs.__getattribute__('C')
        
        plots.append(A=(N, a), B=(N, b), C=(N, c))
        all_results.append((N, results))
        saveload.save(all_results, f"all_results_D={D}", sub_folder="results")

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
    small_res = measure_xyz_expectation_values_with_tn(small_tn, reduce=False)
    print(" ")
    print("Base values")
    print(small_res)

    ## Big network
    distances = []
    Ns = []
    for N in range(min_N, max_N+1):
        big_tn = create_kagome_tn(d=d, D=D, N=N, unit_cell=unit_cell)
        big_tn.connect_random_messages()
        big_res = measure_xyz_expectation_values_with_tn(big_tn, reduce=False)
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
    D = 2,
    N = 3
):

    ## Config:
    bp_config = BPConfig(
        max_swallowing_dim=18        
    )
    
    ## first network:
    tn = create_kagome_tn(d=d, D=D, N=N)

    ## BP:
    tn, messages, stats = belief_propagation(tn, messages=None, bp_config=bp_config)
    tn, messages, stats = belief_propagation(tn, messages=messages, bp_config=bp_config)

    print(stats)


def main_test():
    # single_bp_test()
    # growing_tn_bp_test()
    growing_tn_bp_test2()


if __name__ == "__main__":
    main_test()