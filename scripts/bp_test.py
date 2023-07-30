import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn, UnitCell

# BP:
from algo.belief_propagation import belief_propagation

# Config and containers:
from containers import BPConfig

# Measure core data
from algo.core_measurements import measure_xyz_expectation_values_with_tn

# Numpy for math stuff:
import numpy as np

# usefull utils:
from utils import visuals
from matplotlib import pyplot as plt


def growing_tn_bp_test(
    d = 2,
    D = 2,
    min_N = 2,
    max_N = 13
):
    ## Config:
    bp_config = BPConfig(max_swallowing_dim=8)
    unit_cell = UnitCell.random(d=d, D=D)

    ## small network:
    small_tn = create_kagome_tn(d=d, D=D, N=min_N, unit_cell=unit_cell)
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
                distance += abs(r_small - r_big)**2
        distance = np.sqrt(distance)
        print(f"    Distance={distance}")
        distances.append(distance)
        Ns.append(N)

    ## Print:
    visuals.draw_now()
    plt.plot(N, distances)
    plt.xlabel("N")
    plt.ylabel("Error")

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
    growing_tn_bp_test()


if __name__ == "__main__":
    main_test()