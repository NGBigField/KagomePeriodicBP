import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn, UnitCell

# BP:
from algo.belief_propagation import belief_propagation

# Config and containers:
from containers import BPConfig

# Measure core data
from algo.core_measurements import measure_xyz_expectation_values_with_tn


def growing_tn_bp_test(
    d = 2,
    D = 2,
    min_N = 2,
    max_N = 8
):
    ## Config:
    bp_config = BPConfig(max_swallowing_dim=8)
    unit_cell = UnitCell.random(d=d, D=D)

    ## small network:
    small_tn = create_kagome_tn(d=d, D=D, N=min_N, unit_cell=unit_cell)
    small_tn, messages, stats = belief_propagation(small_tn, messages=None, bp_config=bp_config)
    small_res = measure_xyz_expectation_values_with_tn(small_tn, reduce=False)
    print(small_res)


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