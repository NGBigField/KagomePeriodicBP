import _import_src  ## Needed to import src folders when scripts are called from an outside directory


# useful utils:
from utils import visuals, dicts, saveload

# For TN:
from tensor_networks.construction import kagome_tn_from_unit_cell, UnitCell

# Common types in code:
from containers import TNDimensions
from tensor_networks import CoreTN

# The things we test:
from algo.belief_propagation import belief_propagation, BPConfig
from algo.tn_reduction import reduce_tn

# For testing and showing performance:
from time import perf_counter
from matplotlib import pyplot as plt

# Useful for math:
import numpy as np

d=2

def test_parallel_execution(
    D = 3,
    N = 3,
    parallel : bool = True
)->tuple[float, float]:
    
    print(f"Inputs: D={D} N={N} parallel={parallel!r}")

    ## Prepare TN:
    dims = TNDimensions(physical_dim=d, virtual_dim=D, big_lattice_size=N)
    unit_cell = UnitCell.random(d=d, D=D)
    full_tn = kagome_tn_from_unit_cell(unit_cell, dims)

    ## Prepare config:
    config = BPConfig(
        max_iterations=10, max_swallowing_dim=D**2, target_msg_diff=1e-10, parallel_msgs=parallel
    )

    ## Test BlockBP Performance:
    _, stats = belief_propagation(
        full_tn, None, config, 
        update_plots_between_steps=False, 
        allow_prog_bar=False
    )    
    bp_step_time = stats.execution_time / stats.iterations
    print(f"bp-step average_time   = {bp_step_time}")

    ## Test contraction:
    t1 = perf_counter()
    _ = reduce_tn(full_tn, CoreTN, trunc_dim=2*D**2, parallel=parallel)
    t2 = perf_counter()
    reduction_step_time = t2-t1
    print(f"reduction average_time = {reduction_step_time}")

    return bp_step_time, reduction_step_time



def main_test():
    test_parallel_execution()


if __name__ == "__main__":
    main_test()