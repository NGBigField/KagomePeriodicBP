import _import_src  ## Needed to import src folders when scripts are called from an outside directory


# useful utils:
from utils import visuals, dicts, saveload, csvs

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


def test_parallel_execution_time(
    D = 3,
    N = 6,
    parallel : bool = 1
)->tuple[float, float]:
    
    print(f"Inputs: D={D} N={N} parallel={parallel!r}")

    ## Prepare TN:
    dims = TNDimensions(physical_dim=d, virtual_dim=D, big_lattice_size=N)
    unit_cell = UnitCell.random(d=d, D=D)
    full_tn = kagome_tn_from_unit_cell(unit_cell, dims)

    ## Prepare config:
    config = BPConfig(
        max_iterations=10, max_swallowing_dim=D**2, msg_diff_terminate=1e-10, parallel_msgs=parallel
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



TASK_DATA_KEYS = ["bp-step", "reduction"]


def _get_cell(table, N, D, parallel, task):    
    matching_rows = csvs.get_matching_table_element(table, N=N, D=D, parallel=parallel)
    if len(matching_rows)==0:
        return None
    mean, std = dicts.statistics_along_key(matching_rows, key=task)
    if np.isnan(mean):
        raise ValueError("NaN Value!")
    return mean



def arrange_res_table():
    ## load table:
    f_name = "results_parallel_timings.csv"
    fullpath = csvs.data / "condor" / f_name
    original_table = csvs.read_csv_table(fullpath)

    ## Parse common keys:
    def _sorted_unique_list(s:str)->list[str]:
        set_ = set(original_table[s])
        list_ = list(set_)
        list_.sort()

        return list_
    Ns = _sorted_unique_list("D")
    Ds = _sorted_unique_list("N")

    ## arranged table: 
    N_repeats = [N for N in Ns for _ in Ds]
    D_repeats = [D for _ in Ns for D in Ds]
    new_table = csvs.CSVManager(columns=["task", "parallel", "N"]+N_repeats, name="arranged_timing_results")
    first_row = ["", "", "D"]+D_repeats
    new_table.append(first_row)
    for task in TASK_DATA_KEYS:
        for parallel in ["False", "True"]:
            row = [task, parallel, ""]

            for N in Ns:
                for D in Ds:
                    cell = _get_cell(original_table, N, D, parallel, task)
                    row.append(cell)

            assert len(row)==len(new_table.columns)
            new_table.append(row)
    print(f"new table saved at {new_table.fullpath!r}")




def main_test():
    # test_parallel_execution_time()
    arrange_res_table()


if __name__ == "__main__":
    main_test()