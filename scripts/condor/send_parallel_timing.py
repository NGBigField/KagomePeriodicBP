import _import_main_and_src

from scripts.condor.main_sender import main as main_sender
from scripts.tests.parallel import _single_test_parallel_execution_time_single_bp_step

RESULT_KEYS = ["parallel", 'D', 'N', 'seed', 'bp_step']


def job(
    D : int = 2,
    N : int = 2,
    parallel : int = 0   
) -> dict:
    
    ## Parse:
    if parallel==0:
        parallel = False
    else:
        parallel = True


    ## Run:
    bp_step_time = _single_test_parallel_execution_time_single_bp_step(D=D, N=N, parallel=parallel)

    ## Collect as a dict:
    # res_keys = ["parallel", 'D', 'N', 'seed', 'bp-step', 'reduction']
    results = {
        "parallel" : parallel,
        'D' : D,
        'N' : N, 
        'bp_step' : bp_step_time
    }

    return results


def plot_results(
    target_D = 4,
) -> None:
    from utils.csvs import read_csv_table, PATH_SEP, DATA_FOLDER
    from utils import lists, visuals, strings
    from matplotlib import pyplot as plt

    ## Get data:
    fullpath = str(DATA_FOLDER)+PATH_SEP+"condor"+PATH_SEP+"results_parallel_timings.csv"
    results = read_csv_table(fullpath)

    ## Prepare plots:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.grid()
    visuals.draw_now()
    fig.suptitle(f"Parallel Timing Test\nD={target_D}")
    ax.set_xlabel("N")
    ax.set_ylabel("time [sec]")

    ## Parse data:
    Ns = lists.deep_unique(results["N"])
    Ns = sorted(Ns)
    num_res = len(results["N"])

    ## Plot results:
    for target_parallel in ["False", "True"]:
        values_dict : dict[int, list[float]] = {}
        print(f"parallel={target_parallel}:")

        for target_N in Ns:
            values_dict[target_N] = []

            for i in range(num_res):
                N = results["N"][i]
                D = results["D"][i]
                seed = results["seed"][i]
                bp_step_time = results["bp_step"][i]
                parallel = results["parallel"][i]

                if N!=target_N or D!=target_D or not strings.str_equal_case_insensitive(parallel, target_parallel):
                    continue

                values_dict[target_N].append(bp_step_time)
            
            print(f"    N={int(target_N):>2}: {len(values_dict[target_N]):>2} results")

        visuals.plot_with_spread(values_dict, plt_kwargs=dict(linewidth=6, label=target_parallel))
    

    ax.legend()
    visuals.draw_now()

    print("Done.")


def sender() -> None:
    vals = {}
    vals['D'] = [2, 3, 4]
    vals['N'] = list(range(2, 12))
    vals['chi'] = [1]
    vals['method'] = [-1]
    vals['seed'] = list(range(5))
    vals['parallel'] = [0, 1]
    vals['control'] = [-1]

    main_sender(
        job_type="parallel_timings",
        request_cpus=10,
        request_memory_gb=8,
        vals=vals,
        result_keys=RESULT_KEYS,
        _local_test=True
    )



if __name__ == "__main__":    
    # plot_results()
    sender()