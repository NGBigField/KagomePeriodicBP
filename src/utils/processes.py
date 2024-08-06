import sys, pathlib

if __name__ == "__main__":
    src_folder = pathlib.Path(__file__).parent.parent.__str__()
    if src_folder not in sys.path:
        sys.path.append(src_folder)

    from utils.prints import add_color, PrintColors
else:
    from .prints import add_color, PrintColors

import psutil
import os
import time
from typing import TextIO
import threading 


def _update_state(max_memory, max_cpu, print_out:TextIO):
    s = ""
    s += add_color(f"max_memory", PrintColors.CYAN) +f" = "+ add_color(f"{max_memory!r}", PrintColors.RED) +"[GB]\n"
    s += add_color(f"max_cpu   ", PrintColors.CYAN) +f" = "+ add_color(f"{max_cpu!r}"   , PrintColors.RED) +"[%]\n\n\n\n"
    print(s, file=print_out)


def _thread_job(process:psutil.Process, sleep_time:float=2, cpu_interval_time:float=1, print_out:TextIO=sys.stdout) -> None:
    max_memory = 0
    max_cpu = 0

    while process.is_running():
        ## Get memory and cpu
        memory_info = process.memory_info()
        crnt_memory = memory_info.rss / (1024 ** 3)   # Convert bytes to GB
        crnt_cpu = process.cpu_percent(interval=cpu_interval_time)

        ## Follow max value:
        if crnt_memory > max_memory:
            max_memory = crnt_memory
            _update_state(max_memory, max_cpu, print_out)
        
        if crnt_cpu > max_cpu:
            max_cpu = crnt_cpu
            _update_state(max_memory, max_cpu, print_out)

        # SLeep:
        time.sleep(sleep_time)  # Adjust sleep time as needed

    ## end:
    print(f"max_memory = {max_memory!r}[GB]")
    print(f"max_cpu    = {max_cpu!r}[%]")
    return 


def monitor_crnt_process() -> None:
    process = psutil.Process(os.getpid())
    return monitor_process(process)


def monitor_process(process:psutil.Process, sleep_time:float=2, cpu_interval_time:float=1) -> None:
    """Measures the peak memory usage of a process in GB.

    Args:
    process: The psutil.Process object representing the process.

    Returns:
    The peak memory usage in GB.
    """
    thread = threading.Thread(target=_thread_job, args=(process, sleep_time, cpu_interval_time), daemon=True)
    thread.start()




def _test(max_ram_gb:int=4):
    res = monitor_crnt_process()
    from utils import sizes
    for r in range(max_ram_gb):
        print(f"requested Ram = {r}[GB]")
        sizes.do_computation_by_ram_size(r, repetitions=5, _debug=True)

    print(res)


if __name__ == "__main__":
    _test()