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
from typing import TypedDict, Unpack, NotRequired, Required, Optional, TypeGuard

class MonitorProcessKwargs(TypedDict):
    track_cpu:           NotRequired[bool]
    track_ram:           NotRequired[bool]
    cpu_interval_time:   NotRequired[float]
    sleep_time_interval: NotRequired[float]
    print_out:           NotRequired[TextIO]

class _MonitorProcessKwargsFilled(TypedDict):
    track_cpu:           Required[bool]
    track_ram:           Required[bool]
    cpu_interval_time:   Required[float]
    sleep_time_interval: Required[float]
    print_out:           Required[TextIO]

def fill_defaults(kwargs:MonitorProcessKwargs) -> TypeGuard[_MonitorProcessKwargsFilled]:
    for key, default_value in _monitor_process_kwargs_defaults.items():
        if key not in kwargs:
            kwargs[key] = default_value
    return True

_monitor_process_kwargs_defaults = MonitorProcessKwargs(
    track_cpu=True,
    track_ram=True,
    cpu_interval_time=1,
    sleep_time_interval=15,
    print_out=sys.stdout
)



def _results_str(max_memory, max_cpu, **kwargs:Unpack[_MonitorProcessKwargsFilled]) -> str:
    s = ""
    if kwargs['track_ram']:
        s += add_color(f"max_memory", PrintColors.CYAN) +f" = "+ add_color(f"{max_memory!r}", PrintColors.RED) +"[GB]\n"
    if kwargs['track_cpu']:
        s += add_color(f"max_cpu   ", PrintColors.CYAN) +f" = "+ add_color(f"{max_cpu!r}"   , PrintColors.RED) +"[%]\n\n\n\n"
    return s

def _update_state(max_memory, max_cpu, **kwargs:Unpack[_MonitorProcessKwargsFilled]):
    out = kwargs['print_out']
    s = _results_str(max_memory, max_cpu, **kwargs)
    print(s, file=out)


def _thread_job(process:psutil.Process, **kwargs:Unpack[_MonitorProcessKwargsFilled]) -> None:

    max_memory = 0
    max_cpu = 0

    while process.is_running():
        ## Get memory and cpu
        memory_info = process.memory_info()
        crnt_memory = memory_info.rss / (1024 ** 3)   # Convert bytes to GB
        crnt_cpu = process.cpu_percent(interval=kwargs["cpu_interval_time"])

        ## Follow max value:
        if kwargs["track_ram"] and crnt_memory > max_memory:
            max_memory = crnt_memory
            _update_state(max_memory, max_cpu, **kwargs)
        
        if kwargs['track_cpu'] and crnt_cpu > max_cpu:
            max_cpu = crnt_cpu
            _update_state(max_memory, max_cpu, **kwargs)

        # SLeep:
        time.sleep(kwargs["sleep_time_interval"])  # Adjust sleep time as needed

    ## end:
    s = _results_str(max_memory, max_cpu, **kwargs)
    print(s)
    return 


def monitor_crnt_process(**kwargs:Unpack[MonitorProcessKwargs]) -> None:
    process = psutil.Process(os.getpid())
    return monitor_process(process, **kwargs)


def monitor_process(process:psutil.Process, **kwargs:Unpack[MonitorProcessKwargs]) -> None:
    """Measures the peak memory usage of a process in GB.

    Args:
    process: The psutil.Process object representing the process.

    Returns:
    The peak memory usage in GB.
    """    
    fill_defaults(kwargs)
    thread = threading.Thread(target=_thread_job, args=[process], kwargs=kwargs, daemon=True)
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