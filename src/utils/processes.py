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

from typing import TypedDict, TypeGuard, TypeAlias, Literal

try:  ## Depend on python version
    from typing import Unpack, NotRequired, Required
except ImportError:
    from typing import Optional as NotRequired
    from typing import Optional as Required
    from typing import Optional as Unpack


FileOrIO : TypeAlias = TextIO|str


class MonitorProcessKwargs(TypedDict):
    track_cpu:                  NotRequired[bool]
    track_ram:                  NotRequired[bool]
    print_only_max:             NotRequired[bool]
    cpu_interval_time:          NotRequired[float]
    sleep_time_interval:        NotRequired[float]
    print_out:                  NotRequired[FileOrIO]

class _MonitorProcessKwargsFilled(TypedDict):
    track_cpu:                  Required[bool]
    track_ram:                  Required[bool]
    print_only_max:             Required[bool]
    cpu_interval_time:          Required[float]
    sleep_time_interval:        Required[float]
    print_out:                  Required[FileOrIO]

def fill_defaults(kwargs:MonitorProcessKwargs) -> TypeGuard[_MonitorProcessKwargsFilled]:
    for key, default_value in _monitor_process_kwargs_defaults.items():
        if key not in kwargs:
            kwargs[key] = default_value
    return True

_monitor_process_kwargs_defaults = MonitorProcessKwargs(
    track_cpu=True,
    track_ram=True,
    print_only_max=False,
    cpu_interval_time=1,
    sleep_time_interval=15,
    print_out=sys.stdout
)


def _colored_output_str(mem, cpu, what:str, colored:bool, **kwargs:Unpack[_MonitorProcessKwargsFilled]) -> str:
    #
    def _add_color(_s:str, _color) -> str:
        if colored:
            return add_color(_s, _color)
        else:
            return _s
    #
    s = ""
    if kwargs['track_ram']:
        s += _add_color(what+"-mem", PrintColors.CYAN) +f" = "+ _add_color(f"{mem!r}", PrintColors.RED) +"[GB]\n"
    if kwargs['track_cpu']:
        s += _add_color(what+"-cpu", PrintColors.CYAN) +f" = "+ _add_color(f"{cpu!r}", PrintColors.RED) +"[%]\n"
    return s


def _update_state(mem, cpu, what:Literal['crnt', 'max'], **kwargs:Unpack[_MonitorProcessKwargsFilled]) -> None:
    print_out = kwargs['print_out']
    if hasattr(print_out, 'read') and hasattr(print_out, 'write'):
        s = _colored_output_str(mem, cpu, what, colored=True, **kwargs)
        print(s, file=print_out)  #type: ignore
    elif isinstance(print_out, str):
        s = _colored_output_str(mem, cpu, what, colored=False, **kwargs)
        with open(print_out, 'a') as f:
            print(s, file=f)
    else:
        raise TypeError("Not an expected type.")
    


def _thread_job(process:psutil.Process, **kwargs:Unpack[_MonitorProcessKwargsFilled]) -> None:
    # init values:
    max_memory = 0
    max_cpu = 0

    while process.is_running():
        # SLeep:
        time.sleep(kwargs["sleep_time_interval"]) 

        # init values:
        new_max : bool = False
        crnt_memory = 0
        crnt_cpu = 0

        ## Compute and track:
        if kwargs["track_ram"]:
            # Compute:
            memory_info = process.memory_info()
            crnt_memory = memory_info.rss / (1024 ** 3)   # Convert bytes to GB
            # track:
            if crnt_memory > max_memory:
                max_memory = crnt_memory
                new_max = True
        
        if kwargs['track_cpu']:
            # Compute:
            crnt_cpu = process.cpu_percent(interval=kwargs["cpu_interval_time"])
            # track:
            if crnt_cpu > max_cpu:
                max_cpu = crnt_cpu
                new_max = True

        ## Print always?:
        if kwargs["print_only_max"] and new_max:
            _update_state(max_memory, max_cpu, 'max', **kwargs)
        else:
            _update_state(crnt_memory, crnt_cpu, 'crnt', **kwargs)

    ## end:
    _update_state(max_memory, max_cpu, 'max', **kwargs)
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
    res = monitor_crnt_process(sleep_time_interval=5)
    from utils import sizes
    for r in range(max_ram_gb):
        print(f"requested Ram = {r}[GB]")
        sizes.do_computation_by_ram_size(r, repetitions=5, _debug=True)

    print(res)


if __name__ == "__main__":
    _test()