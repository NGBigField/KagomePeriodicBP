import psutil
import os
import time
from typing import TypeAlias


_ProcessInfoType : TypeAlias = tuple[float, float]



def monitor_crnt_process() -> _ProcessInfoType:
    process = psutil.Process(os.getpid())
    return monitor_process(process)


def monitor_process(process:psutil.Process, sleep_time:float=2, cpu_interval_time:float=1) -> _ProcessInfoType:
    """Measures the peak memory usage of a process in GB.

    Args:
    process: The psutil.Process object representing the process.

    Returns:
    The peak memory usage in GB.
    """

    max_memory = 0
    max_cpu = 0

    while process.is_running():
        memory_info = process.memory_info()
        max_memory = max(max_memory, memory_info.rss)
        max_cpu = max(max_cpu, process.cpu_percent(interval=cpu_interval_time))
        time.sleep(sleep_time)  # Adjust sleep time as needed

    max_memory /= (1024 ** 3)   # Convert bytes to GB
    return max_memory, max_cpu



