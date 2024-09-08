import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from utils import logs, visuals, files, prints
import project_paths

from typing import Literal
import os, pathlib
from dataclasses import dataclass


class _ValidError(Exception): ...

@dataclass
class _LogData:
    D : int
    N : int
    chi : int
    chi_bp : int
    mem : list[float]
    cpu : list[float]


CONFIG_NUM_HEADER_LINES = 65


def _is_existing_folder_fullpath(fullpath:str) -> bool:
    # Check if the full path is an existing directory
    return os.path.isdir(fullpath)


def _derive_folder_fullpath(logs_location:Literal['condor', 'local']) -> str:
    if logs_location == 'local':
        logs_folder = str(project_paths.logs)
        folder_fullpath = os.path.join(logs_folder, "monitor_process")
    elif logs_location == 'condor':
        logs_folder = str(project_paths.condor_paths['io_dir']/'logs')
        folder_fullpath = os.path.join(logs_folder, "monitor_process")
    assert _is_existing_folder_fullpath(logs_folder)
    return folder_fullpath


def _float_value_from_str_with_units(s:str) -> float:
    s, *_ = s.split("[")
    return float(s)

def _parse_strings(
    D_str:list[str], N_str:list[str], trunc_dims:list[str], mem_list:list[str], cpu_list:list[str]
) -> _LogData:
    ## simple
    D = int(D_str[0])
    N = int(N_str[0])
    chi = int(trunc_dims[0])
    chi_bp = int(trunc_dims[1])

    ## lists:
    mem = [_float_value_from_str_with_units(s) for s in mem_list]
    cpu = [_float_value_from_str_with_units(s) for s in cpu_list]

    return _LogData(D, N, chi, chi_bp, mem, cpu)



def _get_log_data(log_fullpath:str) -> _LogData:
    # Get data:
    D, N, trunc_dims = logs.search_words_in_log(log_fullpath, "virtual_dim:", "big_lattice_size:", "trunc_dim:", max_line=CONFIG_NUM_HEADER_LINES)
    mem, cpu = logs.search_words_in_log(log_fullpath, "crnt-mem =", "crnt-cpu =")

    # Parse data:
    try:
        log_data = _parse_strings(D, N, trunc_dims, mem, cpu)
    except (ValueError, IndexError) as e:
        raise _ValidError(e)

    return log_data


def main(
    logs_location: Literal['condor', 'local'] = 'local'
):
    ## Get and check folder:
    folder_fullpath = _derive_folder_fullpath(logs_location)

    ## Init results collections:
    logs_data : list[_LogData] = []

    ## Iterate and get data per log:
    all_logs = files.get_all_files_fullpath_in_folder(folder_full_path=folder_fullpath)
    for log_fullpath in prints.ProgressBar(all_logs):
        assert isinstance(log_fullpath, str)

        try:
            this_data = _get_log_data(log_fullpath)
        except _ValidError:
            continue

        logs_data.append(this_data)


    # Done:
    print("Done.")


if __name__ == "__main__":
    main()