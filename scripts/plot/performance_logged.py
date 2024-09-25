import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from utils import logs, visuals, files, prints, saveload, signal_processing
import project_paths


from typing import Literal
import os, pathlib
from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np


class _ValidError(Exception): ...

@dataclass
class _LogData:
    D : int
    N : int
    chi : int
    chi_bp : int
    mem : list[float]
    cpu : list[float]
    fullpath: str


CONFIG_NUM_HEADER_LINES = 65
T_SAMPLE = 15  # Sample interval [sec]


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

    return _LogData(D, N, chi, chi_bp, mem, cpu, "")



def _get_log_data(log_fullpath:str) -> _LogData:
    # Get data:
    D, N, trunc_dims = logs.search_words_in_log(log_fullpath, "virtual_dim:", "big_lattice_size:", "trunc_dim:", max_line=CONFIG_NUM_HEADER_LINES)
    mem, cpu = logs.search_words_in_log(log_fullpath, "crnt-mem =", "crnt-cpu =")

    # Parse data:
    try:
        log_data : _LogData = _parse_strings(D, N, trunc_dims, mem, cpu)
    except (ValueError, IndexError) as e:
        raise _ValidError(e)

    log_data.fullpath = log_fullpath

    return log_data

def _gather_log_data(
    folder_fullpath:str,
    limited_number_of_logs:int|None
) -> list[_LogData]:
    ## Init results collections:
    logs_data : list[_LogData] = []
    ## Iterate and get data per log:
    all_logs = files.get_all_files_fullpath_in_folder(folder_full_path=folder_fullpath)

    if limited_number_of_logs is not None:
        assert isinstance(limited_number_of_logs, int)
        all_logs = all_logs[:limited_number_of_logs]

    for log_fullpath in prints.ProgressBar(all_logs):
        assert isinstance(log_fullpath, str)

        try:
            this_data = _get_log_data(log_fullpath)
        except _ValidError:
            continue

        logs_data.append(this_data)
    return logs_data


def _new_sub_figures(
    Ds:list[int],
    separate_figures:bool
) -> tuple[visuals.Figure, dict[int, visuals.Axes]]:
    
    axes_dict = {}

    if separate_figures:
        for D in Ds:
            fig, ax = plt.subplots()
            ax.set_title(f"D={D}")            
            axes_dict[D] = ax

    else:
        fig, axes = plt.subplots(1, len(Ds))
        fig.set_figwidth(fig.get_figwidth()*2)

        if isinstance(axes, np.ndarray):
            pass
        else:
            axes = [axes]

        for D, ax in zip(Ds, axes):
            ax : plt.Axes
            ax.set_title(f"D={D}")            
            axes_dict[D] = ax

        fig.tight_layout()

    return fig, axes_dict


def _add_titles(separate_figures:bool, title_text:str, fig:visuals.Figure, axes:visuals.Axes) -> None:
    if separate_figures:
        for ax in axes.values():
            crnt_text = ax.title.get_text()
            ax.title.set_text(title_text+"\n"+crnt_text)
    else:
        fig.suptitle(title_text)


def _plot_data(
    logs_data:list[_LogData],
    separate_figures:bool
):

    ## Gather hyper-data:
    Ds = list({data.D for data in logs_data})
    Ns = list({data.N for data in logs_data})
    colors = list(visuals.distinct_colors(len(Ns)))

    ## Prepare plots::
    cpu_fig, cpu_axes = _new_sub_figures(Ds, separate_figures)
    _add_titles(separate_figures, "CPU utilization", cpu_fig, cpu_axes)
    for i, ax in enumerate(cpu_axes.values()):
        ax.set_xlabel("time [sec]")
        if i==0:
            ax.set_ylabel("cpu[%]")

    mem_fig, mem_axes = _new_sub_figures(Ds, separate_figures)
    _add_titles(separate_figures, "RAM Memory Usage", mem_fig, mem_axes)
    for i, ax in enumerate(mem_axes.values()):
        ax.set_xlabel("time [sec]")
        if i==0:
            ax.set_ylabel("RAM memory [GB]")
    axes_dict : dict[str, dict[int, plt.axes]] = {
        "cpu" : cpu_axes,
        "mem" : mem_axes,
    }

    ## Plot:
    for data in logs_data:        
        ## Get data:
        D = data.D
        N = data.N
        color = colors[Ns.index(N)]

        for key in ["cpu", "mem"]:
            ax = axes_dict[key][D]
            ## Get values:
            values = data.__getattribute__(key)
            time_vec = [i*T_SAMPLE for i, _ in enumerate(values)]
            # Smooth values
            max_values = signal_processing.max_or_min_filter(values, 100, 'max')

            # LabeL:
            label = f"{N}"
            if visuals.check_label_given(ax, label):
               label = None 

            ## Plot:
            ax.plot(time_vec, max_values, color=color, label=label)
    
    ## Last adjustments:
    for axes in axes_dict.values():
        for ax in axes.values():
            ax.legend()
        fig = ax.figure.tight_layout()
    visuals.draw_now()

    return axes_dict


def main(
    load:bool=True,
    logs_location: Literal['condor', 'local'] = 'local',
    limited_number_of_logs:int|None = None
):
    if load:
        logs_data = saveload.load("logs_data")
    else:
        folder_fullpath = _derive_folder_fullpath(logs_location)
        logs_data = _gather_log_data(folder_fullpath, limited_number_of_logs)
        saveload.save(logs_data, "logs_data")

    axes2 = _plot_data(logs_data, separate_figures=True)
    axes1 = _plot_data(logs_data, separate_figures=False)

    # Done:
    print("Done.")


if __name__ == "__main__":
    main()