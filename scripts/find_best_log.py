import _import_src

from utils import files, logs, prints
import project_paths
from dataclasses import dataclass
from typing import Literal

foldersep = files.foldersep
DEFAULT_LOG_FOLDER = project_paths.logs.__str__()

CONFIG_NUM_HEADER_LINES = 65


class _ValidError(Exception): ...


@dataclass
class _LogData:
    D : int
    best_energy : float
    filename : str


def _parse_strings(D_str:list[str], energies_str:list[str]) -> tuple[int, list[float]]:
    D = int(D_str[0])
    energies = [float(e) for e in energies_str]
    return D, energies


def _get_log_data(folder_full_path:str, filename:str) -> _LogData:
    # Get data:
    fullpath = folder_full_path+foldersep+filename
    D = logs.search_words_in_log(fullpath, "virtual_dim:", max_line=CONFIG_NUM_HEADER_LINES)
    energies = logs.search_words_in_log(fullpath, "Mean energy after segment =")

    # Parse data:
    try:
        D, energies = _parse_strings(D, energies)
        best_energy = min(energies)
    except (ValueError, IndexError) as e:
        raise _ValidError(e)

    return _LogData(D=D, best_energy=best_energy, filename=filename)


def main(
    logs_location : Literal['condor', 'local'] = 'local'
):
    if logs_location == 'local':
        folder_full_path = str(project_paths.logs)
    elif logs_location == 'condor':
        folder_full_path = str(project_paths.condor_paths['io_dir']/'logs')
    
    best_logs : dict[int, _LogData] = {}
    all_logs = files.get_all_file_names_in_folder(folder_full_path=folder_full_path)

    for filename in prints.ProgressBar(all_logs):
        assert isinstance(filename, str)

        try:
            this_data = _get_log_data(folder_full_path, filename)
        except _ValidError:
            continue
        D = this_data.D

        # is best?
        if D in best_logs:
            # compare:
            if this_data.best_energy > best_logs[D].best_energy:
                continue

        best_logs[D] = this_data

    # Print:
    print(f"best logs are: ")
    for value in best_logs.values():
        print(f"    {value}")
    print("")


if __name__ == "__main__":
    main()