import _import_src

from utils import files, logs, prints
import project_paths
from dataclasses import dataclass

foldersep = files.foldersep
DEFAULT_LOG_FOLDER = project_paths.logs.__str__()

@dataclass
class _LogData:
    D : int
    best_energy : float
    filename : str


def main(
    logs_location:str=DEFAULT_LOG_FOLDER
):
    
    best_logs : dict[int, _LogData] = {}
    all_logs = files.get_all_file_names_in_folder(folder_full_path=logs_location)

    for filename in prints.ProgressBar(all_logs):
        # Get data:
        fullpath = logs_location+foldersep+filename
        D, energies = logs.search_words_in_log(fullpath, words=["virtual_dim:", "Mean energy after segment ="])

        # Parse data:
        D = int(D[0])
        energies = [float(e) for e in energies]
        try:
            best_energy = min(energies)
        except ValueError:
            continue

        this_data = _LogData(D=D, best_energy=best_energy, filename=filename)


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