import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from utils import logs, visuals
from sys import argv
import os



def _get_last_log_file()->str:
    ## File search:
    folder_full_path = logs.LOGS_FOLDER
    file_names = [file for file in os.listdir(folder_full_path)]
    return file_names[-1]


def main(
    filename:str|None = None
):
    ## Parse inputs:
    if len(argv)>=2 and argv[1] is not None:
        filename = argv[1]

    if filename is None:
        filename = _get_last_log_file()

    logs.plot_log(filename)
    visuals.save_figure(file_name=filename)
    
    print("Done.")


if __name__ == "__main__":
    main()