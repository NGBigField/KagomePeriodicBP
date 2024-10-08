import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from utils import logs, visuals, files
from visualizations.ite import plot_from_log
from sys import argv
import os

def _get_last_log_file()->str:
    ## File search:
    return files.get_last_file_in_folder(logs.DEFAULT_LOGS_FOLDER)



def main(
    filename:str|None = "all"
):
    ## Parse inputs:
    if len(argv)>=2 and argv[1] is not None:
        filename = argv[1]

    if filename is None or ( isinstance(filename, str) and filename=="last" ):
        filename = _get_last_log_file()

    if filename == "all":
        filenames = files.get_all_file_names_in_folder(logs.DEFAULT_LOGS_FOLDER)
        for filename in filenames:
            plot_from_log(filename, save=True)
        return

    else:

        ## Main call:
        plot_from_log(filename, save=True)

    # Done:
    print("Done.")


if __name__ == "__main__":
    main()