import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from utils import logs, visuals
from sys import argv


def main(
    filename:str = "2024.03.26_09.58.31 WJBLUT"
):
    ## Parse inputs:
    if len(argv)>=2 and argv[1] is not None:
        filename = argv[1]


    logs.plot_log(filename)
    visuals.save_figure(file_name=filename)
    
    print("Done.")


if __name__ == "__main__":
    main()