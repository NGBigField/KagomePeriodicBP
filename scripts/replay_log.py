import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from utils import logs, visuals
from matplotlib import pyplot as plt



def main(
    filename:str = "2024.03.23_22.43.49 KNTCBS"
):
    logs.plot_log(filename)
    visuals.save_figure(file_name=filename)
    
    print("Done.")


if __name__ == "__main__":
    main()