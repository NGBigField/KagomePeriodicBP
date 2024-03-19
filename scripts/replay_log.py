import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from utils import logs



def main(
    filename:str = "2024.03.14_17.14.10 EMBKDJ"
):
    logs.plot_log(filename)
    print("Done.")


if __name__ == "__main__":
    main()