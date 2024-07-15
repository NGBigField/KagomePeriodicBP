from typing import Generator, Any
import numpy as np
import pickle, dill


from pathlib import Path
import sys
import os
from os.path import sep as foldersep

## Import itai and roy packages:
this_folder = Path(__file__).parent.__str__()
if this_folder not in sys.path:
    sys.path.append(this_folder)



## From this folder
from itai.kagome_lattice import main as itai_create_kagome_lattice
from itai.Kagome_BPSU_evolution import main as itai_find_ground_state


## Import src:
base_folder =  os.getcwd()
src_folder = str(Path(base_folder)/"src")
if src_folder not in sys.path:
    sys.path.append(src_folder)

from utils import files



simple_update_data_folder = Path(base_folder)/"data"/"simple_update_results"
simple_update_data_folder = simple_update_data_folder.__str__()


d = 2


# from itai.get_saved_results import find_results_by_dims

def _load(fullpath:str) -> Any:
    with open(fullpath, 'rb') as file:
        return dill.load(file)


def _all_saved_results_common(who:str) -> Generator[tuple[str, str], None, None]:
    if who=="itai":
        trgt_prefix = "BPSU"
    elif who=="itai":
        trgt_prefix = "tnsu"
    else:
        raise ValueError("Not an option")

    for fullpath in files.get_all_files_fullpath_in_folder(simple_update_data_folder):
        *_, filename = fullpath.split(foldersep)
        filename, suffix = filename.split(".")
        crnt_prefix, *_ = filename.split("-")
        if crnt_prefix==trgt_prefix:
            yield fullpath, filename

def _all_itai_saved_results() -> Generator[tuple[str, str], None, None]:
    return _all_saved_results_common(who="itai")


def _derive_n_d_from_itai_filename(filename:str) -> tuple[int, int]:
    *_, n_str, d_str = filename.split("-")
    assert n_str[0]=="n"
    assert d_str[0]=="D"
    n = int(n_str[1:])
    D = int(d_str[1:])
    return n, D

def _find_saved_itai_results(N, D) -> list[np.ndarray]:
    for fullpath, name in _all_itai_saved_results():
        crnt_N, crnt_D = _derive_n_d_from_itai_filename(name)
        if crnt_D==D and crnt_N==N:
            return _load(fullpath)
    else:
        return None

def get_itai_tn(N:int, D:int):
    tensor_network = _find_saved_itai_results(N, D)
    if tensor_network is None:
        kagome_block_data = itai_create_kagome_lattice(n=N, _print=False, periodic=True)
        output_dict = itai_find_ground_state(d=d, D=D, N_Kagome=N, lattice_fname=kagome_block_data['lattice']['path'])
        tensor_network = output_dict['data']
    return tensor_network
    

def main(
    N:int=2,
    D:int=2
):
    itai_tn = get_itai_tn(N=N, D=D)
    # get_roy_tn()

    itai_energy = 0


if __name__ == "__main__":
    main()