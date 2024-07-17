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
from roy import roy_get_network

## Import src:
base_folder =  os.getcwd()
src_folder = str(Path(base_folder)/"src")
if src_folder not in sys.path:
    sys.path.append(src_folder)

from utils import files, lists, visuals, csvs
from algo.measurements import calc_measurement_non_unit_cell_kagome_tn, MeasurementsOnUnitCell
from containers import Config
from tensor_networks import KagomeTNArbitrary


_simple_update_data_folder = Path(base_folder)/"data"/"simple_update_results"
simple_update_data_folder = str(_simple_update_data_folder)+foldersep


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


def _derive_n_d_from_filename(filename:str, who:str) -> tuple[int, int]:
    *_, n_str, d_str = filename.split("-")
    assert n_str[0]=="n"
    assert d_str[0]=="D"
    n = int(n_str[1:])
    D = int(d_str[1:])
    return n, D


def _find_saved_tensor_network(D:int, N:int, who:str) -> list[np.ndarray]|None:
    for fullpath, filename in _all_saved_results_common(who=who):
        crnt_N, crnt_D = _derive_n_d_from_filename(filename, who=who)
        if crnt_D==D and crnt_N==N:
            return _load(fullpath)
    else:
        return None 


def get_itai_tn(D:int, N:int) -> list[np.ndarray]:
    tensor_network = _find_saved_tensor_network(D, N, who="itai")
    if tensor_network is None:
        kagome_block_data = itai_create_kagome_lattice(n=N, _print=False, periodic=True, save_at=simple_update_data_folder)
        output_dict = itai_find_ground_state(d=d, D=D, N_Kagome=N, lattice_fname=kagome_block_data['lattice']['path'], save_at=simple_update_data_folder)
        tensor_network = output_dict['data']
    return tensor_network  


def get_roy_tn(D:int, N:int):
    tensor_network = roy_get_network(D=D, size=N, periodic=True)
    return tensor_network.tensors


def measure_energy(
    tn:KagomeTNArbitrary|list[np.ndarray],
    parallel_msgs:bool=False,
    print_:bool=False
) -> float:

    if isinstance(tn, KagomeTNArbitrary):
        pass
    elif isinstance(tn, list) and lists.all_isinstance(tn, np.ndarray):
        tn = KagomeTNArbitrary(tn)
    else:
        raise TypeError(f"Not supporting input `tn` of type {type(tn)!r}")

    ## Config:
    D = tn.dimensions.virtual_dim
    bp_config = Config.BPConfig(
        damping = 0.1,
        trunc_dim = 2*D**2 + 10,
        parallel_msgs=parallel_msgs,
        max_iterations=50,
        msg_diff_terminate=1e-12
    )
    config = Config.derive_from_dimensions(D)
    config.bp = bp_config
    config.trunc_dim = 2*D**2 + 20

    ## Run:
    energy = calc_measurement_non_unit_cell_kagome_tn(
        tn, config=config,
        print_=print_
    )
    return energy


def both_per_D_and_N(
    D:int=2,
    N:int=2
) -> tuple[float, float]:
    itai_tn = get_itai_tn(N=N, D=D)
    itai_energy = measure_energy(itai_tn)
    print(f"Itai energy = {itai_energy}")

    roy_tn = get_roy_tn(N=N, D=D)
    roy_energy = measure_energy(roy_tn)
    print(f"Roy  energy = {roy_energy}")

    return itai_energy, roy_energy


def _add_results_to_plot_and_csv(
    plot:visuals.AppendablePlot, csv:csvs.CSVManager, 
    D:int, N:int, 
    itai_energy:float, roy_energy:float
) -> None:
    ## Plot:
    plot.append(values=dict(
        Itai=(N, itai_energy),
        Roy =(N, roy_energy )
    ))

    ## Csv:
    csv.append([D, N, itai_energy, roy_energy])



def main(
    Ds = [4],
    Ns = [2, 3, 4]
):
    plot = visuals.AppendablePlot()
    csv = csvs.CSVManager(["D", "N", "Itai", "Roy"])

    for D in Ds:
        for N in Ns:
            itai_energy, roy_energy = both_per_D_and_N(N=N, D=D)
            _add_results_to_plot_and_csv(plot, csv, D, N, itai_energy, roy_energy)

    visuals.save_figure(plot.fig)
    print("Finished!")

if __name__ == "__main__":
    main()