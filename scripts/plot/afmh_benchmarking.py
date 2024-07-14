## Import source code:
from pathlib import Path
import sys
base = Path(__file__).parent.parent.parent
src = base/"src"
if src.__str__() not in sys.path:
    sys.path.append(src.__str__())

from utils import visuals, files, saveload, csvs, strings, lists
from dataclasses import dataclass

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from unit_cell import BestUnitCellData, UnitCell
from containers.results import Measurements
from containers.configs import Config

from tensor_networks import KagomeTNRepeatedUnitCell

# For measurements of Kagome Lattice State:
from algo import measurements

d = 2

@dataclass
class RefResult:
    D:int
    SU:float
    VU:float


# Eisert's results from "variPEPS - a versatile tensor network library for variational ground state simulations in two spatial dimensions"
references = [
    RefResult(2,   -0.38620, -0.40454),
    RefResult(3,   -0.41786, -0.42688),
    RefResult(4,   -0.42323, -0.43038),
    RefResult(5,   -0.42866, -0.43286),
    RefResult(6,   -0.43188, -0.43451),
    RefResult(7,   -0.43313, -0.43527),
    RefResult(8,   -0.43391, -0.43552)
]
        #     D      SU        VU


def _collect_references():
    SUs = []
    VUs = []
    Ds  = []
    for res in references:
        Ds.append(res.D)
        SUs.append(res.SU)
        VUs.append(res.VU)
    return Ds, SUs, VUs


def _robust_energy_measurement(unit_cell:UnitCell) -> Measurements:
    ## define configs:
    d, D = unit_cell.derive_dimensions()
    config = Config.derive_from_dimensions(D)
    config.bp.msg_diff_terminate = 1e-12
    config.bp.trunc_dim = 2*D**2 + 10
    config.trunc_dim    = 2*D**2 + 20

    ## Run:
    return measurements.run_converged_measurement_test(unit_cell, config=config)
    

def _collect_best_results():
    Ds = []
    energies = []
    for fullpath in files.get_all_files_fullpath_in_folder(BestUnitCellData._folder_fullpath()):
        data : BestUnitCellData = saveload.save_or_load_with_fullpath(fullpath, saveload.Modes.Read)
        energy_at_run = data.mean_energy
        D = data.D
        unit_cell = data.unit_cell

        ## Verify energy:
        measurements = _robust_energy_measurement(unit_cell)
        energy = measurements.mean_energy
        
        
        ## Keep data:
        Ds.append(D)
        energies.append(energy)

    return Ds, energies


def _plot_line(x, y, linestyle:str="-", label:str|None=None, ) -> Line2D:
    line = plt.plot(x, y, label=label, linewidth=4, linestyle=linestyle)
    return line[0]


def _append_row_to_tabular(tabular:str, row:list[object]) -> str:
    for first, last, item in lists.iterate_with_edge_indicators(row):

        if isinstance(item, str):
            s = item
        elif isinstance(item, int):
            s = f"{item}"
        elif isinstance(item, float):
            s = strings.formatted(item, precision=5, signed=True)

        tabular += s

        if last:
            tabular += " \\\\ "
        else:
            tabular += " & "

    return tabular




def _create_csv(Ds, SUs, VUs, Ds_2, energies) -> None:

    csv = csvs.CSVManager(["D", "SU", "VU", "BlockBP"], name="ITE_AFM_H results compare")
    tabular = _append_row_to_tabular("", csv.columns)
    tabular += " \\hline"
    
    for i, D in enumerate(Ds):
        SU = SUs[i]
        VU = VUs[i]
        row = [D, SU, VU]

        if D in Ds_2:
            row.append(energies[i])
        else:
            row.append("")

        csv.append(row)
        tabular = _append_row_to_tabular(tabular, row)

    print(f"Created an csv file in {csv.fullpath!r}")
    print(f"for latex: \n")
    print(tabular)


def main():
    ## Collect reference results
    Ds, SUs, VUs = _collect_references()

    ## Plot references::
    line1 = _plot_line(Ds, SUs, linestyle=":", label="reference simple-update"     )
    line2 = _plot_line(Ds, VUs, linestyle=":", label="reference variational-update")

    ## collet my results:
    Ds_2, energies = _collect_best_results()
    line3 = _plot_line(Ds_2, energies, label="BlockBP")

    ## Pretty plot:
    ax = line1.axes
    ax.grid()
    ax.legend()

    visuals.draw_now()

    ## add csv:
    _create_csv(Ds, SUs, VUs, Ds_2, energies)

    print("Done.")
    



if __name__ == "__main__":
    main()




