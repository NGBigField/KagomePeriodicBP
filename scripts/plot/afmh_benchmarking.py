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
from containers.results import MeasurementsOnUnitCell
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


def _robust_energy_measurement(unit_cell:UnitCell) -> MeasurementsOnUnitCell:
    ## define configs:
    d, D = unit_cell.derive_dimensions()
    config = Config.derive_from_dimensions(D)
    config.bp.msg_diff_terminate = 1e-12
    config.chi_bp =   D**2 + 10
    config.chi    = 2*D**2 + 20

    ## Run:
    return measurements.run_converged_measurement_test(unit_cell, config=config)
    

def _collect_best_results(measure_again:bool=False):
    Ds = []
    energies = []
    for data in BestUnitCellData.all_best_results():
        energy_at_run = data.mean_energy
        D = data.D
        unit_cell = data.unit_cell

        print(f"D={D}")
        print(f"    Energy at run: {energy_at_run}")

        if measure_again:
            ## Verify energy:
            measurements = _robust_energy_measurement(unit_cell)
            energy = measurements.mean_energy
            print(f"    Energy verified: {energy}")
        else:
            energy = energy_at_run
        
        
        ## Keep data:
        Ds.append(D)
        energies.append(energy)

    return Ds, energies


def _plot_line(x, y, **kwargs) -> Line2D:
    line = plt.plot(x, y, **kwargs)
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


def main(
    measure_again:bool = True
):
    ## Collect reference results
    Ds, SUs, VUs = _collect_references()

    ## Plot references::
    line1 = _plot_line(Ds, SUs, linestyle=":", label="simple-update"     , linewidth=3, alpha=0.8)
    line2 = _plot_line(Ds, VUs, linestyle=":", label="variational-update", linewidth=3, alpha=0.8)

    ## collet my results:
    Ds_2, energies_at_run = _collect_best_results(measure_again=False)
    line3 = _plot_line(Ds_2, energies_at_run, marker="*", markersize=15, linestyle="", label="BlockBP")
    if measure_again:
        Ds_2, energies_at_test = _collect_best_results(measure_again=True)
        line4 = _plot_line(Ds_2, energies_at_test, marker="^", markersize=15, linestyle="", label="BlockBP-at-test")

    ## colors:
    line1.set_color("tab:green")
    line2.set_color("tab:blue")
    line3.set_color("tab:red")
    if measure_again:
        line4.set_color("tab:purple")

    ## Pretty and labels:
    ax = line1.axes
    ax.grid()
    ax.set_xlabel("bond-dimension $D$")
    ax.set_ylabel("Ground-State Energy")
    ax.set_title("Comparing ground state energies on AFM-H Kagome")
    ax.legend()


    visuals.draw_now()

    ## add csv:
    _create_csv(Ds, SUs, VUs, Ds_2, energies_at_run)

    print("Done.")
    



if __name__ == "__main__":
    main()




