## Import source code:
from pathlib import Path
import sys
base = Path(__file__).parent.parent.parent
src = base/"src"
if src.__str__() not in sys.path:
    sys.path.append(src.__str__())

from utils import visuals, files, saveload
from dataclasses import dataclass

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from unit_cell import BestUnitCellData


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


def _collect_best_results():
    Ds = []
    energies = []
    for fullpath in files.get_all_files_fullpath_in_folder(BestUnitCellData._folder_fullpath()):
        data : BestUnitCellData = saveload.save_or_load_with_fullpath(fullpath, saveload.Modes.Read)
        energy = data.mean_energy
        D = data.D
        
        Ds.append(D)
        energies.append(energy)

    return Ds, energies


def _plot_line(x, y, linestyle:str="-", label:str|None=None, ) -> Line2D:
    line = plt.plot(x, y, label=label, linewidth=4, linestyle=linestyle)
    return line[0]


def main():

    ## Collect reference results
    Ds, SUs, VUs = _collect_references()

    ## Plot references::
    line1 = _plot_line(Ds, SUs, linestyle=":", label="reference simple-update"     )
    line2 = _plot_line(Ds, VUs, linestyle=":", label="reference variational-update")

    ## collet my results:
    Ds, energies = _collect_best_results()
    line3 = _plot_line(Ds, energies, label="BlockBP")

    ## Pretty plot:
    ax = line1.axes
    ax.grid()
    ax.legend()

    visuals.draw_now()
    print("Done.")

    



if __name__ == "__main__":
    main()




