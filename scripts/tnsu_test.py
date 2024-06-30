import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from unit_cell import UnitCell, given_by

from utils import strings, csvs, prints
from utils.visuals import AppendablePlot, save_figure
from typing import Iterable



# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians
from tensor_networks.construction import kagome_tn_from_unit_cell
from algo.belief_propagation import robust_belief_propagation
from algo.measurements import measure_energies_and_observables_together, mean_expectation_values

import numpy as np
import time

d = 2



def main(
    D = 2,
    sizes = list(range(1, 3)),
)->None:

    ## Config:
    config = Config.derive_from_physical_dim(D=D)
    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None, None)
    config.bp.damping = 0.1
    config.bp.msg_diff_terminate = 1e-15
    config.bp.max_swallowing_dim = 2*D**2 + 10
    config.dims.big_lattice_size = 3

    ## Prepare plots:
    expectation_plot = AppendablePlot()    
    energies_plot = AppendablePlot()    
    mean_energy_plot = AppendablePlot()    
    for plot in [expectation_plot, energies_plot, mean_energy_plot]:
        plot.axis.grid("on")
        plot.axis.set_xlabel("N")
    expectation_plot.axis.set_title("Expectation Values")
    energies_plot.axis.set_title("Energies")

    table = csvs.CSVManager(columns=["N", "energy", "x", "y", "z", "run-time"])

    ## Increase size for tnsu, measure TN and add to plot:
    for size in sizes:
        ## Get unsu result:
        expectations, energies, run_time = _per_size_get_measurements(D=D, size=size, config=config)

        ## Measure and update plots
        _log_and_update(size=size, table=table, expectation_plot=expectation_plot, energies_plot=energies_plot,  mean_energy_plot=mean_energy_plot, 
                        expectations=expectations, energies=energies, run_time=run_time)

    print("Done")


def _log_and_update(size:int, table:csvs.CSVManager, expectation_plot:AppendablePlot, energies_plot:AppendablePlot, mean_energy_plot:AppendablePlot, expectations, energies, run_time:float)->None:
    expectation_plot.append(**{key:(size, val) for key, val in expectations.items()})
    mean_energy = np.mean([val for val in energies.values()])
    energies_plot.append(mean=(size, mean_energy))
    mean_energy_plot.append(mean=(size, mean_energy))
    for edge_tuple, energy in energies.items():
        edge_name = f"({edge_tuple[0]},{edge_tuple[1]})"
        energies_plot.append(**{edge_name:(size, energy)}, plt_kwargs=dict(alpha=0.5, marker="+"))

    save_figure(expectation_plot.fig, file_name=f"Expectation-size={size}")
    save_figure(energies_plot.fig, file_name=f"Energies-size={size}")
    save_figure(mean_energy_plot.fig, file_name=f"Energies-Mean-size={size}")

    table.append([size, mean_energy, expectations["x"], expectations["y"], expectations["z"], run_time])

    print(
        "size: "+strings.formatted(size, width=2)+" | mean_energy="+strings.formatted(mean_energy, precision=10, width=12, signed=True)+" | run-time="+strings.formatted(run_time, width=8, precision=4)+"[sec]"
    )


def _per_size_get_measurements(D:int, size:int, config:Config):

    ## Get tnsu results:
    t1 = time.perf_counter()
    unit_cell = given_by.tnsu(D=D, size=size)
    t2 = time.perf_counter()
    run_time = t2 - t1

    ## Get :
    full_tn = kagome_tn_from_unit_cell(unit_cell, config.dims)
    _, _ = robust_belief_propagation(full_tn, None, config.bp)

    ## Calculate observables:
    energies, expectations, entanglement = measure_energies_and_observables_together(full_tn, config.ite.interaction_hamiltonian, config.trunc_dim)
    expectations = mean_expectation_values(expectations)

    return expectations, energies, run_time



if __name__ == "__main__":
    main()
    