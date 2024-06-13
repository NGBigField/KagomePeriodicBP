import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from unit_cell import UnitCell, given_by

from utils import strings
from utils.visuals import AppendablePlot
from typing import Iterable


# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians
from tensor_networks.construction import kagome_tn_from_unit_cell
from algo.belief_propagation import robust_belief_propagation
from algo.measurements import measure_energies_and_observables_together, mean_expectation_values

import numpy as np

d = 2



def main(
    D = 2,
    sizes = [2, 4, 6, 8, 10],
)->None:

    ## Config:
    config = Config.derive_from_physical_dim(D=D)
    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None, None)
    config.bp.damping = 0.1
    config.bp.msg_diff_terminate = 1e-12

    ## Prepare plots:
    expectation_plot = AppendablePlot()    
    energies_plot = AppendablePlot()    
    for plot in [expectation_plot, energies_plot]:
        plot.axis.grid("on")
        plot.axis.set_xlabel("N")
    expectation_plot.axis.set_title("Expectation Values")
    energies_plot.axis.set_title("Energies")

    ## Increase size for unsu, measure TN and add to plot:
    for size in sizes:
        ## Get unsu result:
        expectations, energies = _per_size(D=D, size=size, config=config)

        ## Measure
        expectation_plot.append(**{key:(size, val) for key, val in expectations.items()})
        mean_energy = np.mean([val for val in energies.values()])
        energies_plot.append(mean=(size, mean_energy))
        for edge_tuple, energy in energies.items():
            edge_name = f"({edge_tuple[0]},{edge_tuple[1]})"
            energies_plot.append(**{edge_name:(size, energy)}, plt_kwargs=dict(alpha=0.5, marker="+"))

    print("Done")


def _per_size(D:int, size:int, config:Config):

    ## Get tnsu results:
    unit_cell = given_by.tnsu(D=D, size=size)

    ## Get :
    full_tn = kagome_tn_from_unit_cell(unit_cell, config.dims)
    _, _ = robust_belief_propagation(full_tn, None, config.bp)

    ## Calculate observables:
    energies, expectations, mean_energy = measure_energies_and_observables_together(full_tn, config.ite.interaction_hamiltonian, config.trunc_dim)
    expectations = mean_expectation_values(expectations)

    return expectations, energies



if __name__ == "__main__":
    main()
    