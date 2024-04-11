import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from tensor_networks import UnitCell

from utils import strings
from typing import Iterable


# Algos we test here:
from tensor_networks.construction import kagome_tn_from_unit_cell
from algo.belief_propagation import robust_belief_propagation
from algo.measurements import measure_energies_and_observables_together, mean_expectation_values

from physics import hamiltonians
from utils import visuals

import numpy as np

d = 2


def main(
    D = 2,
    N_vals : Iterable[int] = range(2, 12),
    parallel:bool = 0,
    chi_factor : int = 1,
    hamiltonian:str = "AFM",  # Anti-Ferro-Magnetic or Ferro-Magnetic
    damping:float|None = 0.1
)->tuple[float, str]:
    
    live_plots = True
    allow_prog_bar = True

    unit_cell = UnitCell.load("2024.03.24_22.37.58_VWPD - stable start")
    # unit_cell = UnitCell.random_product_state(d, D)

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.bp.damping = damping
    config.bp.max_swallowing_dim = 4*D**2
    config.bp.parallel_msgs = parallel
    config.trunc_dim *= chi_factor
    config.bp.max_swallowing_dim *= chi_factor

    # Interaction:
    match hamiltonian: 
        case "AFM":   config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None)
        case "FM":    config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm, None)
        case "FM-T":  config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm_with_field, None)
        case "Field": config.ite.interaction_hamiltonian = (hamiltonians.field, None)
        case _:
            raise ValueError("Not matching any option.")

    ## Prepare plots:
    expectation_plot = visuals.AppendablePlot()    
    energies_plot = visuals.AppendablePlot()    
    for plot in [expectation_plot, energies_plot]:
        plot.axis.grid("on")
        plot.axis.set_xlabel("N")
    expectation_plot.axis.set_title("Expectation Values")
    energies_plot.axis.set_title("Energies")

    ## Run:
    for N in N_vals:

        config.dims.big_lattice_size = N

        # Get big tn:
        full_tn = kagome_tn_from_unit_cell(unit_cell, config.dims)
        _, _ = robust_belief_propagation(full_tn, None, config.bp, live_plots, allow_prog_bar)

        ## Calculate observables:
        energies, expectations, mean_energy = measure_energies_and_observables_together(full_tn, config.ite.interaction_hamiltonian, config.trunc_dim)
        expectations = mean_expectation_values(expectations)

        expectation_plot.append(**{key:(N, val) for key, val in expectations.items()})
        mean_energy = np.mean([val for val in energies.values()])
        energies_plot.append(mean=(N, mean_energy))
        for edge_tuple, energy in energies.items():
            edge_name = f"({edge_tuple[0]},{edge_tuple[1]})"
            energies_plot.append(**{edge_name:(N, energy)}, plt_kwargs=dict(alpha=0.5, marker="+"))


    print("Done")



if __name__ == "__main__":
    main()