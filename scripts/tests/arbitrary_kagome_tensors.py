import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Import utils:
from utils import saveload, lists
# Import our kagome structure functions:
from tensor_networks import KagomeTNRepeatedUnitCell, KagomeTNArbitrary, LatticeDirection
# Config and Hamiltonians:
from containers import Config
from physics import hamiltonians
# Import our algorithms:
from algo.belief_propagation import belief_propagation, BPStats, MessageDictType
from algo.measurements import measure_energies_and_observables_together
# help types:
from enums import UpdateMode

## Where to load from
from unit_cell.get_from._simple_update import DATA_SUBFOLDER as SIMPLE_UPDATE_DATA_SUBFOLDER



def _print_success(stats:BPStats) -> None:
    if stats.success:
        s = f"Succeeded to converge on an error below {stats.final_config.msg_diff_good_enough!r}"
    else:
        s = f"Failed to converge!"
    s += f" error is now {stats.final_error!r}"
    print(s)


def _get_energy_per_site_per_mode(shifted_tn:KagomeTNArbitrary, h, D, bp_config) -> dict[UpdateMode, float]:
    energy_per_site_per_mode = {}    
    for mode in UpdateMode.all_options():
        print(f"Mode = {mode}")
        energies, expectations, entanglement = measure_energies_and_observables_together(shifted_tn, h, trunc_dim=2*D**2, mode=mode)
        print(energies)
        energy_per_site = sum(energies.values())/3
        print(energy_per_site)
        energy_per_site_per_mode[mode] = energy_per_site

    return energy_per_site_per_mode


def main(
    parallel_msgs : bool = False,
    filename : str = "save-BPSU-Kagome-PEPS-n4-D3.pkl"  # /Kagome-PEPS.pkl / Kagome-PEPS-n2-D3.pkl / save-BPSU-Kagome-PEPS-n4-D3
) -> dict:
    
    ## Create tensor network:
    tensors = saveload.load(filename, sub_folder=SIMPLE_UPDATE_DATA_SUBFOLDER)
    tn = KagomeTNArbitrary(tensors=tensors)

    ## Config:
    D = tn.dimensions.virtual_dim
    bp_config = Config.BPConfig(
        damping=0.1,
        max_swallowing_dim=D**2,
        parallel_msgs=parallel_msgs,
        max_iterations=50
    )
    h = hamiltonians.heisenberg_afm()    

    all_energies = []
    messages = None
    ## For each shift in some direction along the lattice:
    for shifted_tn in tn.all_lattice_shifting_options():
        # for the shifted network: Block BP once:
        messages, stats = belief_propagation(shifted_tn, messages, config=bp_config)
        _print_success(stats)
        # Get energy:
        energies_per_mode = _get_energy_per_site_per_mode(shifted_tn, h, D, bp_config)
        all_energies.append(energies_per_mode)
        break
    
    print("\n"*3)
    for mode in UpdateMode.all_options():
        print(f"Mode = {mode}")
        per_mode_energies = [energies_per_mode[mode] for energies_per_mode in all_energies]
        print(per_mode_energies)
        mean_energy = lists.average(all_energies)
        print(f"mean_energy={mean_energy}")
        print(f"\n")


    print("Done.")

if __name__ == "__main__":
    main()