import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Import utils:
from utils import saveload
# Import our kagome structure functions:
from tensor_networks import KagomeTNRepeatedUnitCell, KagomeTNArbitrary
# Config and Hamiltonians:
from containers import Config
from physics import hamiltonians
# Import our algorithms:
from algo.belief_propagation import belief_propagation, BPStats
from algo.measurements import measure_energies_and_observables_together


D = 4


def _print_success(stats:BPStats) -> None:
    if stats.success:
        s = f"Succeeded to converge on an error below {stats.final_config.msg_diff_good_enough!r}"
    else:
        s = f"Failed to converge!"
    s += f" error is now {stats.final_error!r}"
    print(s)


def main() -> dict:
    ## Config:
    bp_config = Config.BPConfig(
        damping=0.2,
        max_swallowing_dim=4*D**2
    )
    h = hamiltonians.heisenberg_afm()

    ## Create tensor network:
    tensors = saveload.load("Kagome-PEPS.pkl")
    tn = KagomeTNArbitrary(tensors=tensors)
    # tn.plot()

    ## Perform algorithms:
    messages, stats = belief_propagation(tn, config=bp_config)
    _print_success(stats)
    measure_energies_and_observables_together(tn, h, trunc_dim=2*D**2)
    print("Done.")

if __name__ == "__main__":
    main()