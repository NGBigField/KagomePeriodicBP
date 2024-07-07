import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Import utils:
from utils import saveload
# Import our kagome structure functions:
from tensor_networks import KagomeTNRepeatedUnitCell, KagomeTNArbitrary
# Import our algorithms:
from containers import Config
from algo.belief_propagation import belief_propagation
from algo.measurements import measure_energies_and_observables_together



def main() -> dict:
    ## Config:
    # config = Config.BPConfig()

    ## Create tensor network:
    tensors = saveload.load("Kagome-PEPS.pkl")
    tn = KagomeTNArbitrary(tensors=tensors)
    # tn.plot()
    messages, stats = belief_propagation(tn)
    measure_energies_and_observables_together(tn)
    print("Done.")

if __name__ == "__main__":
    main()