import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Import our kagome structure functions:
from tensor_networks import KagomeTNRepeatedUnitCell, KagomeTNArbitrary
from utils import saveload


def main(size:int) -> dict:
    tensors = saveload.load("Kagome-PEPS.pkl")
    tn = KagomeTNArbitrary(tensors=tensors)
    print("Done.")

if __name__ == "__main__":
    main(size=2)