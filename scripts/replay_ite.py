import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import ITEProgressTracker

from matplotlib import pyplot as plt



def main(
    filename:str = "ite-tracker_2024.03.14_17.14.10 OFSJV"
):
    tracker : ITEProgressTracker = ITEProgressTracker.load(filename)
    size = tracker.memory_usage
    print(f"size={size} [bytes]")
    tracker.plot()
    print("Done replaying ITE")


if __name__ == "__main__":
    main()