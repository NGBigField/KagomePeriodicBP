from dataclasses import dataclass
from containers._meta import container_repr

@dataclass
class VisualsConfig: 
    live_plots : bool = True  # Show results as they come in
    progress_bars : bool = True  # Show progress bars to get a feel for expected execution time
    verbose : bool = False  # Print any tiny detail throught the code execution 

    def __repr__(self) -> str:
        return container_repr(self)
