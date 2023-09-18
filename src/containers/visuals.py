from dataclasses import dataclass, field
from containers._meta import container_repr



def _DEFAULT_PLOTS_TO_SHOW():
    return [True, True, True]



@dataclass
class VisualsConfig: 

    live_plots : bool = True  # Show results as they come in
    progress_bars : bool = True  # Show progress bars to get a feel for expected execution time
    verbose : bool = False  # Print any tiny detail throught the code execution 

    _what_plots : list[bool] = field(default_factory=_DEFAULT_PLOTS_TO_SHOW)

    def __repr__(self) -> str:
        return container_repr(self)
