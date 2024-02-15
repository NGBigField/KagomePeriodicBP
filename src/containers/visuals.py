from dataclasses import dataclass, field
from containers._meta import container_repr
from typing import Iterable
from utils import assertions



_DEFAULT_PLOTS_TO_SHOW = (True, True, False)


@dataclass
class VisualsConfig(): 

    progress_bars : bool = True  # Show progress bars to get a feel for expected execution time
    verbose : bool = False  # Print any tiny detail through the code execution 
    
    _plots_to_show : tuple[bool] = _DEFAULT_PLOTS_TO_SHOW

    @property
    def live_plots(self) -> bool:
        return True if any(self._plots_to_show) else False
    
    @live_plots.setter
    def live_plots(self, value:bool|Iterable[bool]) -> None:
        self._plots_to_show_setter(value)

    def _plots_to_show_setter(self, value) -> None:
        _assert_msg = "live_plots attribute must be a bool or list of bools"

        if isinstance(value, Iterable):
            _new_list = []
            for x in value:
                x = assertions.logical(x, reason=_assert_msg) 
                _new_list.append(x)
            self._plots_to_show = tuple(_new_list)

        elif isinstance(value, bool):
            if value is True:
                self._plots_to_show = _DEFAULT_PLOTS_TO_SHOW
            else:
                self._plots_to_show = (False, False, False)

        else:
            raise ValueError(_assert_msg)

    def __repr__(self) -> str:
        return container_repr(self)
