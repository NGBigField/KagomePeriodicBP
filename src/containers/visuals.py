from dataclasses import dataclass, field
from containers._meta import _ConfigClass
from typing import Iterable, TypeAlias
from utils import assertions



_Logical : TypeAlias = int|bool
_DEFAULT_PLOTS_TO_SHOW = (True, True, False)


@dataclass
class VisualsConfig(_ConfigClass): 

    progress_bars : bool = True  # Show progress bars to get a feel for expected execution time
    verbose : bool = False  # Print any tiny detail through the code execution 
    energies_print_decimal_point_length : int = 8
    
    _plots_to_show : tuple[bool] = _DEFAULT_PLOTS_TO_SHOW

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def live_plots(self) -> bool:
        return True if any(self._plots_to_show) else False
    
    @live_plots.setter
    def live_plots(self, value:_Logical|Iterable[_Logical]) -> None:
        self._plots_to_show_setter(value)

    def _plots_to_show_setter(self, value) -> None:
        _assert_msg = "live_plots attribute must be a bool or list of bools"

        if isinstance(value, Iterable):
            _new_list = []
            for x in value:
                x = assertions.logical(x, reason=_assert_msg) 
                _new_list.append(x)
            self._plots_to_show = tuple(_new_list)

        elif isinstance(value, bool) or value in [0, 1]:
            if value is True or value==1:
                self._plots_to_show = _DEFAULT_PLOTS_TO_SHOW
            else:
                self._plots_to_show = (False, False, False)

        else:
            raise ValueError(_assert_msg)
