from dataclasses import dataclass, field
from containers._meta import _ConfigClass
from typing import Iterable, TypeAlias, Literal, Final
from utils import assertions
from enum import Enum, auto, unique


_Logical : TypeAlias = int|bool
_DEFAULT_PLOTS_TO_SHOW = (True, True, False)


ProgBarPrintLevel : TypeAlias           = Literal['bubblecon', 'blockBP', 'ITE-per-mode', 'ITE-per-segment', 'ITE-per-delta-t', 'ITE-Main']
_ProgBarPrintLevelHierarchy : Final[list[str]] = ['bubblecon', 'blockBP', 'ITE-per-mode', 'ITE-per-segment', 'ITE-per-delta-t', 'ITE-Main']

@unique
class ProgressBarSConfig(Enum):
    ALL_ACTIVE = auto()
    ALL_DISABLED = auto()
    ONLY_MAIN = auto()


    @staticmethod
    def _main_levels_threshold() -> str:
        return "ITE-per-delta-t"

    def __bool__(self) -> bool:
        match self:
            case ProgressBarSConfig.ALL_ACTIVE:
                return True
            case ProgressBarSConfig.ALL_DISABLED:
                return False
        raise AttributeError("No valid response. Use method `is_active_at`. For exact result")
    
    def is_active_at(self, func_level:ProgBarPrintLevel) -> bool:
        if self is ProgressBarSConfig.ALL_ACTIVE:
            return True
        
        elif self is ProgressBarSConfig.ALL_DISABLED:
            return False
        
        elif self is ProgressBarSConfig.ONLY_MAIN:
            level = _ProgBarPrintLevelHierarchy.index(func_level)
            threshold = _ProgBarPrintLevelHierarchy.index(ProgressBarSConfig._main_levels_threshold())
            return level >= threshold
        
        else:
            raise ValueError("Not an optional type")


@dataclass
class VisualsConfig(_ConfigClass): 

    
    verbose : bool = False       # Print any tiny detail through the code execution 
    energies_print_decimal_point_length : int = 8
    
    _progress_bars : ProgressBarSConfig = ProgressBarSConfig.ALL_ACTIVE   # Show progress bars to get a feel for expected execution time
    _plots_to_show : tuple[bool, bool, bool] = _DEFAULT_PLOTS_TO_SHOW

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def live_plots(self) -> bool:
        return True if any(self._plots_to_show) else False
    
    @live_plots.setter
    def live_plots(self, value:_Logical|Iterable[_Logical]) -> None:
        self._plots_to_show_setter(value)

    @property
    def progress_bars(self) -> ProgressBarSConfig:
        return self._progress_bars
    
    @progress_bars.setter
    def progress_bars(self, value:Literal['all_active', 'all_disabled', 'only_main']|bool) -> None:
        self._progress_bars = _parse_progress_bar_config(value)

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


def _parse_progress_bar_config(value:Literal['all_active', 'all_disabled', 'only_main']|bool) -> ProgressBarSConfig:
        if isinstance(value, str):
            match value: 
                case 'all_active':
                    return ProgressBarSConfig.ALL_ACTIVE
                case 'all_disabled':
                    return ProgressBarSConfig.ALL_DISABLED
                case 'only_main':
                    return ProgressBarSConfig.ONLY_MAIN
                case _:
                    raise ValueError("Not an expected input")
        elif isinstance(value, bool):
            if value is True:
                return ProgressBarSConfig.ALL_ACTIVE
            else:
                return ProgressBarSConfig.ALL_DISABLED
        else:
            raise TypeError("Not an expected input type")