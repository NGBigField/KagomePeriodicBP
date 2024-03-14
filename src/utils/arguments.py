from typing import Any, Optional, TypeVar, Union, Callable
from dataclasses import dataclass
from collections.abc import Mapping, Container
from utils import saveload, strings, size
from sys import getsizeof
from os.path import getsize
_T = TypeVar('_T')

@dataclass
class Stats():
    execution_time : float = None  # type: ignore
    size_of_inputs : int = None
    size_of_outputs : int = None

    def __post_init__(self):
        pass

    @property
    def memory_usage(self)->int:
        if self.size_of_inputs is None or self.size_of_outputs is None:
            return None
        return self.size_of_inputs + self.size_of_outputs


def default_value(
    arg:Union[None, _T], 
    default:_T=None, 
    default_factory:Optional[Callable[[], _T ]]=None
) -> _T :
    if arg is not None:
        return arg
    if default is not None:
        return default
    if default_factory is not None:
        return default_factory()
    raise ValueError(f"Must provide either `default` value or function `default_factory` that generates a value")
    
