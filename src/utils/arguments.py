from typing import Any, Optional, TypeVar, Union, Callable
from dataclasses import dataclass
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
    

def only_single_input_allowed(function_name:str|None=None, **kwargs:_T)->tuple[str, _T]:
    only_key, only_value = None, None
    for key, value in kwargs.items():
        if value is not None:
            if only_key is not None:
                ## Raise error:
                err_msg = "Only a single input is allowed out of inputs "
                err_msg += f"{list(kwargs.keys())} "
                if function_name is not None:
                    err_msg += f"in function {function_name!r} "
                raise ValueError(err_msg)

            else:
                ## Tag as the only key and value:
                only_key, only_value = key, value

    return only_key, only_value