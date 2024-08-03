from numpy import ndarray as np_ndarray
from dataclasses import fields
from copy import deepcopy

# For type annotations
from typing import TypeVar
_SpecificConfigClass = TypeVar("_SpecificConfigClass", bound="_ConfigClass")

def _space(level:int) -> str:
    assert level>0
    spaces = 4*level
    return " "*spaces

def container_repr(obj, level:int=1)->str:
    s = f"{obj.__class__.__name__}:"
    for field in fields(obj):
        s += "\n"
        value = getattr(obj, field.name)
        if isinstance(value, np_ndarray):
            value_str = f"ndarray of shape {value.shape}"
        else:
            value_str = str(value)
        s += _space(level)+f"{field.name}: {value_str}"
    return s


class _ConfigClass():
    def __setattr__(self, k, v): 
        is_annotation = k in self.__annotations__
        is_existing_attr = hasattr(self, k)
        if not (is_annotation or is_existing_attr): 
            raise AttributeError(f'{self.__class__.__name__} dataclass has no field {k}')
        super().__setattr__(k, v)

    def __repr__(self, **kwargs) -> str:
        return container_repr(self, **kwargs)
    
    def copy(self:_SpecificConfigClass)->_SpecificConfigClass:
        return deepcopy(self)