from numpy import ndarray as np_ndarray
from dataclasses import fields

def container_repr(obj)->str:
    s = f"{obj.__class__.__name__}:"
    for field in fields(obj):
        s += "\n"
        value = getattr(obj, field.name)
        if isinstance(value, np_ndarray):
            value_str = f"ndarray of shape {value.shape}"
        else:
            value_str = str(value)
        s += f"    {field.name}: {value_str}"
    return s