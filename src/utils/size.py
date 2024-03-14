from dataclasses import dataclass, is_dataclass
from utils import iterations
import sys
import numpy as np


def _is_simple_native_python_type(obj)->bool:
    return isinstance(obj, (str, int, float, bool, complex, np.ndarray))


def get_object_size(obj, visited_objects:set|None=None)->int:
    """Estimates the total memory size of the object and its attributes.

    Args:
        obj: The object whose size to be estimated.

    Returns:
        The estimated size of the object in bytes.
    """
    ## Trimming tree against recursions:
    if visited_objects is None:
        visited_objects = set()
    elif id(obj) in visited_objects:
        return 0
    visited_objects.add(id(obj))

    ## Get the native size of the object:
    size = sys.getsizeof(obj)  # get object's size without its attributes
    if _is_simple_native_python_type(obj):
        return size
    
    ## Prepare for iteration over inner attributes:
    # Choose simpler iteration methods if exist in native structures:
    if isinstance(obj, dict):
        iterator = obj.values()
    elif is_dataclass(obj):
        iterator = iterations.iterate_dataclass_attributes(obj)
    elif isinstance(obj, (list, tuple, set)):
        iterator = obj
    else:
        iterator = iterations.iterate_objects_attributes(obj)

    ## Iterate with recursion:
    for attr_value in iterator:

        attr_size = get_object_size(attr_value, visited_objects=visited_objects)
        size += attr_size

    return size

    