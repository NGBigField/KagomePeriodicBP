

from typing import TypeAlias, TypeVar
_T = TypeVar("_T")

from .main import iterate_with_edge_indicators, iterate_with_periodic_prev_next_items

SortedList : TypeAlias = list[_T]

def unique_values(l:SortedList) -> bool:
    for _, crnt, next in iterate_with_periodic_prev_next_items(l):
        if crnt == next:
            return False
    return True