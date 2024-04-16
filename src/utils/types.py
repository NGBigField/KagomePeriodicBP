from typing import Generator, Callable, Any


def all_methods(obj, excption_func:Callable[[str], bool]|None=None)->Generator[Callable, None, None]:
    for attr_name in dir(obj):
        if attr_name[:2] == "__":
            continue

        if excption_func is not None and excption_func(attr_name)==True:
            continue
        
        attr = getattr(obj, attr_name)

        if callable(attr):
            yield attr