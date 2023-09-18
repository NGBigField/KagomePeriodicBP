import sys
from pathlib import Path


src : Path = Path(__file__).parent
base : Path = src.parent
scripts : Path = base/"scripts"
data : Path = base/"data"


def _unique_paths()->None:
    s = set(sys.path)
    sys.path = list(s)


def _add_path(path:Path)->None:
    s = path.__str__()
    if s not in sys.path:
        sys.path.append(s)


def add_base()->None:
    _add_path(base)


def add_src()->None:
    _add_path(src)


def add_scripts(add_necessary_imports_for_scripts:bool=True)->None:
    _add_path(scripts)
    if add_necessary_imports_for_scripts:
        add_base()
        add_src()