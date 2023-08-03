import sys
from pathlib import Path


src = Path(__file__).parent
base = src.parent
scripts = base/"scripts"
data = base/"data"


def _unique_paths()->None:
    s = set(sys.path)
    sys.path = list(s)


def _add_path(path:Path)->None:
    s = path.__str__()
    if s not in sys.path:
        sys.path.append(s)


def add_base()->None:
    _add_path(base)


def add_scripts()->None:
    _add_path(scripts)


def add_src()->None:
    _add_path(src)