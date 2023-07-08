import sys
from pathlib import Path


src = Path(__file__).parent
base = src.parent
scripts = base/"scripts"


def _add_path(path:Path)->None:
    sys.path.append(path.__str__())


def add_base()->None:
    _add_path(base)


def add_scripts()->None:
    _add_path(scripts)


def add_src()->None:
    _add_path(src)