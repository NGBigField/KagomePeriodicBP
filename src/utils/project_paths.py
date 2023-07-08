import sys
from pathlib import Path



utils = Path(__file__).parent
src = utils.parent
base = src.parent
scripts = base/"scripts"


def _add_path(path:Path)->None:
    sys.path.append(path)


def add_scripts()->None:
    _add_path(scripts)


def add_src()->None:
    _add_path(src)