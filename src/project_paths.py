import sys
from pathlib import Path
from typing import Final
import yaml


src     : Final[Path] = Path(__file__).parent
base    : Final[Path] = src.parent
scripts : Final[Path] = base/"scripts"
logs    : Final[Path] = base/"logs"
data    : Final[Path] = base/"data"


_condor_config_path = scripts / "condor" / "configurations.yml"
with open(str(_condor_config_path), 'r') as f:
    _condor_config = yaml.safe_load(f)
condor_paths : dict[str, Path] = {key:Path(path) for key,path in _condor_config['directories'].items()}


def _unique_paths()->None:
    s = set(sys.path)
    sys.path = list(s)


def _add_path(path:Path)->None:
    s = path.__str__()
    if s not in sys.path:
        sys.path.insert(0, s)


def add_base()->None:
    _add_path(base)


def add_src()->None:
    _add_path(src)


def add_scripts(add_necessary_imports_for_scripts:bool=True)->None:
    _add_path(scripts)
    if add_necessary_imports_for_scripts:
        add_base()
        add_src()