from pathlib import Path
import sys, os

## Import src:
base_folder =  os.getcwd()
src_folder = str(Path(base_folder)/"src")
if src_folder not in sys.path:
    sys.path.append(src_folder)

from unit_cell.get_from._simple_update import _load_or_compute_tnsu_network as roy_get_network