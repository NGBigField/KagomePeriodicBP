## For finding imports of tests and sources:
import pathlib, sys
this_folder = pathlib.Path(__file__).parent.__str__()
if this_folder not in sys.path:
    sys.path.append(this_folder)
import _import_src
sys.path.remove(this_folder)