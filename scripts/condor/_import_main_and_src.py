from pathlib import Path
import sys

def _add_path(path:Path):
    path_str = path.__str__()
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

main = Path(__file__).parent.parent.parent
_add_path(main)

src = main / "src"
_add_path(src)

    
