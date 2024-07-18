from pathlib import Path
import sys

base = Path(__file__).parent.parent.parent
src = base/"src"
if src.__str__() not in sys.path:
    sys.path.append(src.__str__())