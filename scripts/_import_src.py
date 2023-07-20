from pathlib import Path
import sys

base = Path(__file__).parent.parent
sys.path.append(base.__str__())
src = base/"src"
sys.path.append(src.__str__())