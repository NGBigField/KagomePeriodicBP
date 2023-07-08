from pathlib import Path
import sys

src = Path(__file__).parent.parent/"src"
sys.path.append(src.__str__())