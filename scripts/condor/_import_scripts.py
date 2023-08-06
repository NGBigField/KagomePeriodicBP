from pathlib import Path
import sys

scripst = Path(__file__).parent.parent
if scripst.__str__() not in sys.path:
    sys.path.append(scripst.__str__())
    
