"""Software tests. Run from the repository root: ``python -m unittest discover -s testCases -t .``"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
