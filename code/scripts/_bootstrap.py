"""Make the repository root importable regardless of the working directory.

Every entry script imports this first so ``import wireless_twin`` works whether
you launch from the repo root or from inside ``scripts/`` (Linux or Windows).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
