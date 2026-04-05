"""Run Restricted Zone Monitor from the repository root: ``python main.py run``.

See ``python main.py --help`` for all commands.
"""

import sys
from pathlib import Path

# Allow running from a fresh checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from restricted_zone_monitor.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
