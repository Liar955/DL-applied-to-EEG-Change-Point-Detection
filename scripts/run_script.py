"""Backward-compatible entry point.

The recommended command-line runner is ``scripts/run_modular.py``. This file is
kept so older README snippets or local notes that call ``run_script.py`` still
work.
"""

from __future__ import annotations

from run_modular import main


if __name__ == "__main__":
    main()

