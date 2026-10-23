"""Runs lib/entry_asymmetry.py (return-to-work rates by sex and child under 5)."""
import runpy

import paths

runpy.run_path(str(paths.LIB / "entry_asymmetry.py"), run_name="__main__")
