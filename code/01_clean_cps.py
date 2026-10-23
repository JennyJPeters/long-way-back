"""Clean the CPS extract and build the calibration moments (data/clean/moments.json)."""
import runpy

import paths

runpy.run_path(str(paths.LIB / "clean_cps.py"), run_name="__main__")
