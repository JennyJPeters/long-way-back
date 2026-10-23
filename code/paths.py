"""Directory layout shared by every script in the package."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
LIB = CODE / "lib"

RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

CPS_FILE = RAW / "cps_00012.dat"
ASEC_FILE = RAW / "asec" / "cps_00013.dat"
ASEC_CLEAN = CLEAN / "asec"

MOMENTS = CLEAN / "moments.json"
MOMENT_SE = CLEAN / "moment_se.json"
CHECKPOINT_OLD = CLEAN / "smm_joint_checkpoint.npy"
CHECKPOINT_START = CLEAN / "smm_joint_scar_checkpoint.npy"
CHECKPOINT = CLEAN / "smm_joint_scar_med_checkpoint.npy"
CALIBRATION_RESULT = CLEAN / "smm_joint_scar_med_result.json"

if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

for directory in (CLEAN, ASEC_CLEAN, TABLES, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)
