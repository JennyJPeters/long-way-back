"""Regenerate every table and figure of the paper."""
import argparse
import subprocess
import sys
import time

from paths import ASEC_FILE, CHECKPOINT, CODE, CPS_FILE, MOMENT_SE, MOMENTS

PYTHON = sys.executable


def run(script, description, extra=()):
    command = [PYTHON, "-u", str(CODE / script), *extra]
    print(f"\n{'=' * 78}\n>> {description}\n   $ {' '.join(command)}\n{'=' * 78}", flush=True)
    started = time.time()
    if subprocess.run(command, cwd=str(CODE)).returncode != 0:
        sys.exit(f"FAILED: {description}")
    print(f"   finished in {time.time() - started:.0f}s", flush=True)


def require(path, stage):
    if not path.exists():
        sys.exit(f"{stage} needs {path}, which is missing; see README.md for how to obtain it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facts", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--checks", action="store_true")
    args = parser.parse_args()
    has_cps = CPS_FILE.exists()

    if args.rebuild_data:
        require(CPS_FILE, "--rebuild-data")
        require(ASEC_FILE, "--rebuild-data")
        run("01_clean_cps.py", "CPS -> moments.json")
        run("02_clean_asec.py", "ASEC -> cross-section and panel frames")
        run("03_estimate_delta.py", "ASEC -> depreciation and growth rates")
        run("04_bootstrap_moments.py", "Bootstrap covariance of the 37 moments -> moment_se.json", ["--B", "400"])

    if args.facts:
        require(CPS_FILE, "--facts")
        run("05_exit_asymmetry.py", "Figure: exit asymmetry (E->N, E->U)")
        run("06_reentry_flows.py", "Figures: activation and job-finding hazards")
        run("07_entry_asymmetry.py", "Figure: entry asymmetry")
        run("08_child_age_profile.py", "Figure: child penalty by age of youngest child")

    if args.recalibrate:
        require(MOMENTS, "--recalibrate")
        require(MOMENT_SE, "--recalibrate")
        run("09_calibrate_fast.py", "SMM, fast grid, bounded HC scar")
        run("10_calibrate_medium.py", "SMM, medium grid, confirmed at the reporting grid")

    if args.checks:
        run("00_checks.py", "HC-floor preflight", ["--preflight-only"])

    for path, stage in ((MOMENTS, "model stages"), (MOMENT_SE, "model stages"), (CHECKPOINT, "model stages")):
        require(path, stage)
    run("11_wage_gaps.py", "Model solve at the checkpoint and wage gaps",
        [] if has_cps else ["--model-only"])
    run("12_make_fit_tables.py", "Fit tables")
    run("13_counterfactuals.py", "Policy counterfactuals (GE)")
    run("14_decompose.py", "Child-penalty decomposition")
    run("15_decompose_ge.py", "GE check of the matching friction")
    run("16_decompose_dn.py", "Depreciation and re-entry friction (figure)")
    run("17_make_model_tables.py", "Calibration, counterfactual and GE decomposition tables")
    print("\nDONE. Tables are in output/tables, figures in output/figures.")


if __name__ == "__main__":
    main()
