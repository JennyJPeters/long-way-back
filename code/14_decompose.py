"""Decompose the mother-father child penalty into re-entry frictions."""
import argparse
import json

import numpy as np

from paths import CHECKPOINT, CLEAN, TABLES

from model_extended import run_extended_model
from smm_joint import make_base, vec_to_params, load_targets
MODES = ['baseline', 'activation_off', 'activation_off_ge', 'frictionless']

# Type indices: child = 0-3, no child = 4-7; within block 0=UnM 1=UnF 2=SkM 3=SkF.
SKILLS = {
    'Non-college': dict(father=0, mother=1),
    'College':     dict(father=2, mother=3),
}


def solve_all(res):
    """Solve the model under each re-entry mode."""
    tgt = load_targets()
    x = np.load(CHECKPOINT)
    sol = {}
    for mode in MODES:
        p = vec_to_params(x, make_base(res))
        p.delta = np.clip(tgt['eu'].copy(), 1e-4, 0.3)  # E->U rate taken from data
        p.reentry_mode = mode
        r = run_extended_model(p, verbose=False)
        sol[mode] = dict(emp=np.asarray(r['emp_rate']),
                         earn=np.asarray(r['earnings_pc']))
        print(f"  solved {mode:14s} (mother UnF_c emp={r['emp_rate'][1]:.3f})")
    return sol


def decompose_gap(sol, ref_idx, mother_idx, outcome):
    """Return (baseline_gap, activation, matching, withdrawal, activation_ge) as shares of the gap."""
    def gap(mode):
        return sol[mode][outcome][ref_idx] - sol[mode][outcome][mother_idx]
    g0, g1, g2 = gap('baseline'), gap('activation_off'), gap('frictionless')
    g1_ge = gap('activation_off_ge')
    if abs(g0) < 1e-9:
        return g0, float('nan'), float('nan'), float('nan'), float('nan')
    activation = (g0 - g1) / g0
    matching = (g1 - g2) / g0
    withdrawal = g2 / g0            # residual that survives a frictionless return
    # GE counterpart of activation, from mode activation_off_ge.
    activation_ge = (g0 - g1_ge) / g0
    return g0, activation, matching, withdrawal, activation_ge


def emp_pp(sol, ref_idx, mother_idx):
    """Baseline employment gap in percentage points."""
    return 100. * (sol['baseline']['emp'][ref_idx] - sol['baseline']['emp'][mother_idx])


def earn_pct(sol, ref_idx, mother_idx):
    """Baseline earnings gap as % of the reference's earnings."""
    ref = sol['baseline']['earn'][ref_idx]
    return 100. * (ref - sol['baseline']['earn'][mother_idx]) / max(ref, 1e-12)


def build_rows(sol):
    """Decomposition rows by skill and outcome."""
    rows = []
    for skill, idx in SKILLS.items():
        ref, mom = idx['father'], idx['mother']
        for outcome, unit, size in [('emp', 'pp', emp_pp(sol, ref, mom)),
                                    ('earn', '%', earn_pct(sol, ref, mom))]:
            g0, act, mat, wdl, act_ge = decompose_gap(sol, ref, mom, outcome)
            saturated = bool(wdl < 0)
            rows.append(dict(skill=skill,
                             outcome=('Employment' if outcome == 'emp' else 'Earnings'),
                             gap_size=size, gap_unit=unit,
                             activation=act, matching=mat, withdrawal=wdl,
                             activation_ge=act_ge,
                             activation_ge_delta=act_ge - act,
                             reentry_total=act + mat, saturated=saturated))
    return rows


def print_table(rows):
    print("\n" + "=" * 88)
    print("  DECOMPOSITION OF THE CHILD PENALTY (mother vs father among parents)")
    print("  shares of the baseline gap. Re-entry = activation (N->U) + matching (U->E);")
    print("  withdrawal = residual surviving frictionless return.  * = gap over-closes (>=100%)")
    print("=" * 88)
    hdr = f"{'Skill':>11} | {'Outcome':>10} | {'Gap':>9} | " \
          f"{'Activation':>10} | {'Act(GE)':>8} | {'Matching':>9} | " \
          f"{'Re-entry':>9} | {'Withdraw':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        star = ' *' if r['saturated'] else ''
        print(f"{r['skill']:>11} | {r['outcome']:>10} | "
              f"{r['gap_size']:>6.1f}{r['gap_unit']:<2} | "
              f"{100 * r['activation']:>9.1f}% | {100 * r['activation_ge']:>7.1f}% | "
              f"{100 * r['matching']:>8.1f}% | "
              f"{100 * r['reentry_total']:>8.1f}% | {100 * r['withdrawal']:>7.1f}%{star}")


def write_tex(rows):
    L = [r"\begin{tabular}{ll r rrrr r}", r"\toprule",
         r"& & & \multicolumn{4}{c}{Re-entry friction} & Withdrawal \\",
         r"\cmidrule(lr){4-7}",
         r"Skill & Outcome & Baseline & Activation & \;\;(GE) & Matching & Total & (residual) \\",
         r"\midrule"]
    for r in rows:
        unit = r"pp" if r['gap_unit'] == 'pp' else r"\%"
        note = r"$^{\dagger}$" if r['saturated'] else ""
        L.append(f"{r['skill']} & {r['outcome']} & "
                 f"{r['gap_size']:.1f} {unit} & "
                 f"{100 * r['activation']:.0f}\\% & {100 * r['activation_ge']:.0f}\\% & "
                 f"{100 * r['matching']:.0f}\\% & "
                 f"{100 * r['reentry_total']:.0f}\\% & {100 * r['withdrawal']:.0f}\\%{note} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / 'tab_decomposition.tex').write_text("\n".join(L) + "\n")
    print("\n[wrote output/tables/tab_decomposition.tex]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='medium', choices=['fast', 'medium', 'full'],
                    help='solving resolution (full=100x600 OOMs on 3 solves; medium=30x200)')
    ap.add_argument('--full', action='store_true', help='alias for --res full')
    args = ap.parse_args()
    res = 'full' if args.full else args.res
    print(f"Solving three re-entry regimes at {res} resolution ...")
    sol = solve_all(res)
    rows = build_rows(sol)
    print_table(rows)
    write_tex(rows)
    out = dict(resolution=res, note='mother-penalty decomposition; shares of baseline gap',
               rows=rows,
               emp=dict((m, sol[m]['emp'].tolist()) for m in MODES),
               earn=dict((m, sol[m]['earn'].tolist()) for m in MODES))
    json.dump(out, open(CLEAN / 'decompose_results.json', 'w'), indent=2)
    print("[wrote data/clean/decompose_results.json]\nDONE_DECOMP")


if __name__ == '__main__':
    main()
