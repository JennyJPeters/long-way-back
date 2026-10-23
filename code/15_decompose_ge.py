"""GE check of the matching friction in the child penalty."""
import argparse
import json

import numpy as np

from paths import CHECKPOINT, CLEAN

from model_extended import (run_extended_model, build_hc_grids,
                            solve_worker_shares_ext)
from smm_joint import make_base, vec_to_params, load_targets

SKILLS = {'Non-college': dict(father=0, mother=1),
          'College':     dict(father=2, mother=3)}


def base_params(mode, res, x, tgt):
    p = vec_to_params(x, make_base(res))
    p.delta = np.clip(tgt['eu'].copy(), 1e-4, 0.3)
    p.reentry_mode = mode
    return p


def earnings_emp(res_dict):
    return np.asarray(res_dict['earnings_pc']), np.asarray(res_dict['emp_rate'])


def earnings_emp_from_ss(ss, wage):
    n = wage.shape[0]
    tm = ss['E'] + ss['U'] + ss['N']
    earn = np.array([(ss['E_mat'][i, :] @ wage[i, :]) / max(tm[i], 1e-12) for i in range(n)])
    emp = ss['E'] / np.maximum(tm, 1e-12)
    return earn, emp


def contributions(earn_b, emp_b, earn_mo, emp_mo):
    """Matching-friction share of the mother-father child penalty, per skill/outcome."""
    rows = []
    for skill, idx in SKILLS.items():
        dad, mom = idx['father'], idx['mother']
        for name, base_arr, mo_arr, unit, size in [
                ('Earnings', earn_b, earn_mo, '%',
                 100. * (earn_b[dad] - earn_b[mom]) / max(earn_b[dad], 1e-12)),
                ('Employment', emp_b, emp_mo, 'pp',
                 100. * (emp_b[dad] - emp_b[mom]))]:
            g0 = base_arr[dad] - base_arr[mom]
            g1 = mo_arr[dad] - mo_arr[mom]
            share = (g0 - g1) / g0 if abs(g0) > 1e-12 else float('nan')
            rows.append((skill, name, size, unit, 100. * share))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='medium', choices=['fast', 'medium', 'full'])
    ap.add_argument('--full', action='store_true', help='alias for --res full')
    args = ap.parse_args()
    res = 'full' if args.full else args.res
    tgt = load_targets(); x = np.load(CHECKPOINT)

    print(f"Solving baseline + matching_off (mechanical & GE) at {res} resolution ...")
    base = run_extended_model(base_params('baseline', res, x, tgt), verbose=False)
    earn_b, emp_b = earnings_emp(base)

    # GE: full re-solve under matching_off
    ge = run_extended_model(base_params('matching_off', res, x, tgt), verbose=False)
    earn_ge, emp_ge = earnings_emp(ge)

    # Mechanical: baseline cutoffs and wages, stationary flows under matching_off
    g, wage = base['g'], base['wage_dollar_by_k']
    ia, in_, iq, pvec = (base['int_x_a_p'], base['int_x_n_p'],
                         base['int_x_q_p'], base['p_vec_full'])
    p_mo = base_params('matching_off', res, x, tgt)
    hc = build_hc_grids(p_mo)
    ss_mech = solve_worker_shares_ext(p_mo, g, hc, ia, in_, iq, pvec)
    earn_mech, emp_mech = earnings_emp_from_ss(ss_mech, wage)

    rows_mech = contributions(earn_b, emp_b, earn_mech, emp_mech)
    rows_ge = contributions(earn_b, emp_b, earn_ge, emp_ge)

    print("\n" + "=" * 74)
    print("  MATCHING FRICTION contribution to the child penalty: mechanical vs GE")
    print("=" * 74)
    print(f"{'Skill':>11} | {'Outcome':>10} | {'Gap':>8} | {'Mechanical':>11} | {'GE':>8}")
    print("-" * 62)
    for (sk, nm, sz, un, m), (_, _, _, _, ge_s) in zip(rows_mech, rows_ge):
        print(f"{sk:>11} | {nm:>10} | {sz:>5.1f}{un:<2} | {m:>10.1f}% | {ge_s:>6.1f}%")

    out = dict(resolution=res,
               note='matching-friction contribution to mother-father child penalty; '
                    'mechanical (frozen VFI) vs GE (full re-solve)',
               mechanical=[dict(skill=r[0], outcome=r[1], gap=r[2], unit=r[3], share=r[4]) for r in rows_mech],
               ge=[dict(skill=r[0], outcome=r[1], gap=r[2], unit=r[3], share=r[4]) for r in rows_ge])
    (CLEAN / 'decompose_ge.json').write_text(json.dumps(out, indent=2))
    print("\n[wrote data/clean/decompose_ge.json]\nDONE_GE")


if __name__ == '__main__':
    main()
