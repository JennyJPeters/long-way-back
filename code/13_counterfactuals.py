"""Policy counterfactuals at the calibrated parameters, each re-solved in GE."""
import argparse
import json
import numpy as np

from paths import CHECKPOINT, CLEAN
from model_extended import run_extended_model
from smm_joint import make_base, vec_to_params, load_targets

# mother / father type indices by skill
PAIRS = {'Non-college': dict(mother=1, father=0, nochild_w=5),
         'College':     dict(mother=3, father=2, nochild_w=7)}


def base_params(res):
    tgt = load_targets()
    p = vec_to_params(np.load(CHECKPOINT), make_base(res))
    p.delta = np.clip(tgt['eu'].copy(), 1e-4, 0.3)
    return p


def childcare_subsidy(s):
    """Lower mothers' K toward the childless-woman K by fraction s (s=1 => equal)."""
    def f(p):
        p.K = p.K.copy()
        p.K[1] = p.K[5] + (1. - s) * (p.K[1] - p.K[5])   # UnF-c toward UnF-0
        p.K[3] = p.K[7] + (1. - s) * (p.K[3] - p.K[7])   # SkF-c toward SkF-0
    return f


def guaranteed_return(p):
    # The job is held: activation sends her straight to E, skipping matching.
    p.reentry_mode = 'matching_off'


SCENARIOS = [
    ('Baseline', lambda p: None),
    ('Childcare subsidy (half)', childcare_subsidy(0.5)),
    ('Childcare subsidy (full)', childcare_subsidy(1.0)),
    ('Parental leave (guaranteed return)', guaranteed_return),
]


def outcomes(r):
    lfp = np.asarray(r['model_lfp']); emp = np.asarray(r['emp_rate'])
    earn = np.asarray(r['earnings_pc']); V = np.asarray(r['V_entry'])
    return dict(lfp=lfp, emp=emp, earn=earn, V=V, gpg=float(r['gender_pay_gap']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--full', action='store_true')
    res = 'full' if ap.parse_args().full else 'medium'
    print(f"Solving {len(SCENARIOS)} scenarios at {res} resolution (GE re-solve each) ...")

    sol = {}
    for name, mod in SCENARIOS:
        p = base_params(res); mod(p)
        sol[name] = outcomes(run_extended_model(p, verbose=False))
        print(f"  solved: {name}")
    base = sol['Baseline']

    def cp_earn(o, sk):
        i = PAIRS[sk]; return 100. * (o['earn'][i['father']] - o['earn'][i['mother']]) / max(o['earn'][i['father']], 1e-12)

    print("\n" + "=" * 96)
    print("  POLICY COUNTERFACTUALS (GE): mothers, vs baseline")
    print("=" * 96)
    hdr = f"{'scenario':34s} {'skill':>11} | {'LFP':>13} | {'emp':>13} | {'earn %chg':>9} | {'child pen':>10} | {'welfare%':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, _ in SCENARIOS:
        o = sol[name]
        for sk in PAIRS:
            i = PAIRS[sk]; m = i['mother']
            lfp_s = f"{o['lfp'][m]:.3f}" + ("" if name == 'Baseline' else f" ({100*(o['lfp'][m]-base['lfp'][m]):+.1f}pp)")
            emp_s = f"{o['emp'][m]:.3f}" + ("" if name == 'Baseline' else f" ({100*(o['emp'][m]-base['emp'][m]):+.1f}pp)")
            earn_chg = 100. * (o['earn'][m] / base['earn'][m] - 1.)
            cp = cp_earn(o, sk)
            welf = 100. * (o['V'][m] / base['V'][m] - 1.) if base['V'][m] != 0 else float('nan')
            print(f"{name:34s} {sk:>11} | {lfp_s:>13} | {emp_s:>13} | "
                  f"{earn_chg:>+8.1f}% | {cp:>9.1f}% | {welf:>+7.1f}%")
            rows.append(dict(scenario=name, skill=sk, lfp=float(o['lfp'][m]), emp=float(o['emp'][m]),
                             earn_pct_chg=float(earn_chg), child_penalty_earn=float(cp),
                             welfare_pct_chg=float(welf)))
        if name != 'Baseline':
            print(f"{'':34s} {'(GPG '+format(o['gpg'],'.3f')+' vs '+format(base['gpg'],'.3f')+')':>60}")
    json.dump(dict(resolution=res, rows=rows,
                   note='GE policy counterfactuals; childcare subsidy = lower mother K_child; '
                        'parental leave = frictionless N->E; welfare = ex-ante entry value'),
              open(CLEAN / 'counterfactuals.json', 'w'), indent=2)
    print("\n[wrote data/clean/counterfactuals.json]\nDONE_CF")


if __name__ == '__main__':
    main()
