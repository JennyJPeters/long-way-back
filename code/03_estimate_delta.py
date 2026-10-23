"""Estimate human-capital rates delta_e, delta_u, delta_n from the cleaned ASEC frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ASEC_CLEAN

CROSS = ASEC_CLEAN / 'asec_cross.csv'
PANEL = ASEC_CLEAN / 'asec_panel.csv'

WEEKS_PER_MONTH = 52.0 / 12.0
YNGCH_BINS = [0, 1, 3, 6, 12, 18, 99]
YNGCH_LABELS = ['0', '1-2', '3-5', '6-11', '12-17', '18+']


def _wls_sandwich(x: np.ndarray, y: np.ndarray, w: np.ndarray, fit_intercept=True
                  ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """WLS with robust covariance; returns (beta, cov, indices of kept columns)."""
    if fit_intercept:
        x = np.column_stack([np.ones(len(x)), x])
    # Drop collinear columns so the design stays invertible.
    _, r = np.linalg.qr(x)
    tol = 1e-9 * max(1.0, abs(r).max())
    kept = [i for i in range(x.shape[1]) if abs(r[i, i]) > tol]
    xk = x[:, kept]
    n, k = xk.shape
    xtwx = xk.T @ (xk * w[:, None])
    xtwx_inv = np.linalg.inv(xtwx)
    beta = xtwx_inv @ (xk.T @ (w * y))
    resid = y - xk @ beta
    scaled = xk * (w * resid)[:, None]
    meat = scaled.T @ scaled
    cov = (n / (n - k)) * (xtwx_inv @ meat @ xtwx_inv)
    return beta, cov, kept


def _combo(beta, cov, coef_by_pos: dict) -> tuple[float, float]:
    """Point estimate and robust SE of a linear combination of beta."""
    c = np.zeros(len(beta))
    for pos, coef in coef_by_pos.items():
        c[pos] = coef
    return float(c @ beta), float(np.sqrt(c @ cov @ c))


def _weekly_to_monthly_delta(slope_per_week: float) -> float:
    """Monthly delta from a weekly log slope; negative means growth."""
    return float(1.0 - np.exp(slope_per_week * WEEKS_PER_MONTH))


def _delta_with_se(slope: float, se_slope: float) -> dict:
    delta = _weekly_to_monthly_delta(slope)
    se = abs(WEEKS_PER_MONTH * (1.0 - delta)) * se_slope     # delta method
    return {'slope_per_week': slope, 'slope_se': se_slope,
            'delta_monthly': delta, 'delta_se': se}


def estimate_panel(panel: pd.DataFrame) -> dict:
    """Within-person wage-change regression on weeks in each state, by skill."""
    cols = ['weeks_employed', 'weeks_unemployed', 'weeks_out_of_lf']
    inter = [f'{c}_skill' for c in cols]
    x = pd.DataFrame({c: panel[c] for c in cols})
    for c in cols:
        x[f'{c}_skill'] = panel[c] * panel['skill']
    order = cols + inter
    x = x[order].to_numpy(dtype=float)
    y = panel['dln_real_wage'].to_numpy(dtype=float)
    w = panel['asecwt'].to_numpy(dtype=float)

    beta, cov, kept = _wls_sandwich(x, y, w, fit_intercept=False)
    pos = {order[c]: j for j, c in enumerate(kept)}   # name -> position in beta

    def rate(base: str, skilled: bool) -> dict:
        combo = {pos[base]: 1.0}
        inter_name = f'{base}_skill'
        if skilled and inter_name in pos:
            combo[pos[inter_name]] = 1.0
        slope, se = _combo(beta, cov, combo)
        return _delta_with_se(slope, se)

    return {
        'method': 'before/after panel, WLS (ASECWT) design-based robust SE, no intercept',
        'n_pairs': int(len(panel)),
        'noncollege': {
            'delta_e': rate('weeks_employed', False),
            'delta_u': rate('weeks_unemployed', False),
            'delta_n': rate('weeks_out_of_lf', False),
        },
        'college': {
            'delta_e': rate('weeks_employed', True),
            'delta_u': rate('weeks_unemployed', True),
            'delta_n': rate('weeks_out_of_lf', True),
        },
    }


def estimate_cross(cross: pd.DataFrame) -> dict:
    df = cross.copy()
    df['age2'] = df['age'] ** 2
    df['age3'] = df['age'] ** 3
    df['age4'] = df['age'] ** 4
    base = pd.DataFrame({
        'weeks_unemployed': df['weeks_unemployed'],
        'weeks_out_of_lf': df['weeks_out_of_lf'],
        'weeks_unemployed_skill': df['weeks_unemployed'] * df['skill'],
        'weeks_out_of_lf_skill': df['weeks_out_of_lf'] * df['skill'],
        'age': df['age'], 'age2': df['age2'], 'age3': df['age3'], 'age4': df['age4'],
        'skill': df['skill'], 'is_female': df['is_female'],
    })
    year_fe = pd.get_dummies(df['earn_year'], prefix='yr', drop_first=True).astype(float)
    design = pd.concat([base, year_fe], axis=1)
    order = list(design.columns)

    beta, cov, kept = _wls_sandwich(
        design.to_numpy(dtype=float),
        df['log_real_wage'].to_numpy(dtype=float),
        df['asecwt'].to_numpy(dtype=float),
        fit_intercept=True)
    # Position 0 is the intercept added inside _wls_sandwich.
    pos = {}
    for j, c in enumerate(kept):
        if c == 0:
            pos['const'] = j
        else:
            pos[order[c - 1]] = j

    def rate(base_name: str, skilled: bool) -> dict:
        combo = {pos[base_name]: 1.0}
        inter = f'{base_name}_skill'
        if skilled and inter in pos:
            combo[pos[inter]] = 1.0
        slope, se = _combo(beta, cov, combo)
        return _delta_with_se(slope, se)

    return {
        'method': 'cross-section log-wage level on last-year weeks (rel. to employed)',
        'n_obs': int(len(df)),
        'noncollege': {'delta_u': rate('weeks_unemployed', False),
                       'delta_n': rate('weeks_out_of_lf', False)},
        'college': {'delta_u': rate('weeks_unemployed', True),
                    'delta_n': rate('weeks_out_of_lf', True)},
    }


def recovery_profile(cross: pd.DataFrame) -> dict:
    women = cross[cross['is_female'] == 1].copy()
    mothers = women[women['yngch_valid'].notna()].copy()
    if mothers.empty:
        return {'note': 'no mothers with valid youngest-child age'}
    mothers['bin'] = pd.cut(mothers['yngch_valid'], bins=YNGCH_BINS,
                            right=False, labels=YNGCH_LABELS)
    prof = {}
    for lab, g in mothers.groupby('bin', observed=True):
        w = g['asecwt'].to_numpy(float)
        prof[str(lab)] = {'n': int(len(g)),
                          'mean_log_real_wage': float(np.average(
                              g['log_real_wage'].to_numpy(float), weights=w))}
    return prof


def _fmt(d: dict) -> str:
    return (f"slope {d['slope_per_week']:+.5f} (se {d['slope_se']:.5f})  "
            f"-> monthly delta {d['delta_monthly']:+.4f} (se {d['delta_se']:.4f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cross', type=Path, default=CROSS)
    ap.add_argument('--panel', type=Path, default=PANEL)
    ap.add_argument('--out', type=Path, default=ASEC_CLEAN / 'delta_estimates.json')
    args = ap.parse_args()
    for p in (args.cross, args.panel):
        if not p.exists():
            print(f"ERROR: {p} not found. Run clean_asec.py first.")
            return 1

    cross = pd.read_csv(args.cross)
    panel = pd.read_csv(args.panel)

    result = {
        'panel_delta_e_u_n': estimate_panel(panel),
        'cross_section_check_delta_u_n': estimate_cross(cross),
        'recovery_profile_by_youngest_child_age': recovery_profile(cross),
    }
    args.out.write_text(json.dumps(result, indent=2))

    pan = result['panel_delta_e_u_n']
    print("=" * 78)
    print(f"PANEL (delta_e, delta_u, delta_n)  n={pan['n_pairs']:,} linked pairs")
    print("  [negative monthly delta = HC GROWS; positive = HC depreciates]")
    for grp in ('noncollege', 'college'):
        print(f"  {grp}:")
        for name in ('delta_e', 'delta_u', 'delta_n'):
            print(f"    {name}: {_fmt(pan[grp][name])}")
    cx = result['cross_section_check_delta_u_n']
    print("-" * 78)
    print(f"CROSS-SECTION check (delta_u, delta_n rel. to employed)  n={cx['n_obs']:,}")
    for grp in ('noncollege', 'college'):
        print(f"  {grp}:")
        for name in ('delta_u', 'delta_n'):
            print(f"    {name}: {_fmt(cx[grp][name])}")
    print("-" * 78)
    print("Recovery profile (working mothers, mean log real wage by youngest-child age):")
    for lab, cell in result['recovery_profile_by_youngest_child_age'].items():
        if isinstance(cell, dict) and 'mean_log_real_wage' in cell:
            print(f"    youngest {lab:>5}: {cell['mean_log_real_wage']:+.3f}  (n={cell['n']})")
    print("=" * 78)
    print(f"Wrote {args.out}")
    print("NOTE: reduced-form rates ignore return-selection; the STRUCTURAL deltas "
          "come from making the model reproduce these (indirect inference).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
