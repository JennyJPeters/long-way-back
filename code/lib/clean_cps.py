"""Clean the IPUMS-CPS Basic Monthly extract and build the calibration moments."""
import json
import numpy as np
import pandas as pd

from paths import CLEAN, CPS_FILE, MOMENTS

DATA_FILE  = str(CPS_FILE)
OUT_JSON   = str(MOMENTS)
OUT_MD     = str(CLEAN / 'moments_summary.md')

COLSPECS = {
    'year': (0, 4), 'serial': (4, 9), 'month': (9, 11), 'hwtfinl': (11, 21),
    'cpsid': (21, 35), 'asecflag': (35, 36), 'mish': (36, 37), 'pernum': (37, 39),
    'wtfinl': (39, 53), 'cpsidp': (53, 67), 'cpsidv': (67, 82), 'age': (82, 84),
    'sex': (84, 85), 'race': (85, 88),
    'nchild': (88, 89), 'nchlt5': (89, 90), 'yngch': (90, 92),
    'hispan': (92, 95), 'empstat': (95, 97), 'labforce': (97, 98),
    'educ': (98, 101), 'earnwt': (101, 111), 'lnkfw1mwt': (111, 125),
    'paidhour': (125, 126), 'earnweek': (126, 134), 'uhrsworkorg': (134, 137),
}
GROUP_LABELS  = ['UnM', 'UnF', 'SkM', 'SkF']
# '-C' = own child under 5 present, '-0' = no young child
TYPE8_LABELS  = ['UnM-C', 'UnF-C', 'SkM-C', 'SkF-C',
                 'UnM-0', 'UnF-0', 'SkM-0', 'SkF-0']

def load_cps(path: str = DATA_FILE) -> pd.DataFrame:
    """Return the prime-age civilian person-month frame with group codes added."""
    names = list(COLSPECS)
    specs = list(COLSPECS.values())
    print(f"  reading {path} ...", flush=True)
    df = pd.read_fwf(path, colspecs=specs, names=names)
    print(f"  read {len(df):,} rows", flush=True)

    # IPUMS stores weights with 4 implied decimals and earnings with 2.
    for col in ('wtfinl', 'earnwt', 'lnkfw1mwt', 'hwtfinl'):
        df[col] = df[col] / 10000.0
    df['earnweek'] = df['earnweek'] / 100.0

    df['ym'] = df['year'] * 12 + df['month']

    # E = employed, U = unemployed, N = not in the labour force.
    lf = pd.Series(pd.NA, index=df.index, dtype='object')
    lf[df['empstat'].isin([10, 12])] = 'E'
    lf[df['empstat'].isin([20, 21, 22])] = 'U'
    lf[df['empstat'].between(30, 36)] = 'N'
    df['lf_state'] = lf

    df['is_female'] = (df['sex'] == 2).astype(int)
    df['skill'] = (df['educ'] >= 111).astype(int)
    df['has_child'] = (df['nchlt5'] >= 1).astype(int)
    df['fertile_age'] = df['age'].between(25, 44).astype(int)
    df['group4'] = 2 * df['skill'] + df['is_female']
    # type8 = 4 * no_child + group4: codes 0-3 child-present, 4-7 no young child.
    no_child = (df['has_child'] == 0).astype(int)
    df['type8'] = 4 * no_child + df['group4']

    keep = (df['age'].between(25, 54)
            & df['sex'].isin([1, 2])
            & df['lf_state'].notna())        # drops armed forces and NIU
    df = df[keep].copy()
    print(f"  prime-age civilian person-months: {len(df):,}", flush=True)
    return df


def _wsum(df, mask):
    return float(df.loc[mask, 'wtfinl'].sum())

def compute_stocks(df: pd.DataFrame, by: str):
    """Weighted LFP, unemployment and sample size by group; `by` is group4 or type8."""
    labels = GROUP_LABELS if by == 'group4' else TYPE8_LABELS
    n = len(labels)
    lfp, unemp, nobs = {}, {}, {}
    for g in range(n):
        sub = df[df[by] == g]
        E = _wsum(sub, sub['lf_state'] == 'E')
        U = _wsum(sub, sub['lf_state'] == 'U')
        N = _wsum(sub, sub['lf_state'] == 'N')
        lab = labels[g]
        lfp[lab]   = (E + U) / (E + U + N) if (E + U + N) > 0 else float('nan')
        unemp[lab] = U / (E + U) if (E + U) > 0 else float('nan')
        nobs[lab]  = int(len(sub))
    return lfp, unemp, nobs


def compute_flows(df: pd.DataFrame, by: str):
    """Monthly UE rate, 3x3 transition matrix and link count by source-month group."""
    cols = ['cpsidp', 'ym', 'lf_state', 'sex', 'age', by,
            'lnkfw1mwt', 'wtfinl']
    a = df[cols].copy()
    b = df[cols].copy()
    b['ym_prev'] = b['ym'] - 1            # b is the next-month record
    merged = a.merge(b, left_on=['cpsidp', 'ym'],
                     right_on=['cpsidp', 'ym_prev'],
                     suffixes=('_t', '_t1'))
    valid = ((merged['sex_t'] == merged['sex_t1'])
             & (merged['age_t1'] - merged['age_t']).between(0, 1))
    merged = merged[valid].copy()

    # Link weight, or the first-month final weight where the link weight is missing or zero.
    w = merged['lnkfw1mwt_t1'].where(merged['lnkfw1mwt_t1'] > 0,
                                     merged['wtfinl_t'])
    merged['w'] = w

    labels = GROUP_LABELS if by == 'group4' else TYPE8_LABELS
    grp_t  = by + '_t'
    ue, trans, nlink = {}, {}, {}
    for g in range(len(labels)):
        lab = labels[g]
        sub = merged[merged[grp_t] == g]
        nlink[lab] = int(len(sub))
        src_u = sub[sub['lf_state_t'] == 'U']
        wu = src_u['w'].sum()
        ue[lab] = float(src_u.loc[src_u['lf_state_t1'] == 'E', 'w'].sum() / wu) \
                  if wu > 0 else float('nan')
        # Rows sum to 1.
        tm = {}
        for s in ('E', 'U', 'N'):
            src = sub[sub['lf_state_t'] == s]
            ws = src['w'].sum()
            for d in ('E', 'U', 'N'):
                tm[f'{s}{d}'] = float(src.loc[src['lf_state_t1'] == d, 'w'].sum() / ws) \
                                if ws > 0 else float('nan')
        trans[lab] = tm
    return ue, trans, nlink


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * w.sum()
    return float(v[np.searchsorted(cw, cutoff)])

def compute_gpg(df: pd.DataFrame):
    """Weighted median full-time weekly earnings by sex x skill and the gender pay gap."""
    e = df[(df['lf_state'] == 'E')
           & (df['earnweek'] > 0) & (df['earnweek'] < 9999.99)
           & (df['uhrsworkorg'].between(35, 168))
           & (df['earnwt'] > 0)].copy()

    def med(sub):
        return _weighted_median(sub['earnweek'].to_numpy(),
                                sub['earnwt'].to_numpy()) if len(sub) else float('nan')

    med_m = med(e[e['is_female'] == 0])
    med_f = med(e[e['is_female'] == 1])
    gpg = {
        'overall_women_over_men': med_f / med_m if med_m else float('nan'),
        'overall_model_men_over_women': med_m / med_f if med_f else float('nan'),
        'median_weekly_male': med_m, 'median_weekly_female': med_f,
        'n_obs': int(len(e)),
    }
    for sk, name in [(0, 'unskilled'), (1, 'skilled')]:
        mm = med(e[(e['is_female'] == 0) & (e['skill'] == sk)])
        mf = med(e[(e['is_female'] == 1) & (e['skill'] == sk)])
        gpg[f'{name}_women_over_men'] = mf / mm if mm else float('nan')
        gpg[f'{name}_model_men_over_women'] = mm / mf if mf else float('nan')
    return gpg


def compute_shares(df: pd.DataFrame):
    """Population shares of the 8 types and child-present shares by group."""
    tot = df['wtfinl'].sum()
    pop8 = {TYPE8_LABELS[g]: float(df.loc[df['type8'] == g, 'wtfinl'].sum() / tot)
            for g in range(8)}
    child = {}
    for g in range(4):
        sub = df[df['group4'] == g]
        wc = sub.loc[sub['has_child'] == 1, 'wtfinl'].sum()
        wt = sub['wtfinl'].sum()
        child[GROUP_LABELS[g]] = float(wc / wt) if wt > 0 else float('nan')
    child['overall'] = float(df.loc[df['has_child'] == 1, 'wtfinl'].sum() / tot)
    return pop8, child


def main():
    print("Cleaning IPUMS-CPS 2015-2019 and building moments ...")
    df = load_cps()

    lfp4, unemp4, nobs4   = compute_stocks(df, 'group4')
    lfp8, unemp8, nobs8   = compute_stocks(df, 'type8')
    ue4, trans4, nlink4   = compute_flows(df, 'group4')
    ue8, trans8, nlink8   = compute_flows(df, 'type8')
    gpg                   = compute_gpg(df)
    pop8, child           = compute_shares(df)

    years = sorted(df['year'].unique().tolist())
    moments = {
        'meta': {
            'source': 'IPUMS-CPS Basic Monthly, data/cps_00012.dat',
            'years': years, 'age_range': [25, 54],
            'skill_def': "college = educ >= 111 (bachelor's or higher)",
            'childcare_def': 'child-present = own child under 5 (nchlt5>=1); '
                             'else no-child. block 0-3=child-present, 4-7=no-child',
            'n_person_months': int(len(df)),
            'data_moments_are': 'RAW (not misclassification-corrected); '
                                'model simulates misclassification, see the module docstring',
            'flow_weight': 'lnkfw1mwt (fallback wtfinl)',
        },
        'lfp_4': lfp4, 'unemp_4': unemp4, 'ue_4': ue4,
        'lfp_8': lfp8, 'unemp_8': unemp8, 'ue_8': ue8,
        'transitions_4': trans4, 'transitions_8': trans8,
        'gpg': gpg, 'pop_shares_8': pop8, 'child_shares': child,
        'n_obs': {'stocks_4': nobs4, 'stocks_8': nobs8,
                  'links_4': nlink4, 'links_8': nlink8},
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(moments, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    _write_summary(moments)
    print(f"Wrote {OUT_MD}")
    return moments


def _write_summary(m: dict):
    L = GROUP_LABELS
    def row(name, d, keys, fmt='{:.3f}'):
        return f"| {name} | " + " | ".join(fmt.format(d[k]) for k in keys) + " |"
    lines = []
    lines.append("# CPS Calibration Moments, 2015-2019 (pre-COVID)\n")
    lines.append(f"Source: `{m['meta']['source']}`  ")
    lines.append(f"Years: {m['meta']['years']}, ages 25-54, "
                 f"{m['meta']['n_person_months']:,} person-months  ")
    lines.append(f"Data moments are **{m['meta']['data_moments_are']}**\n")

    lines.append("## 4-group targets\n")
    lines.append("| Moment | UnM | UnF | SkM | SkF |")
    lines.append("|--------|-----|-----|-----|-----|")
    lines.append(row('LFP',   m['lfp_4'],   L))
    lines.append(row('Unemp', m['unemp_4'], L))
    lines.append(row('UE',    m['ue_4'],    L))
    lines.append(row('n (stocks)', m['n_obs']['stocks_4'], L, '{:,}'))
    lines.append(row('n (links)',  m['n_obs']['links_4'],  L, '{:,}'))

    lines.append("\n## 8-type targets (child-present -C / no-child -0)\n")
    lines.append("| Moment | " + " | ".join(TYPE8_LABELS) + " |")
    lines.append("|" + "---|" * (len(TYPE8_LABELS) + 1))
    for nm, key in [('LFP', 'lfp_8'), ('Unemp', 'unemp_8'), ('UE', 'ue_8')]:
        lines.append("| " + nm + " | "
                     + " | ".join(f"{m[key][t]:.3f}" for t in TYPE8_LABELS) + " |")

    g = m['gpg']
    lines.append("\n## Gender wage gap (median FT weekly earnings)\n")
    lines.append(f"- Overall: women/men = **{g['overall_women_over_men']:.3f}** "
                 f"(model men/women = {g['overall_model_men_over_women']:.3f}); "
                 f"median M=${g['median_weekly_male']:.0f}, F=${g['median_weekly_female']:.0f}")
    lines.append(f"- Unskilled: women/men = {g['unskilled_women_over_men']:.3f} "
                 f"(model {g['unskilled_model_men_over_women']:.3f})")
    lines.append(f"- Skilled: women/men = {g['skilled_women_over_men']:.3f} "
                 f"(model {g['skilled_model_men_over_women']:.3f})")
    lines.append(f"- n = {g['n_obs']:,} full-time employed earners")

    lines.append("\n## Population & child shares\n")
    lines.append("| Type | " + " | ".join(TYPE8_LABELS) + " |")
    lines.append("|" + "---|" * (len(TYPE8_LABELS) + 1))
    lines.append("| Pop share | "
                 + " | ".join(f"{m['pop_shares_8'][t]:.3f}" for t in TYPE8_LABELS) + " |")
    lines.append("\nChild-present (own child <5) share within group: "
                 + ", ".join(f"{k}={v:.3f}" for k, v in m['child_shares'].items()))
    lines.append("\n**Child penalty (LFP, women): no-child minus child-present**  ")
    unf = m['lfp_8']['UnF-0'] - m['lfp_8']['UnF-C']
    skf = m['lfp_8']['SkF-0'] - m['lfp_8']['SkF-C']
    unm = m['lfp_8']['UnM-0'] - m['lfp_8']['UnM-C']
    skm = m['lfp_8']['SkM-0'] - m['lfp_8']['SkM-C']
    lines.append(f"- Unskilled women: {unf:+.3f}   Skilled women: {skf:+.3f}  ")
    lines.append(f"- (men, for contrast) Unskilled: {unm:+.3f}   Skilled: {skm:+.3f}")
    with open(OUT_MD, 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
