"""Monthly exit rates from employment (E -> N withdrawal, E -> U job loss) by sex and presence of a child under 5."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import CLEAN, CPS_FILE, FIGURES
from fact_rates import clustered_rate

DAT = CPS_FILE
FIGDIR = FIGURES
COLS = {'year': (0, 4), 'month': (9, 11), 'cpsidp': (53, 67), 'age': (82, 84),
        'sex': (84, 85), 'nchlt5': (89, 90), 'empstat': (95, 97),
        'educ': (98, 101), 'lnkfw1mwt': (111, 125)}
BLUE, ORANGE, INK, INK2, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'


def load_and_link() -> pd.DataFrame:
    df = pd.read_fwf(DAT, colspecs=list(COLS.values()), names=list(COLS.keys()))
    df['lnkfw1mwt'] = df['lnkfw1mwt'] / 10000.0
    df['ym'] = df['year'] * 12 + df['month']
    lf = pd.Series(pd.NA, index=df.index, dtype='object')
    lf[df['empstat'].isin([10, 12])] = 'E'
    lf[df['empstat'].isin([20, 21, 22])] = 'U'
    lf[df['empstat'].between(30, 36)] = 'N'
    df['lf'] = lf
    df['skill'] = (df['educ'] >= 111).astype(int)
    df['child'] = (df['nchlt5'] >= 1).astype(int)          # own child < 5
    df = df[(df['age'].between(25, 54)) & df['lf'].notna()].copy()
    a = df[['cpsidp', 'ym', 'lf', 'sex', 'skill', 'child', 'age', 'lnkfw1mwt']]
    b = df[['cpsidp', 'ym', 'lf', 'age']].rename(columns={'lf': 'lf_n', 'age': 'age_n'})
    b = b.assign(ym=b['ym'] - 1)
    m = a.merge(b, on=['cpsidp', 'ym'])
    return m[(m['age_n'] - m['age']).between(0, 1)]


def _rate(sub, cond):
    return clustered_rate(sub, cond)


def exit_rates(m: pd.DataFrame) -> dict:
    """E -> N and E -> U rates by sex and child under 5."""
    out = {}
    for sx, sn in ((1, 'men'), (2, 'women')):
        for ch, cn in ((0, 'no_young_child'), (1, 'young_child')):
            s = m[(m['sex'] == sx) & (m['child'] == ch) & (m['lf'] == 'E')]
            en, en_se = _rate(s, (s['lf_n'] == 'N'))
            eu, eu_se = _rate(s, (s['lf_n'] == 'U'))
            out[f'{sn}_{cn}'] = {'E_to_N': en, 'E_to_N_se': en_se,
                                 'E_to_U': eu, 'E_to_U_se': eu_se, 'n': len(s)}
    return out


def make_one(rt, flow, title_txt, ymax, outpath, legend_loc):
    """One bar panel for one flow."""
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    x = np.arange(2); wbar = 0.40; order = ['men', 'women']
    for i, (cn, col, lab) in enumerate([('no_young_child', MUTED, 'No young child'),
                                        ('young_child', BLUE, 'Young child ($<$5)')]):
        vals = [rt[f'{g}_{cn}'][flow] for g in order]
        errs = [1.96 * rt[f'{g}_{cn}'][flow + '_se'] for g in order]
        bars = ax.bar(x + (i - 0.5) * wbar, vals, wbar, color=col, label=lab,
                      yerr=errs, error_kw=dict(ecolor=INK2, lw=0.9, capsize=3))
        for b, v, e in zip(bars, vals, errs):          # label above the CI whisker
            ax.text(b.get_x() + b.get_width() / 2, v + e + ymax * 0.025, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(['Men', 'Women'])
    ax.set_ylim(0, ymax); ax.grid(True, axis='y', lw=0.6)
    ax.set_title(title_txt, fontsize=13, color=INK, pad=8)
    ax.set_ylabel('Monthly transition rate')
    ax.legend(frameon=False, fontsize=10, loc=legend_loc)
    fig.tight_layout(); fig.savefig(outpath, dpi=300, bbox_inches='tight'); plt.close(fig)


def _style(usetex: bool):
    plt.rcParams.update({
        'text.usetex': usetex, 'font.family': 'serif', 'font.size': 12,
        'mathtext.fontset': 'cm', 'figure.dpi': 300, 'savefig.dpi': 300,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': MUTED,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GRID, 'grid.linewidth': 0.7, 'axes.axisbelow': True})


def main() -> int:
    m = load_and_link()
    rt = exit_rates(m)
    (CLEAN / 'exit_asymmetry.json').write_text(json.dumps(rt, indent=2))
    YMAX = 0.040                                   # common scale for both flows
    figs = [('E_to_N', r'Withdraw to non-participation $(E\!\to\!N)$', FIGDIR / 'exit_EN.png', 'upper left'),
            ('E_to_U', r'Lose job to unemployment $(E\!\to\!U)$', FIGDIR / 'exit_EU.png', 'upper right')]
    for usetex in (True, False):
        try:
            _style(usetex)
            for flow, title, out, loc in figs:
                make_one(rt, flow, title, YMAX, out, loc)
            break
        except Exception as e:
            print(f"  usetex={usetex} failed ({type(e).__name__}); retrying"); plt.close('all')
    print("Monthly exit rates (pooled skill):")
    for g in ('men_young_child', 'women_young_child', 'men_no_young_child', 'women_no_young_child'):
        print(f"  {g:>22}: E->N {rt[g]['E_to_N']:.4f}  E->U {rt[g]['E_to_U']:.4f}")
    print(f"Wrote {FIGDIR}/exit_EN.png and exit_EU.png")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
