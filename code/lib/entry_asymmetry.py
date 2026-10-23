"""Return to work of the non-employed (U or N -> E) and the N share of the non-employed, by sex and child under 5."""
from __future__ import annotations

import json
from pathlib import Path

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


def entry_rates(m: pd.DataFrame) -> dict:
    """Keys: return (monthly rate of non-employed -> E) and out_of_lf (N share of non-employed)."""
    out = {}
    for sx, sn in ((1, 'men'), (2, 'women')):
        for ch, cn in ((0, 'no_young_child'), (1, 'young_child')):
            s = m[(m['sex'] == sx) & (m['child'] == ch) & (m['lf'].isin(['U', 'N']))]
            ret, ret_se = _rate(s, (s['lf_n'] == 'E'))
            olf, olf_se = _rate(s, (s['lf'] == 'N'))
            out[f'{sn}_{cn}'] = {'return': ret, 'return_se': ret_se,
                                 'out_of_lf': olf, 'out_of_lf_se': olf_se, 'n': len(s)}
    return out


def make_figure(rt: dict, path: Path) -> None:
    fig, (axN, axU) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(2); wbar = 0.36
    order = ['men', 'women']

    def panel(ax, flow, title_txt, ymax, ylab):
        for i, (cn, col, lab) in enumerate([('no_young_child', MUTED, 'No young child'),
                                            ('young_child', BLUE, 'Young child ($<$5)')]):
            vals = [rt[f'{g}_{cn}'][flow] for g in order]
            ses = [1.96 * rt[f'{g}_{cn}'][flow + '_se'] for g in order]
            bars = ax.bar(x + (i - 0.5) * wbar, vals, wbar, color=col, label=lab,
                          yerr=ses, error_kw=dict(ecolor=INK2, lw=0.9, capsize=3))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.03, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(['Men', 'Women'])
        ax.set_ylim(0, ymax); ax.grid(True, axis='y', lw=0.6)
        ax.set_title(title_txt, fontsize=12, color=INK, pad=8)
        ax.set_ylabel(ylab)

    panel(axN, 'return', r'Return to work $(U\text{ or }N \to E)$', 0.24,
          'Monthly return-to-work rate')
    panel(axU, 'out_of_lf', r'Non-employment that is out-of-LF $(N\text{-share})$', 1.0,
          'Share of the non-employed')
    axN.legend(frameon=False, fontsize=10, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


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
    rt = entry_rates(m)
    (CLEAN / 'entry_asymmetry.json').write_text(json.dumps(rt, indent=2))
    path = FIGDIR / 'entry_asymmetry.png'
    try:
        _style(True); make_figure(rt, path)
    except Exception as e:
        print(f"  usetex failed ({type(e).__name__}); mathtext-CM fallback")
        plt.close('all'); _style(False); make_figure(rt, path)
    print("Return to work and non-employment composition (pooled skill):")
    for g in ('men_young_child', 'women_young_child', 'men_no_young_child', 'women_no_young_child'):
        print(f"  {g:>22}: return->E {rt[g]['return']:.3f}  out-of-LF share {rt[g]['out_of_lf']:.3f}"
              f"  (n={rt[g]['n']})")
    print(f"Wrote {path} and data/entry_asymmetry.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
