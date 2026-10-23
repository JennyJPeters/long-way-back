"""LFP (left) and monthly E -> N withdrawal among the employed (right) by age of the youngest child, mothers vs fathers, with childless baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import CPS_FILE, FIGURES
from fact_rates import clustered_rate

DAT = CPS_FILE
COLS = {'year': (0, 4), 'month': (9, 11), 'cpsidp': (53, 67), 'age': (82, 84),
        'sex': (84, 85), 'nchild': (88, 89), 'nchlt5': (89, 90), 'yngch': (90, 92),
        'empstat': (95, 97), 'educ': (98, 101), 'lnkfw1mwt': (111, 125)}
BLUE, ORANGE, INK, INK2, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
MAROON = '#b23a48'   # mothers
AGES = list(range(0, 13))


def load():
    df = pd.read_fwf(DAT, colspecs=list(COLS.values()), names=list(COLS.keys()))
    df['lnkfw1mwt'] = df['lnkfw1mwt'] / 10000.0
    df['ym'] = df['year'] * 12 + df['month']
    lf = pd.Series(pd.NA, index=df.index, dtype='object')
    lf[df['empstat'].isin([10, 12])] = 'E'
    lf[df['empstat'].isin([20, 21, 22])] = 'U'
    lf[df['empstat'].between(30, 36)] = 'N'
    df['lf'] = lf
    df = df[(df['age'].between(25, 54)) & df['lf'].notna()].copy()
    a = df[['cpsidp', 'ym', 'lf', 'sex', 'nchild', 'yngch', 'lnkfw1mwt']]
    b = df[['cpsidp', 'ym', 'lf']].rename(columns={'lf': 'lf_n'}).assign(ym=lambda d: d['ym'] - 1)
    return a.merge(b, on=['cpsidp', 'ym'])


def _wmean(sub, cond):
    return clustered_rate(sub, cond)


def series(m, sx, kind):
    """Kind is 'lfp' (share in E or U) or 'en' (E -> N hazard among the employed)."""
    ys, ses = [], []
    for age in AGES:
        s = m[(m['sex'] == sx) & (m['yngch'] == age)]
        if kind == 'en':
            s = s[s['lf'] == 'E']
            p, se = _wmean(s, (s['lf_n'] == 'N'))
        else:
            p, se = _wmean(s, s['lf'].isin(['E', 'U']))
        ys.append(p); ses.append(se)
    return np.array(ys), np.array(ses)


def baseline(m, sx, kind):
    s = m[(m['sex'] == sx) & (m['nchild'] == 0)]      # childless
    if kind == 'en':
        s = s[s['lf'] == 'E']
        return _wmean(s, (s['lf_n'] == 'N'))[0]
    return _wmean(s, s['lf'].isin(['E', 'U']))[0]


def main():
    m = load()
    for usetex in (True, False):
        try:
            _style(usetex)
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
            for ax, kind, ylab, title in (
                    (axL, 'lfp', 'Labour-force participation', 'Participation by age of youngest child'),
                    (axR, 'en', r'Monthly $E\!\to\!N$ (among employed)', 'Withdrawal by age of youngest child')):
                for sx, col, lab in ((2, MAROON, 'Mothers'), (1, BLUE, 'Fathers')):
                    y, se = series(m, sx, kind)
                    ax.plot(AGES, y, '-o', color=col, lw=2, ms=5, label=lab, zorder=3)
                    ax.fill_between(AGES, y - 1.96 * se, y + 1.96 * se, color=col, alpha=0.12, zorder=1)
                    b = baseline(m, sx, kind)
                    ax.axhline(b, color=col, ls=(0, (4, 3)), lw=1.1, alpha=0.7, zorder=2)
                ax.axvline(5, color=MUTED, ls=':', lw=1.0)
                ax.set_xlabel('Age of youngest child'); ax.set_ylabel(ylab)
                ax.set_title(title, fontsize=12, color=INK); ax.grid(True, lw=0.6)
                ax.margins(x=0.02)
            axL.text(5.15, axL.get_ylim()[0] + 0.02, 'school entry', color=MUTED, fontsize=8.5, rotation=90, va='bottom')
            axL.legend(frameon=False, fontsize=10, loc='lower right')
            axL.text(0.02, 0.06, 'dashed = childless baseline', transform=axL.transAxes,
                     color=MUTED, fontsize=8.5)
            fig.tight_layout()
            fig.savefig(FIGURES / 'child_age_profile.png', dpi=300, bbox_inches='tight')
            plt.close(fig); break
        except Exception as e:
            print(f"  usetex={usetex} failed ({type(e).__name__})"); plt.close('all')
    print("Wrote output/figures/child_age_profile.png")
    for sx, nm in ((2, 'mothers'), (1, 'fathers')):
        y, _ = series(m, sx, 'lfp')
        print(f"  {nm} LFP by child age 0..12: {np.round(y, 3).tolist()}  (childless {baseline(m,sx,'lfp'):.3f})")


def _style(usetex):
    plt.rcParams.update({
        'text.usetex': usetex, 'font.family': 'serif', 'font.size': 12,
        'mathtext.fontset': 'cm', 'figure.dpi': 300, 'savefig.dpi': 300,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': MUTED,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GRID, 'grid.linewidth': 0.7, 'axes.axisbelow': True})


if __name__ == '__main__':
    main()
