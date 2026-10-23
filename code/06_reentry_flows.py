"""Monthly hazards conditional on the source state: activation (N -> U | N), job finding (U -> E | U) and direct N -> E | N, by sex and child under 5."""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import CLEAN, FIGURES
import entry_asymmetry as ea
load_and_link, _rate = ea.load_and_link, ea._rate
BLUE, ORANGE, INK, INK2, MUTED, GRID = ea.BLUE, ea.ORANGE, ea.INK, ea.INK2, ea.MUTED, ea.GRID


def reentry_flows(m) -> dict:
    """Hazards by sex x young child; the denominator is person-months in the source state."""
    out = {}
    for sx, sn in ((1, 'men'), (2, 'women')):
        for ch, cn in ((0, 'no_young_child'), (1, 'young_child')):
            g = m[(m['sex'] == sx) & (m['child'] == ch)]
            in_N = g[g['lf'] == 'N']
            in_U = g[g['lf'] == 'U']
            nu, nu_se = _rate(in_N, in_N['lf_n'] == 'U')
            ne, ne_se = _rate(in_N, in_N['lf_n'] == 'E')
            ue, ue_se = _rate(in_U, in_U['lf_n'] == 'E')
            out[f'{sn}_{cn}'] = {'N_to_U': nu, 'N_to_U_se': nu_se,
                                 'U_to_E': ue, 'U_to_E_se': ue_se,
                                 'N_to_E': ne, 'N_to_E_se': ne_se,
                                 'n_N': len(in_N), 'n_U': len(in_U)}
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
    ax.set_ylabel('Monthly transition probability')
    ax.legend(frameon=False, fontsize=10, loc=legend_loc)
    fig.tight_layout(); fig.savefig(outpath, dpi=300, bbox_inches='tight'); plt.close(fig)


def main() -> int:
    m = load_and_link()
    rt = reentry_flows(m)
    (CLEAN / 'reentry_flows.json').write_text(json.dumps(rt, indent=2))
    def _ymax(flow):
        return 1.30 * max(rt[f'{g}_{c}'][flow] + 1.96 * rt[f'{g}_{c}'][flow + '_se']
                          for g in ('men', 'women') for c in ('no_young_child', 'young_child'))
    figs = [('N_to_U', r'Activation: become a searcher $(N\!\to\!U \mid N)$',
             FIGURES / 'reentry_NU.png', 'upper right'),
            ('U_to_E', r'Job finding $(U\!\to\!E \mid U)$',
             FIGURES / 'reentry_UE.png', 'upper right')]
    for usetex in (True, False):
        try:
            ea._style(usetex)
            for flow, title, out, loc in figs:
                make_one(rt, flow, title, _ymax(flow), out, loc)
            break
        except Exception as e:
            print(f"  usetex={usetex} failed ({type(e).__name__}); retrying"); plt.close('all')
    print("Source-conditional monthly hazards (pooled skill), SE clustered by person:")
    for g in ('men_young_child', 'women_young_child', 'men_no_young_child', 'women_no_young_child'):
        d = rt[g]
        print(f"  {g:>22}: N->U|N {d['N_to_U']:.4f}({d['N_to_U_se']:.4f})  "
              f"U->E|U {d['U_to_E']:.4f}({d['U_to_E_se']:.4f})  "
              f"[raw N->E|N {d['N_to_E']:.4f}({d['N_to_E_se']:.4f})]")
    print("Wrote output/figures/reentry_NU.png, reentry_UE.png and data/clean/reentry_flows.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
