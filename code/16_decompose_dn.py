"""HC depreciation out of the labour force and the re-entry friction, PE vs GE."""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import CHECKPOINT, CLEAN, FIGURES

from model_extended import (run_extended_model, build_hc_grids,
                            solve_worker_shares_ext)
from smm_joint import make_base, vec_to_params, load_targets

FIGDIR = FIGURES
BLUE, ORANGE, INK, INK2, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'

PAIRS = {'Non-college': (1, 0), 'College': (3, 2)}   # (mother, father) type idx
DN_GRID = np.array([0.0, 0.005, 0.010, 0.013, 0.020, 0.030, 0.040, 0.050])
DELTA_U = 0.03
RES = 'medium'


def base_params(delta_n, mode, x, tgt):
    p = vec_to_params(x, make_base(RES))
    p.delta = np.clip(tgt['eu'].copy(), 1e-4, 0.3)
    p.delta_n = float(delta_n)
    p.delta_n_by_skill = (float(delta_n), float(delta_n))
    p.reentry_mode = mode
    return p


def earnings_from_ss(ss, wage_dollar_by_k):
    """Per-type per-capita earnings = (E_mat @ wage) / type_mass."""
    n = wage_dollar_by_k.shape[0]
    tm = ss['E'] + ss['U'] + ss['N']
    return np.array([(ss['E_mat'][i, :] @ wage_dollar_by_k[i, :]) / max(tm[i], 1e-12)
                     for i in range(n)])


def friction_cost(earn_base, earn_fric):
    """Friction earnings cost per skill = (mother_fric - mother_base)/father_base, %."""
    out = {}
    for sk, (mom, dad) in PAIRS.items():
        out[sk] = 100. * (earn_fric[mom] - earn_base[mom]) / max(earn_base[dad], 1e-12)
    return out


def run_ge(x, tgt):
    """GE: full re-solve at each delta_n."""
    cost = {sk: [] for sk in PAIRS}
    for dn in DN_GRID:
        b = run_extended_model(base_params(dn, 'baseline', x, tgt), verbose=False)
        f = run_extended_model(base_params(dn, 'frictionless', x, tgt), verbose=False)
        fc = friction_cost(b['earnings_pc'], f['earnings_pc'])
        for sk in PAIRS:
            cost[sk].append(fc[sk])
        print(f"  [GE] delta_n={dn:.3f}: NC {fc['Non-college']:5.1f}%  C {fc['College']:5.1f}%")
    return {sk: np.array(v) for sk, v in cost.items()}


def run_pe(x, tgt):
    """PE: calibrated cutoffs and wages frozen, only HC decay varies."""
    cal = run_extended_model(base_params(0.013, 'baseline', x, tgt), verbose=False)
    g, wage = cal['g'], cal['wage_dollar_by_k']
    ia, in_, iq, pvec = (cal['int_x_a_p'], cal['int_x_n_p'],
                         cal['int_x_q_p'], cal['p_vec_full'])   # frozen behaviour
    cost = {sk: [] for sk in PAIRS}
    for dn in DN_GRID:
        hc_dn = build_hc_grids(base_params(dn, 'baseline', x, tgt))
        pb = base_params(dn, 'baseline', x, tgt)
        pf = base_params(dn, 'frictionless', x, tgt)
        ss_b = solve_worker_shares_ext(pb, g, hc_dn, ia, in_, iq, pvec)
        ss_f = solve_worker_shares_ext(pf, g, hc_dn, ia, in_, iq, pvec)
        eb = earnings_from_ss(ss_b, wage)
        ef = earnings_from_ss(ss_f, wage)
        fc = friction_cost(eb, ef)
        for sk in PAIRS:
            cost[sk].append(fc[sk])
        print(f"  [PE] delta_n={dn:.3f}: NC {fc['Non-college']:5.1f}%  C {fc['College']:5.1f}%")
    return {sk: np.array(v) for sk, v in cost.items()}


def make_figure(pe, ge, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, (sk, col) in zip(axes, [('Non-college', ORANGE), ('College', BLUE)]):
        ax.fill_between(DN_GRID, pe[sk], ge[sk], color=col, alpha=0.10, zorder=1)
        ax.plot(DN_GRID, pe[sk], '--', color=col, lw=2, ms=5, marker='o',
                label='PE (behaviour fixed)', zorder=3)
        ax.plot(DN_GRID, ge[sk], '-', color=col, lw=2, ms=5, marker='s',
                label='GE (full model)', zorder=3)
        ax.axvline(DELTA_U, color=MUTED, ls=':', lw=1.0, zorder=1)
        ax.set_title(sk, color=INK, fontsize=12, loc='left')
        ax.set_xlabel(r'HC decay while non-participating, $\delta_n$')
        ax.legend(frameon=False, fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.6)
        ax.margins(x=0.02)
        ylo, yhi = ax.get_ylim()
        ax.text(DELTA_U + 0.0008, ylo + 0.06 * (yhi - ylo), r'$\delta_u$',
                color=INK2, fontsize=11)
    axes[0].set_ylabel(r'Re-entry friction earnings cost' + '\n' + r'(\% of father earnings)')
    fig.suptitle('Faster HC decay out of work: compounding (PE) vs. withdrawal-deterrence (GE)',
                 color=INK, fontsize=13, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _style(usetex):
    plt.rcParams.update({
        'text.usetex': usetex, 'font.family': 'serif', 'font.size': 12,
        'mathtext.fontset': 'cm', 'figure.dpi': 300, 'savefig.dpi': 300,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': MUTED,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GRID, 'grid.linewidth': 0.7, 'axes.axisbelow': True})


def main():
    tgt = load_targets(); x = np.load(CHECKPOINT)
    print(f"delta_n sweep at {RES} resolution ({len(DN_GRID)} points) ...")
    pe = run_pe(x, tgt)
    ge = run_ge(x, tgt)
    (CLEAN / 'decompose_dn.json').write_text(json.dumps(dict(
        delta_n=DN_GRID.tolist(), delta_u=DELTA_U,
        pe={sk: v.tolist() for sk, v in pe.items()},
        ge={sk: v.tolist() for sk, v in ge.items()},
        note='re-entry friction earnings cost (% of father earnings); PE=behaviour '
             'fixed (compounds), GE=full model (withdrawal deterrence dominates)'),
        indent=2))
    path = FIGDIR / 'dn_compounding.png'
    try:
        _style(True); make_figure(pe, ge, path)
    except Exception as e:
        print(f"  usetex failed ({type(e).__name__}); mathtext-CM fallback")
        plt.close('all'); _style(False); make_figure(pe, ge, path)
    for sk in PAIRS:
        print(f"  {sk}: PE {pe[sk][0]:.0f}%->{pe[sk][-1]:.0f}% (delta_n 0->0.05); "
              f"GE {ge[sk][0]:.0f}%->{ge[sk][-1]:.0f}%")
    print(f"Wrote {path} and data/clean/decompose_dn.json\nDONE_DN")


if __name__ == '__main__':
    main()
