"""Targets, base parameters, parameter mapping and objective for the SMM."""
import json
import os
import time
import argparse
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

from paths import CLEAN, MOMENTS, MOMENT_SE
from model_extended import ExtParameters, run_extended_model

DATA_ORDER = ['UnM-C', 'UnF-C', 'SkM-C', 'SkF-C', 'UnM-0', 'UnF-0', 'SkM-0', 'SkF-0']
_CKPT = str(CLEAN / 'smm_joint_checkpoint.npy')
_SE_FILE = str(MOMENT_SE)
_KAPPA_LEVEL = 0.03    # LFP levels and GPG
_KAPPA_FLOW  = 0.30    # NU labour-force-entry flows


def load_targets(path=MOMENTS):
    m = json.load(open(path))
    lfp = np.array([m['lfp_8'][k] for k in DATA_ORDER])
    nu  = np.array([m['transitions_8'][k]['NU'] for k in DATA_ORDER])  # targeted (-> lambda)
    gpg = m['gpg']['overall_model_men_over_women']
    tgt = dict(
        lfp=lfp, nu=nu, gpg=gpg,
        unemp=np.array([m['unemp_8'][k] for k in DATA_ORDER]),
        ue=np.array([m['ue_8'][k] for k in DATA_ORDER]),            # untargeted (-> external mu)
        en=np.array([m['transitions_8'][k]['EN'] for k in DATA_ORDER]),
        eu=np.array([m['transitions_8'][k]['EU'] for k in DATA_ORDER]),  # -> external delta
        child_share=m['child_shares']['overall'],
    )
    # Group UE averages the child and no-child types (indices g and g + 4).
    ue_grp = np.array([0.5 * (tgt['ue'][g] + tgt['ue'][g + 4]) for g in range(4)])
    tgt['ue_grp'] = ue_grp
    tgt['data_vec'] = np.concatenate([lfp, nu, tgt['en'], tgt['eu'], ue_grp, [gpg]])
    if not os.path.exists(_SE_FILE):
        raise FileNotFoundError(
            f"{_SE_FILE} not found; run 04_bootstrap_moments.py first "
            f"(the weighting needs the bootstrap standard errors).")
    se = np.array(json.load(open(_SE_FILE))['se'])
    if len(se) != len(tgt['data_vec']):
        raise ValueError(
            f"moment_se.json has {len(se)} SEs, expected {len(tgt['data_vec'])}; "
            f"re-run 04_bootstrap_moments.py (it produces all 37).")
    # Model-error tolerance per moment, in data_vec order (flows use the larger kappa).
    kappa = np.concatenate([np.full(8, _KAPPA_LEVEL),
                            np.full(24, _KAPPA_FLOW),
                            np.full(4, _KAPPA_LEVEL),
                            [_KAPPA_LEVEL]])
    tgt['scale'] = np.sqrt(se ** 2 + (kappa * tgt['data_vec']) ** 2)
    return tgt


def make_base(resolution='medium', theta_cap=40):
    """Base parameters for a grid resolution; theta_cap bounds the outer tightness loop."""
    p = ExtParameters()
    p.n_k = 11
    p.hc_floor_k = 0.58
    p.alpha = 0.5; p.gamma = 0.5           # Hosios split
    _KAPPA_CC = 15.0
    p.cc = _KAPPA_CC * p.y_hl.copy()
    p.max_theta_iter = theta_cap
    p.tau_w_f = 1.0
    p.mu_vec = np.zeros(4)
    p.g_emp   = 0.0018192392022343
    p.g_emp_by_skill = (0.0018192392022343, 0.0030735260404167)
    p.delta_u = 0.0263
    p.delta_n = 0.0143
    p.delta_u_by_skill = (0.0263, 0.0379)
    p.delta_n_by_skill = (0.0143, 0.0132)
    p.w_min   = 0.60
    if resolution == 'fast':
        p.n_w, p.n_x, p.vfi_max_iter = 12, 90, 500
    elif resolution == 'medium':
        p.n_w, p.n_x, p.vfi_max_iter = 30, 200, 1500
    elif resolution == 'full':
        p.n_w, p.n_x, p.vfi_max_iter = 100, 600, 2000
    else:
        raise ValueError(resolution)
    return p


def setup_external(p_base, tgt, verbose=True):
    """Fix delta at the observed EU rates."""
    p_base = deepcopy(p_base)
    p_base.delta = np.clip(tgt['eu'].copy(), 1e-4, 0.3)
    if verbose:
        print("  external: delta = observed EU rates (mu_vec is estimated in the SMM)")
    return p_base


_KMAX = 12.0

def vec_to_params(x, p_base):
    p = deepcopy(p_base)
    # lambda differs by child status: one value per type.
    lam_c = np.clip(np.exp(x[0:4]), 1e-5, 0.4)             # child types  (UnM_c,UnF_c,SkM_c,SkF_c)
    lam_n = np.clip(np.exp(x[4:8]), 1e-5, 0.4)             # no-child types(UnM_0,UnF_0,SkM_0,SkF_0)
    p.lambda_grid = np.concatenate([lam_c, lam_n])
    Km  = np.clip(np.exp(x[8:10]),  0.01, _KMAX)            # UnM, SkM
    Kfc = np.clip(np.exp(x[10:12]), 0.01, _KMAX)            # UnF-c, SkF-c
    Kfn = np.clip(np.exp(x[12:14]), 0.01, _KMAX)            # UnF-0, SkF-0
    p.K = np.array([Km[0], Kfc[0], Km[1], Kfc[1], Km[0], Kfn[0], Km[1], Kfn[1]])
    mu_raw   = np.clip(np.exp(x[14:18]), 0.01, 3.0)         # [UnM, (UnF), SkM, (SkF)]
    p.mu_vec = np.array([mu_raw[0], mu_raw[0], mu_raw[2], mu_raw[2]])  # tie F->M by skill
    p.mu = float(p.mu_vec[0])
    p.phi_c = 0.0
    return p


def warm_start():
    return np.array([
        # lambda x8: same values for child and no-child types (log scale).
        np.log(0.009), np.log(0.026), np.log(0.018), np.log(0.004),  # child
        np.log(0.009), np.log(0.026), np.log(0.018), np.log(0.004),  # no-child
        np.log(0.9), np.log(1.5),                                    # K_male
        np.log(1.6), np.log(1.9),                                    # K_fem_child
        np.log(1.3), np.log(1.6),                                    # K_fem_nochild
        np.log(0.402), np.log(0.414), np.log(0.095), np.log(0.109),
    ])


# (slot, types it moves).
_LAM_MAP = [(i, [i]) for i in range(8)]
_K_MAP   = [(8, [0, 4]), (9, [2, 6]), (10, [1]), (11, [3]), (12, [5]), (13, [7])]


def smart_warm_start(p_base, tgt, passes=2, biters=9):
    """Start point: bisect each K to its LFP target, then each lambda to its NU target."""
    pf = deepcopy(p_base)
    pf.n_w, pf.n_x, pf.vfi_max_iter = 12, 90, 500
    x = warm_start().copy()

    def lfp_nu(xv):
        r = run_extended_model(vec_to_params(xv, pf), verbose=False)
        return r['model_lfp'], r['NU_hat']

    for pnum in range(passes):
        for xi, types in _K_MAP:                        # K: LFP decreasing in K
            target = float(np.mean([tgt['lfp'][t] for t in types]))
            lo, hi = np.log(0.05), np.log(12.0)
            for _ in range(biters):
                x[xi] = 0.5 * (lo + hi)
                m = float(np.mean([lfp_nu(x)[0][t] for t in types]))
                if m > target: lo = x[xi]              # too much participation -> raise K
                else:          hi = x[xi]
            x[xi] = 0.5 * (lo + hi)
        for xi, types in _LAM_MAP:                      # lambda: NU increasing in lambda
            target = float(np.mean([tgt['nu'][t] for t in types]))
            lo, hi = np.log(1e-4), np.log(0.3)
            for _ in range(biters):
                x[xi] = 0.5 * (lo + hi)
                m = float(np.mean([lfp_nu(x)[1][t] for t in types]))
                if m < target: lo = x[xi]              # too little entry -> raise lambda
                else:          hi = x[xi]
            x[xi] = 0.5 * (lo + hi)
        print(f"    smart-init pass {pnum+1}/{passes} done", flush=True)
    return x


def confirm_full(x_opt, tgt, p_search, polish_iters=3):
    """Full-grid check of the optimum with a short Powell polish; returns (params, x, result)."""
    print("\n  === FULL-RESOLUTION CONFIRMATION (n_w=100, n_x=600) ===", flush=True)
    pf = deepcopy(p_search)
    pf.n_w, pf.n_x, pf.vfi_max_iter = 100, 600, 2000
    _best['val'] = np.inf; _best['x'] = x_opt.copy(); _best['n'] = 0
    if polish_iters > 0:
        minimize(objective, x_opt, args=(pf, tgt), method='Powell',
                 options=dict(maxiter=polish_iters, xtol=1e-3, ftol=1e-4, disp=True))
    x_full = _best['x'] if _best['x'] is not None else x_opt
    r = run_extended_model(vec_to_params(x_full, pf), verbose=False)
    return pf, x_full, r


_best = {'val': np.inf, 'x': None, 'n': 0}


def objective(x, p_base, tgt):
    _best['n'] += 1
    try:
        p = vec_to_params(x, p_base)
        r = run_extended_model(p, verbose=False)
        # Each moment error is divided by tgt['scale'] (built in load_targets).
        ue_g = np.array([0.5 * (r['UE_hat'][g] + r['UE_hat'][g + 4]) for g in range(4)])
        model_vec = np.concatenate([r['model_lfp'], r['NU_hat'], r['EN_hat'],
                                    r['EU_hat'], ue_g, [float(r['gender_pay_gap'])]])
        errs = (model_vec - tgt['data_vec']) / tgt['scale']
        val = float(np.sqrt(np.mean(errs ** 2)))
        mu_eff = np.array([(p.mu_vec[i % 4] if (len(p.mu_vec) > i % 4 and p.mu_vec[i % 4] > 0.) else p.mu)
                           for i in range(8)])
        th = np.asarray(r['theta'], float)
        # Penalty when the unclipped job-finding probability mu * theta^(1 - alpha) exceeds 1.
        p_unclip = mu_eff * th ** (1. - p.alpha)
        val = val + 2.0 * float(np.sum(np.maximum(0., p_unclip - 1.0) ** 2))
        if not np.isfinite(val):
            val = 10.0
    except Exception:
        val = 10.0
    if val < _best['val']:
        _best['val'] = val; _best['x'] = x.copy(); np.save(_CKPT, x)
        try:
            p = vec_to_params(x, p_base); r = run_extended_model(p, verbose=False)
            print(f"  eval {_best['n']:5d}  RMSE={val:.4f}  LFP={r['model_lfp'].round(2)}  "
                  f"GPG={r['gender_pay_gap']:.3f}  phi_c={p.phi_c:.3f} "
                  f"du={p.delta_u_by_skill}", flush=True)
        except Exception:
            print(f"  eval {_best['n']:5d}  RMSE={val:.4f}", flush=True)
    return val


def run(resolution='medium', maxiter=1200):
    p0 = make_base(resolution)
    tgt = load_targets()
    print(f"Two-tier SMM | resolution={resolution} n_w={p0.n_w} n_x={p0.n_x} vfi={p0.vfi_max_iter}")
    p_base = setup_external(p0, tgt)
    print(f"  18 free params (lambda x8, K x6, mu_vec x4 jointly estimated); HC rates "
          f"fixed from ASEC; targets LFP8 + NU8 + EN8 + EU8 + UE_group4 + GPG (37 moments)")

    if os.path.exists(_CKPT):
        x0 = np.load(_CKPT)
        print(f"  warm-started from {_CKPT}")
    else:
        print("  smart initialisation (solving each K->LFP, lambda->NU on a fast grid) ...")
        x0 = smart_warm_start(p_base, tgt)
    _best['val'] = np.inf; _best['x'] = x0.copy(); _best['n'] = 0

    t0 = time.time()
    minimize(objective, x0, args=(p_base, tgt), method='Nelder-Mead',
             options=dict(maxiter=maxiter, adaptive=True, xatol=1e-3, fatol=1e-4, disp=True))
    x_best = _best['x']
    print("\n  Powell polish ...")
    minimize(objective, x_best, args=(p_base, tgt), method='Powell',
             options=dict(maxiter=30, xtol=1e-3, ftol=1e-4, disp=True))
    x_opt = _best['x']
    print(f"\n  done in {(time.time()-t0)/60:.1f} min, {_best['n']} evals, best RMSE={_best['val']:.4f}")

    rm = run_extended_model(vec_to_params(x_opt, p_base), verbose=False)

    # The full-grid confirmation gives the reported numbers.
    if resolution != 'full':
        pf, x_full, r = confirm_full(x_opt, tgt, p_base)
    else:
        pf, x_full, r = p_base, x_opt, rm
    p = vec_to_params(x_full, pf)

    out = dict(
        search_resolution=resolution, reported_resolution='full',
        rmse_search=float(_best['val']), x_opt=x_full.tolist(),
        external=dict(delta=p.delta.tolist(), mu=p.mu, mu_vec=p.mu_vec.tolist(),
                      alpha=p.alpha, gamma=p.gamma, cc=float(p.cc[0]),
                      rho_enter=p.rho_enter, rho_exit=p.rho_exit),
        estimated=dict(lambda_grid=p.lambda_grid.tolist(), K=p.K.tolist(),
                       phi_c=p.phi_c, delta_u=p.delta_u, delta_n=p.delta_n,
                       delta_u_by_skill=list(p.delta_u_by_skill),
                       delta_n_by_skill=list(p.delta_n_by_skill)),
        moments_full=dict(lfp=r['model_lfp'].tolist(), nu=r['NU_hat'].tolist(),
                          gpg=float(r['gender_pay_gap'])),
        moments_medium=dict(lfp=rm['model_lfp'].tolist(), nu=rm['NU_hat'].tolist(),
                            gpg=float(rm['gender_pay_gap'])),
        targets=dict(lfp=tgt['lfp'].tolist(), nu=tgt['nu'].tolist(), gpg=tgt['gpg']),
        untargeted=dict(  # validation only - not in the objective
            unemp_model=r['model_unemp'].tolist(), unemp_data=tgt['unemp'].tolist(),
            ue_model=r['UE_hat'].tolist(), ue_data=tgt['ue'].tolist(),
            en_model=r['EN_hat'].tolist(), en_data=tgt['en'].tolist()),
    )
    json.dump(out, open(str(CLEAN / 'smm_joint_results.json'), 'w'), indent=2)
    print("  saved smm_joint_results.json")
    lfp_shift = float(np.max(np.abs(np.array(rm['model_lfp']) - r['model_lfp'])))
    print(f"  [FULL]  LFP model={r['model_lfp'].round(3)}")
    print(f"  [data]  LFP      ={tgt['lfp'].round(3)}")
    print(f"  [FULL]  GPG model={r['gender_pay_gap']:.3f}  data={tgt['gpg']:.3f}")
    print(f"  [check] unemp={r['model_unemp'].round(3)} (data {tgt['unemp'].round(3)}); "
          f"EN={r['EN_hat'].round(3)} (data {tgt['en'].round(3)})")
    print(f"  [resolution] max |LFP_full - LFP_medium| = {lfp_shift:.4f} "
          f"({'stable' if lfp_shift < 0.01 else 'CHECK: >1pp shift, consider more full polish'})")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--medium', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--maxiter', type=int, default=1200)
    a = ap.parse_args()
    resolution = 'fast' if a.fast else 'full' if a.full else 'medium'
    run(resolution=resolution, maxiter=a.maxiter)
