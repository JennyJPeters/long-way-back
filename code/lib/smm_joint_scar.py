"""Re-estimate the 18 SMM parameters under the bounded human-capital floor."""
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

from paths import CHECKPOINT_OLD, CHECKPOINT_START, CLEAN
import smm_joint
import hc_scar

N_K = 11
FLOOR_K = 0.58
CKPT_SCAR = str(CHECKPOINT_START)
INERT_SLOTS = (15, 17)
LAMBDA_SLOTS = list(range(8))
K_SLOTS = [(8, [0, 4]), (9, [2, 6]), (10, [1]), (11, [3]), (12, [5]), (13, [7])]
TYPE_NAMES = smm_joint.DATA_ORDER


def objective_value(x_vec, p_base, targets):
    """Smm_joint objective with its checkpoint write switched off."""
    smm_joint._best['val'] = -np.inf
    return smm_joint.objective(x_vec, p_base, targets)


def solve(x_vec, p_base):
    return smm_joint.run_extended_model(smm_joint.vec_to_params(x_vec, p_base), verbose=False)


def bisect_init(x_start, p_base, targets, iterations):
    """Step A: interior start."""
    x_vec = x_start.copy()
    for slot, types in K_SLOTS:
        target = float(np.mean([targets['lfp'][t] for t in types]))
        low, high = np.log(0.01), np.log(12.0)
        for _ in range(iterations):
            x_vec[slot] = 0.5 * (low + high)
            model = float(np.mean([solve(x_vec, p_base)['model_lfp'][t] for t in types]))
            low, high = (x_vec[slot], high) if model > target else (low, x_vec[slot])   # too much LFP -> raise K
        x_vec[slot] = 0.5 * (low + high)
        print(f"    init K slot {slot}: log K = {x_vec[slot]:+.3f}", flush=True)
    for slot in LAMBDA_SLOTS:
        target = float(targets['nu'][slot])
        low, high = np.log(1e-5), np.log(0.4)
        for _ in range(iterations):
            x_vec[slot] = 0.5 * (low + high)
            model = float(solve(x_vec, p_base)['NU_hat'][slot])
            low, high = (x_vec[slot], high) if model < target else (low, x_vec[slot])   # too little entry -> raise lambda
        x_vec[slot] = 0.5 * (low + high)
        print(f"    init lambda slot {slot}: log lam = {x_vec[slot]:+.3f}", flush=True)
    return x_vec


def dead_parameter_test(x_vec, p_base, targets, candidates):
    """Step B: bump each candidate, assert the objective moves."""
    base_value = objective_value(x_vec, p_base, targets)
    live, dead = [], []
    for slot in candidates:
        bumped = x_vec.copy(); bumped[slot] += 0.5
        change = abs(objective_value(bumped, p_base, targets) - base_value)
        (live if change > 1e-6 else dead).append(slot)
        print(f"    slot {slot:2d}: |d objective| = {change:.5f}  -> {'LIVE' if change > 1e-6 else 'DEAD'}", flush=True)
    return live, dead


def economics_checks(x_vec, p_base, targets, result):
    """Step D: comparative statics the model must reproduce; returns the failed check names."""
    lfp, nu = result['model_lfp'], result['NU_hat']
    params = smm_joint.vec_to_params(x_vec, p_base)
    failures = []

    def check(name, condition, detail=''):
        print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}", flush=True)
        if not condition:
            failures.append(name)

    for label, child, no_child in (('UnF', 1, 5), ('SkF', 3, 7)):
        check(f'mothers LFP < non-mothers ({label})', lfp[child] < lfp[no_child],
              f"({lfp[child]:.3f} vs {lfp[no_child]:.3f}; data {targets['lfp'][child]:.3f} vs {targets['lfp'][no_child]:.3f})")
    for label, child, no_child in (('UnM', 0, 4), ('SkM', 2, 6)):
        check(f'fathers LFP > non-fathers ({label})', lfp[child] > lfp[no_child],
              f"({lfp[child]:.3f} vs {lfp[no_child]:.3f}; data {targets['lfp'][child]:.3f} vs {targets['lfp'][no_child]:.3f})")
    check('gender pay gap > 1 (men earn more)', float(result['gender_pay_gap']) > 1.0,
          f"({float(result['gender_pay_gap']):.3f}; data {targets['gpg']:.3f})")
    check('mothers K_child > K_no_child (child raises home value)',
          bool(np.all(params.K[[1, 3]] > params.K[[5, 7]])), f"(K={np.round(params.K, 3)})")
    mu_eff = np.array([params.mu_vec[i % 4] for i in range(8)])
    p_unclipped = mu_eff * np.asarray(result['theta'], float) ** (1. - params.alpha)
    check('job-finding p <= 1 (not on the cap)', bool(np.all(p_unclipped <= 1.0 + 1e-9)),
          f"(max unclipped p = {p_unclipped.max():.3f})")
    interior = lambda value, low, high: low * 1.5 < value < high * 0.9
    stuck = [name for name, value, low, high in (
        [(f'lam[{i}]', params.lambda_grid[i], 1e-5, 0.4) for i in range(8)] +
        [(f'K[{i}]', params.K[i], 0.01, 12.0) for i in range(8)]) if not interior(value, low, high)]
    check('no parameter pinned on its clip', not stuck, f"(on a clip: {stuck})")
    return failures


def print_fit(result, targets, label):
    print(f"\n  {label}")
    print(f"  {'type':6s} {'LFP mod':>8s} {'LFP dat':>8s} {'diff pp':>8s} | {'NU mod':>7s} {'NU dat':>7s} | {'unemp mod':>9s} {'dat':>6s} | {'UE mod':>7s} {'dat':>6s}")
    for i, name in enumerate(TYPE_NAMES):
        print(f"  {name:6s} {result['model_lfp'][i]:8.3f} {targets['lfp'][i]:8.3f} {100*(result['model_lfp'][i]-targets['lfp'][i]):8.1f} | "
              f"{result['NU_hat'][i]:7.3f} {targets['nu'][i]:7.3f} | {result['model_unemp'][i]:9.3f} {targets['unemp'][i]:6.3f} | "
              f"{result['UE_hat'][i]:7.3f} {targets['ue'][i]:6.3f}")
    print(f"  GPG model {float(result['gender_pay_gap']):.3f}  data {targets['gpg']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxfev', type=int, default=1500)
    parser.add_argument('--smoke', action='store_true', help='tiny pipeline test')
    args = parser.parse_args()

    p_base, targets = hc_scar.setup(N_K, FLOOR_K)
    x_checkpoint = np.load(CHECKPOINT_OLD)
    print(f"smm_joint_scar | n_k={N_K} floor={FLOOR_K} | fast grid n_w={p_base.n_w} n_x={p_base.n_x} | "
          f"theta cap {p_base.max_theta_iter} | {'SMOKE' if args.smoke else 'FULL RUN'}", flush=True)
    print(f"  OLD checkpoint under the floor: RMSE {objective_value(x_checkpoint, p_base, targets):.4f}", flush=True)

    started = time.time()
    print("\nA. interior initialisation", flush=True)
    x_start = bisect_init(x_checkpoint, p_base, targets, iterations=1 if args.smoke else 8)
    start_value = objective_value(x_start, p_base, targets)
    print(f"  start RMSE after initialisation: {start_value:.4f}  ({(time.time()-started)/60:.1f} min)", flush=True)

    print("\nB. dead-parameter test", flush=True)
    candidates = [s for s in range(18) if s not in INERT_SLOTS]
    if args.smoke:
        candidates = candidates[:3]
    live, dead = dead_parameter_test(x_start, p_base, targets, candidates)
    print(f"  live: {live}\n  dead (held fixed): {dead}  [inert by construction: {list(INERT_SLOTS)}]", flush=True)
    assert live, "no live parameters"

    print("\nC. Nelder-Mead over the live parameters", flush=True)
    best = dict(val=np.inf, x=x_start.copy(), n=0)

    def objective_free(z_vec):
        x_vec = x_start.copy(); x_vec[live] = z_vec
        value = objective_value(x_vec, p_base, targets)
        best['n'] += 1
        if value < best['val']:
            best['val'], best['x'] = value, x_vec.copy()
            np.save(CKPT_SCAR, x_vec)                          # own checkpoint file
            print(f"  eval {best['n']:5d}  RMSE={value:.4f}  ({(time.time()-started)/60:.0f} min)", flush=True)
        return value

    z_start = x_start[live].copy()
    simplex = np.vstack([z_start] + [z_start + 0.3 * np.eye(len(live))[i] for i in range(len(live))])
    minimize(objective_free, z_start, method='Nelder-Mead',
             options=dict(maxfev=10 if args.smoke else args.maxfev, adaptive=True, xatol=1e-3, fatol=1e-4,
                          initial_simplex=simplex))
    print(f"\n  search done: {best['n']} evals, best RMSE {best['val']:.4f} "
          f"(start {start_value:.4f}), {(time.time()-started)/60:.1f} min", flush=True)

    print("\nD. economics checks and cap-40 re-evaluation", flush=True)
    result = solve(best['x'], p_base)
    print_fit(result, targets, 'SCAR CALIBRATION (cap 10, fast grid)')
    failures = economics_checks(best['x'], p_base, targets, result)
    p_cap40, _ = hc_scar.setup(N_K, FLOOR_K)
    p_cap40.max_theta_iter = 40
    value_cap40 = objective_value(best['x'], p_cap40, targets)
    print(f"\n  RMSE at cap 40 (fast grid): {value_cap40:.4f}   (cap 10: {best['val']:.4f})", flush=True)
    json.dump(dict(n_k=N_K, floor_k=FLOOR_K, rmse_cap10_fast=best['val'], rmse_cap40_fast=value_cap40,
                   x=best['x'].tolist(), live_slots=live, dead_slots=dead, economics_failures=failures,
                   lfp=result['model_lfp'].tolist(), nu=result['NU_hat'].tolist(),
                   gpg=float(result['gender_pay_gap'])),
              open(CLEAN / 'smm_joint_scar_result.json', 'w'), indent=2)
    print(f"\n  economics checks failed: {failures if failures else 'none'}", flush=True)


if __name__ == '__main__':
    main()
