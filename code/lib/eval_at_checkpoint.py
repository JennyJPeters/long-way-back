import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy
from io import StringIO

import numpy as np

from paths import CHECKPOINT_OLD, CLEAN

import smm_joint


def solve_once(x_vec, p_base, tgt, max_theta_iter, n_w, n_x, vfi_max_iter):
    """SMM objective at x_vec on one grid and theta cap; returns (rmse, result, trace_text, seconds)."""
    p_full = deepcopy(p_base)
    p_full.n_w, p_full.n_x, p_full.vfi_max_iter = n_w, n_x, vfi_max_iter
    p_full.max_theta_iter = max_theta_iter

    captured = {}
    original_solver = smm_joint.run_extended_model
    theta_log = StringIO()
    real_stdout = sys.stdout

    def recording_solver(params, verbose=False):
        # verbose=True so the solver prints the theta trace that is parsed below.
        sys.stdout = theta_log
        try:
            result = original_solver(params, verbose=True)
        finally:
            sys.stdout = real_stdout
        captured['result'] = result
        return result

    smm_joint.run_extended_model = recording_solver
    smm_joint._best['val'] = -np.inf
    try:
        start_time = time.time()
        rmse = smm_joint.objective(x_vec, p_full, tgt)
        elapsed = time.time() - start_time
    finally:
        smm_joint.run_extended_model = original_solver

    return rmse, captured['result'], theta_log.getvalue(), elapsed


def theta_converged(trace_text):
    """True if the outer tightness loop stopped on its tolerance, not its cap."""
    return 'Theta loop stopped early' not in trace_text


def theta_iterations(trace_text):
    """Number of outer tightness iterations executed."""
    return len(re.findall(r'theta iter\s+\d+', trace_text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=str(CHECKPOINT_OLD))
    parser.add_argument('--caps', type=int, nargs='+', default=[10, 40],
                        help='max_theta_iter values to evaluate (first is the reference)')
    parser.add_argument('--search-res', default='fast', choices=['fast', 'medium', 'full'],
                        help="resolution whose external delta/mu_vec the search used")
    parser.add_argument('--n-w', type=int, default=100, help='reported-grid n_w')
    parser.add_argument('--n-x', type=int, default=600, help='reported-grid n_x')
    parser.add_argument('--vfi', type=int, default=2000, help='reported-grid vfi_max_iter')
    parser.add_argument('--out', default=str(CLEAN / 'eval_at_checkpoint.json'))
    args = parser.parse_args()

    x_vec = np.load(args.ckpt)
    print(f"checkpoint   {args.ckpt}", flush=True)
    print(f"  x (18)     {np.array2string(x_vec, precision=5, max_line_width=130)}", flush=True)

    targets = smm_joint.load_targets()
    p_search = smm_joint.make_base(args.search_res, theta_cap=10)
    p_base = smm_joint.setup_external(p_search, targets, verbose=False)

    rmse_search, res_search, trace_search, secs_search = solve_once(
        x_vec, p_base, targets, p_base.max_theta_iter,
        p_search.n_w, p_search.n_x, p_search.vfi_max_iter)
    print(f"\n[search grid n_w={p_search.n_w} n_x={p_search.n_x} cap={p_base.max_theta_iter}]"
          f"  RMSE={rmse_search:.4f}  ({secs_search:.1f} s, "
          f"{theta_iterations(trace_search)} theta iters, "
          f"converged={theta_converged(trace_search)})", flush=True)

    runs = {}
    for cap in args.caps:
        rmse, result, trace, secs = solve_once(
            x_vec, p_base, targets, cap,
            n_w=args.n_w, n_x=args.n_x, vfi_max_iter=args.vfi)
        converged = theta_converged(trace)
        runs[cap] = dict(rmse=float(rmse), result=result, converged=converged,
                         theta_iters=theta_iterations(trace), seconds=secs)
        print(f"\n[FULL n_w={args.n_w} n_x={args.n_x} cap={cap}]  RMSE={rmse:.4f}  ({secs/60:.1f} min, "
              f"{theta_iterations(trace)} theta iters, converged={converged})", flush=True)
        print(f"  LFP model = {result['model_lfp'].round(3)}", flush=True)
        print(f"  LFP data  = {targets['lfp'].round(3)}", flush=True)
        print(f"  GPG model = {result['gender_pay_gap']:.3f}   data = {targets['gpg']:.3f}", flush=True)
        print(f"  theta     = {np.asarray(result['theta'], float).round(4)}", flush=True)

    reference_cap = args.caps[0]
    for cap in args.caps[1:]:
        d_lfp = np.max(np.abs(runs[cap]['result']['model_lfp']
                              - runs[reference_cap]['result']['model_lfp']))
        d_gpg = abs(runs[cap]['result']['gender_pay_gap']
                    - runs[reference_cap]['result']['gender_pay_gap'])
        d_rmse = runs[cap]['rmse'] - runs[reference_cap]['rmse']
        print(f"\n[theta truncation bias cap {reference_cap} -> {cap}]  "
              f"max|dLFP| = {d_lfp:.4f} ({100*d_lfp:.2f}pp)   "
              f"|dGPG| = {d_gpg:.4f}   dRMSE = {d_rmse:+.4f}", flush=True)

    # Resolution stability: search grid versus the last cap.
    best_cap = args.caps[-1]
    lfp_shift = float(np.max(np.abs(res_search['model_lfp']
                                    - runs[best_cap]['result']['model_lfp'])))
    print(f"\n[resolution] max |LFP_full - LFP_{args.search_res}| = {lfp_shift:.4f} "
          f"({'stable' if lfp_shift < 0.01 else 'CHECK: >1pp shift'})", flush=True)

    params_full = smm_joint.vec_to_params(x_vec, p_base)
    payload = dict(
        source_checkpoint=os.path.abspath(args.ckpt),
        produced='eval_at_checkpoint.py (one solve per cap, NO polish)',
        search_resolution=args.search_res,
        rmse_search_grid=float(rmse_search),
        x_opt=x_vec.tolist(),
        external=dict(delta=params_full.delta.tolist(), mu=params_full.mu,
                      mu_vec=params_full.mu_vec.tolist(), alpha=params_full.alpha,
                      gamma=params_full.gamma, cc=float(params_full.cc[0]),
                      rho_enter=params_full.rho_enter, rho_exit=params_full.rho_exit),
        estimated=dict(lambda_grid=params_full.lambda_grid.tolist(),
                       K=params_full.K.tolist(), phi_c=params_full.phi_c,
                       delta_u=params_full.delta_u, delta_n=params_full.delta_n,
                       delta_u_by_skill=list(params_full.delta_u_by_skill),
                       delta_n_by_skill=list(params_full.delta_n_by_skill)),
        targets=dict(lfp=targets['lfp'].tolist(), nu=targets['nu'].tolist(),
                     gpg=targets['gpg'], unemp=targets['unemp'].tolist(),
                     ue=targets['ue'].tolist(), en=targets['en'].tolist()),
        by_theta_cap={
            str(cap): dict(
                rmse=run['rmse'], theta_converged=bool(run['converged']),
                theta_iters=run['theta_iters'], seconds=run['seconds'],
                lfp=run['result']['model_lfp'].tolist(),
                nu=run['result']['NU_hat'].tolist(),
                gpg=float(run['result']['gender_pay_gap']),
                theta=np.asarray(run['result']['theta'], float).tolist(),
                unemp=run['result']['model_unemp'].tolist(),
                ue=run['result']['UE_hat'].tolist(),
                en=run['result']['EN_hat'].tolist(),
                eu=run['result']['EU_hat'].tolist())
            for cap, run in runs.items()},
    )
    json.dump(payload, open(args.out, 'w'), indent=2)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == '__main__':
    main()
