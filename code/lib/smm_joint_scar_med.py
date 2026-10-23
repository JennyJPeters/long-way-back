"""Re-calibrate the scar model on the medium grid, confirm at full resolution."""
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

from paths import CHECKPOINT, CALIBRATION_RESULT
import smm_joint
import hc_scar
import smm_joint_scar as scar
import eval_at_checkpoint as evaluator

CKPT_MED = str(CHECKPOINT)
RESULT_MED = str(CALIBRATION_RESULT)
LIVE_SLOTS = [slot for slot in range(18) if slot not in scar.INERT_SLOTS]
MEDIUM = dict(n_w=30, n_x=200, vfi=1500)
FULL = dict(n_w=100, n_x=600, vfi=2000)


def set_grid(params, grid):
    params.n_w, params.n_x, params.vfi_max_iter = grid['n_w'], grid['n_x'], grid['vfi']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxfev', type=int, default=300)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    p_base, targets = hc_scar.setup(scar.N_K, scar.FLOOR_K)
    search_grid = dict(n_w=12, n_x=90, vfi=500) if args.smoke else MEDIUM
    final_grid = dict(n_w=12, n_x=90, vfi=500) if args.smoke else FULL
    set_grid(p_base, search_grid)
    x_start = np.load(scar.CKPT_SCAR)
    started = time.time()
    start_value = scar.objective_value(x_start, p_base, targets)
    print(f"smm_joint_scar_med | n_k={scar.N_K} floor={scar.FLOOR_K} | search grid {search_grid} | cap "
          f"{p_base.max_theta_iter} | {'SMOKE' if args.smoke else 'FULL RUN'}\n  start RMSE (fast-grid optimum on this grid): "
          f"{start_value:.4f}", flush=True)

    best = dict(val=np.inf, x=x_start.copy(), n=0)

    def objective_free(z_vec):
        x_vec = x_start.copy(); x_vec[LIVE_SLOTS] = z_vec
        value = scar.objective_value(x_vec, p_base, targets)
        best['n'] += 1
        if value < best['val']:
            best['val'], best['x'] = value, x_vec.copy()
            np.save(CKPT_MED, x_vec)                              # own checkpoint file
            print(f"  eval {best['n']:5d}  RMSE={value:.4f}  ({(time.time()-started)/60:.0f} min)", flush=True)
        return value

    z_start = x_start[LIVE_SLOTS].copy()
    simplex = np.vstack([z_start] + [z_start + 0.1 * np.eye(len(LIVE_SLOTS))[i] for i in range(len(LIVE_SLOTS))])
    minimize(objective_free, z_start, method='Nelder-Mead',
             options=dict(maxfev=3 if args.smoke else args.maxfev, adaptive=True, xatol=1e-3, fatol=1e-4,
                          initial_simplex=simplex))
    print(f"\n  search done: {best['n']} evals, best {best['val']:.4f} (start {start_value:.4f}), "
          f"{(time.time()-started)/60:.1f} min", flush=True)

    # Cap-40 re-check on the search grid, then one reporting-grid solve at cap 40.
    p_cap40, _ = hc_scar.setup(scar.N_K, scar.FLOOR_K)
    set_grid(p_cap40, search_grid); p_cap40.max_theta_iter = 40
    value_cap40 = scar.objective_value(best['x'], p_cap40, targets)
    print(f"  search-grid RMSE at cap 40: {value_cap40:.4f} (cap 10: {best['val']:.4f})", flush=True)
    rmse_final, result, trace, seconds = evaluator.solve_once(best['x'], p_base, targets, 40,
                                                              final_grid['n_w'], final_grid['n_x'], final_grid['vfi'])
    converged = evaluator.theta_converged(trace)
    print(f"\nREPORTING GRID {final_grid} cap 40: RMSE {rmse_final:.4f}  ({seconds/60:.1f} min, theta converged={converged})",
          flush=True)
    scar.print_fit(result, targets, 'MEDIUM-GRID SCAR CALIBRATION at the reporting grid')
    failures = scar.economics_checks(best['x'], p_base, targets, result)
    params = smm_joint.vec_to_params(best['x'], p_base)
    json.dump(dict(n_k=scar.N_K, floor_k=scar.FLOOR_K, search_grid=search_grid, final_grid=final_grid,
                   rmse_search_cap10=best['val'], rmse_search_cap40=value_cap40, rmse_final=float(rmse_final),
                   theta_converged=bool(converged), x=best['x'].tolist(), lambda_grid=params.lambda_grid.tolist(),
                   K=params.K.tolist(), mu_vec=params.mu_vec.tolist(), delta=params.delta.tolist(),
                   lfp=result['model_lfp'].tolist(), nu=result['NU_hat'].tolist(), en=result['EN_hat'].tolist(),
                   eu=result['EU_hat'].tolist(), ue=result['UE_hat'].tolist(), unemp=result['model_unemp'].tolist(),
                   gpg=float(result['gender_pay_gap']), economics_failures=failures),
              open(RESULT_MED if not args.smoke else RESULT_MED.replace('.json', '_SMOKE.json'), 'w'), indent=2)
    print(f"\n  economics checks failed: {failures if failures else 'none'}\nDONE", flush=True)


if __name__ == '__main__':
    main()
