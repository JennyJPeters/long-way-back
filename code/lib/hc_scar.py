"""Setup helpers, preflight checks and a floor scan for the bounded human-capital scar."""
import argparse
import time

import numpy as np

from paths import CHECKPOINT_OLD
import smm_joint
import model_extended as me


def setup(n_k=6, floor_k=None):
    """Fast-grid base parameters and targets with n_k and the HC floor set."""
    targets = smm_joint.load_targets()
    p_base = smm_joint.setup_external(smm_joint.make_base('fast', theta_cap=10), targets, verbose=False)
    p_base.n_k = n_k
    p_base.hc_floor_k = floor_k
    return p_base, targets


def evaluate(p_base, targets, x_vec):
    """Smm_joint objective (checkpoint write off), model result and E/U/N stocks per HC node."""
    captured = {}
    original = me.misclassify_ext

    def spy(i, E, U, N, flows, p):
        captured[i] = {m: np.array(flows[m][i]) for m in ('E_mat', 'U_mat', 'N_mat')}
        return original(i, E, U, N, flows, p)

    me.misclassify_ext = spy
    try:
        smm_joint._best['val'] = -np.inf
        value = smm_joint.objective(x_vec, p_base, targets)
        result = smm_joint.run_extended_model(smm_joint.vec_to_params(x_vec, p_base), verbose=False)
    finally:
        me.misclassify_ext = original
    return value, result, captured


def accept_weighted(result, captured, type_index=0):
    """U-stock-weighted acceptance and share of the U stock at the lowest node, for one type."""
    acceptance = np.asarray(result['int_x_a_p'][type_index])
    unemployed = captured[type_index]['U_mat']
    return float((acceptance * unemployed).sum() / unemployed.sum()), float(unemployed[0] / unemployed.sum())


def preflight(x_checkpoint):
    failures = []

    def check(name, condition, detail=''):
        print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}", flush=True)
        if not condition:
            failures.append(name)

    p0, tgt = setup(6, None)
    hc_orig = me.build_hc_grids(p0)
    p0.hc_floor_k = p0.k_min
    hc_floor0 = me.build_hc_grids(p0)
    check('floor at node 0 == original P_U_by_type', np.allclose(hc_floor0['P_U_by_type'], hc_orig['P_U_by_type'], atol=1e-14))
    check('floor at node 0 == original P_N_by_type', np.allclose(hc_floor0['P_N_by_type'], hc_orig['P_N_by_type'], atol=1e-14))
    base_value, base_result, base_stocks = evaluate(p0, tgt, x_checkpoint)
    p0.hc_floor_k = None
    none_value, _, _ = evaluate(p0, tgt, x_checkpoint)
    check('no floor reproduces smm_joint fast-grid RMSE 1.3717', abs(none_value - 1.3717) < 2e-3, f'(got {none_value:.4f})')
    check('floor at node 0 gives the same RMSE as no floor', abs(base_value - none_value) < 1e-9)

    p1, _ = setup(6, 0.58)
    hc1 = me.build_hc_grids(p1)
    node = hc1['hc_floor_node']
    for name in ('P_U_by_type', 'P_N_by_type'):
        mats = hc1[name]
        check(f'{name}: rows stochastic, non-negative', np.allclose(mats.sum(axis=2), 1.0) and bool((mats >= -1e-15).all()))
        check(f'{name}: no mass lands below the floor node',
              bool((mats[:, node:, :node] < 1e-15).all()))
        check(f'{name}: nodes <= floor are identity rows (intended)',
              all(np.allclose(mats[i, k], np.eye(6)[k]) for i in range(8) for k in range(node + 1)))
        check(f'{name}: no accidental identity row above the floor',
              all(not np.allclose(mats[i, k], np.eye(6)[k]) for i in range(8) for k in range(node + 1, 6)))

    value1, result1, stocks1 = evaluate(p1, tgt, x_checkpoint)
    acc0, share0 = accept_weighted(base_result, base_stocks)
    acc1, share1 = accept_weighted(result1, stocks1)
    check('floor empties the lowest node in the U stock (UnM-C)', share1 < 1e-6 < share0, f'({share0:.3f} -> {share1:.3f})')
    check('floor raises U-stock-weighted acceptance (UnM-C)', acc1 > acc0 + 0.05, f'({acc0:.3f} -> {acc1:.3f})')
    if failures:
        raise AssertionError(f"preflight failed: {failures}")
    print("  preflight: all checks passed\n", flush=True)


def scan(x_checkpoint):
    """Print the UnM-C cell and the objective under alternative floors; no re-optimisation."""
    tgt0 = smm_joint.load_targets()
    print(f"data: LFP UnM-C {tgt0['lfp'][0]:.3f} unemp UnM-C {tgt0['unemp'][0]:.3f} NU UnM-C {tgt0['nu'][0]:.3f} "
          f"UE grp UnM {tgt0['ue_grp'][0]:.3f} | LFP UnF-C {tgt0['lfp'][1]:.3f} UnF-0 {tgt0['lfp'][5]:.3f} GPG {tgt0['gpg']:.3f}")
    print(" n_k floor(node k) lam_UnM_c |  RMSE | LFP UnM-C unemp  NU   UE grp | U@node0  accept(U) | LFP UnF-C UnF-0  GPG | sec")
    cases = [(6, None), (6, 0.58), (6, 1.0),
             (11, None), (11, 0.58), (11, 0.82)]
    for n_k, floor_k in cases:
        for lam in (1e-5, 1e-2):
            x_vec = x_checkpoint.copy(); x_vec[0] = np.log(lam)
            try:
                p_base, tgt = setup(n_k, floor_k)
                started = time.time()
                value, result, stocks = evaluate(p_base, tgt, x_vec)
                hc = me.build_hc_grids(p_base)
                node = hc.get('hc_floor_node', 0)
                accept_u, share_node0 = accept_weighted(result, stocks)
                print(f"{n_k:4d} {('none' if floor_k is None else f'{floor_k:.2f}'):>5s}({hc['k_grid'][node]:.2f}) {lam:9.0e} | "
                      f"{value:6.3f} | {result['model_lfp'][0]:8.3f} {result['model_unemp'][0]:5.3f} {result['NU_hat'][0]:5.3f} "
                      f"{0.5*(result['UE_hat'][0]+result['UE_hat'][4]):6.3f} | {share_node0:7.3f} {accept_u:9.3f} | "
                      f"{result['model_lfp'][1]:8.3f} {result['model_lfp'][5]:5.3f} {float(result['gender_pay_gap']):5.3f} | "
                      f"{time.time()-started:4.0f}", flush=True)
            except Exception as error:                         # report the failure and continue
                print(f"{n_k:4d} {str(floor_k):>5s} {lam:9.0e} | FAILED: {type(error).__name__}: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    x_checkpoint = np.load(CHECKPOINT_OLD)
    print("hc_scar | fast grid | checkpoint parameters, NOTHING re-optimised, delta NOT re-estimated")
    preflight(x_checkpoint)
    if not args.preflight_only:
        scan(x_checkpoint)


if __name__ == '__main__':
    main()
