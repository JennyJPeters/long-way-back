"""Untargeted wage gaps (CPS ORG data vs model) and the model fit moments."""
import argparse
import json
import numpy as np

from paths import CHECKPOINT, CLEAN

OUT = CLEAN / 'wage_gaps.json'
GRIDS = dict(medium=(30, 200, 1500), full=(100, 600, 2000))   # (n_w, n_x, vfi_max_iter)
CKPT = CHECKPOINT
# Type order: 0-3 child-present, 4-7 no child; within each block UnM UnF SkM SkF
TYPE_INDEX = {(skill, female, child): (0 if child else 4) + 2 * skill + female
              for skill in (0, 1) for female in (0, 1) for child in (0, 1)}


def data_cells():
    """Weighted median weekly earnings by (skill, female, child)."""
    import clean_cps
    df = clean_cps.load_cps()
    e = df[(df['lf_state'] == 'E') & (df['earnweek'] > 0) & (df['earnweek'] < 9999.99)
           & (df['uhrsworkorg'].between(35, 168)) & (df['earnwt'] > 0)]
    def med(sub):
        return clean_cps._weighted_median(sub['earnweek'].to_numpy(), sub['earnwt'].to_numpy())
    cells, pooled = {}, {}
    for (skill, female, child) in TYPE_INDEX:
        sub = e[(e['skill'] == skill) & (e['is_female'] == female) & (e['has_child'] == child)]
        cells[(skill, female, child)] = (med(sub), int(len(sub)))
    for skill in (0, 1):
        for female in (0, 1):
            pooled[(skill, female)] = med(e[(e['skill'] == skill) & (e['is_female'] == female)])
    return cells, pooled


def model_cells(grid):
    """Employed-weighted mean wage per type, and employed mass."""
    import smm_joint_scar as scar
    import hc_scar
    import eval_at_checkpoint as evaluator
    x_vec = np.load(CKPT)
    p_base, targets = hc_scar.setup(scar.N_K, scar.FLOOR_K)
    n_w, n_x, vfi = GRIDS[grid]
    rmse, result, trace, seconds = evaluator.solve_once(x_vec, p_base, targets, 40, n_w, n_x, vfi)
    print(f"model solve: RMSE {rmse:.4f}, {seconds/60:.1f} min, theta converged="
          f"{evaluator.theta_converged(trace)}, GPG {result['gender_pay_gap']:.4f}", flush=True)
    wage = np.asarray(result['avg_wage_emp']); mass = np.asarray(result['E'])
    cells = {key: (float(wage[i]), float(mass[i])) for key, i in TYPE_INDEX.items()}
    pooled = {}
    for skill in (0, 1):
        for female in (0, 1):
            idx = [TYPE_INDEX[(skill, female, c)] for c in (0, 1)]
            pooled[(skill, female)] = float(mass[idx] @ wage[idx] / mass[idx].sum())
    fit = dict(grid=grid, rmse=float(rmse), theta_converged=bool(evaluator.theta_converged(trace)),
               lfp=result['model_lfp'].tolist(), nu=result['NU_hat'].tolist(), en=result['EN_hat'].tolist(),
               eu=result['EU_hat'].tolist(), ue=result['UE_hat'].tolist(), unemp=result['model_unemp'].tolist(),
               gpg=float(result['gender_pay_gap']), w_nash=np.asarray(result['w_nash']).tolist(),
               avg_wage_emp=wage.tolist(), avg_k_E=np.asarray(result['avg_k_E']).tolist())
    return cells, pooled, float(result['gender_pay_gap']), float(rmse), fit


def gaps(cells, pooled):
    """Ratios: men/women within skill; no-child/child (women); child/no-child (men)."""
    out = {}
    for skill, name in ((0, 'noncollege'), (1, 'college')):
        out[f'gpg_{name}'] = pooled[(skill, 0)] / pooled[(skill, 1)]
        out[f'motherhood_{name}'] = cells[(skill, 1, 0)][0] / cells[(skill, 1, 1)][0]
        out[f'fatherhood_{name}'] = cells[(skill, 0, 1)][0] / cells[(skill, 0, 0)][0]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--model-only', action='store_true',
                        help='keep the stored data block (no CPS file needed) and recompute the model block')
    parser.add_argument('--grid', default='medium', choices=list(GRIDS))
    args = parser.parse_args()
    if args.model_only:
        stored = json.loads(OUT.read_text())
        record = {key: stored[key] for key in ('data_gaps', 'data_cells_median_weekly')}
    else:
        d_cells, d_pooled = data_cells()
        record = {'data_gaps': gaps(d_cells, d_pooled),
                  'data_cells_median_weekly': {str(k): v for k, v in d_cells.items()}}
    if not args.data_only:
        m_cells, m_pooled, m_gpg, rmse, fit = model_cells(args.grid)
        record.update(model_gaps=gaps(m_cells, m_pooled), model_gpg=m_gpg, model_rmse=rmse, model_fit=fit,
                      model_cells_mean_wage=[str(k) + f' {v}' for k, v in m_cells.items()])
    for key, val in record['data_gaps'].items():
        mv = record.get('model_gaps', {}).get(key, float('nan'))
        print(f"  {key:22s} data {val:.3f}   model {mv:.3f}")
    json.dump(record, open(OUT, 'w'), indent=2, default=str)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
