"""Cluster bootstrap (resampling persons by cpsidp) of the 37 targeted CPS moments."""
import json
import argparse
import numpy as np

from paths import MOMENT_SE
from clean_cps import load_cps, _weighted_median

OUT = str(MOMENT_SE)
DATA_ORDER = ['UnM-C', 'UnF-C', 'SkM-C', 'SkF-C', 'UnM-0', 'UnF-0', 'SkM-0', 'SkF-0']
_UEG = ['UE_UnM', 'UE_UnF', 'UE_SkM', 'UE_SkF']   # average of child / no-child UE
NAMES = ([f'LFP_{t}' for t in DATA_ORDER] + [f'NU_{t}' for t in DATA_ORDER]
         + [f'EN_{t}' for t in DATA_ORDER] + [f'EU_{t}' for t in DATA_ORDER]
         + _UEG + ['GPG'])


def _person_group_matrix(df, pid_codes, n_pers, value):
    """Persons x 8 matrix of summed `value` by (person, type8)."""
    M = np.zeros((n_pers, 8))
    np.add.at(M, (pid_codes, df['type8'].to_numpy()), value)
    return M


def build_arrays(df):
    """Per-person numerator / denominator matrices for all moments, plus the GPG earner table."""
    pid = df['cpsidp'].astype('category')
    df = df.assign(_pcode=pid.cat.codes.to_numpy())
    n_pers = pid.cat.categories.size

    w = df['wtfinl'].to_numpy()
    inLF = df['lf_state'].isin(['E', 'U']).to_numpy().astype(float)
    LFP_den = _person_group_matrix(df, df['_pcode'].to_numpy(), n_pers, w)
    LFP_num = _person_group_matrix(df, df['_pcode'].to_numpy(), n_pers, w * inLF)

    # Flows use linked adjacent months; denominators are counts of pairs from the source state.
    cols = ['cpsidp', 'ym', 'lf_state', 'sex', 'age', 'type8', 'lnkfw1mwt', 'wtfinl', '_pcode']
    a = df[cols]; b = df[cols].copy(); b['ym_prev'] = b['ym'] - 1
    mg = a.merge(b, left_on=['cpsidp', 'ym'], right_on=['cpsidp', 'ym_prev'], suffixes=('_t', '_t1'))
    valid = (mg['sex_t'] == mg['sex_t1']) & (mg['age_t1'] - mg['age_t']).between(0, 1)
    mg = mg[valid]
    wl = mg['lnkfw1mwt_t1'].where(mg['lnkfw1mwt_t1'] > 0, mg['wtfinl_t']).to_numpy()
    pc_t = mg['_pcode_t'].to_numpy(); tp_t = mg['type8_t'].to_numpy()
    st_t = mg['lf_state_t'].to_numpy(); st_t1 = mg['lf_state_t1'].to_numpy()

    def _flow(from_state, to_state=None):
        """Persons x 8 matrix of link-weighted pairs from `from_state` (to `to_state` if given)."""
        m = np.zeros((n_pers, 8))
        w = wl * (st_t == from_state)
        if to_state is not None:
            w = w * (st_t1 == to_state)
        np.add.at(m, (pc_t, tp_t), w)
        return m

    E_den  = _flow('E')
    EN_num = _flow('E', 'N')
    EU_num = _flow('E', 'U')
    U_den  = _flow('U')
    UE_num = _flow('U', 'E')
    N_den  = _flow('N')
    NU_num = _flow('N', 'U')

    e = df[(df['lf_state'] == 'E') & (df['earnweek'] > 0) & (df['earnweek'] < 9999.99)
           & (df['uhrsworkorg'].between(35, 168)) & (df['earnwt'] > 0)]
    earn = dict(pcode=e['_pcode'].to_numpy(), val=e['earnweek'].to_numpy(),
                wt=e['earnwt'].to_numpy(), female=(e['is_female'] == 1).to_numpy())
    return dict(n_pers=n_pers, LFP_num=LFP_num, LFP_den=LFP_den,
                NU_num=NU_num, N_den=N_den, EN_num=EN_num, EU_num=EU_num, E_den=E_den,
                UE_num=UE_num, U_den=U_den, earn=earn)


def moments_from_mult(A, mult):
    """The 37-moment vector for a per-person multiplicity vector `mult`."""
    lfp = (mult @ A['LFP_num']) / np.maximum(mult @ A['LFP_den'], 1e-12)
    nu  = (mult @ A['NU_num'])  / np.maximum(mult @ A['N_den'],   1e-12)
    en  = (mult @ A['EN_num'])  / np.maximum(mult @ A['E_den'],   1e-12)
    eu  = (mult @ A['EU_num'])  / np.maximum(mult @ A['E_den'],   1e-12)
    ue8 = (mult @ A['UE_num'])  / np.maximum(mult @ A['U_den'],   1e-12)
    ue_grp = np.array([0.5 * (ue8[g] + ue8[g + 4]) for g in range(4)])
    e = A['earn']; ew = e['wt'] * mult[e['pcode']]
    fem = e['female']
    med_m = _weighted_median(e['val'][~fem], ew[~fem])
    med_f = _weighted_median(e['val'][fem],  ew[fem])
    gpg = med_m / med_f if med_f > 0 else np.nan
    return np.concatenate([lfp, nu, en, eu, ue_grp, [gpg]])


def main(B=400):
    print("Cluster bootstrap of SMM target moments (by person) ...")
    df = load_cps()
    A = build_arrays(df)
    n = A['n_pers']
    print(f"  persons={n:,}; building point estimate + {B} bootstrap resamples")

    ones = np.ones(n)
    point = moments_from_mult(A, ones)

    boot = np.empty((B, len(NAMES)))
    rng = np.random.RandomState(12345)
    for bcnt in range(B):
        idx = rng.randint(0, n, n)
        mult = np.bincount(idx, minlength=n).astype(float)
        boot[bcnt] = moments_from_mult(A, mult)
        if (bcnt + 1) % 100 == 0:
            print(f"    {bcnt+1}/{B}", flush=True)

    Sigma = np.cov(boot, rowvar=False)
    se = np.sqrt(np.diag(Sigma))
    out = dict(
        names=NAMES, point=point.tolist(), se=se.tolist(),
        cov=Sigma.tolist(), n_boot=int(B), n_persons=int(n),
        note='cluster bootstrap by cpsidp; 37 moments, order LFP8,NU8,EN8,EU8,UE_group4,GPG',
    )
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"\n  wrote {OUT}")
    print(f"  {'moment':10s} {'point':>9} {'se':>9} {'se/point %':>10}")
    for nm, p, s in zip(NAMES, point, se):
        print(f"  {nm:10s} {p:9.4f} {s:9.5f} {100*s/abs(p) if p else float('nan'):10.2f}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=400)
    main(B=ap.parse_args().B)
