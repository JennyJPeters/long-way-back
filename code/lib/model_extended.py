"""Search model with human capital (HC), child-rearing lifecycle and eight worker types."""

import numpy as np
from scipy.stats import genpareto
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

try:
    import numba as _numba
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

# Threads for the (wage x type) VFI loop; set to 1 to disable parallelism.
_N_VFI_WORKERS = 6


@dataclass
class ExtParameters:
    """Model parameters."""

    n_k:    int   = 6
    hc_floor_k: float = None   # None = no floor; else decay stops at the node nearest this value
    k_min:  float = 0.10
    k_max:  float = 2.50
    k_init: float = 1.00    # HC at labour-market entry

    delta_u: float = 0.03
    delta_n: float = 0.013
    delta_u_by_skill: tuple = (0.0263, 0.0379)  # per skill [unskilled, skilled]
    delta_n_by_skill: tuple = (0.0143, 0.0132)  # per skill [unskilled, skilled]
    g_emp:   float = 0.0025
    g_emp_by_skill: tuple = (0.0018192392022343, 0.0030735260404167)
    psi_h:   float = 0.00432581257173400

    # Child-rearing chain; stationary child-present share = rho_enter / (rho_enter + rho_exit).
    rho_enter: float = 1./281.  # prob no-child -> child-present per month
    rho_exit:  float = 1./60.
    # Child-exit rate by skill [non-college, college].
    rho_exit_by_skill: np.ndarray = field(default_factory=lambda: np.array(
        [0.01811, 0.01413]))
    rho_death: float = 1./480.  # monthly prob of reset to a new no-child worker at k_init

    pi_E: float = 0.0

    # Childcare cost phi_c, paid while employed by child-present women (types 1, 3).
    phi_c: float = 0.10         # per-period cost

    # Separation rate by type (child-present types first).
    delta: np.ndarray = field(default_factory=lambda: np.array([
        0.00461, 0.000986, 0.01286, 0.00866,
        0.00461, 0.000986, 0.01286, 0.00866,
    ]))

    # Permanent-exit probability per period (zero for all types).
    lambda_y: np.ndarray = field(default_factory=lambda: np.zeros(8))

    # Probability of a new x draw per period, by type.
    lambda_grid: np.ndarray = field(default_factory=lambda: np.array([
        0.009, 0.026, 0.018, 0.004,
        0.005, 0.010, 0.010, 0.002,
    ]))

    e:    float = 0.625   # h/T - working
    s:    float = 0.125   # s/T - searching
    beta: float = 0.996

    alpha: float = 0.72
    mu:    float = 0.15
    gamma: float = 0.72

    # Matching efficiency by group [UnM, UnF, SkM, SkF]; if zero, scalar mu is used.
    mu_vec: np.ndarray = field(default_factory=lambda: np.zeros(4))

    b_frac: float = 0.40

    w_min: float = 0.0

    reentry_mode: str = 'baseline'
    eta_act: float = 0.0

    # Population shares by type.
    Ntype: np.ndarray = field(default_factory=lambda: np.array([
        0.141, 0.126, 0.082, 0.099,
        0.176, 0.173, 0.094, 0.110,
    ]))

    # Productivity by skill; HC k scales it.
    y_hl: np.ndarray = field(default_factory=lambda: np.array(
        [1., 1., 2., 2., 1., 1., 2., 2.]))

    y_ratio_f: float = 1.0

    # tau_w_f < 1 lowers women's received wages (worker side only, not firm output).
    tau_w_f: float = 1.0

    # Opportunity cost x ~ generalized Pareto with shape K by type.
    K: np.ndarray = field(default_factory=lambda: np.array([
        0.700, 25.96, 94.83, 2.079,
        0.700,  5.00, 10.00, 2.079,
    ]))
    x_upper_h: float = 12.313
    xmax_ind:  float = 7.13 / 9.73

    cc: np.ndarray = field(default_factory=lambda: 8.5 * np.ones(8))

    # q_ab: probability that true state a is reported as b (same for both child statuses).
    q_nu: np.ndarray = field(default_factory=lambda: np.tile(
        [0.0041, 0.0024, 0.0041, 0.0024], 2))
    q_ne: np.ndarray = field(default_factory=lambda: np.tile(
        [0.0066, 0.0042, 0.0066, 0.0042], 2))
    q_en: np.ndarray = field(default_factory=lambda: np.tile(
        [0.0065, 0.0154, 0.0065, 0.0154], 2))
    q_eu: np.ndarray = field(default_factory=lambda: np.tile(
        [0.0019, 0.0020, 0.0019, 0.0020], 2))
    q_un: np.ndarray = field(default_factory=lambda: np.tile(
        [0.078,  0.1146, 0.078,  0.1146], 2))
    q_ue: np.ndarray = field(default_factory=lambda: np.tile(
        [0.023,  0.0147, 0.023,  0.0147], 2))

    # Data targets by type: LFP, unemployment rate, UE flow rate.
    data_lfp:   np.ndarray = field(default_factory=lambda: np.array(
        [0.81, 0.65, 0.94, 0.84, 0.82, 0.68, 0.93, 0.82]))
    data_unemp: np.ndarray = field(default_factory=lambda: np.array(
        [0.0454, 0.0537, 0.0202, 0.0202, 0.0310, 0.0329, 0.0196, 0.0211]))
    data_UE:    np.ndarray = field(default_factory=lambda: np.array(
        [0.2651, 0.2630, 0.2479, 0.2535, 0.2901, 0.2899, 0.2713, 0.2729]))

    use_transfer: bool = False

    # Numerical resolution; smaller values run faster but less accurately.
    n_w:           int = 100    # wage grid points
    n_x:           int = 600    # opportunity-cost grid points
    vfi_max_iter:  int = 2000   # VFI iteration cap per (wage, type) pair
    max_theta_iter: int = 10    # outer market-tightness loop cap


def _lottery_row(k_target, k_grid):
    """Mean-preserving two-node lottery over k_grid with mean k_target (clipped to the grid)."""
    n_k = len(k_grid)
    row = np.zeros(n_k)
    clipped = float(np.clip(k_target, k_grid[0], k_grid[-1]))
    upper = int(np.searchsorted(k_grid, clipped, side='left'))
    if upper <= 0:
        row[0] = 1.0
    elif upper >= n_k:
        row[-1] = 1.0
    else:
        lower = upper - 1
        weight_upper = (clipped - k_grid[lower]) / (k_grid[upper] - k_grid[lower])
        row[lower] = 1.0 - weight_upper
        row[upper] = weight_upper
    return row


def floored_matrix(k_grid, depreciation_rate, floor_node):
    """Row-stochastic depreciation matrix; nodes at or below floor_node keep their level."""
    floor_value = k_grid[floor_node]
    rows = []
    for node, k in enumerate(k_grid):
        if node <= floor_node:
            row = np.zeros(len(k_grid))
            row[node] = 1.0
        else:
            row = _lottery_row(max((1. - depreciation_rate) * k, floor_value), k_grid)
        rows.append(row)
    return np.array(rows)


def apply_hc_floor(hc: dict, p: ExtParameters) -> dict:
    """Rebuild unemployed and non-participating HC transitions with a floor node."""
    if p.hc_floor_k is None:
        return hc
    k_grid = hc['k_grid']
    floor_node = int(np.argmin(np.abs(k_grid - p.hc_floor_k)))
    hc['hc_floor_node'] = floor_node
    n_types = hc['P_U_by_type'].shape[0]
    hc['P_U_by_type'] = np.array([floored_matrix(k_grid, hc['delta_u_by_type'][i], floor_node)
                                  for i in range(n_types)])
    hc['P_N_by_type'] = np.array([floored_matrix(k_grid, hc['delta_n_by_type'][i], floor_node)
                                  for i in range(n_types)])
    hc['P_U'] = floored_matrix(k_grid, p.delta_u, floor_node)
    hc['P_N'] = floored_matrix(k_grid, p.delta_n, floor_node)
    return hc


def build_hc_grids(p: ExtParameters) -> dict:
    """HC grid and per-type transition matrices for employed, unemployed, non-participating."""
    k_grid = np.linspace(p.k_min, p.k_max, p.n_k)

    # Employed HC grows multiplicatively at g_emp; no depreciation while employed.
    growth_E = float(np.exp(p.g_emp))
    h_psi = growth_E

    def nearest_idx(val):
        return int(np.argmin(np.abs(k_grid - np.clip(val, p.k_min, p.k_max))))

    k_E_idx = np.array([nearest_idx(growth_E * k)          for k in k_grid])
    k_U_idx = np.array([nearest_idx((1. - p.delta_u) * k)  for k in k_grid])
    k_N_idx = np.array([nearest_idx((1. - p.delta_n) * k)  for k in k_grid])

    # Row-stochastic lottery transitions (source node -> destination node).
    def lottery_row(k_target: float) -> np.ndarray:
        """Row over grid nodes with mean k_target (clipped to the grid)."""
        row = np.zeros(p.n_k)
        kt = float(np.clip(k_target, k_grid[0], k_grid[-1]))
        hi = int(np.searchsorted(k_grid, kt, side='left'))
        if hi <= 0:
            row[0] = 1.0
        elif hi >= p.n_k:
            row[-1] = 1.0
        else:
            lo = hi - 1
            w = (kt - k_grid[lo]) / (k_grid[hi] - k_grid[lo])
            row[lo] = 1.0 - w
            row[hi] = w
        return row

    P_E = np.array([lottery_row(growth_E * k)          for k in k_grid])
    P_U = np.array([lottery_row((1. - p.delta_u) * k)  for k in k_grid])
    P_N = np.array([lottery_row((1. - p.delta_n) * k)  for k in k_grid])

    n_types_hc = 8
    du_by_skill = getattr(p, 'delta_u_by_skill', None)
    dn_by_skill = getattr(p, 'delta_n_by_skill', None)

    def _rate_for_type(i, by_skill, pooled):
        """Rate for type i: skill-specific (skill = (i % 4) // 2) if supplied, else pooled."""
        if by_skill is None:
            return float(pooled)
        return float(by_skill[(i % 4) // 2])

    ge_by_skill = getattr(p, 'g_emp_by_skill', None)
    P_E_by_type = np.array([
        [lottery_row(np.exp(_rate_for_type(i, ge_by_skill, p.g_emp)) * k)
         for k in k_grid]
        for i in range(n_types_hc)])
    g_emp_by_type = np.array([_rate_for_type(i, ge_by_skill, p.g_emp)
                              for i in range(n_types_hc)])

    P_U_by_type = np.array([
        [lottery_row((1. - _rate_for_type(i, du_by_skill, p.delta_u)) * k)
         for k in k_grid]
        for i in range(n_types_hc)])
    P_N_by_type = np.array([
        [lottery_row((1. - _rate_for_type(i, dn_by_skill, p.delta_n)) * k)
         for k in k_grid]
        for i in range(n_types_hc)])
    delta_u_by_type = np.array([_rate_for_type(i, du_by_skill, p.delta_u)
                                for i in range(n_types_hc)])
    delta_n_by_type = np.array([_rate_for_type(i, dn_by_skill, p.delta_n)
                                for i in range(n_types_hc)])

    k_ss_E = p.k_max
    k_ss_U = p.k_min

    hc = dict(n_k=p.n_k, k_grid=k_grid, h_psi=h_psi,
              k_E_idx=k_E_idx, k_U_idx=k_U_idx, k_N_idx=k_N_idx,
              P_E=P_E, P_U=P_U, P_N=P_N,
              P_E_by_type=P_E_by_type,
              P_U_by_type=P_U_by_type, P_N_by_type=P_N_by_type,
              g_emp_by_type=g_emp_by_type,
              delta_u_by_type=delta_u_by_type,
              delta_n_by_type=delta_n_by_type,
              k_init_idx=nearest_idx(p.k_init),
              k_ss_E=k_ss_E, k_ss_U=k_ss_U)
    return apply_hc_floor(hc, p)


def build_grids(p: ExtParameters) -> dict:
    """Wage grid (as wage rates), x-grid and x PMFs for the 8 types."""
    n_types = 8
    n_w     = p.n_w
    # w_grid holds wage rates in [0, 1]; the wage paid is w_rate * y(k).
    w_grid  = np.linspace(0., 1., n_w)
    # Benefit b does not vary with the wage grid.
    b_grid  = p.b_frac * np.ones(n_w)
    # Transfer I is zero on the grid.
    I_grid  = np.zeros(n_w)

    n_x    = p.n_x
    x_grid = np.linspace(0., p.x_upper_h, n_x)
    xmax   = round(p.xmax_ind * n_x)

    gp_grid = np.zeros((n_x, n_types))
    for i in range(n_types):
        cdf = genpareto.cdf(x_grid, c=p.K[i], loc=0., scale=1.)
        pdf = np.diff(cdf); pdf = np.append(pdf, 0.)
        pdf[-1] = 0.
        pdf = pdf + (1. - pdf.sum()) / n_x
        if i % 2 == 0:    # male types: 0,2,4,6
            pdf[:xmax] += pdf[xmax:].sum() / xmax
            pdf[xmax:]  = 0.
        gp_grid[:, i] = pdf

    return dict(n_w=n_w, w_grid=w_grid, b_grid=b_grid, I_grid=I_grid,
                n_x=n_x, x_grid=x_grid, xmax=xmax, gp_grid=gp_grid)


def _make_vfi_core():
    """Build the Bellman iteration kernel, JIT-compiled with Numba when available."""
    def _core(flow_W, flow_S, flow_H,
               gp, P_E, P_U, P_N,
               lam, ly, delta, beta, pi_i,
               vfi_max_iter, reentry_code, W0, S0, H0,
               rho, Tc, clean_u, eta_act):
        n_x, n_k = flow_W.shape
        # Warm start from the previous tightness iteration (zeros on a cold start).
        W = W0.copy()
        S = S0.copy()
        H = H0.copy()

        err = np.inf
        it  = 0
        while err > 1e-4 and it < vfi_max_iter:
            it += 1
            Wn = np.zeros((n_x, n_k))
            Sn = np.zeros((n_x, n_k))
            Hn = np.zeros((n_x, n_k))

            # Max over labour states at each destination node, once per iteration.
            T_all   = np.maximum(np.maximum(W, S), H)   # continue-anywhere value
            sep_all = np.maximum(S, H)
            O_all   = np.maximum(W, S)

            for ki in range(n_k):
                pE = P_E[ki]; pU = P_U[ki]; pN = P_N[ki]   # (n_k,) dest weights

                W_kE   = W @ pE
                sep_kE = sep_all @ pE
                T_kE   = T_all @ pE
                sep_kU = sep_all @ pU
                T_kU   = T_all @ pU
                O_kU   = O_all @ pU
                S_kU   = S @ pU
                sep_kN = sep_all @ pN
                H_kN   = H @ pN
                T_kN   = T_all @ pN

                ET_kE = np.dot(gp, T_kE)
                ES_kE = np.dot(gp, sep_kE)
                ET_kU = np.dot(gp, T_kU)
                ES_kU = np.dot(gp, sep_kU)   # E_x[max(U,N)] over a fresh draw after U
                ES_kN = np.dot(gp, sep_kN)
                ET_kN = np.dot(gp, T_kN)   # E_x[max(W,S,H)] over the N HC-lottery

                # Employed: no on-the-job search; lam is the arrival rate of a new x draw.
                QW = (lam   * (1. - ly) * ((1. - delta) * ET_kE + delta * ES_kE)
                    + (1. - lam) * (1. - ly) * (
                        (1. - delta) * W_kE
                        + delta * sep_kE))

                qs_noffer = (ES_kU + 0.0 * sep_kU) if clean_u == 1 else sep_kU
                QS = (lam   * (1. - ly) * (pi_i * ET_kU + (1. - pi_i) * qs_noffer)
                    + (1. - lam) * (1. - ly) * (pi_i * O_kU + (1. - pi_i) * S_kU))

                qh_stay = eta_act * sep_kN + (1. - eta_act) * H_kN
                if reentry_code == 1:
                    QH = (1. - ly) * (lam * ET_kN + (1. - lam) * qh_stay)
                else:
                    QH = (1. - ly) * (lam * ES_kN + (1. - lam) * qh_stay)

                # Child-status switch (prob rho): re-optimise as the counterpart type at the rank-matched x.
                sw_E = Tc @ P_E[ki]     # (n_x,) rank-matched counterpart value after E
                sw_U = Tc @ P_U[ki]
                sw_N = Tc @ P_N[ki]
                Wn[:, ki] = flow_W[:, ki] + beta * ((1. - rho) * QW + rho * sw_E)
                Sn[:, ki] = flow_S[:, ki] + beta * ((1. - rho) * QS + rho * sw_U)
                Hn[:, ki] = flow_H[:, ki] + beta * ((1. - rho) * QH + rho * sw_N)

            err = (np.sum(np.abs(Wn - W)) + np.sum(np.abs(Sn - S))
                 + np.sum(np.abs(Hn - H)))
            W = Wn
            S = Sn
            H = Hn

        return W, S, H

    if _NUMBA_OK:
        return _numba.njit(_core, cache=True, nogil=True)
    return _core


_solve_vfi_core = _make_vfi_core()
def _vfi_and_cutoffs(j, i, w_rate, worker_w_rate, b, I, pi_i, phi_c_i, p, g, hc,
                     W0=None, S0=None, H0=None, rho=0.0, Tc=None):
    """Solve VFI, cutoffs and cutoff integrals for wage index j and type i."""
    W2d, S2d, H2d = solve_vfi_2d(worker_w_rate, b, I, i, pi_i, phi_c_i, p, g, hc,
                                 W0, S0, H0, rho, Tc)
    iq, xq, in_, xn, ia, xa = find_cutoffs_2d(W2d, S2d, H2d, g['x_grid'])
    gp  = g['gp_grid'][:, i]
    n_k = hc['n_k']

    def _int(idx):
        idx = int(idx)
        return float(gp[:idx].sum()) if idx > 1 else 0.

    ix_q = np.array([_int(iq[ki]) for ki in range(n_k)])
    ix_n = np.array([_int(in_[ki]) for ki in range(n_k)])
    ix_a = np.array([_int(ia[ki]) for ki in range(n_k)])
    return j, i, W2d, S2d, H2d, iq, in_, ia, ix_q, ix_n, ix_a

def solve_vfi_2d(worker_w_rate: float, b: float, I: float,
                 i: int, pi_i: float, phi_c_i: float,
                 p: ExtParameters, g: dict, hc: dict,
                 W0=None, S0=None, H0=None,
                 rho: float = 0.0, Tc=None) -> tuple:
    """VFI for one (wage rate, type) pair; returns W, S, H of shape (n_x, n_k)."""
    n_x    = g['n_x']
    n_k    = hc['n_k']
    x_grid = g['x_grid']
    k_grid = hc['k_grid']
    gp     = np.ascontiguousarray(g['gp_grid'][:, i])   # (n_x,) PMF over x'

    P_E = np.ascontiguousarray(hc['P_E_by_type'][i], dtype=np.float64)
    P_U = np.ascontiguousarray(hc['P_U_by_type'][i], dtype=np.float64)
    P_N = np.ascontiguousarray(hc['P_N_by_type'][i], dtype=np.float64)

    lam   = float(p.lambda_grid[i])
    ly    = float(p.lambda_y[i]) if hasattr(p, 'lambda_y') else 0.
    delta = float(p.delta[i])
    beta  = float(p.beta)

    y_k = p.y_hl[i] * k_grid

    flow_W = (worker_w_rate * y_k)[None, :] + (1. - p.e) * x_grid[:, None] + I - phi_c_i
    flow_S = b + (1. - p.s) * x_grid[:, None] + np.zeros((n_x, n_k))
    flow_H = x_grid[:, None] + np.zeros((n_x, n_k))

    zk = np.zeros((n_x, n_k))
    W0c = np.ascontiguousarray(W0) if W0 is not None else zk
    S0c = np.ascontiguousarray(S0) if S0 is not None else zk.copy()
    H0c = np.ascontiguousarray(H0) if H0 is not None else zk.copy()

    ic = i + 4 if i < 4 else i - 4
    if Tc is not None:
        Tc_raw  = np.ascontiguousarray(Tc, dtype=np.float64)
        cdf_own = np.cumsum(gp)
        cdf_c   = np.cumsum(np.ascontiguousarray(g['gp_grid'][:, ic]))
        rank_map = np.clip(np.searchsorted(cdf_c, cdf_own, side='left'), 0, n_x - 1)
        Tc_arr  = np.ascontiguousarray(Tc_raw[rank_map, :])
    else:
        Tc_arr = np.zeros((n_x, n_k))
    reentry_code = 1 if getattr(p, 'reentry_mode', 'baseline') == 'matching_off' else 0
    eta_act = (1.0 if getattr(p, 'reentry_mode', 'baseline') == 'activation_off_ge'
               else float(getattr(p, 'eta_act', 0.0)))

    return _solve_vfi_core(
        np.ascontiguousarray(flow_W),
        np.ascontiguousarray(flow_S),
        np.ascontiguousarray(flow_H),
        gp, P_E, P_U, P_N,
        lam, ly, delta, beta,
        float(pi_i),
        int(p.vfi_max_iter),
        int(reentry_code),
        W0c, S0c, H0c,
        float(rho), Tc_arr,
        1 if getattr(p, 'clean_u', True) else 0,
        float(eta_act))


def find_cutoffs_2d(W: np.ndarray, S: np.ndarray, H: np.ndarray,
                    x_grid: np.ndarray) -> tuple:
    """Cutoff index and x-value per k where W=H (quit), S=H (participate), W=S (accept)."""
    n_x, n_k = W.shape

    def _cut(V1, V2):
        idxs = np.empty(n_k, dtype=int); vals = np.empty(n_k)
        for k in range(n_k):
            diff = V1[:, k] - V2[:, k]
            if diff.max() <= 0.:
                idxs[k] = 1;    vals[k] = x_grid[0]
            elif diff.min() >= 0.:
                idxs[k] = n_x;  vals[k] = x_grid[-1]
            else:
                idx = int(np.argmin(np.abs(diff)))
                idxs[k] = idx;  vals[k] = x_grid[idx]
        return idxs, vals

    iq, xq = _cut(W, H)
    in_, xn = _cut(S, H)
    ia, xa  = _cut(W, S)
    return iq, xq, in_, xn, ia, xa


def nash_bargaining_2d(i: int, p: ExtParameters, g: dict, hc: dict,
                       WWW, SSS, HHH,
                       ii_q_all, ii_a_all, int_x_q_all, int_x_n_all, int_x_a_all,
                       k_init_dist: np.ndarray) -> tuple:
    """Nash bargaining over the wage grid for male type i, with surplus averaged over k_init_dist."""
    n_w  = g['n_w']
    n_x  = g['n_x']
    n_k  = hc['n_k']
    gp   = g['gp_grid'][:, i]

    lam   = p.lambda_grid[i]
    ly    = getattr(p, 'lambda_y', np.zeros(8))[i] if hasattr(p,'lambda_y') else 0.
    delta = p.delta[i]
    beta  = p.beta
    gamma = p.gamma

    barg_obj = np.zeros(n_w)
    JJ_arr   = np.zeros(n_w)
    int_JJ_  = np.zeros(n_w)

    for j in range(n_w):
        w_rate = g['w_grid'][j]

        # Firm value J(k) = y(k)*(1 - w_rate)/denom; w_grid holds wage rates (share of output).

        J_k = np.zeros(n_k)
        for ki in range(n_k):
            y_k    = p.y_hl[i] * hc['k_grid'][ki]
            iq     = ii_q_all[j, i, ki]
            ia     = ii_a_all[j, i, ki]
            cut    = min(int(iq), int(ia))
            QJ     = gp[:cut].sum() if cut > 1 else 0.
            denom  = max(1. - beta*(1.-delta)*(lam*(1.-ly)*QJ + (1.-lam)*(1.-ly)), 1e-10)
            J_k[ki] = max(0., y_k * (1. - g['w_grid'][j]) / denom)

        JJ_arr[j]  = k_init_dist @ J_k
        int_JJ_[j] = (k_init_dist * np.array([
            gp[:min(int(ii_q_all[j,i,ki]), int(ii_a_all[j,i,ki]))].sum()
            if min(int(ii_q_all[j,i,ki]), int(ii_a_all[j,i,ki])) > 1 else 0.
            for ki in range(n_k)])).sum()

        int_bo = 0.
        for ki in range(n_k):
            cut = min(int(ii_q_all[j, i, ki]), int(ii_a_all[j, i, ki]))
            if cut > 1:
                mask = np.zeros(n_x); mask[:cut] = 1.
                Wt = WWW[:, i, ki, j]; St = SSS[:, i, ki, j]; Ht = HHH[:, i, ki, j]
                int_bo += k_init_dist[ki] * (mask * gp * np.maximum(0., Wt - np.maximum(St, Ht))).sum()

        barg_obj[j] = (int_bo ** gamma) * (JJ_arr[j] ** (1. - gamma))

    j_opt = int(np.argmax(barg_obj))
    if 1 <= j_opt <= n_w - 2:
        dw         = g['w_grid'][1] - g['w_grid'][0]
        fa, fb, fc = barg_obj[j_opt - 1], barg_obj[j_opt], barg_obj[j_opt + 1]
        curv       = fa - 2. * fb + fc      # negative => concave (valid max)
        if curv < -1e-14:
            x_star = float(np.clip(-0.5 * (fc - fa) / curv, -1., 1.))
            w_opt  = float(g['w_grid'][j_opt] + x_star * dw)
            j_lo   = j_opt if x_star >= 0. else j_opt - 1
            t      = x_star if x_star >= 0. else x_star + 1.
            t1     = 1. - t
            return (w_opt,
                    t1 * WWW[:, i, :, j_lo]       + t * WWW[:, i, :, j_lo + 1],
                    t1 * SSS[:, i, :, j_lo]       + t * SSS[:, i, :, j_lo + 1],
                    t1 * HHH[:, i, :, j_lo]       + t * HHH[:, i, :, j_lo + 1],
                    t1 * JJ_arr[j_lo]             + t * JJ_arr[j_lo + 1],
                    t1 * int_JJ_[j_lo]            + t * int_JJ_[j_lo + 1],
                    t1 * int_x_q_all[j_lo, i, :] + t * int_x_q_all[j_lo + 1, i, :],
                    t1 * int_x_n_all[j_lo, i, :] + t * int_x_n_all[j_lo + 1, i, :],
                    t1 * int_x_a_all[j_lo, i, :] + t * int_x_a_all[j_lo + 1, i, :])

    # Boundary or non-concave optimum: return exact grid values.
    return (float(g['w_grid'][j_opt]),
            WWW[:, i, :, j_opt], SSS[:, i, :, j_opt], HHH[:, i, :, j_opt],
            JJ_arr[j_opt], int_JJ_[j_opt],
            int_x_q_all[j_opt, i, :], int_x_n_all[j_opt, i, :], int_x_a_all[j_opt, i, :])


def solve_worker_shares_ext(p: ExtParameters, g: dict, hc: dict,
                             int_x_a_p, int_x_n_p, int_x_q_p,
                             p_vec: np.ndarray) -> dict:
    """Stationary E, U, N by type and HC node, with child-status and HC transitions."""
    n_types = 8
    n_k     = hc['n_k']
    P_E_by_type = hc['P_E_by_type']
    P_U_by_type = hc['P_U_by_type']
    P_N_by_type = hc['P_N_by_type']
    k_init  = hc['k_init_idx']

    rho_enter = p.rho_enter   # no-child -> child-present
    rho_exit  = p.rho_exit    # child-present -> no-child
    rho_death = p.rho_death   # any -> reborn no-child at k_init

    # Cutoff integrals by (type, HC node): acceptance and participation.
    int_a = int_x_a_p   # (n_types, n_k)
    int_n = int_x_n_p   # (n_types, n_k)

    def _flow_probs(i, ki):
        lam   = p.lambda_grid[i]
        delta = p.delta[i]
        pi_i  = p_vec[i]
        ia    = int_a[i, ki]
        in_   = int_n[i, ki]

        # Probabilities conditional on the source state; each row sums to one.
        EE_ = (1.-delta) * (lam*ia + 1.-lam)          # survives sep + prefers E
        EU_ = (1.-delta)*lam*(in_-ia) + delta*(lam*in_ + 1.-lam)  # goes to U
        EN_ = 1. - EE_ - EU_                               # residual goes to N

        UE_ = pi_i * ia
        UN_ = 1. - in_
        UU_ = 1. - UE_ - UN_     # = in_ - pi_i*ia

        if p.reentry_mode in ('activation_off', 'activation_off_ge'):
            NE_ = 0.
            NU_ = in_
            NN_ = 1. - in_
        elif p.reentry_mode == 'frictionless':
            NE_ = ia
            NU_ = in_ - ia
            NN_ = 1. - in_
        elif p.reentry_mode == 'matching_off':
            # Direct N->E on the lam-shock if she would accept; this mode also changes the VFI.
            NE_ = lam * ia
            NU_ = lam * (in_ - ia)
            NN_ = 1. - lam * in_
        else:  # 'baseline'
            NE_ = 0.
            NU_ = lam * in_
            NN_ = 1. - NU_
        return dict(EE=EE_, EU=EU_, EN=EN_, UE=UE_, UU=UU_, UN=UN_,
                    NE=NE_, NU=NU_, NN=NN_)

    Ns = n_types * n_k * 3

    def _idx(i, ki, s):
        return (i * n_k + ki) * 3 + s

    T = np.zeros((Ns, Ns))
    surv = 1. - rho_death
    for i in range(n_types):
        g          = i % 4
        is_child   = (i < 4)
        partner    = i + 4 if is_child else i - 4
        move       = (p.rho_exit_by_skill[g // 2] if is_child else rho_enter)
        no_child_g = 4 + g
        for ki in range(n_k):
            fl = _flow_probs(i, ki)
            # Per-type HC matrices (skill-specific rates).
            pE = P_E_by_type[i][ki]; pU = P_U_by_type[i][ki]; pN = P_N_by_type[i][ki]
            # Per source labour state (0=E, 1=U, 2=N): (dest state, flow prob, HC lottery row).
            src_flows = (
                ((0, fl['EE'], pE), (1, fl['EU'], pE), (2, fl['EN'], pE)),
                ((0, fl['UE'], pU), (1, fl['UU'], pU), (2, fl['UN'], pU)),
                ((0, fl['NE'], pN), (1, fl['NU'], pN), (2, fl['NN'], pN)),
            )
            for s in range(3):
                src = _idx(i, ki, s)
                T[_idx(no_child_g, k_init, 1), src] += rho_death   # death -> reborn no-child, k_init, U
                for ds, fp, prow in src_flows[s]:
                    coeff = surv * fp
                    for kp in range(n_k):
                        w = coeff * prow[kp]
                        T[_idx(i,       kp, ds), src] += (1. - move) * w  # stay fecundity state
                        T[_idx(partner, kp, ds), src] += move        * w  # switch fecundity state
    mu_flat = np.zeros(Ns)
    for grp in range(4):
        idxs = np.array([_idx(t, ki, s) for t in (grp, grp + 4)
                         for ki in range(n_k) for s in range(3)])
        Tg = T[np.ix_(idxs, idxs)]
        Ag = Tg - np.eye(len(idxs)); Ag[0, :] = 1.0
        bg = np.zeros(len(idxs)); bg[0] = 1.0
        mug = np.maximum(np.linalg.solve(Ag, bg), 0.)
        mu_flat[idxs] = mug / mug.sum() * 0.25            # 1/4 of total mass per group
    mu = mu_flat.reshape(n_types, n_k, 3)
    mu /= mu.sum()

    E_mat = mu[:, :, 0]   # (n_types, n_k)
    U_mat = mu[:, :, 1]
    N_mat = mu[:, :, 2]

    E_type = E_mat.sum(axis=1)   # (n_types,)
    U_type = U_mat.sum(axis=1)
    N_type = N_mat.sum(axis=1)

    tot = E_type + U_type + N_type
    tot = np.maximum(tot, 1e-12)
    E_share = E_type / tot
    U_share = U_type / tot
    N_share = N_type / tot

    flows_agg = {}
    for key in ('EE','EU','EN','UE','UU','UN','NE','NU','NN'):
        vals = np.zeros(n_types)
        for i in range(n_types):
            w_E   = E_mat[i, :].sum()
            w_U   = U_mat[i, :].sum()
            w_N   = N_mat[i, :].sum()
            w_tot = w_E + w_U + w_N
            if w_tot < 1e-12:
                vals[i] = 0.
                continue
            acc = 0.
            for ki in range(n_k):
                fl = _flow_probs(i, ki)
                src_state = key[0]  # 'E', 'U', or 'N'
                wgt = (E_mat[i,ki] if src_state=='E'
                       else U_mat[i,ki] if src_state=='U'
                       else N_mat[i,ki])
                acc += wgt * fl[key]
            # Divide by source-state mass to get a conditional rate.
            src_state = key[0]
            if src_state == 'E':
                denom = max(w_E, 1e-12)
            elif src_state == 'U':
                denom = max(w_U, 1e-12)
            else:
                denom = max(w_N, 1e-12)
            vals[i] = acc / denom
        flows_agg[key] = vals

    return dict(E=E_share, U=U_share, N=N_share,
                E_mat=E_mat, U_mat=U_mat, N_mat=N_mat,
                **flows_agg)


def misclassify_ext(i: int, E: float, U: float, N: float,
                    flows: dict, p: ExtParameters) -> tuple:
    qeu = p.q_eu[i]; qen = p.q_en[i]
    que = p.q_ue[i]; qun = p.q_un[i]
    qne = p.q_ne[i]; qnu = p.q_nu[i]

    E_h = E*(1.-qeu-qen) + U*que       + N*qne
    U_h = E*qeu          + U*(1.-que-qun) + N*qnu
    N_h = E*qen          + U*qun        + N*(1.-qne-qnu)

    EE = flows['EE'][i]; EU = flows['EU'][i]; EN = flows['EN'][i]
    UE = flows['UE'][i]; UU = flows['UU'][i]; UN = flows['UN'][i]
    NE = flows['NE'][i]; NU = flows['NU'][i]; NN = flows['NN'][i]

    M1 = np.array([
        [E*(1.-qeu-qen)/max(E_h,1e-12), U*que/max(E_h,1e-12), N*qne/max(E_h,1e-12)],
        [E*qeu/max(U_h,1e-12), U*(1.-que-qun)/max(U_h,1e-12), N*qnu/max(U_h,1e-12)],
        [E*qen/max(N_h,1e-12), U*qun/max(N_h,1e-12), N*(1.-qne-qnu)/max(N_h,1e-12)],
    ])
    M2 = np.array([
        [EE*(1.-qeu-qen)+EU*que+EN*qne, EU*(1.-que-qun)+EN*qnu+EE*qeu, EN*(1.-qne-qnu)+EU*qun+EE*qen],
        [UE*(1.-qeu-qen)+UU*que+UN*qne, UU*(1.-que-qun)+UN*qnu+UE*qeu, UN*(1.-qne-qnu)+UU*qun+UE*qen],
        [NE*(1.-qeu-qen)+NU*que+NN*qne, NU*(1.-que-qun)+NN*qnu+NE*qeu, NN*(1.-qne-qnu)+NU*qun+NE*qen],
    ])
    M3 = M1 @ M2
    return (E_h, U_h, N_h,
            M3[0,0], M3[0,1], M3[0,2],
            M3[1,0], M3[1,1], M3[1,2],
            M3[2,0], M3[2,1], M3[2,2])


def run_extended_model(p: ExtParameters = None, verbose: bool = False) -> dict:
    """Solve the extended model; returns equilibrium objects and model moments."""
    if p is None:
        p = ExtParameters()

    g   = build_grids(p)
    hc  = build_hc_grids(p)
    n_types = 8
    n_w = g['n_w']; n_x = g['n_x']; n_k = hc['n_k']

    # New workers start at k_init.
    k_init_dist = np.zeros(n_k)
    k_init_dist[hc['k_init_idx']] = 1.

    theta   = np.ones(n_types)
    theta_p = 2. * np.ones(n_types)

    # Storage: W, S, H indexed (x, type, k, wage)
    WWW = np.zeros((n_x, n_types, n_k, n_w))
    SSS = np.zeros((n_x, n_types, n_k, n_w))
    HHH = np.zeros((n_x, n_types, n_k, n_w))

    ii_q_all  = np.ones( (n_w, n_types, n_k), dtype=int)
    ii_n_all  = np.ones( (n_w, n_types, n_k), dtype=int)
    ii_a_all  = np.ones( (n_w, n_types, n_k), dtype=int)
    int_x_q_all = np.zeros((n_w, n_types, n_k))
    int_x_n_all = np.zeros((n_w, n_types, n_k))
    int_x_a_all = np.zeros((n_w, n_types, n_k))

    iter_n = 0
    while np.sum(np.abs(theta_p - theta)) > 1e-3:
        iter_n += 1
        if iter_n > p.max_theta_iter:
            if verbose: print("Theta loop stopped early.")
            break

        theta = theta_p.copy()
        # Matching efficiency by type: mu_vec[group] if positive, else scalar mu.
        mu_eff = np.array([
            (p.mu_vec[i % 4] if (len(p.mu_vec) > i % 4 and p.mu_vec[i % 4] > 0.) else p.mu)
            for i in range(n_types)
        ])
        # Cap at 1: the job-finding rate is a probability.
        p_vec = np.clip(mu_eff * theta ** (1. - p.alpha), 0., 1.)

        if verbose:
            print(f"  theta iter {iter_n}: theta={theta.round(3)}, p={p_vec.round(3)}")

        # VFI for all (wage, type) pairs in parallel; each pair writes its own output slice.
        futures = []
        n_workers = min(_N_VFI_WORKERS, n_w * n_types)
        with ThreadPoolExecutor(max_workers=n_workers) as _pool:
            for j in range(n_w):
                for i in range(n_types):
                    phi_c_i = p.phi_c if i in (1, 3) else 0.
                    worker_w_rate = g['w_grid'][j]
                    if i in (1, 3, 5, 7):   # female types
                        worker_w_rate *= p.tau_w_f

                    # Warm start from the previous tightness iteration (cold start on the first).
                    W0 = WWW[:, i, :, j] if iter_n > 1 else None
                    S0 = SSS[:, i, :, j] if iter_n > 1 else None
                    H0 = HHH[:, i, :, j] if iter_n > 1 else None

                    # Anticipated child-status switch: rho_i is her switch rate, ic the counterpart type.
                    if i < 4:
                        ic = i + 4
                        rho_i = float(p.rho_exit_by_skill[i // 2])
                    else:
                        ic = i - 4
                        rho_i = float(p.rho_enter)
                    # no_worker_fertility: the worker ignores her own switch; the distribution is unchanged.
                    if getattr(p, 'no_worker_fertility', False):
                        rho_i = 0.0
                    Tc = (np.maximum(np.maximum(WWW[:, ic, :, j], SSS[:, ic, :, j]),
                                     HHH[:, ic, :, j]) if iter_n > 1 else None)
                    futures.append(_pool.submit(
                        _vfi_and_cutoffs,
                        j, i,
                        g['w_grid'][j],          # gross wage rate for bargaining grid
                        worker_w_rate,           # net worker wage rate for VFI
                        g['b_grid'][j], g['I_grid'][j],
                        float(p_vec[i]), float(phi_c_i),
                        p, g, hc, W0, S0, H0, rho_i, Tc))
        for fut in futures:
            j, i, W2d, S2d, H2d, iq, in_, ia, ix_q, ix_n, ix_a = fut.result()
            WWW[:, i, :, j]    = W2d
            SSS[:, i, :, j]    = S2d
            HHH[:, i, :, j]    = H2d
            ii_q_all[j, i, :]  = iq
            ii_a_all[j, i, :]  = ia
            ii_n_all[j, i, :]  = in_
            int_x_q_all[j, i, :] = ix_q
            int_x_n_all[j, i, :] = ix_n
            int_x_a_all[j, i, :] = ix_a

        w_nash    = np.zeros(n_types)
        J_p       = np.zeros(n_types)
        int_J_p   = np.zeros(n_types)
        int_x_q_p = np.zeros((n_types, n_k))
        int_x_n_p = np.zeros((n_types, n_k))
        int_x_a_p = np.zeros((n_types, n_k))

        for i in [0, 2, 4, 6]:   # male types (fertile + infertile)
            (w_opt, W_o, S_o, H_o, J_o, intJ_o,
             ixq, ixn, ixa) = nash_bargaining_2d(
                i, p, g, hc, WWW, SSS, HHH,
                ii_q_all, ii_a_all, int_x_q_all, int_x_n_all, int_x_a_all,
                k_init_dist)
            w_nash[i]        = w_opt
            J_p[i]           = J_o
            int_J_p[i]       = intJ_o
            int_x_q_p[i, :]  = ixq
            int_x_n_p[i, :]  = ixn
            int_x_a_p[i, :]  = ixa

        beta   = p.beta
        rho_en = p.rho_enter
        for grp in (0, 1):                 # 0 = unskilled women, 1 = college women
            rho_ex = float(p.rho_exit_by_skill[grp])
            i_c   = 2 * grp + 1            # child-present female type (1, 3)
            i_n   = i_c + 4               # no-child female type (5, 7)
            lm_c  = 2 * grp              # male child type (0, 2)
            lm_n  = lm_c + 4            # male no-child type (4, 6)
            lam_c, lam_n = p.lambda_grid[i_c], p.lambda_grid[i_n]
            a_c,  a_n  = beta * (1. - p.delta[i_c]), beta * (1. - p.delta[i_n])
            gp_c, gp_n = g['gp_grid'][:, i_c], g['gp_grid'][:, i_n]

            JJ_c = np.zeros(n_w); JJ_n = np.zeros(n_w)
            for j in range(n_w):
                Jc_k = np.zeros(n_k); Jn_k = np.zeros(n_k)
                for ki in range(n_k):
                    y_k  = p.y_hl[i_c] * hc['k_grid'][ki]   # same productivity both states
                    cutc = min(int(ii_q_all[j, i_c, ki]), int(ii_a_all[j, i_c, ki]))
                    cutn = min(int(ii_q_all[j, i_n, ki]), int(ii_a_all[j, i_n, ki]))
                    QJc  = gp_c[:cutc].sum() if cutc > 1 else 0.
                    QJn  = gp_n[:cutn].sum() if cutn > 1 else 0.
                    Sc   = lam_c * QJc + (1. - lam_c)
                    Sn   = lam_n * QJn + (1. - lam_n)
                    fc   = y_k * (1. - g['w_grid'][j])
                    # 2x2 firm-value system  [[m11,m12],[m21,m22]] [Jc,Jn]^T = [fc,fc]^T
                    m11 = 1. - a_c * Sc * (1. - rho_ex); m12 = -a_c * Sc * rho_ex
                    m21 = -a_n * Sn * rho_en;            m22 = 1. - a_n * Sn * (1. - rho_en)
                    det = m11 * m22 - m12 * m21
                    if abs(det) < 1e-12:
                        Jc_k[ki] = max(0., fc); Jn_k[ki] = max(0., fc)
                    else:
                        Jc_k[ki] = max(0., fc * (m22 - m12) / det)
                        Jn_k[ki] = max(0., fc * (m11 - m21) / det)
                JJ_c[j] = k_init_dist @ Jc_k
                JJ_n[j] = k_init_dist @ Jn_k

            y_k_avg = float((p.y_hl[i_c] * hc['k_grid']) @ k_init_dist)
            # Wage floor as a rate: max(b_frac, w_min) divided by average output.
            floor_dollar = max(p.b_frac, p.w_min)
            w_floor = float(np.clip(floor_dollar / max(y_k_avg, 1e-6), 0., 1.))

            # Women's wage solves JJ_f(w_f) = J_m of the man with the same child status.
            for i_f, JJ_f, JJ_m in ((i_c, JJ_c, J_p[lm_c]), (i_n, JJ_n, J_p[lm_n])):
                if JJ_m >= JJ_f[0]:
                    w_f_cont = float(g['w_grid'][0])
                elif JJ_m <= JJ_f[-1]:
                    w_f_cont = float(g['w_grid'][-1])
                else:
                    jx = 1
                    while jx < n_w - 1 and JJ_f[jx] > JJ_m:
                        jx += 1
                    f0, f1 = JJ_f[jx - 1], JJ_f[jx]
                    frac = (f0 - JJ_m) / max(f0 - f1, 1e-12)
                    w_f_cont = float(g['w_grid'][jx - 1] + frac * (g['w_grid'][jx] - g['w_grid'][jx - 1]))
                w_f_cont = max(w_f_cont, w_floor)          # apply reservation-wage floor
                j_lo_f = int(np.clip(int(np.searchsorted(g['w_grid'], w_f_cont, side='right')) - 1, 0, n_w - 2))
                dw_f = float(g['w_grid'][j_lo_f + 1] - g['w_grid'][j_lo_f])
                t_f  = float(np.clip((w_f_cont - g['w_grid'][j_lo_f]) / max(dw_f, 1e-10), 0., 1.)); t1_f = 1. - t_f
                w_nash[i_f]       = w_f_cont
                # Firm value at the actual (possibly floored) wage; below J_m when the floor binds.
                J_p[i_f]          = float(t1_f * JJ_f[j_lo_f] + t_f * JJ_f[j_lo_f + 1])
                int_x_q_p[i_f, :] = t1_f * int_x_q_all[j_lo_f, i_f, :] + t_f * int_x_q_all[j_lo_f + 1, i_f, :]
                int_x_n_p[i_f, :] = t1_f * int_x_n_all[j_lo_f, i_f, :] + t_f * int_x_n_all[j_lo_f + 1, i_f, :]
                int_x_a_p[i_f, :] = t1_f * int_x_a_all[j_lo_f, i_f, :] + t_f * int_x_a_all[j_lo_f + 1, i_f, :]

        # Damped fixed-point update of theta, clipped to [1e-4, 1e4].
        _THETA_DAMP = 0.5
        skill_members = [[0, 1, 4, 5], [2, 3, 6, 7]]   # unskilled, skilled (type idx)
        for members in skill_members:
            Jbar  = float(np.mean([max(J_p[i], 0.) for i in members]))
            cc_sk = float(p.cc[members[0]])
            target = (Jbar * p.beta / cc_sk) ** (1. / p.alpha)
            target = min(max(target, 1e-4), 1e4)
            for i in members:
                theta_p[i] = (1. - _THETA_DAMP) * theta[i] + _THETA_DAMP * target

    # Dollar wage by HC node: w_rate * worker wedge * y(k).
    worker_wedge = np.array([p.tau_w_f if i in (1, 3, 5, 7) else 1.0
                             for i in range(n_types)])
    wage_dollar_by_k = np.vstack([
        (w_nash[i] * worker_wedge[i]) * p.y_hl[i] * hc['k_grid']
        for i in range(n_types)
    ])
    # Dollar wage at entry HC k_init.
    k_init_val = hc['k_grid'][hc['k_init_idx']]
    w_dollar = np.array([
        (w_nash[i] * worker_wedge[i]) * p.y_hl[i] * k_init_val
        for i in range(n_types)
    ])

    ss = solve_worker_shares_ext(p, g, hc, int_x_a_p, int_x_n_p, int_x_q_p, p_vec)

    # Misclassified (reported) shares and flows.
    E_hat = np.zeros(n_types); U_hat = np.zeros(n_types); N_hat = np.zeros(n_types)
    EE_hat = np.zeros(n_types); EU_hat = np.zeros(n_types); EN_hat = np.zeros(n_types)
    UE_hat = np.zeros(n_types); UU_hat = np.zeros(n_types); UN_hat = np.zeros(n_types)
    NE_hat = np.zeros(n_types); NU_hat = np.zeros(n_types); NN_hat = np.zeros(n_types)

    for i in range(n_types):
        (E_hat[i], U_hat[i], N_hat[i],
         EE_hat[i], EU_hat[i], EN_hat[i],
         UE_hat[i], UU_hat[i], UN_hat[i],
         NE_hat[i], NU_hat[i], NN_hat[i]) = misclassify_ext(
            i, ss['E'][i], ss['U'][i], ss['N'][i], ss, p)

    model_lfp   = U_hat + E_hat
    model_unemp = U_hat / np.maximum(U_hat + E_hat, 1e-12)
    moment = np.concatenate([
        (model_lfp   - p.data_lfp)   ** 2,
        (model_unemp - p.data_unemp) ** 2,
        (UE_hat      - p.data_UE)    ** 2,
    ])
    momfun = np.sqrt(moment.mean())

    # Gender pay gap: ratio of employed-weighted mean male to female wage, all child statuses.
    employed_mass = ss['E_mat'].sum(axis=1)
    avg_wage_emp = np.array([
        (ss['E_mat'][i, :] @ wage_dollar_by_k[i, :]) / max(employed_mass[i], 1e-12)
        for i in range(n_types)
    ])
    male_idxs   = np.array([0, 2, 4, 6])
    female_idxs = np.array([1, 3, 5, 7])
    male_wage = employed_mass[male_idxs] @ avg_wage_emp[male_idxs] / max(employed_mass[male_idxs].sum(), 1e-12)
    female_wage = employed_mass[female_idxs] @ avg_wage_emp[female_idxs] / max(employed_mass[female_idxs].sum(), 1e-12)
    gpg = male_wage / max(female_wage, 1e-12)

    avg_k_E = np.array([(ss['E_mat'][i,:] @ hc['k_grid']) / max(ss['E_mat'][i,:].sum(),1e-12)
                         for i in range(n_types)])
    avg_k_U = np.array([(ss['U_mat'][i,:] @ hc['k_grid']) / max(ss['U_mat'][i,:].sum(),1e-12)
                         for i in range(n_types)])

    # Per-type earnings: employment times wage over the HC distribution (true, not reported, shares).
    type_mass   = ss['E'] + ss['U'] + ss['N']
    emp_rate    = ss['E'] / np.maximum(type_mass, 1e-12)
    earnings_pc = np.array([
        (ss['E_mat'][i, :] @ wage_dollar_by_k[i, :]) / max(type_mass[i], 1e-12)
        for i in range(n_types)
    ])

    # Entry value V_entry = E_x[S(x, k_init)] at the equilibrium wage, interpolated over the wage grid.
    w_grid = g['w_grid']; ki0 = hc['k_init_idx']; gp_all = g['gp_grid']
    V_entry = np.zeros(n_types)
    for i in range(n_types):
        j_hi = min(max(int(np.searchsorted(w_grid, w_nash[i])), 1), n_w - 1)
        j_lo = j_hi - 1
        t = np.clip((w_nash[i] - w_grid[j_lo]) / max(w_grid[j_hi] - w_grid[j_lo], 1e-12), 0., 1.)
        S_entry = (1. - t) * SSS[:, i, ki0, j_lo] + t * SSS[:, i, ki0, j_hi]
        V_entry[i] = float(gp_all[:, i] @ S_entry)

    return dict(
        momfun=momfun, w_nash=w_nash, w_dollar=w_dollar, theta=theta, p_vec=p_vec,
        E=ss['E'], U=ss['U'], N=ss['N'],
        E_hat=E_hat, U_hat=U_hat, N_hat=N_hat,
        EE_hat=EE_hat, EU_hat=EU_hat, EN_hat=EN_hat,
        UE_hat=UE_hat, UU_hat=UU_hat, UN_hat=UN_hat,
        NE_hat=NE_hat, NU_hat=NU_hat, NN_hat=NN_hat,
        model_lfp=model_lfp, model_unemp=model_unemp,
        gender_pay_gap=gpg,
        emp_rate=emp_rate, earnings_pc=earnings_pc, avg_wage_emp=avg_wage_emp,
        V_entry=V_entry,
        avg_k_E=avg_k_E, avg_k_U=avg_k_U,
        k_grid=hc['k_grid'],
        int_x_a_p=int_x_a_p, int_x_n_p=int_x_n_p, int_x_q_p=int_x_q_p,
        g=g, hc=hc, wage_dollar_by_k=wage_dollar_by_k, p_vec_full=p_vec,
    )


if __name__ == '__main__':
    import time
    labels = ['UnM-F','UnF-F','SkM-F','SkF-F','UnM-I','UnF-I','SkM-I','SkF-I']
    t0 = time.time()
    res = run_extended_model(verbose=True)
    print(f"\nConverged in {time.time()-t0:.1f}s")
    print(f"Objective:   {res['momfun']:.6f}")
    print(f"Nash wages:  {dict(zip(labels, res['w_nash'].round(4)))}")
    print(f"Gender pay gap (fertile):  {res['gender_pay_gap']:.4f}")
    print(f"Avg HC (employed):  {dict(zip(labels, res['avg_k_E'].round(3)))}")
    print(f"Avg HC (unemployed):{dict(zip(labels, res['avg_k_U'].round(3)))}")
    print(f"LFP:    {res['model_lfp'].round(4)}")
    print(f"Unemp:  {res['model_unemp'].round(4)}")
