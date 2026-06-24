"""Constant-mu bivariate sum-of-exponentials Hawkes calibration (single window).

Two timescales (fast beta1=100, slow beta2=1) and a 2x2 buy/sell excitation.
`params` are log-parameters (10): [mu_B, mu_S, then 8 kernel alphas BB1,SS1,BS1,SB1,
BB2,SS2,BS2,SB2]. fit_window_soe_fast returns the kernel norms phi_* = alpha/beta and
the branching ratio eta (spectral radius of the phi matrix).

Be careful when modifying soe_loglik_fast as tiny numerical differences can change the path taken by the L-BFGS-B optimizer and lead into a different local optimum."""

import numpy as np
from numba import njit
from scipy.optimize import minimize


@njit(nogil=True)
def soe_loglik_fast(params, times, types, T_B, T_S, beta1, beta2):
    # exp of clipped log-params: baselines mu_B/mu_S, then the 8 kernel alphas
    mu_B  = np.exp(min(max(params[0], -15), 15))
    mu_S  = np.exp(min(max(params[1], -15), 15))
    a_BB1 = np.exp(min(max(params[2], -15), 15))
    a_SS1 = np.exp(min(max(params[3], -15), 15))
    a_BS1 = np.exp(min(max(params[4], -15), 15))
    a_SB1 = np.exp(min(max(params[5], -15), 15))
    a_BB2 = np.exp(min(max(params[6], -15), 15))
    a_SS2 = np.exp(min(max(params[7], -15), 15))
    a_BS2 = np.exp(min(max(params[8], -15), 15))
    a_SB2 = np.exp(min(max(params[9], -15), 15))

    A_B1 = A_S1 = A_B2 = A_S2 = 0.0
    last_t = 0.0
    loglik = 0.0

    for i in range(len(times)):
        t     = times[i]
        etype = types[i]
        dt    = t - last_t
        if dt > 0.0:
            d1 = np.exp(-beta1 * dt)
            d2 = np.exp(-beta2 * dt)
            A_B1 *= d1;  A_S1 *= d1
            A_B2 *= d2;  A_S2 *= d2

        if etype == 0.0:
            lam = mu_B + a_BB1*A_B1 + a_SB1*A_S1 + a_BB2*A_B2 + a_SB2*A_S2
            if lam <= 1e-300:
                return 1e10
            loglik += np.log(lam)
            A_B1 += 1.0;  A_B2 += 1.0
        else:
            lam = mu_S + a_BS1*A_B1 + a_SS1*A_S1 + a_BS2*A_B2 + a_SS2*A_S2
            if lam <= 1e-300:
                return 1e10
            loglik += np.log(lam)
            A_S1 += 1.0;  A_S2 += 1.0
        last_t = t

    T_end = times[-1]
    for i in range(len(T_B)):
        v1 = 1.0 - np.exp(-beta1*(T_end - T_B[i]))
        v2 = 1.0 - np.exp(-beta2*(T_end - T_B[i]))
        loglik -= (a_BB1/beta1)*v1 + (a_BB2/beta2)*v2
        loglik -= (a_BS1/beta1)*v1 + (a_BS2/beta2)*v2
    for i in range(len(T_S)):
        v1 = 1.0 - np.exp(-beta1*(T_end - T_S[i]))
        v2 = 1.0 - np.exp(-beta2*(T_end - T_S[i]))
        loglik -= (a_SS1/beta1)*v1 + (a_SS2/beta2)*v2
        loglik -= (a_SB1/beta1)*v1 + (a_SB2/beta2)*v2
    loglik -= (mu_B + mu_S) * T_end

    return -loglik


def fit_window_soe_fast(T_B, T_S, beta1=100.0, beta2=1.0, n_inits=3):
    """Multi-start MLE of the constant-mu Hawkes model on one window.

    T_B/T_S are buy/sell event times in seconds. Returns a dict of fitted baselines
    (mu_B, mu_S), kernel norms (phi_*), branching ratio (eta) and log-likelihood, or
    None if the window is too sparse (< 20 events per side)."""
    if len(T_B) < 20 or len(T_S) < 20:
        return None

    times = np.concatenate([T_B, T_S])
    types = np.concatenate([np.zeros(len(T_B)), np.ones(len(T_S))])
    order = np.argsort(times, kind='stable')
    times = times[order].astype(np.float64)
    types = types[order].astype(np.float64)
    T_B   = np.sort(T_B).astype(np.float64)
    T_S   = np.sort(T_S).astype(np.float64)

    def obj(params):
        return soe_loglik_fast(params, times, types, T_B, T_S, beta1, beta2)

    rate_B  = len(T_B) / (T_B[-1] - T_B[0] + 1e-6)
    rate_S  = len(T_S) / (T_S[-1] - T_S[0] + 1e-6)
    x0_base = np.array([np.log(rate_B*0.1), np.log(rate_S*0.1),
                         np.log(0.2), np.log(0.2), np.log(0.05), np.log(0.05),
                         np.log(0.2), np.log(0.2), np.log(0.05), np.log(0.05)])

    best_val, best_x = np.inf, None
    for i in range(n_inits):
        noise = np.random.randn(10)*0.5 if i > 0 else np.zeros(10)
        x0 = x0_base + noise
        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                           options={'maxiter': 300})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_x = res.fun, res.x
        except:
            continue

    if best_x is None:
        return None

    p = np.clip(best_x, -15, 15)
    a_BB1,a_SS1,a_BS1,a_SB1 = np.exp(p[2]),np.exp(p[3]),np.exp(p[4]),np.exp(p[5])
    a_BB2,a_SS2,a_BS2,a_SB2 = np.exp(p[6]),np.exp(p[7]),np.exp(p[8]),np.exp(p[9])
    phi_mat = (np.array([[a_BB1,a_SB1],[a_BS1,a_SS1]])/beta1 +
               np.array([[a_BB2,a_SB2],[a_BS2,a_SS2]])/beta2)
    eta = float(np.max(np.abs(np.linalg.eigvals(phi_mat))))

    return {
        'mu_B': float(np.exp(p[0])), 'mu_S': float(np.exp(p[1])), #bug fix: return mue so that GOF doesn't use default values. 
        'phi_BB1': a_BB1/beta1, 'phi_SS1': a_SS1/beta1,
        'phi_BS1': a_BS1/beta1, 'phi_SB1': a_SB1/beta1,
        'phi_BB2': a_BB2/beta2, 'phi_SS2': a_SS2/beta2,
        'phi_BS2': a_BS2/beta2, 'phi_SB2': a_SB2/beta2,
        'eta': eta, 'loglik': -best_val
    }
