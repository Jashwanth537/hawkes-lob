"""
Piecewise-linear (spline) time-varying baseline μ(t) for the SOE Hawkes model.

The baseline intensity is represented as a continuous linear spline with S+1
equally spaced knots over the observation window. Knot values are constrained
to be non-negative and are optimized in log space.

This parameterization was introduced for the Phase 1b baseline-flexibility
study, providing a smooth, data-driven baseline instead of arbitrary
piecewise-constant bins. The motivation follows prior work showing the
importance of separating genuine self-excitation from slowly varying
background activity.

The baseline contribution μ(t_i) enters the log-intensity as usual. Since
μ(t) is linear between knots, the baseline compensator ∫₀ᵀ μ(t)dt is computed
exactly using the trapezoidal rule. The excitation kernel (8 exponentials with
β = [100, 1]) and its compensator remain identical to the implementations in
model.py and timevarying.py.

Validation:
    test_reduces_to_constant verifies that when all knot values are equal,
    the spline collapses to a constant baseline and the resulting log-likelihood
    matches the constant-μ implementation (src.model.soe_loglik_fast) to
    machine precision. Exact bitwise equality is not expected because separate
    Numba-compiled functions may generate slightly different floating-point
    instruction sequences.
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize

_LOG_CLIP = 15.0
_MIN_EVENTS = 50
_KERNEL_X0 = np.array([0.2, 0.2, 0.05, 0.05, 0.2, 0.2, 0.05, 0.05])


@njit(nogil=True)
def _soe_loglik_spline(params, times, types, T_B, T_S, beta1, beta2, knots):
    """Negative log-likelihood, piecewise-linear baseline.
    params = [muB_knot_0..muB_knot_S, muS_knot_0..muS_knot_S, 8 kernel alphas]
    knots  = S+1 equispaced knot times over [0, window]."""
    S = len(knots) - 1
    nk = S + 1
    muB = np.empty(nk); muS = np.empty(nk)
    for j in range(nk):
        muB[j] = np.exp(min(max(params[j],      -_LOG_CLIP), _LOG_CLIP))
        muS[j] = np.exp(min(max(params[nk + j], -_LOG_CLIP), _LOG_CLIP))
    ker = 2 * nk
    a_BB1 = np.exp(min(max(params[ker + 0], -_LOG_CLIP), _LOG_CLIP))
    a_SS1 = np.exp(min(max(params[ker + 1], -_LOG_CLIP), _LOG_CLIP))
    a_BS1 = np.exp(min(max(params[ker + 2], -_LOG_CLIP), _LOG_CLIP))
    a_SB1 = np.exp(min(max(params[ker + 3], -_LOG_CLIP), _LOG_CLIP))
    a_BB2 = np.exp(min(max(params[ker + 4], -_LOG_CLIP), _LOG_CLIP))
    a_SS2 = np.exp(min(max(params[ker + 5], -_LOG_CLIP), _LOG_CLIP))
    a_BS2 = np.exp(min(max(params[ker + 6], -_LOG_CLIP), _LOG_CLIP))
    a_SB2 = np.exp(min(max(params[ker + 7], -_LOG_CLIP), _LOG_CLIP))

    seg_len = knots[1] - knots[0]          # equispaced
    A_B1 = A_S1 = A_B2 = A_S2 = 0.0
    last_t = 0.0
    loglik = 0.0
    for i in range(len(times)):
        t = times[i]; etype = types[i]; dt = t - last_t
        if dt > 0.0:
            d1 = np.exp(-beta1 * dt); d2 = np.exp(-beta2 * dt)
            A_B1 *= d1; A_S1 *= d1; A_B2 *= d2; A_S2 *= d2
        # linear-interpolated baseline at t
        k = int(t / seg_len)
        if k >= S:
            k = S - 1
        frac = (t - knots[k]) / seg_len
        muB_t = muB[k] + (muB[k + 1] - muB[k]) * frac
        muS_t = muS[k] + (muS[k + 1] - muS[k]) * frac
        if etype == 0.0:
            lam = muB_t + a_BB1*A_B1 + a_SB1*A_S1 + a_BB2*A_B2 + a_SB2*A_S2
            if lam <= 1e-300:
                return 1e10
            loglik += np.log(lam); A_B1 += 1.0; A_B2 += 1.0
        else:
            lam = muS_t + a_BS1*A_B1 + a_SS1*A_S1 + a_BS2*A_B2 + a_SS2*A_S2
            if lam <= 1e-300:
                return 1e10
            loglik += np.log(lam); A_S1 += 1.0; A_S2 += 1.0
        last_t = t

    T_end = times[-1]
    for i in range(len(T_B)):
        v1 = 1.0 - np.exp(-beta1*(T_end - T_B[i])); v2 = 1.0 - np.exp(-beta2*(T_end - T_B[i]))
        loglik -= (a_BB1/beta1)*v1 + (a_BB2/beta2)*v2 + (a_BS1/beta1)*v1 + (a_BS2/beta2)*v2
    for i in range(len(T_S)):
        v1 = 1.0 - np.exp(-beta1*(T_end - T_S[i])); v2 = 1.0 - np.exp(-beta2*(T_end - T_S[i]))
        loglik -= (a_SS1/beta1)*v1 + (a_SS2/beta2)*v2 + (a_SB1/beta1)*v1 + (a_SB2/beta2)*v2
    # baseline compensator: exact trapezoid of the linear baseline over [0, T_end],
    # matching the constant engine (which integrates to times[-1], not the nominal window)
    for k in range(S):
        a = knots[k]
        if a >= T_end:
            break
        b = knots[k + 1]
        bb = b if b < T_end else T_end
        fracb = (bb - a) / seg_len
        fbB = muB[k] + (muB[k + 1] - muB[k]) * fracb
        fbS = muS[k] + (muS[k + 1] - muS[k]) * fracb
        loglik -= 0.5 * (muB[k] + fbB) * (bb - a)
        loglik -= 0.5 * (muS[k] + fbS) * (bb - a)
    return -loglik


def _phi_eta(p, ker, beta1, beta2):
    a_BB1, a_SS1, a_BS1, a_SB1 = np.exp(p[ker+0]), np.exp(p[ker+1]), np.exp(p[ker+2]), np.exp(p[ker+3])
    a_BB2, a_SS2, a_BS2, a_SB2 = np.exp(p[ker+4]), np.exp(p[ker+5]), np.exp(p[ker+6]), np.exp(p[ker+7])
    phi = (np.array([[a_BB1, a_SB1], [a_BS1, a_SS1]]) / beta1 +
           np.array([[a_BB2, a_SB2], [a_BS2, a_SS2]]) / beta2)
    eta = float(np.max(np.abs(np.linalg.eigvals(phi))))
    return {'phi_BB1': a_BB1/beta1, 'phi_SS1': a_SS1/beta1,
            'phi_SB2': a_SB2/beta2, 'phi_BS2': a_BS2/beta2, 'eta': eta}


def fit_spline(T_B, T_S, n_seg, beta1=100.0, beta2=1.0, window_sec=3600.0,
               n_inits=10, seed=0):
    """Fit the linear-spline-baseline model with `n_seg` segments (n_seg+1 knots).
    Fixed per-window seed -> reproducible multistart. Returns kernel norms + eta +
    mean baselines + loglik + n_params, or None if too sparse."""
    if len(T_B) < _MIN_EVENTS or len(T_S) < _MIN_EVENTS:
        return None
    times = np.concatenate([T_B, T_S])
    types = np.concatenate([np.zeros(len(T_B)), np.ones(len(T_S))])
    o = np.argsort(times, kind='stable')
    times = times[o].astype(np.float64); types = types[o].astype(np.float64)
    t_buy = np.sort(T_B).astype(np.float64); t_sell = np.sort(T_S).astype(np.float64)
    knots = np.linspace(0.0, window_sec, n_seg + 1)
    nk = n_seg + 1
    rate_B = max(len(t_buy), 1) / window_sec
    rate_S = max(len(t_sell), 1) / window_sec
    x0 = np.concatenate([np.full(nk, np.log(rate_B * 0.5)),
                         np.full(nk, np.log(rate_S * 0.5)),
                         np.log(_KERNEL_X0)])
    rng = np.random.default_rng(seed)
    best, bx = np.inf, None
    for j in range(n_inits):
        start = x0 + (rng.standard_normal(len(x0)) * 0.3 if j > 0 else 0.0)
        r = minimize(lambda p: _soe_loglik_spline(p, times, types, t_buy, t_sell, beta1, beta2, knots),
                     start, method='L-BFGS-B', options={'maxiter': 500})
        if np.isfinite(r.fun) and r.fun < best:
            best, bx = r.fun, r.x
    if bx is None:
        return None
    p = np.clip(bx, -_LOG_CLIP, _LOG_CLIP)
    ker = 2 * nk
    out = _phi_eta(p, ker, beta1, beta2)
    out['mu_B'] = float(np.exp(p[:nk]).mean())
    out['mu_S'] = float(np.exp(p[nk:2*nk]).mean())
    out['loglik'] = -best
    out['n_params'] = 2 * nk + 8
    out['n_seg'] = n_seg
    return out


def select_n_seg_bic(T_B, T_S, candidates=(1, 2, 3, 4, 6, 8), **kw):
    """Pick n_seg minimizing BIC = k*ln(n) - 2*loglik. Returns (best_n_seg, table)."""
    n = len(T_B) + len(T_S)
    table = []
    best_seg, best_bic = None, np.inf
    for s in candidates:
        r = fit_spline(T_B, T_S, s, **kw)
        if r is None:
            continue
        bic = r['n_params'] * np.log(n) - 2 * r['loglik']
        table.append({'n_seg': s, 'loglik': r['loglik'], 'n_params': r['n_params'], 'bic': bic})
        if bic < best_bic:
            best_bic, best_seg = bic, s
    return best_seg, table


def test_reduces_to_constant(seed=0, tol=1e-7):
    """Degenerate check: flat spline baseline == constant-mu engine loglik."""
    from .model import soe_loglik_fast
    rng = np.random.default_rng(seed)
    T_B = np.sort(rng.uniform(0, 3600, 400)); T_S = np.sort(rng.uniform(0, 3600, 350))
    times = np.concatenate([T_B, T_S]); types = np.concatenate([np.zeros(len(T_B)), np.ones(len(T_S))])
    o = np.argsort(times, kind='stable'); times = times[o]; types = types[o]
    # constant params: mu_B,mu_S + 8 alphas
    cp = np.log(np.array([0.3, 0.25, 0.2, 0.2, 0.05, 0.05, 0.2, 0.2, 0.05, 0.05]))
    ll_const = soe_loglik_fast(cp, times, types, np.sort(T_B), np.sort(T_S), 100.0, 1.0)
    # spline with S segments, all knots equal to the same level -> flat
    for S in (1, 4, 8):
        nk = S + 1
        sp = np.concatenate([np.full(nk, np.log(0.3)), np.full(nk, np.log(0.25)),
                             np.log(np.array([0.2, 0.2, 0.05, 0.05, 0.2, 0.2, 0.05, 0.05]))])
        knots = np.linspace(0.0, 3600.0, nk)
        ll_sp = _soe_loglik_spline(sp, times, types, np.sort(T_B), np.sort(T_S), 100.0, 1.0, knots)
        rel = abs(ll_sp - ll_const) / abs(ll_const)
        assert rel < tol, f"S={S}: spline {ll_sp} != const {ll_const} (rel {rel:.2e})"
    return ll_const
