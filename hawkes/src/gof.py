"""Goodness-of-fit via the time-rescaling theorem (Ogata 1988; Brown et al. 2002).

compute_compensator_increments -> Exp(1) under correct specification; the
plotting helpers take a Matplotlib Axes so this module has no pyplot dependency."""

import numpy as np
from scipy.stats import ks_1samp  # One-sample Kolmogorov-Smirnov GOF test
from scipy.stats import expon     # Standard exponential distribution

def compute_compensator_increments(T_B, T_S, result, process='buy', beta1=100.0, beta2=1.0):
    """
    process='buy'  → Λ_B increments, uses μ_B, φ_BB, φ_SB
    process='sell' → Λ_S increments, uses μ_S, φ_SS, φ_BS
    Under a correctly-specified model, increments ~ i.i.d. Exp(1).
    """
    betas = np.array([beta1, beta2])

    if process == 'buy':
        mu        = result['mu_B']
        phi_self  = np.array([result['phi_BB1'], result['phi_BB2']])
        phi_cross = np.array([result['phi_SB1'], result['phi_SB2']])
        target    = 0.0
    else:
        mu        = result['mu_S']
        phi_self  = np.array([result['phi_SS1'], result['phi_SS2']])
        phi_cross = np.array([result['phi_BS1'], result['phi_BS2']])
        target    = 1.0

    all_t = np.concatenate([T_B, T_S])
    all_e = np.concatenate([np.zeros(len(T_B)), np.ones(len(T_S))])
    order = np.argsort(all_t, kind='stable')
    all_t = all_t[order];  all_e = all_e[order]

    increments, A_self, A_cross = [], np.zeros(2), np.zeros(2)
    last_t = in_interval = comp = 0.0

    for k in range(len(all_t)):
        t, etype, dt = all_t[k], all_e[k], all_t[k] - last_t
        if dt > 0.0:
            e = np.exp(-betas * dt)
            if in_interval:
                comp += mu * dt + np.dot(phi_self,  A_self  * (1.0 - e)) \
                                + np.dot(phi_cross, A_cross * (1.0 - e))
            A_self *= e;  A_cross *= e
        if etype == target:
            if in_interval:
                increments.append(comp)
            comp = 0.0;  in_interval = True;  A_self += 1.0
        else:
            A_cross += 1.0
        last_t = t

    return np.array(increments)


def time_rescaling_test(T_B, T_S, result, ax, process='buy', title='', beta1=100.0, beta2=1.0):
    inc = compute_compensator_increments(T_B, T_S, result, process, beta1, beta2)
    ks_d, ks_p = ks_1samp(inc, expon.cdf)
    print(f"  {title} {process.upper():4s}  n={len(inc):,}  mean={inc.mean():.4f}  "
          f"KS={ks_d:.4f}  p={ks_p:.2e}")

    q  = np.linspace(0.005, 0.995, min(len(inc), 1500))
    th = expon.ppf(q);  em = np.quantile(inc, q)
    lim = max(th.max(), em.max()) * 1.05
    ax.scatter(th, em, alpha=0.4, s=15, label='Empirical')
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='Perfect fit')
    ax.set_xlim(0, lim);  ax.set_ylim(0, lim)
    ax.set_xlabel('Theoretical Exp(1) quantiles', fontsize=11)
    ax.set_ylabel('Empirical compensator increments', fontsize=11)
    ax.set_title(f'{title} — {process.upper()} process', fontsize=11, fontweight='bold')
    ax.legend();  ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.05, f'KS = {ks_d:.4f}\np  = {ks_p:.2e}\nmean = {inc.mean():.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    return inc


def _qq_panel(ax, increments, title, ks_d, ks_p, color):
    """
    QQ plot for time-rescaling GOF:
    compare empirical compensator increments Λ(T_i)-Λ(T_{i-1})
    against Exp(1) quantiles predicted by the time-rescaling theorem.
    Diagonal alignment => better Hawkes fit.
    """
    q  = np.linspace(0.005, 0.995, min(len(increments), 1500))
    th = expon.ppf(q);  em = np.quantile(increments, q)
    lim = max(th.max(), em.max()) * 1.05
    ax.scatter(th, em, alpha=0.40, s=12, color=color, label='Empirical')
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='y = x  (perfect fit)')
    ax.set_xlim(0, lim);  ax.set_ylim(0, lim)
    ax.set_xlabel('Theoretical Exp(1) quantiles', fontsize=11)
    ax.set_ylabel('Empirical compensator increments', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.05,
            f'KS = {ks_d:.4f}\np  = {ks_p:.2e}\nmean = {increments.mean():.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
