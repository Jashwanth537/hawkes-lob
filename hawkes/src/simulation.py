"""Bivariate sum-of-exponentials Hawkes simulator (Ogata thinning)."""

import numpy as np
from numba import njit

@njit
def simulate_bivariate_soe(mu_B, mu_S, 
                            a_BB1, a_SS1, a_BS1, a_SB1,
                            a_BB2, a_SS2, a_BS2, a_SB2,
                            beta1, beta2, T_max):
    """Ogata thinning for bivariate SoE Hawkes"""
    T_B, T_S = [], []
    A_B1 = A_S1 = A_B2 = A_S2 = 0.0
    t = 0.0

    while t < T_max:
        # upper bound on total intensity
        lam_B = mu_B + a_BB1*A_B1 + a_SB1*A_S1 + a_BB2*A_B2 + a_SB2*A_S2
        lam_S = mu_S + a_BS1*A_B1 + a_SS1*A_S1 + a_BS2*A_B2 + a_SS2*A_S2
        lam_max = lam_B + lam_S

        # draw next candidate event time
        dt = -np.log(np.random.random()) / lam_max
        t += dt

        if t >= T_max:
            break

        # decay sums
        d1 = np.exp(-beta1 * dt)
        d2 = np.exp(-beta2 * dt)
        A_B1 *= d1;  A_S1 *= d1
        A_B2 *= d2;  A_S2 *= d2

        # thinning: accept or reject
        lam_B_new = mu_B + a_BB1*A_B1 + a_SB1*A_S1 + a_BB2*A_B2 + a_SB2*A_S2
        lam_S_new = mu_S + a_BS1*A_B1 + a_SS1*A_S1 + a_BS2*A_B2 + a_SS2*A_S2
        u = np.random.random() * lam_max

        if u < lam_B_new:       # BUY event
            T_B.append(t)
            A_B1 += 1.0;  A_B2 += 1.0
        elif u < lam_B_new + lam_S_new:  # SELL event
            T_S.append(t)
            A_S1 += 1.0;  A_S2 += 1.0
        # else: rejected

    return np.array(T_B), np.array(T_S)
