#include "hawkes.hpp"
#include <LBFGS.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <iostream>

using TVec = Eigen::Matrix<double, Eigen::Dynamic, 1>;

// ── softplus helpers ───────────────────────────────────────────────────────────
static inline double sp(double z)  { return std::log1p(std::exp(std::min(z, 30.0))); }
static inline double dsp(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// ── spectral radius of 2x2 excitation matrix / beta ───────────────────────────
static double spectral_radius(double aBB, double aSS, double aBS, double aSB, double beta) {
    // eigenvalues of [[aBB, aSB],[aBS, aSS]] / beta
    double a = aBB / beta, d = aSS / beta;
    double b = aSB / beta, c = aBS / beta;
    double tr = a + d;
    double det = a * d - b * c;
    double disc = tr * tr - 4.0 * det;
    if (disc < 0) disc = 0;
    return 0.5 * (tr + std::sqrt(disc));
}

// ── negative log-likelihood + gradient ────────────────────────────────────────
double HawkesNegLogLik::operator()(const TVec& x, TVec& grad) const {
    // unpack params
    const double mu_B    = std::exp(x[0]);
    const double mu_S    = std::exp(x[1]);
    const double a_BB    = sp(x[2]);
    const double a_SS    = sp(x[3]);
    const double a_BS    = sp(x[4]);
    const double a_SB    = sp(x[5]);
    const double beta    = std::exp(x[6]) + 0.1;

    // stability penalty
    double sr = spectral_radius(a_BB, a_SS, a_BS, a_SB, beta);
    if (sr >= 0.99) {
        grad.setZero();
        return 1e6 + sr * 1e4;
    }

    const int N = static_cast<int>(evs_.size());

    // Running recursion sums.
    // R_BB, R_SB: contributions of past B events  (decay on B gaps / S gaps)
    // R_SS, R_BS: contributions of past S events  (decay on S gaps / B gaps)
    double R_BB = 0, R_SB = 0;   // last updated at last B event
    double R_SS = 0, R_BS = 0;   // last updated at last S event
    double last_t_B = 0, last_t_S = 0;

    // For gradient we also need dR/d(log_beta).
    // We track the derivative of each R w.r.t. beta via chain rule.
    // dR_BB/d(beta) maintained analogously.
    double dR_BB_db = 0, dR_SB_db = 0;
    double dR_SS_db = 0, dR_BS_db = 0;

    double loglik = 0.0;

    // Gradient accumulators (w.r.t. unconstrained x)
    double g0=0, g1=0, g2=0, g3=0, g4=0, g5=0, g6=0;

    // compensator accumulators for gradient (alpha terms)
    double cBB = 0, cSB = 0, cSS = 0, cBS = 0;

    for (int k = 0; k < N; ++k) {
        double t   = evs_[k].t;
        bool   isB = (evs_[k].type == 0);

        if (isB) {
            double gap = t - last_t_B;
            double eB  = std::exp(-beta * gap);

            // decay
            double R_BB_new   = eB * (R_BB + 1.0);   // +1 for THIS event
            double R_SB_new   = eB * (R_SB + 1.0);

            // intensity BEFORE adding this event's contribution
            double lam = mu_B + a_BB * R_BB + a_SB * R_BS;
            if (lam <= 0) lam = 1e-300;
            loglik += std::log(lam);

            // gradient of log(lam) w.r.t. x
            double inv_lam = 1.0 / lam;
            g0 += mu_B * inv_lam;                     // d/d(log mu_B)
            g2 += a_BB * dsp(x[2]) * R_BB * inv_lam; // d/d(x[2])
            g5 += a_SB * dsp(x[5]) * R_BS * inv_lam; // d/d(x[5])
            // beta gradient via dR terms
            g6 += (a_BB * dR_BB_db + a_SB * dR_BS_db) * inv_lam * beta; // chain d/d(x[6])

            // update dR for beta gradient
            // dR_BB/d(beta) = -gap*eB*(R_BB+1) + eB*dR_BB_prev
            double dR_BB_db_new = eB * (dR_BB_db - gap * (R_BB + 1.0));
            double dR_SB_db_new = eB * (dR_SB_db - gap * (R_SB + 1.0));

            R_BB = R_BB_new;
            R_SB = R_SB_new;
            dR_BB_db = dR_BB_db_new;
            dR_SB_db = dR_SB_db_new;
            last_t_B = t;

            // compensator accumulator
            cBB += (1.0 - std::exp(-beta * (T_ - t)));
            cBS += (1.0 - std::exp(-beta * (T_ - t)));

        } else {  // SELL
            double gap = t - last_t_S;
            double eS  = std::exp(-beta * gap);

            double R_SS_new = eS * (R_SS + 1.0);
            double R_BS_new = eS * (R_BS + 1.0);

            double lam = mu_S + a_SS * R_SS + a_BS * R_SB;
            if (lam <= 0) lam = 1e-300;
            loglik += std::log(lam);

            double inv_lam = 1.0 / lam;
            g1 += mu_S * inv_lam;
            g3 += a_SS * dsp(x[3]) * R_SS * inv_lam;
            g4 += a_BS * dsp(x[4]) * R_SB * inv_lam;
            g6 += (a_SS * dR_SS_db + a_BS * dR_SB_db) * inv_lam * beta;

            double dR_SS_db_new = eS * (dR_SS_db - gap * (R_SS + 1.0));
            double dR_BS_db_new = eS * (dR_BS_db - gap * (R_BS + 1.0));

            R_SS = R_SS_new;
            R_BS = R_BS_new;
            dR_SS_db = dR_SS_db_new;
            dR_BS_db = dR_BS_db_new;
            last_t_S = t;

            cSS += (1.0 - std::exp(-beta * (T_ - t)));
            cSB += (1.0 - std::exp(-beta * (T_ - t)));
        }
    }

    // compensator (integral of intensities)
    // integral_B = mu_B*T + (a_BB/beta)*cBB + (a_SB/beta)*cSB
    // integral_S = mu_S*T + (a_SS/beta)*cSS + (a_BS/beta)*cBS
    double comp_B = mu_B * T_ + (a_BB / beta) * cBB + (a_SB / beta) * cSB;
    double comp_S = mu_S * T_ + (a_SS / beta) * cSS + (a_BS / beta) * cBS;
    double comp   = comp_B + comp_S;
    loglik -= comp;

    // gradient of compensator
    g0 -= mu_B * T_;                          // d/d(log mu_B)
    g1 -= mu_S * T_;
    g2 -= dsp(x[2]) * cBB / beta;            // d/d(x[2]) for alpha_BB
    g3 -= dsp(x[3]) * cSS / beta;
    g4 -= dsp(x[4]) * cBS / beta;
    g5 -= dsp(x[5]) * cSB / beta;

    // beta gradient of compensator: d/d(beta) of (alpha/beta)*sum(1-exp(-beta*(T-t)))
    // = (-alpha/beta^2)*sum(1-exp(..)) + (alpha/beta)*sum((T-t)*exp(-beta*(T-t)))
    // Collect for each alpha term; chain rule for x[6]: d(beta)/d(x[6])=beta
    {
        double d_cBB_dbeta = 0, d_cSB_dbeta = 0, d_cSS_dbeta = 0, d_cBS_dbeta = 0;
        for (auto& ev : evs_) {
            double rem = T_ - ev.t;
            double ex  = std::exp(-beta * rem);
            if (ev.type == 0) {
                d_cBB_dbeta += rem * ex;
                d_cBS_dbeta += rem * ex;
            } else {
                d_cSS_dbeta += rem * ex;
                d_cSB_dbeta += rem * ex;
            }
        }
        double dc_db = (a_BB * (-cBB / beta + d_cBB_dbeta)
                      + a_SB * (-cSB / beta + d_cSB_dbeta)
                      + a_SS * (-cSS / beta + d_cSS_dbeta)
                      + a_BS * (-cBS / beta + d_cBS_dbeta)) / beta;
        g6 -= dc_db * beta;   // chain rule: x[6] = log(beta - 0.1) ≈ log(beta), d(beta)/d(x[6])=beta
    }

    // Return NEGATIVE log-likelihood for minimisation
    grad.resize(7);
    grad[0] = -g0; grad[1] = -g1; grad[2] = -g2; grad[3] = -g3;
    grad[4] = -g4; grad[5] = -g5; grad[6] = -g6;

    return -loglik;
}

// ── calibrate: 5 random inits, take best ──────────────────────────────────────
HawkesResult calibrate(const std::vector<Event>& events, double T_window) {
    if (events.empty()) {
        HawkesResult r{}; r.converged = false; r.loglik = -1e300; return r;
    }

    // Count empirical rates
    int nB = 0, nS = 0;
    for (auto& e : events) { if (e.type == 0) nB++; else nS++; }
    double rB = nB / T_window, rS = nS / T_window;

    HawkesNegLogLik functor(events, T_window);

    LBFGSpp::LBFGSParam<double> param;
    param.max_iterations = 200;
    param.epsilon        = 1e-6;
    LBFGSpp::LBFGSSolver<double> solver(param);

    // Base init
    TVec x0(7);
    x0[0] = std::log(std::max(rB * 0.1, 1e-6));
    x0[1] = std::log(std::max(rS * 0.1, 1e-6));
    x0[2] = 0.0;   // softplus(0) ≈ 0.693
    x0[3] = 0.0;
    x0[4] = 0.0;
    x0[5] = 0.0;
    x0[6] = std::log(2.0);   // beta ≈ 2.1

    HawkesResult best;
    best.loglik = -std::numeric_limits<double>::infinity();

    std::mt19937 rng(12345);
    std::normal_distribution<double> noise(0.0, 0.5);

    for (int init = 0; init < 5; ++init) {
        TVec x = x0;
        if (init > 0) {
            for (int i = 0; i < 7; ++i) x[i] += noise(rng);
        }

        TVec grad(7);
        double fval;
        try {
            solver.minimize(functor, x, fval);
        } catch (...) {
            continue;
        }

        double ll = -fval;
        if (!std::isfinite(ll)) continue;

        double mu_B    = std::exp(x[0]);
        double mu_S    = std::exp(x[1]);
        double a_BB    = sp(x[2]);
        double a_SS    = sp(x[3]);
        double a_BS    = sp(x[4]);
        double a_SB    = sp(x[5]);
        double beta    = std::exp(x[6]) + 0.1;

        double sr = spectral_radius(a_BB, a_SS, a_BS, a_SB, beta);

        if (ll > best.loglik) {
            best.mu_B = mu_B; best.mu_S = mu_S;
            best.alpha_BB = a_BB; best.alpha_SS = a_SS;
            best.alpha_BS = a_BS; best.alpha_SB = a_SB;
            best.beta  = beta;
            best.loglik = ll;
            best.spectral_radius = sr;
            best.phi_BB = a_BB / beta;
            best.phi_SS = a_SS / beta;
            best.phi_BS = a_BS / beta;
            best.phi_SB = a_SB / beta;
            best.eta_total = (a_BB + a_SS + a_BS + a_SB) / beta;
            best.converged = (sr < 0.99);
        }
    }
    return best;
}

// ── Ogata thinning simulation ──────────────────────────────────────────────────
std::vector<Event> simulate_hawkes(
    double mu_B, double mu_S,
    double alpha_BB, double alpha_SS, double alpha_BS, double alpha_SB,
    double beta, double T, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::exponential_distribution<double>  expdist;

    std::vector<Event> events;
    double t = 0.0;
    double R_BB = 0, R_SS = 0, R_BS = 0, R_SB = 0;
    double last_t_B = 0, last_t_S = 0;

    while (t < T) {
        // Upper bound on intensity: lambda* = current total intensity
        double lam_B = mu_B + alpha_BB * R_BB + alpha_SB * R_BS;
        double lam_S = mu_S + alpha_SS * R_SS + alpha_BS * R_SB;
        double lam_star = lam_B + lam_S;
        if (lam_star <= 0) lam_star = mu_B + mu_S;

        expdist = std::exponential_distribution<double>(lam_star);
        double dt = expdist(rng);
        t += dt;
        if (t >= T) break;

        // Decay all R values
        double eB = std::exp(-beta * (t - last_t_B));
        double eS = std::exp(-beta * (t - last_t_S));
        R_BB = eB * R_BB;
        R_SB = eB * R_SB;
        R_SS = eS * R_SS;
        R_BS = eS * R_BS;

        double new_lam_B = mu_B + alpha_BB * R_BB + alpha_SB * R_BS;
        double new_lam_S = mu_S + alpha_SS * R_SS + alpha_BS * R_SB;
        double new_lam   = new_lam_B + new_lam_S;

        if (uni(rng) > new_lam / lam_star) continue;  // thinned

        // Accept: determine if B or S
        bool isB = (uni(rng) < new_lam_B / new_lam);
        events.push_back({t, isB ? 0 : 1});

        if (isB) {
            R_BB = R_BB + 1.0;
            R_SB = R_SB + 1.0;
            last_t_B = t;
        } else {
            R_SS = R_SS + 1.0;
            R_BS = R_BS + 1.0;
            last_t_S = t;
        }
    }
    return events;
}
