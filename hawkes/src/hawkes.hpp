#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>

// ── event types ────────────────────────────────────────────────────────────────
struct Event {
    double t;      // seconds, zero-referenced to window start
    int    type;   // 0 = BUY, 1 = SELL
};

// ── calibration result ─────────────────────────────────────────────────────────
struct HawkesResult {
    double mu_B, mu_S;
    double alpha_BB, alpha_SS, alpha_BS, alpha_SB;
    double beta;
    double phi_BB, phi_SS, phi_BS, phi_SB, eta_total;
    double loglik;
    double spectral_radius;
    bool   converged;
};

// ── log-likelihood functor for LBFGSpp ────────────────────────────────────────
// x[7]: unconstrained parameters
//   mu_B    = exp(x[0])
//   mu_S    = exp(x[1])
//   alpha_BB = softplus(x[2])
//   alpha_SS = softplus(x[3])
//   alpha_BS = softplus(x[4])
//   alpha_SB = softplus(x[5])
//   beta    = exp(x[6]) + 0.1
class HawkesNegLogLik {
public:
    using Scalar = double;
    using TVec   = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    explicit HawkesNegLogLik(const std::vector<Event>& events, double T)
        : evs_(events), T_(T) {}

    double operator()(const TVec& x, TVec& grad) const;

    static double softplus(double z)      { return std::log1p(std::exp(z)); }
    static double d_softplus(double z)    { return 1.0 / (1.0 + std::exp(-z)); }

private:
    const std::vector<Event>& evs_;
    double T_;
};

// ── public API ─────────────────────────────────────────────────────────────────
HawkesResult calibrate(const std::vector<Event>& events, double T_window);

// Ogata thinning simulation (for --test)
std::vector<Event> simulate_hawkes(
    double mu_B, double mu_S,
    double alpha_BB, double alpha_SS, double alpha_BS, double alpha_SB,
    double beta, double T, unsigned seed = 42);
