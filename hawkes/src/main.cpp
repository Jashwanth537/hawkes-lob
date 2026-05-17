#include "hawkes.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

// ── CSV loading (streaming by hour window) ────────────────────────────────────
// Read the entire MO CSV once; store as flat vector of (t_us, type).
struct RawEvent { int64_t t_us; int type; };

static std::vector<RawEvent> load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string line;
    std::getline(f, line); // skip header
    std::vector<RawEvent> out;
    out.reserve(5'000'000);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // timestamp_us,event_type,...
        const char* p = line.c_str();
        char* end;
        int64_t ts = std::strtoll(p, &end, 10);
        if (*end != ',') continue;
        int et = (int)std::strtol(end + 1, nullptr, 10);
        out.push_back({ts, et});
    }
    return out;
}

// ── --test mode ───────────────────────────────────────────────────────────────
static int run_test() {
    const double mu_B=0.3, mu_S=0.3;
    const double a_BB=0.4, a_SS=0.4, a_BS=0.1, a_SB=0.1, beta=3.0;
    const double T = 3600.0;

    std::cout << "Generating synthetic bivariate Hawkes data...\n";
    auto evs = simulate_hawkes(mu_B, mu_S, a_BB, a_SS, a_BS, a_SB, beta, T, 42);
    int nB=0, nS=0;
    for (auto& e : evs) { if (e.type==0) nB++; else nS++; }
    std::cout << "  Generated: " << nB << " BUY, " << nS << " SELL events\n";

    std::cout << "Running MLE...\n";
    auto res = calibrate(evs, T);

    std::cout << "\nTrue vs Recovered:\n";
    std::printf("  mu_B:     true=%.3f  recovered=%.3f\n", mu_B,  res.mu_B);
    std::printf("  mu_S:     true=%.3f  recovered=%.3f\n", mu_S,  res.mu_S);
    std::printf("  alpha_BB: true=%.3f  recovered=%.3f\n", a_BB,  res.alpha_BB);
    std::printf("  alpha_SS: true=%.3f  recovered=%.3f\n", a_SS,  res.alpha_SS);
    std::printf("  alpha_BS: true=%.3f  recovered=%.3f\n", a_BS,  res.alpha_BS);
    std::printf("  alpha_SB: true=%.3f  recovered=%.3f\n", a_SB,  res.alpha_SB);
    std::printf("  beta:     true=%.3f  recovered=%.3f\n", beta,  res.beta);
    std::printf("  loglik:   %.2f   converged: %s\n", res.loglik, res.converged ? "YES" : "NO");

    // Pass criterion: all alphas within 20%
    bool pass = true;
    auto within = [](double truth, double rec) {
        return std::abs(rec - truth) / truth < 0.20;
    };
    if (!within(a_BB, res.alpha_BB)) { std::cout << "  FAIL: alpha_BB\n"; pass=false; }
    if (!within(a_SS, res.alpha_SS)) { std::cout << "  FAIL: alpha_SS\n"; pass=false; }
    if (!within(a_BS, res.alpha_BS)) { std::cout << "  FAIL: alpha_BS\n"; pass=false; }
    if (!within(a_SB, res.alpha_SB)) { std::cout << "  FAIL: alpha_SB\n"; pass=false; }

    std::cout << (pass ? "\n[PASS] All alphas within 20%\n" : "\n[FAIL]\n");
    return pass ? 0 : 1;
}

// ── main calibration loop ─────────────────────────────────────────────────────
static int run_calibration(const std::string& input_path,
                           const std::string& output_path) {
    std::cout << "Loading " << input_path << " ...\n" << std::flush;
    auto raw = load_csv(input_path);
    std::cout << "  Loaded " << raw.size() << " events\n" << std::flush;

    // Sort by timestamp (should already be sorted, but guarantee it)
    std::sort(raw.begin(), raw.end(), [](auto& a, auto& b){ return a.t_us < b.t_us; });

    if (raw.empty()) { std::cerr << "No events loaded.\n"; return 1; }

    double t_start_us = static_cast<double>(raw.front().t_us);
    double t_end_us   = static_cast<double>(raw.back().t_us);
    double t_start    = t_start_us / 1e6;
    double t_end      = t_end_us   / 1e6;

    // Snap to hour boundary
    double win_start = std::floor(t_start / 3600.0) * 3600.0;
    const double WIN = 3600.0;

    std::ofstream out(output_path);
    if (!out) { std::cerr << "Cannot open output: " << output_path << "\n"; return 1; }
    out << "window_start_unix,utc_hour,n_B,n_S,"
           "mu_B,mu_S,alpha_BB,alpha_SS,alpha_BS,alpha_SB,beta,"
           "phi_BB,phi_SS,phi_BS,phi_SB,eta_total,loglik,converged\n";

    int n_windows = 0, n_done = 0, n_skipped = 0;
    size_t pos = 0; // index into raw

    auto t0 = std::chrono::steady_clock::now();

    for (double ws = win_start; ws < t_end; ws += WIN) {
        double we = ws + WIN;
        ++n_windows;

        // Collect events in [ws, we)
        std::vector<Event> evs;
        // advance pos to ws
        while (pos < raw.size() && raw[pos].t_us / 1e6 < ws) ++pos;
        size_t pos2 = pos;
        while (pos2 < raw.size() && raw[pos2].t_us / 1e6 < we) {
            double t_rel = raw[pos2].t_us / 1e6 - ws;
            evs.push_back({t_rel, raw[pos2].type});
            ++pos2;
        }

        int nB = 0, nS = 0;
        for (auto& e : evs) { if (e.type==0) nB++; else nS++; }

        if (nB < 20 || nS < 20) {
            ++n_skipped;
            continue;
        }

        auto res = calibrate(evs, WIN);

        int utc_hour = static_cast<int>(std::fmod(ws, 86400.0) / 3600.0);

        out << std::fixed;
        out << static_cast<int64_t>(ws) << ','
            << utc_hour << ','
            << nB << ',' << nS << ','
            << res.mu_B << ',' << res.mu_S << ','
            << res.alpha_BB << ',' << res.alpha_SS << ','
            << res.alpha_BS << ',' << res.alpha_SB << ','
            << res.beta << ','
            << res.phi_BB << ',' << res.phi_SS << ','
            << res.phi_BS << ',' << res.phi_SB << ','
            << res.eta_total << ','
            << res.loglik << ','
            << (res.converged ? 1 : 0) << '\n';
        out.flush();

        ++n_done;
        if (n_done % 10 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count();
            double rate = n_done / std::max(elapsed, 1L);
            int    left = static_cast<int>((n_windows - n_done) / std::max(rate, 0.01));
            std::printf("  [%3d done | %3d skipped]  ~%ds remaining\n",
                        n_done, n_skipped, left);
            std::fflush(stdout);
        }
    }

    std::printf("Finished: %d windows calibrated, %d skipped.\n", n_done, n_skipped);
    std::printf("Output: %s\n", output_path.c_str());
    return 0;
}

// ── entry point ───────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc >= 2 && std::string(argv[1]) == "--test") {
        return run_test();
    }

    std::string input_path, output_path;
    for (int i = 1; i < argc - 1; ++i) {
        std::string a(argv[i]);
        if (a == "--input")  input_path  = argv[i+1];
        if (a == "--output") output_path = argv[i+1];
    }
    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: hawkes --input <csv> --output <csv>\n"
                  << "       hawkes --test\n";
        return 1;
    }
    return run_calibration(input_path, output_path);
}
