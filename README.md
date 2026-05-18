# hawkes-lob

**Hawkes Process Calibration of Cryptocurrency Limit Order Book Data**  
Bivariate Sum-of-Exponentials Hawkes calibration on Binance BTC/USDT and ETH/USDT market order flow. Self-excitation is 10 to 19% stronger at UTC 13:30 (NYSE open) than the rest-of-day baseline for both assets, and the branching ratio η correlates with realized volatility at ρ = 0.84, confirming that high-η windows are also the high-volatility windows.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Key Results

### 1 · Sweep aggregation fixes the GOF, consistently across all UTC hours

A single aggressive market order on Binance sweeps multiple price levels, generating several `aggTrade` rows within ~100µs. The 10ms Hawkes kernel cannot resolve these as distinct events, causing the Kolmogorov-Smirnov (KS) test to fail on raw data (KS ≈ 0.50 to 0.62). Collapsing sub-100µs same-direction bursts into single events reduces the KS statistic by **91 to 97%** across 12 windows sampled from all UTC hours.

| | Raw stream | Aggregated (100µs) |
|---|---|---|
| BTC first window | KS = 0.620 | KS = 0.020 |
| Mean across 12 sampled windows | KS = 0.502 | KS = 0.026 |
| Range across 12 sampled windows | [0.383, 0.646] | [0.016, 0.038] |

Approximately 62% of raw `aggTrade` BUY events in the first window are sub-100µs sweep fills from the same originating market order, not distinct aggressive orders.

![GOF sweep comparison](results/gof_sweep_comparison.png)

### 2 · Branching ratio η predicts realized volatility within the hour

![η vs Realized Volatility](results/eta_vs_rv.png)

| Asset | ρ(η, RV) | p-value | n windows |
|-------|----------|---------|-----------|
| BTC/USDT | 0.840 | <0.0001 | 144 |
| ETH/USDT | 0.818 | <0.0001 | 145 |

UTC hour alone explains **35 to 39%** of cross-window variance in η (OLS F-test, p < 0.001), meaning a model that assumes time-homogeneous order flow misses roughly a third of the within-day structure.

### 3 · Market order self-excitation varies with the UTC clock

Despite operating 24/7 with no formal session boundary, crypto market microstructure is not time-homogeneous. Self-excitation peaks at UTC 13:30 (NYSE regular open, 9:30 AM EDT) for both assets.

![Kernel norms by UTC hour](results/kernel_norms_by_hour.png)

> Magnitudes are from raw `aggTrade` streams. Approximately 62% of raw BUY events in the first window are sub-100µs sweep fills (see Goodness-of-Fit below); the fraction varies by UTC hour. The qualitative UTC-13:30 pattern is expected to persist after sweep aggregation, but absolute magnitudes would shrink.

**Targeted Mann-Whitney test, UTC hour 13 vs all other hours:**

| Parameter | BTC Δ% | ETH Δ% | MW p-value (BTC / ETH) |
|-----------|--------|--------|------------------------|
| η (branching ratio) | +12.4% | +9.5% | 0.004 / 0.001 |
| φ_BB1 (fast BUY self-excitation) | +18.9% | +9.5% | 0.002 / 0.004 |
| φ_SS1 (fast SELL self-excitation) | +12.6% | +7.0% | 0.006 / 0.020 |
| φ_SB2 (slow SELL→BUY cross) | +60.3% | +115.5% | 0.042 / 0.001 |

The NYSE-open hypothesis was pre-specified in Section 1 of [hawkes_lob_summary.md](hawkes_lob_summary.md) before any post-hoc analysis. OLS regression (η on 23 UTC-hour dummies plus asset dummy): R² = 0.352, F = 2.56, p(bootstrap) = 0.001.

ETH shows a sharper UTC-13:30 spike in slow SELL→BUY cross-excitation than BTC, roughly double its rest-of-day baseline, with no BTC counterpart. See [hawkes_lob_summary.md](hawkes_lob_summary.md) §5 for discussion.

## Model

Bivariate Sum-of-Exponentials Hawkes process with U = 2 kernel components and fixed decay rates β₁ = 100 s⁻¹ (10ms) and β₂ = 1 s⁻¹ (1s). The BUY intensity at time t is:

$$\lambda_B(t) = \mu_B + \sum_{u=1}^{2} \alpha_{BB}^u \sum_{t_j^B < t} e^{-\beta_u(t-t_j^B)} + \sum_{u=1}^{2} \alpha_{SB}^u \sum_{t_k^S < t} e^{-\beta_u(t-t_k^S)}$$

The SELL intensity λ_S(t) is symmetric with parameters μ_S, α_SS, α_BS. The two-timescale split captures fast momentum (sub-100ms) and slow momentum (sub-second) simultaneously.

10 free parameters per window: [μ_B, μ_S, α_BB1, α_SS1, α_BS1, α_SB1, α_BB2, α_SS2, α_BS2, α_SB2]. MLE via L-BFGS-B with log-space reparameterisation and 3 random initialisations per window. Log-likelihood inner loop compiled with Numba JIT: ~2.5s per window vs ~60s pure Python.

## Validation

Synthetic parameter recovery from simulated data (Ogata thinning):

| Parameter | True | Recovered | Error |
|-----------|------|-----------|-------|
| φ_BB1 | 0.700 | 0.694 | 0.9% |
| φ_SS1 | 0.800 | 0.805 | 0.6% |
| φ_BB2 | 0.040 | 0.043 | 7.5% |
| φ_SB2 | 0.025 | 0.027 | 8.0% |
| η | ~0.89 | 0.853 | 4.2% |

See `calibration.ipynb` Cell 5 for the full synthetic recovery test.

## Goodness-of-Fit

The time-rescaling test (Ogata 1988) checks whether compensator increments Λ_B(t_{i-1}, t_i) are i.i.d. Exp(1). Raw `aggTrade` streams contain sweep fills: a single aggressive order sweeping N price levels generates N rows within ~100µs. The 10ms kernel cannot resolve these, producing heavy-tailed GOF failure (KS ≈ 0.5 to 0.6) across all hours.

Collapsing sub-100µs same-direction events into single events (a heuristic approximation of the order-book filtration approach in [Anantha, Jain and Maiti 2025](https://arxiv.org/abs/2507.22712)) drops KS to 0.016 to 0.038 across all sampled windows.

![GOF robustness](results/gof_robustness.png)

## Reproducibility

### Dependencies

```bash
conda create -n hawkes python=3.11
conda activate hawkes
python -m ipykernel install --user --name hawkes --display-name "Python (hawkes)"
pip install -r requirements.txt
```

### Data collection

The collector streams Binance `aggTrade` and `bookTicker` combined feeds per symbol. Built with CMake on a GCP VM (asia-northeast1), using IXWebSocket for TLS WebSocket handling and nlohmann/json for message parsing.

```bash
# On GCP VM: install system dependencies and build
sudo apt install -y libeigen3-dev libssl-dev
cd ~/collector
mkdir -p build && cd build
cmake .. && make -j1          # -j1 to stay within 1 GB RAM; swap required
```

```bash
# Run in a persistent tmux session (auto-started on reboot via crontab @reboot)
tmux new-session -s lob
cd ~/collector/build
./collector --outdir ~/lob_data/
# Ctrl-B D to detach
```

The collector writes one row per event to `~/lob_data/<symbol>_events.csv`: `timestamp_us, event_type, price, quantity, symbol`. Timestamps are in microseconds; monotone ordering is enforced at write time. Event types: 0 = BUY market order, 1 = SELL market order, 2 = OFI positive, 3 = OFI negative. OFI events (~93% of rows) come from `bookTicker`; market orders from `aggTrade`.

Data collected from May 10 (20:32 UTC) to May 16 (20:27 UTC), 2026 (143.9 usable hours). An 8.7-hour connectivity gap (May 16 20:27 to May 17 05:09 UTC) is excluded automatically by the rolling-window calibration.

### Preprocessing

Filter from raw events (60M rows, ~3 GB) to market orders only (~4.3M BTC, ~3.5M ETH):

```bash
cd ~/collector/hawkes
python3 preprocess.py
```

Reads `~/lob_data/btcusdt_events.csv` and `ethusdt_events.csv`, streams line-by-line without loading into memory, writes to `~/collector/hawkes/data/btcusdt_mo.csv` and `ethusdt_mo.csv` with the same schema. Prints row counts and BUY:SELL ratio on completion.

### Calibration

Open `calibration.ipynb` and run cells in order:

| Cell | Purpose |
|------|---------|
| 0 to 2 | Imports, data load, buy/sell split |
| 3 | First-hour window extraction |
| 4 | `soe_loglik_fast` (Numba JIT) + `fit_window_soe_fast` |
| 5 | Ogata simulation + synthetic recovery test |
| 6 | Unit tests |
| 7 | `aggregate_sweeps`, `compute_compensator_increments`, `time_rescaling_test_corrected` |
| 8 | KS vs sweep threshold, selects optimal 100µs aggregation gap |
| 9 | Sweep comparison + baseline GOF (2×2 grid) |
| 10 | Rolling calibration (run once, ~12 min; saves CSVs) |
| 11, 12 | Exploratory parameter time-series plots (optional) |
| 13 | KW + Mann-Whitney UTC seasonality tests |
| 14 | OLS regression + bootstrap F-test |
| 15 | η vs realized volatility |
| 16 | `plot_kernel_norms_by_hour` |

Numba warmup ~30s on first call. Per-window fit ~2.5s. Full rolling calibration ~12 minutes for 289 windows.

## Repository Structure

```
hawkes-lob/
├── collector/                  # C++ WebSocket client
│   ├── CMakeLists.txt
│   └── main.cpp
├── data/                       # Not tracked in git due to size; regenerate via preprocess.py
│   ├── btcusdt_mo.csv          # preprocessed market orders (4.3M rows)
│   └── ethusdt_mo.csv          # preprocessed market orders (3.5M rows)
├── results/
│   ├── results_btcusdt.csv     # 144 windows × 12 parameters
│   ├── results_ethusdt.csv     # 145 windows × 12 parameters
│   ├── kernel_norms_by_hour.png
│   ├── gof_baseline.png
│   ├── gof_sweep_comparison.png
│   ├── gof_robustness.png
│   ├── ks_vs_threshold.png
│   └── eta_vs_rv.png
├── calibration.ipynb
├── preprocess.py
├── requirements.txt
├── hawkes_lob_summary.md
├── CITATION.cff
├── README.md
└── LICENSE
```

Preprocessed CSVs (`data/*_mo.csv`) are excluded from git tracking due to size. Regenerate them by running `preprocess.py` against the raw collector output, or contact the author for direct access.

## References

- Anantha, A. N. and Jain, S. (2025). *Forecasting high frequency order flow imbalance using Hawkes processes.* Computational Economics. [arXiv:2408.03594](https://arxiv.org/abs/2408.03594)
- Anantha, A. N., Jain, S. and Maiti, P. (2025). *Order Book Filtration and Directional Signal Extraction at High Frequency.* [arXiv:2507.22712](https://arxiv.org/abs/2507.22712)
- Joseph, S. S. and Jain, S. (2024). Non-parametric Hawkes on cryptocurrency LOB. [arXiv:2402.04740](https://arxiv.org/abs/2402.04740)
- Ogata, Y. (1988). Statistical models for earthquake occurrences. *JASA* 83(401).
- Bacry, E., Jaisson, T. and Muzy, J.-F. (2016). *Quantitative Finance* 16(8).
- Hardiman, S. J., Bercot, N. and Bouchaud, J.-P. (2013). *European Physical Journal B* 86(10).
- Filimonov, V. and Sornette, D. (2012). [arXiv:1204.2406](https://arxiv.org/abs/1204.2406)
- Almgren, R. and Chriss, N. (2001). *Journal of Risk* 3(2).
- Cartea, Á., Jaimungal, S. and Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge.

## Citation

```bibtex
@misc{hawkes-lob-2026,
  author = {Mummalaneni, Jashwanth},
  title  = {Hawkes Process Calibration of Cryptocurrency Limit Order Book Data},
  year   = {2026},
  url    = {https://github.com/Jashwanth537/hawkes-lob}
}
```

## License

MIT. See [LICENSE](LICENSE).
