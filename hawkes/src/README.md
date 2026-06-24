# `hawkes/src`: calibration engine

Bivariate sum-of-exponentials Hawkes model of buy/sell market-order flow, calibrated
per 1-hour window.

## Modules

| Module | What it does | Key functions |
|---|---|---|
| `simulation.py` | Ogata-thinning simulator (ground truth for tests) | `simulate_bivariate_soe` |
| `preprocessing.py` | sweep aggregation, windowing, µs→s | `aggregate_sweeps`, `to_seconds`, `_first_hour_us` |
| `data.py` | Binance aggTrades loaders (parquet-first, agg_id dedup) | `load_one_file`, `load_aggtrades` |
| `model.py` | constant-µ single-window MLE | `soe_loglik_fast`, `fit_window_soe_fast` |
| `calibration.py` | threaded rolling calibration; streaming over months | `rolling_calibration`, `streaming_rolling_calibration` |
| `gof.py` | time-rescaling goodness-of-fit (Ogata 1988; Brown 2002) | `compute_compensator_increments`, `time_rescaling_test` |
| `timevarying.py` | piecewise-µ(t) fit + within-window LR/AIC/BIC test | `fit_pw_full`, `within_window_mu_test` |
| `survival.py` | φ_SB2 UTC-open survival test (fit + contrast + report) | `fit_survival_windows`, `survival_report` |

## Model

Intensity for buys (B) and sells (S), two timescales β = [100, 1] s⁻¹:

    λ_X(t) = µ_X + Σ_d Σ_Y α_{YX,d} · Σ_{t_Y < t} exp(-β_d (t - t_Y))

Parameters are fit in log-space. Kernel norms `φ = α/β`; branching ratio `η` is the
spectral radius of the φ matrix (η < 1 ⇒ stationary). `µ` is the exogenous baseline.

## Conventions & gotchas

- **event_type**: 0 = taker buy, 1 = taker sell (Binance `is_buyer_maker` False/True).
- **Threading**: the njit log-likelihoods run `nogil`, so rolling/survival fits use
  threads (flat RAM). Multi-start uses `np.random`, whose draw *order* across threads is
  not deterministic → per-window estimates wobble ~few % run-to-run. Pass `n_workers=1`
  for bit-reproducibility; otherwise judge results by effect size, not the 3rd decimal.
- **Do not reformat the `@njit` log-likelihood bodies** (`soe_loglik_fast`,
  `_soe_loglik_pw`). LLVM contracts float ops (FMA) in a layout-sensitive way, and
  L-BFGS-B amplifies a 1-ULP change into a different local optimum. Comments/blank lines
  are safe; renaming variables or splitting expressions is not.
- **µ(t) test**: H0 constant-µ vs H1 piecewise-µ over K sub-intervals (kernel shared).
  LR ~ χ²(2(K-1)); with ~10⁴ events/window LR rejects ~always, so lean on BIC + effect
  size. 
