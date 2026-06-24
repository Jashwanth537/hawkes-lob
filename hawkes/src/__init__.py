"""hawkes.src -- bivariate sum-of-exponentials Hawkes calibration engine.

Module map:
  simulation    Ogata-thinning simulator (ground-truth for tests)
  preprocessing sweep aggregation, windowing, us->s conversion
  data          Binance aggTrades loaders (parquet-first, agg_id dedup)
  model         constant-mu single-window MLE (soe_loglik_fast, fit_window_soe_fast)
  calibration   threaded rolling calibration over windows / streaming over files
  gof           time-rescaling goodness-of-fit (Ogata 1988)
  timevarying   piecewise-mu(t) fit + within-window LR/AIC/BIC test
  survival      phi_SB2 UTC-open survival test: fit + contrast + report
"""
from .simulation import simulate_bivariate_soe
from .preprocessing import aggregate_sweeps, to_seconds, _first_hour_us
from .gof import compute_compensator_increments, time_rescaling_test, _qq_panel
from .model import soe_loglik_fast, fit_window_soe_fast
from .data import load_one_file, load_aggtrades
from .calibration import rolling_calibration, streaming_rolling_calibration, _rv_in
from .timevarying import _soe_loglik_pw, _fit_pw, fit_pw_full, within_window_mu_test
from .survival import (fit_survival_windows, contrast, survival_verdict, survival_report,
                       SURVIVAL_VARS, OPEN_HOUR_UTC, PLACEBO_HOUR_UTC)

__all__ = [
    "simulate_bivariate_soe",
    "aggregate_sweeps", "to_seconds", "_first_hour_us",
    "compute_compensator_increments", "time_rescaling_test", "_qq_panel",
    "soe_loglik_fast", "fit_window_soe_fast",
    "load_one_file", "load_aggtrades",
    "rolling_calibration", "streaming_rolling_calibration", "_rv_in",
    "_soe_loglik_pw", "_fit_pw", "fit_pw_full", "within_window_mu_test",
    "fit_survival_windows", "contrast", "survival_verdict", "survival_report",
    "SURVIVAL_VARS", "OPEN_HOUR_UTC", "PLACEBO_HOUR_UTC",
]
