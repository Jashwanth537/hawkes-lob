# mu_stationarity branch — is the NYSE-open excitation endogenous, or baseline mu?

**Question.** Under a constant-baseline Hawkes fit, all four kernels (fast self
phi_BB1/phi_SS1, slow cross phi_SB2/phi_BS2) appear to rise at the US equity open. Is that
genuine endogenous excitation, or an exogenous activity surge absorbed into the kernels
(the Filimonov-Sornette mu-confound)?

**Finding.** The open effect is real and localized to the equity open (hour 14 UTC under
EST, above the 8:30-ET macro hour and the 10:00-ET data hour). But when the baseline mu(t)
is allowed to vary (spline, BIC-selected knots), the phi_SB2 open contrast collapses by
about 74% and is no longer distinguishable from zero, including against an activity-matched
control. Conclusion: the apparent open-time excitation is primarily exogenous (baseline mu),
not a robust endogenous channel. All estimates are conditional on the 2-exp kernel
(beta=[100,1]).

Full narrative with hypotheses, CIs, and verdicts: `DECISIONS.md` (Phase 0 -> 1a -> 1b).

## src/ — calibration engine (single source of truth; do not reformat the njit cores)
- `model.py` — constant-mu single-window MLE (`soe_loglik_fast`, `fit_window_soe_fast`). Verified bit-identical to the main-branch engine.
- `spline.py` — time-varying baseline mu(t) (piecewise-linear; verified to reduce to constant-mu exactly).
- `timevarying.py` — piecewise-constant mu(t) and the within-window LR test (now superseded; see DECISIONS).
- `simulation.py` — Ogata-thinning simulator (ground truth for tests).
- `gof.py` — time-rescaling goodness-of-fit.
- `data.py` — Binance aggTrades loaders (parquet-first, agg_id dedup).
- `calibration.py` — threaded rolling calibration / streaming over months.
- `preprocessing.py` — sweep aggregation, windowing, microsecond-to-second.
- `survival.py` — phi_SB2 contrast/report helper (superseded by phase1b_sweep).

## scripts/ — analysis (each caches to results/, n_workers=1 for reported numbers)
- `csv_to_parquet.py` — build/verify the parquet archive from raw aggTrades CSVs.
- `phase0_baseline.py` — constant-mu per-UTC-hour baseline + tie fractions.
  Writes `results/phase0_*` (const fits, kernel-norms figure, tie table, open contrast).
- `phase1a_killshots.py` — window-definition / placebo / multiplicity / tie tests on phi_SB2.
  Writes `results/phase1a_*` (hourly backbone Dec-Feb, aligned windows).
- `phase1b_sweep.py` — mu-flexibility sweep A->B->C + activity-matched control.
  Writes `results/phase1b_sweep_matched.csv` and the sweep figure.
- `make_figures.py` — regenerate the 3 headline figures from caches (no refitting).
- `phase1b_injection_check.py` — spline-validity check (does the spline absorb a planted real effect?).

## How to reproduce (clean checkout)
```
cd hawkes
pip install -r requirements.txt          # frozen lockfile (numpy, pandas, scipy, numba, pyarrow, psutil, matplotlib, jupyterlab)
# data: place the BTCUSDT monthly aggTrades parquets in data/download/ :
#   BTCUSDT-aggTrades-2025-12.parquet, -2026-01.parquet, -2026-02.parquet
#   (regenerate from the raw Binance CSVs with: python scripts/csv_to_parquet.py)
python scripts/phase0_baseline.py        # ~30 min  (n_workers=1)
python scripts/phase1a_killshots.py      # ~40 min
python scripts/phase1b_sweep.py          # ~2.5 h   (spline sweep, n_inits=10)
python scripts/make_figures.py           # seconds, from caches -> results/figures/
```
All scripts find data relative to this directory; no env var or manual path edit needed.
`requirements_unfrozen.txt` is the loose top-level dependency list; `requirements.txt` is
the pinned install source of truth.

## Pointers
- `DECISIONS.md` — append-only decision log, the Phase 0 -> 1a -> 1b arc.
- `results/figures/` — the 3 headline figures.
- `results/legacy/` — outputs from the initial single-notebook approach; superseded.
- `calibration.ipynb` — legacy exploratory notebook (the original single-file approach);
  current work lives in `scripts/` + `src/`.
- `notebooks/_scratch_unverified.ipynb` — the phi_SB2 piecewise "survival" lead; SUPERSEDED
  by the Phase 1b clean negative.
