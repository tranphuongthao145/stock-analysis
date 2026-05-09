# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An event-study research project measuring abnormal returns of global stock indices around trade-war announcement dates. All code lives in two Jupyter notebooks under `src/`; there is no Python package, no test suite, and no build system.

## Running the notebooks

The notebooks are written for **Google Colab** — Cell 2 calls `from google.colab import drive; drive.mount(...)` and reads `DATA_PATH = '/content/drive/MyDrive/data/data_full.xlsx'`. To run locally, both that mount cell and `DATA_PATH` must be edited to point at a local Excel workbook (the local `data/full.xlsx` is a candidate but has not been verified to match the Colab sheet schema).

Dependencies are installed in-cell by Cell 1: `%pip install -q pandas openpyxl numpy statsmodels arch matplotlib`. There is no `requirements.txt`/`pyproject.toml` — when adding deps, update Cell 1.

The notebooks are intended to be run top-to-bottom; Cells 6/7/8 each populate a results dict (`results_am`, `results_mm`, `results_garch`) consumed by the export cell at the end, so skipping a model cell breaks the Excel export.

## Two notebooks, one codebase

`src/stock_analyze.ipynb` and `src/stock_analyze_teacher.ipynb` share **identical helper functions and cell structure**. They differ only in Cell 3 (`DATE_RANGE`, `EVENT_DATES`) and Cell 9 (CAR chart's `smpl_start`/`smpl_end` and shaded-region/vertical-line dates):

- `stock_analyze_teacher.ipynb` — original 2017-2020 US-China trade war study (37 events).
- `stock_analyze.ipynb` — extended 2024-2026 reconfiguration (26 events) — this is the working/active notebook (see `.omc/project-memory.json` hotPaths).

When changing analysis logic (helpers, model definitions, export format), the change must be applied to **both** notebooks to keep them in sync. When changing only event dates / date ranges / chart annotations, edit only the relevant notebook.

## Data shape and the `Change %` shortcut

The Excel workbook is multi-sheet — one sheet per index. Cell 2 loads all sheets via `pd.read_excel(..., sheet_name=None, usecols=['Date', 'Price', 'Change %'])`, parses each `Change %` column (stripping `%`, coercing to float), and indexes by date.

Crucially, **returns are read directly from the `Change %` column**, not computed from `Price`. The commented-out `compute_returns()` helper in Cell 4a is dead code; do not "fix" the code by switching to price-derived returns without confirming this with the user — the source data already contains percentage returns and re-deriving them from `Price` would double-compute.

`INDICES` is the list of indices under study; `BENCHMARK` (currently `MIWD00000PUS`, the MSCI World) provides the market return used by all three models. Both must exist as sheet names in the workbook.

## Event-study architecture

`run_event_study()` in Cell 4a is the single entry point. For each event date it:

1. Snaps to the first trading day on/after the event.
2. Picks an estimation window of `ESTIMATION_WINDOW` (61) trading days ending `EVENT_WINDOW + 1` (6) days before the event — i.e., the estimation window does **not** overlap the event window.
3. Picks an event window of ±`EVENT_WINDOW` (5) trading days around the event.
4. Computes abnormal returns using one of three `model_type` values:
   - `'am'` (Market Adjusted): `AR = r_index - r_benchmark`.
   - `'mm'` (Market Model): OLS `r_index = α + β·r_benchmark` over the estimation window, then `AR = r_index - (α + β·r_benchmark)`.
   - `'garch'`: ARX(0)-GARCH(1,1) via the `arch` library, mean = `c + β·r_benchmark`, no GARCH-in-mean.
5. Returns `(ar_df, tstat_df)` indexed by relative day `-5..+5`, columns are event numbers `1..N`.

`build_results_table()` / `build_all_tables()` / `export_to_excel()` flatten these per-model dicts into a single multi-sheet workbook. Sheet names are truncated to 31 chars (Excel limit) — keep model labels short when adding new ones.

`fit_garch_subperiod()` (Cell 10) is a separate analysis: AR(1)-AR(9)+GARCH(1,1) on each index over hardcoded `SUB_PERIODS`, then a refitted model keeping only AR lags with `p < 0.05`. Output is regression summary tables, not an event study.

## Gotchas

- `DATE_RANGE` in `stock_analyze.ipynb` extends to 2026-05-09 but several sheets only go to 2026-03-20 — `returns.dropna(subset=...)` inside `run_event_study` handles this per-index, so events at the tail end may be silently skipped if neither the index nor the benchmark has post-event data for the full ±5 window.
- `warnings.filterwarnings('ignore')` is global — convergence warnings from `arch` and `statsmodels` are suppressed. If GARCH results look wrong, comment that line out before re-running.
- The notebooks contain large embedded outputs (~700 KB each). Prefer `Read` with `limit` or the `python3 -c "import json; ..."` pattern when inspecting cells; do not pipe full notebooks through `cat`.
- `display(...)` is Jupyter-only — running the cells outside Jupyter requires replacing `display(df)` with `print(df)`.
