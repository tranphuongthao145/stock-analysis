# Event-Study Notebook Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three SE/t-stat bugs and add an AAR/CAAR aggregation panel with the BMP cross-sectional test in `src/stock_analyze.ipynb`, leaving `src/stock_analyze_teacher.ipynb` untouched.

**Architecture:** Surgical edits to Cell 4a (helpers) propagate via a 4-tuple return signature change; Cells 6/7/8 adapt to the new signature; a new Cell 8b is inserted to compute AAR/CAAR/BMP; Cells 9/10/11 receive small targeted fixes. No new files are created in `src/`.

**Tech Stack:** Jupyter notebook (.ipynb JSON), Python 3, `pandas`, `numpy`, `statsmodels`, `arch` (GARCH), `scipy.stats`, `matplotlib`, `openpyxl`. Notebook is normally executed on Google Colab; for validation we use the `NotebookEdit` tool to edit cells and a standalone Python smoke script that loads `data/data_full.xlsx` directly (the local sheet schema matches Colab — same source).

**Spec:** `docs/superpowers/specs/2026-05-09-event-study-fixes-design.md`. Read it before starting if unfamiliar.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `src/stock_analyze.ipynb` | Modify (cells 1, 4a, 6, 7, 8, 9, 10, 11) + insert one new cell (8b) | Active student notebook — all changes happen here |
| `src/stock_analyze_teacher.ipynb` | **DO NOT TOUCH** | Reference notebook |
| `/tmp/smoke_event_study.py` | Create (temporary, removed after Task 8) | Standalone smoke test: imports data, runs new helpers, asserts shapes |
| `docs/superpowers/specs/2026-05-09-event-study-fixes-design.md` | Read-only reference | Design spec already approved |

**Cell index in `src/stock_analyze.ipynb` after all changes:**

| Index | Title | Status |
|---|---|---|
| 0 | Cell 1 — Install & Import | modified (add scipy import) |
| 1 | Cell 2 — Mount Drive & Load Data | unchanged |
| 2 | Cell 3 — Configuration | unchanged |
| 3 | Cell 4a — Helper Functions | rewritten |
| 4 | Cell 5 — Load Returns | unchanged |
| 5 | Cell 6 — Market Adjusted Model | modified (unpack) |
| 6 | Cell 7 — Market Model | modified (unpack) |
| 7 | Cell 8 — GARCH Model | modified (unpack) |
| **8** | **Cell 8b — AAR/CAAR Aggregation (new)** | **inserted** |
| 9 | Cell 9 — CAR Charts | modified (label + title) |
| 10 | `returns_smpl.head()` | unchanged |
| 11 | Cell 10 — GARCH Sub-Period | modified (sub-periods) |
| 12 | Cell 11 — Export | modified (CAAR sheets + dropped sheet) |

**Critical:** insertion of Cell 8b shifts later indices by +1. The `NotebookEdit` tool with `edit_mode=insert` and `cell_id` of Cell 8 handles this automatically.

---

## Task 1: Cell 1 — Add `scipy.stats.norm` import

**Files:**
- Modify: `src/stock_analyze.ipynb` cell index 0

- [ ] **Step 1: Read current Cell 1 to confirm contents**

Run:
```bash
python3 -c "import json; nb = json.load(open('src/stock_analyze.ipynb')); print(''.join(nb['cells'][0]['source']))"
```

Expected: a block starting `# Cell 1 — Install & Import` with `%pip install` and `import` statements but no `from scipy.stats import norm`.

- [ ] **Step 2: Replace Cell 1 with new content via NotebookEdit**

Tool call:
```
NotebookEdit(
  notebook_path="/Users/khango/work/stock-analysis/src/stock_analyze.ipynb",
  cell_id="<cell_id of cell 0 — read from JSON if needed>",
  edit_mode="replace",
  cell_type="code",
  new_source="""# Cell 1 — Install & Import
%pip install -q pandas openpyxl numpy statsmodels arch matplotlib scipy

import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
"""
)
```

If the executor's `NotebookEdit` requires the cell's `id` field, fetch it with:
```bash
python3 -c "import json; nb = json.load(open('src/stock_analyze.ipynb')); print(nb['cells'][0].get('id', '(no id)'))"
```

If the cell has no `id` field, use a positional approach: read the cell, modify the `source` array directly via Python:
```bash
python3 <<'EOF'
import json
path = '/Users/khango/work/stock-analysis/src/stock_analyze.ipynb'
nb = json.load(open(path))
nb['cells'][0]['source'] = """# Cell 1 — Install & Import
%pip install -q pandas openpyxl numpy statsmodels arch matplotlib scipy

import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
""".splitlines(keepends=True)
json.dump(nb, open(path, 'w'), indent=1)
EOF
```

- [ ] **Step 3: Verify the edit**

Run:
```bash
python3 -c "import json; nb = json.load(open('src/stock_analyze.ipynb')); src = ''.join(nb['cells'][0]['source']); assert 'from scipy.stats import norm' in src, 'import not found'; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/stock_analyze.ipynb
git commit -m "Cell 1: add scipy.stats.norm import for BMP test"
```

---

## Task 2: Cell 4a — Rewrite helpers (Fix #1, Fix #2, Fix #3, add bmp_test)

This is the largest task. We replace the entire Cell 4a content in one atomic edit. Sub-changes inside:

1. **Fix #1** — AM uses its own estimation-window SE.
2. **Fix #2** — `compute_ar_garch` returns per-day conditional σ as a Series.
3. **Fix #3** — `run_event_study` returns `(ar_df, tstat_df, se_df, dropped)`.
4. **build_all_tables** — reads `se_df` directly (no back-computation hack).
5. **`bmp_test`** — new helper, sum-of-variances standardizer, two-tailed normal p-value.

**Files:**
- Modify: `src/stock_analyze.ipynb` cell index 3

- [ ] **Step 1: Read current Cell 4a to confirm starting point**

```bash
python3 -c "import json; nb = json.load(open('src/stock_analyze.ipynb')); print(''.join(nb['cells'][3]['source'])[:200])"
```

Expected output starts with `# Cell 4a — Helper Functions`.

- [ ] **Step 2: Replace Cell 4a with new content**

Use `NotebookEdit` (replace mode) on cell index 3. Full new source:

```python
# Cell 4a — Helper Functions

# def compute_returns(df: pd.DataFrame, columns: list[str]):
#     """Compute percentage returns against last non-null value: (P_t - P_{t-1}) / P_{t-1} * 100"""
#     prices = df[columns].copy()
#     returns = pd.DataFrame(index=prices.index)
#     for col in columns:
#         col_prices = prices[col].dropna()
#         returns[col] = col_prices.pct_change() * 100
#     return returns


def estimate_market_model_ols(returns, index_col, benchmark_col, est_start, est_end):
    """OLS r_index = alpha + beta * r_benchmark over [est_start, est_end]."""
    est_data = returns.loc[est_start:est_end, [index_col, benchmark_col]].dropna()
    y = est_data[index_col]
    X = sm.add_constant(est_data[benchmark_col])
    model = sm.OLS(y, X).fit()
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    residual_se = np.sqrt(model.mse_resid)
    return alpha, beta, residual_se


def compute_ar_market_adjusted(returns, index_col, benchmark_col):
    """AR = r_index - r_benchmark."""
    return returns[index_col] - returns[benchmark_col]


def compute_ar_market_model(returns, index_col, benchmark_col, alpha, beta):
    """AR = r_index - (alpha + beta * r_benchmark)."""
    return returns[index_col] - (alpha + beta * returns[benchmark_col])


def compute_ar_garch(returns, index_col, benchmark_col, est_start, est_end, event_dates_window):
    """
    GARCH(1,1) with mean: r_index = c + beta * r_benchmark.
    Estimated only on [est_start, est_end].

    Returns: (ar series over full returns, conditional_vol Series indexed by event_dates_window)
    """
    est_data = returns.loc[est_start:est_end, [index_col, benchmark_col]].dropna()
    y = est_data[index_col]
    x_exog = est_data[[benchmark_col]]

    am = arch_model(y, x=x_exog, mean='ARX', lags=0, vol='GARCH', p=1, q=1)
    res = am.fit(disp='off')

    alpha = res.params.get('Const', 0.0)
    beta_bench = res.params.get(benchmark_col, 0.0)
    fitted_mean = alpha + beta_bench * returns[benchmark_col]
    ar = returns[index_col] - fitted_mean

    horizon = len(event_dates_window)
    fc = res.forecast(horizon=horizon, reindex=False)
    cond_var = fc.variance.iloc[-1].values
    cond_vol = pd.Series(np.sqrt(cond_var), index=event_dates_window)

    if cond_vol.std() < 1e-8 or cond_vol.isna().any():
        unconditional_se = float(np.sqrt(np.sum(res.resid ** 2) / (len(res.resid) - len(res.params))))
        cond_vol = pd.Series(unconditional_se, index=event_dates_window)
        print(f"  WARNING: GARCH cond-vol forecast collapsed for {index_col}; using unconditional SE.")

    return ar, cond_vol


def run_event_study(returns: pd.DataFrame, event_dates, index_col, benchmark_col,
                    model_type, estimation_window=61, event_window=5):
    """
    Returns: (ar_df, tstat_df, se_df, dropped)
      - ar_df, tstat_df, se_df: relative_day x event_number DataFrames
      - dropped: list of (event_date, reason) for events skipped
    """
    trading_dates = returns.dropna(subset=[index_col, benchmark_col]).index
    ar_dict, tstat_dict, se_dict = {}, {}, {}
    dropped = []

    for k, event_date in enumerate(event_dates, start=1):
        valid_dates = trading_dates[trading_dates >= event_date]
        if len(valid_dates) == 0:
            dropped.append((event_date, 'no_trading_day_on_or_after_event'))
            continue

        event_idx_date = valid_dates[0]
        event_pos = trading_dates.get_loc(event_idx_date)
        est_end_pos = event_pos - event_window - 1
        est_start_pos = est_end_pos - estimation_window + 1

        if est_start_pos < 0 or est_end_pos < 0:
            dropped.append((event_date, 'estimation_window_starts_before_data'))
            continue

        est_start = trading_dates[est_start_pos]
        est_end = trading_dates[est_end_pos]
        ev_start_pos = event_pos - event_window
        ev_end_pos = event_pos + event_window

        if ev_start_pos < 0 or ev_end_pos >= len(trading_dates):
            dropped.append((event_date, 'event_window_extends_past_data_tail'))
            continue

        ev_dates = trading_dates[ev_start_pos:ev_end_pos + 1]
        alpha, beta, ols_se = estimate_market_model_ols(
            returns, index_col, benchmark_col, est_start, est_end
        )

        if model_type == 'am':
            ar_full = compute_ar_market_adjusted(returns, index_col, benchmark_col)
            se = ar_full.loc[est_start:est_end].std(ddof=1)
        elif model_type == 'mm':
            ar_full = compute_ar_market_model(returns, index_col, benchmark_col, alpha, beta)
            se = ols_se
        elif model_type == 'garch':
            ar_full, se = compute_ar_garch(
                returns, index_col, benchmark_col, est_start, est_end,
                event_dates_window=ev_dates
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        ar_window = ar_full.loc[ev_dates]
        relative_days = list(range(-event_window, event_window + 1))

        if len(ar_window) != len(relative_days):
            dropped.append((event_date, 'event_window_partially_missing_returns'))
            continue

        if isinstance(se, pd.Series):
            se_window = se.loc[ev_dates].values
        else:
            se_window = np.full(len(ev_dates), float(se))

        ar_dict[k] = pd.Series(ar_window.values, index=relative_days)
        tstat_dict[k] = pd.Series(ar_window.values / se_window, index=relative_days)
        se_dict[k] = pd.Series(se_window, index=relative_days)

    ar_df = pd.DataFrame(ar_dict)
    ar_df.index.name = 'relative_day'
    tstat_df = pd.DataFrame(tstat_dict)
    tstat_df.index.name = 'relative_day'
    se_df = pd.DataFrame(se_dict)
    se_df.index.name = 'relative_day'
    return ar_df, tstat_df, se_df, dropped


def build_results_table(ar_df, se_df):
    """Combine AR / SE / t-stat sections into a display-ready DataFrame."""
    ar_section = ar_df.copy()
    ar_section.index = [f"AR(t={d:+d})" for d in ar_section.index]

    se_section = se_df.copy()
    se_section.index = [f"SE(t={d:+d})" for d in se_section.index]

    tstat_section = (ar_df / se_df).copy()
    tstat_section.index = [f"t(t={d:+d})" for d in ar_df.index]

    sep1 = pd.DataFrame({c: ['---'] for c in ar_df.columns}, index=['---'])
    sep2 = pd.DataFrame({c: ['---'] for c in ar_df.columns}, index=['---2'])
    return pd.concat([ar_section, sep1, se_section, sep2, tstat_section])


def build_all_tables(results_dict, model_label):
    """Build display tables for all indices in a results dict."""
    tables = {}
    for name, (ar_df, _tstat_df, se_df, _dropped) in results_dict.items():
        if ar_df.empty:
            continue
        tables[f"{model_label}_{name}"] = build_results_table(ar_df, se_df)
    return tables


def export_to_excel(tables_dict, filename):
    """Write result tables to a multi-sheet Excel file."""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, df in tables_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31])
    print(f"Exported {len(tables_dict)} sheets to {filename}")


def bmp_test(ar_df, se_df, window):
    """
    Boehmer-Musumeci-Poulsen (1991) cross-sectional t-test for CAAR over [window[0], window[1]].

    SCAR_k = CAR_k / sqrt(sum(sigma_t^2))   # sum-of-variances standardizer
    t = mean(SCAR) / (std(SCAR) / sqrt(N))
    """
    relative_days = list(range(window[0], window[1] + 1))

    scar_list = []
    for event in ar_df.columns:
        car = ar_df.loc[relative_days, event].sum()
        sigma_squared_sum = (se_df.loc[relative_days, event] ** 2).sum()
        if sigma_squared_sum <= 0 or np.isnan(sigma_squared_sum):
            continue
        scar_list.append(car / np.sqrt(sigma_squared_sum))

    if len(scar_list) < 2:
        return {'caar': float('nan'), 'scar_mean': float('nan'),
                't_stat': float('nan'), 'p_value': float('nan'),
                'n_events': len(scar_list)}

    scar = np.array(scar_list)
    n = len(scar)
    mean_scar = scar.mean()
    std_scar = scar.std(ddof=1)
    t = mean_scar / (std_scar / np.sqrt(n))
    p = 2 * (1 - norm.cdf(abs(t)))
    caar = ar_df.loc[relative_days].sum().mean()

    return {'caar': caar, 'scar_mean': mean_scar, 't_stat': t,
            'p_value': p, 'n_events': n}


def fit_garch_subperiod(returns, index_col, period_start, period_end, max_ar_lags=9):
    """
    Sub-period AR-GARCH(1,1) fit.
    Step 1: AR(1)-AR(max_ar_lags) + GARCH(1,1)
    Step 2: refit with only AR lags significant at p<0.05
    """
    sub_returns = returns.loc[period_start:period_end, index_col].dropna()

    full_model = arch_model(sub_returns, mean='ARX', lags=max_ar_lags, vol='GARCH', p=1, q=1)
    full_result = full_model.fit(disp='off')

    significant_lags = []
    for i in range(1, max_ar_lags + 1):
        for param_name in [f'y[{i}]', f'AR[{i}]']:
            if param_name in full_result.pvalues.index:
                if full_result.pvalues[param_name] < 0.05:
                    significant_lags.append(i)
                break

    if significant_lags:
        refined_model = arch_model(sub_returns, mean='ARX', lags=significant_lags, vol='GARCH', p=1, q=1)
    else:
        refined_model = arch_model(sub_returns, mean='Constant', vol='GARCH', p=1, q=1)
    refined_result = refined_model.fit(disp='off')
    return full_result, refined_result
```

- [ ] **Step 3: Verify Cell 4a parses and the public symbols exist**

Run:
```bash
python3 <<'EOF'
import json, ast
nb = json.load(open('/Users/khango/work/stock-analysis/src/stock_analyze.ipynb'))
src = ''.join(nb['cells'][3]['source'])
tree = ast.parse(src)
defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
expected = {'estimate_market_model_ols','compute_ar_market_adjusted','compute_ar_market_model',
            'compute_ar_garch','run_event_study','build_results_table','build_all_tables',
            'export_to_excel','bmp_test','fit_garch_subperiod'}
missing = expected - defs
assert not missing, f"Missing functions: {missing}"
print("OK — all expected functions defined")
EOF
```

Expected: `OK — all expected functions defined`.

- [ ] **Step 4: Smoke-test helpers against synthetic data**

Create `/tmp/smoke_event_study.py`:

```python
"""Standalone smoke test for Cell 4a helpers — synthetic returns."""
import json, ast, sys
import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import norm
import statsmodels.api as sm
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Extract Cell 4a source from notebook and exec into globals
nb = json.load(open('/Users/khango/work/stock-analysis/src/stock_analyze.ipynb'))
exec(''.join(nb['cells'][3]['source']), globals())

# Synthetic data: 400 trading days, 2 indices + benchmark
np.random.seed(42)
dates = pd.bdate_range('2024-01-01', periods=400)
returns = pd.DataFrame({
    'IDX_A': np.random.randn(400) * 1.0,
    'IDX_B': np.random.randn(400) * 1.5,
    'BENCH': np.random.randn(400) * 0.8,
}, index=dates)

# IDX_A correlated with bench, IDX_B independent
returns['IDX_A'] = 0.7 * returns['BENCH'] + 0.5 * np.random.randn(400)

events = [date(2024, 6, 1), date(2024, 9, 15), date(2025, 1, 10)]

for model in ['am', 'mm', 'garch']:
    ar, tstat, se, dropped = run_event_study(returns, events, 'IDX_A', 'BENCH',
                                              model_type=model,
                                              estimation_window=61, event_window=5)
    assert ar.shape == (11, len(events) - len(dropped)), f"{model}: ar shape wrong: {ar.shape}"
    assert se.shape == ar.shape, f"{model}: se shape wrong"
    assert tstat.shape == ar.shape, f"{model}: tstat shape wrong"
    if model == 'garch':
        # Per-day SE should vary across the 11-day window for at least one event
        col = ar.columns[0]
        assert se[col].std() > 1e-6 or len(dropped) > 0, "GARCH per-day cond σ is flat — Fix #2 broken"
    print(f"{model}: ar.shape={ar.shape}, dropped={len(dropped)}, se.mean={se.mean().mean():.4f}")

# AM SE should differ from MM SE for IDX_A (correlated with bench)
ar_am, _, se_am, _ = run_event_study(returns, events, 'IDX_A', 'BENCH', model_type='am')
ar_mm, _, se_mm, _ = run_event_study(returns, events, 'IDX_A', 'BENCH', model_type='mm')
assert (se_am.iloc[0, 0] != se_mm.iloc[0, 0]), "Fix #1 broken: AM SE == MM SE"
print(f"Fix #1 OK: AM SE={se_am.iloc[0,0]:.4f}, MM SE={se_mm.iloc[0,0]:.4f}")

# BMP test smoke
result = bmp_test(ar_am, se_am, (-1, 1))
assert np.isfinite(result['t_stat']), "BMP t_stat is not finite"
assert result['n_events'] == ar_am.shape[1], "BMP n_events mismatch"
print(f"BMP test OK: caar={result['caar']:.4f}, t={result['t_stat']:.2f}, n={result['n_events']}")

print("\nALL SMOKE TESTS PASSED")
```

Run:
```bash
python3 /tmp/smoke_event_study.py
```

Expected: `ALL SMOKE TESTS PASSED` and no `AssertionError`.

If the GARCH per-day σ assertion fails, see spec §6.2 — the forecast horizon must equal `len(event_dates_window)` (11 days).

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyze.ipynb
git commit -m "Cell 4a: fix AM SE bug, GARCH cond σ, return signature; add bmp_test"
```

---

## Task 3: Cells 6/7/8 — adapt to new return signature

**Files:**
- Modify: `src/stock_analyze.ipynb` cell indices 5, 6, 7

- [ ] **Step 1: Replace Cell 6 (Market Adjusted) source**

Use `NotebookEdit` on cell index 5 with new source:

```python
# Cell 6 — Market Adjusted Model (AM)
results_am = {}

print("=" * 60)
print("MARKET ADJUSTED MODEL — Abnormal Returns")
print("=" * 60)

for col in INDICES:
    print(f"\n--- {col.upper()} ---")
    ar_df, tstat_df, se_df, dropped = run_event_study(
        returns, EVENT_DATES, col, BENCHMARK,
        model_type='am',
        estimation_window=ESTIMATION_WINDOW,
        event_window=EVENT_WINDOW
    )
    results_am[col] = (ar_df, tstat_df, se_df, dropped)
    display(ar_df.round(4))
    if dropped:
        print(f"  Dropped events: {len(dropped)} — see Dropped_Events sheet for reasons.")

print(f"\nCompleted Market Adjusted model for {len(results_am)} indices.")
```

- [ ] **Step 2: Replace Cell 7 (Market Model) source**

Use `NotebookEdit` on cell index 6:

```python
# Cell 7 — Market Model (MM)
results_mm = {}

print("=" * 60)
print("MARKET MODEL — Abnormal Returns")
print("=" * 60)

for col in INDICES:
    print(f"\n--- {col.upper()} ---")
    ar_df, tstat_df, se_df, dropped = run_event_study(
        returns, EVENT_DATES, col, BENCHMARK,
        model_type='mm',
        estimation_window=ESTIMATION_WINDOW,
        event_window=EVENT_WINDOW
    )
    results_mm[col] = (ar_df, tstat_df, se_df, dropped)
    display(ar_df.round(4))
    if dropped:
        print(f"  Dropped events: {len(dropped)} — see Dropped_Events sheet for reasons.")

print(f"\nCompleted Market Model for {len(results_mm)} indices.")
```

- [ ] **Step 3: Replace Cell 8 (GARCH) source**

Use `NotebookEdit` on cell index 7:

```python
# Cell 8 — GARCH Model (G) — Event Study
results_garch = {}

print("=" * 60)
print("GARCH(1,1) MODEL — Abnormal Returns")
print("=" * 60)
print("(ARX(0)-mean GARCH(1,1) — per-day conditional σ used for t-stats)")

for col in INDICES:
    print(f"\n--- {col.upper()} ---")
    try:
        ar_df, tstat_df, se_df, dropped = run_event_study(
            returns, EVENT_DATES, col, BENCHMARK,
            model_type='garch',
            estimation_window=ESTIMATION_WINDOW,
            event_window=EVENT_WINDOW
        )
        results_garch[col] = (ar_df, tstat_df, se_df, dropped)
        display(ar_df.round(4))
        if dropped:
            print(f"  Dropped events: {len(dropped)} — see Dropped_Events sheet for reasons.")
    except Exception as e:
        print(f"  GARCH estimation failed for {col}: {e}")
        results_garch[col] = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])

print(f"\nCompleted GARCH model for {len(results_garch)} indices.")
```

- [ ] **Step 4: Verify all three cells parse and reference the new 4-tuple**

Run:
```bash
python3 <<'EOF'
import json, ast
nb = json.load(open('/Users/khango/work/stock-analysis/src/stock_analyze.ipynb'))
for idx, label in [(5, 'Cell 6'), (6, 'Cell 7'), (7, 'Cell 8')]:
    src = ''.join(nb['cells'][idx]['source'])
    ast.parse(src)
    assert 'ar_df, tstat_df, se_df, dropped' in src, f"{label}: 4-tuple unpack missing"
    assert 'results_' in src and '(ar_df, tstat_df, se_df, dropped)' in src, f"{label}: dict assignment wrong"
    print(f"{label}: OK")
EOF
```

Expected: `Cell 6: OK`, `Cell 7: OK`, `Cell 8: OK`.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyze.ipynb
git commit -m "Cells 6/7/8: adapt to 4-tuple return from run_event_study"
```

---

## Task 4: Insert new Cell 8b — AAR/CAAR aggregation

**Files:**
- Modify: `src/stock_analyze.ipynb` — insert new cell after index 7

- [ ] **Step 1: Insert new cell via NotebookEdit**

Use `NotebookEdit` with `edit_mode=insert`, `cell_id=<id of cell index 7>` (the GARCH cell), `cell_type=code`. New source:

```python
# Cell 8b — AAR/CAAR aggregation with BMP cross-sectional test
#
# NOTE on event clustering:
# Several events in this study (notably 2025-04-02 / 04-05 / 04-09 / 04-15) fall within
# overlapping ±5 trading-day windows. Standardized abnormal returns across these events
# are positively correlated, which biases the BMP t-statistic toward rejection of the
# null. CAAR magnitudes are unaffected; only the significance levels should be
# interpreted with this caveat.

CAAR_WINDOWS = [(-5, -1), (-1, 1), (0, 5), (-5, 5)]


def build_caar_panel(results_dict):
    caar_rows, tstat_rows, n_rows = {}, {}, {}
    for index_name, (ar_df, _tstat, se_df, _dropped) in results_dict.items():
        if ar_df.empty:
            continue
        caar_row, tstat_row, n_row = {}, {}, {}
        for window in CAAR_WINDOWS:
            r = bmp_test(ar_df, se_df, window)
            label = f"CAAR[{window[0]:+d},{window[1]:+d}]"
            caar_row[label] = r['caar']
            tstat_row[label] = r['t_stat']
            n_row[label] = r['n_events']
        caar_rows[index_name] = caar_row
        tstat_rows[index_name] = tstat_row
        n_rows[index_name] = n_row
    return (pd.DataFrame(caar_rows).T,
            pd.DataFrame(tstat_rows).T,
            pd.DataFrame(n_rows).T)


def add_significance_stars(caar_df, tstat_df):
    out = caar_df.copy().astype(object)
    for i in caar_df.index:
        for c in caar_df.columns:
            t = abs(tstat_df.loc[i, c])
            stars = '***' if t > 2.576 else ('**' if t > 1.96 else ('*' if t > 1.645 else ''))
            out.loc[i, c] = f"{caar_df.loc[i, c]:+.2f}{stars}"
    return out


caar_am,    tstat_caar_am,    n_caar_am    = build_caar_panel(results_am)
caar_mm,    tstat_caar_mm,    n_caar_mm    = build_caar_panel(results_mm)
caar_garch, tstat_caar_garch, n_caar_garch = build_caar_panel(results_garch)

for label, caar, tstat, n in [
    ('AM',    caar_am,    tstat_caar_am,    n_caar_am),
    ('MM',    caar_mm,    tstat_caar_mm,    n_caar_mm),
    ('GARCH', caar_garch, tstat_caar_garch, n_caar_garch),
]:
    print(f"\n{'='*60}\n{label} — CAAR panel (stars: * p<0.10, ** p<0.05, *** p<0.01)")
    print(f"{'='*60}")
    display(add_significance_stars(caar, tstat))
    print(f"BMP t-statistics ({label}):")
    display(tstat.round(2))
    print(f"N events used per window ({label}):")
    display(n)

# Sanity plot: GARCH conditional σ over event window for Liberation Day on S&P500
target_event_date = date.fromisoformat('2025-04-02')
target_event_idx = next((k for k, ev in enumerate(EVENT_DATES, start=1)
                         if ev == target_event_date), None)
if (target_event_idx is not None
    and 'S&P500' in results_garch
    and target_event_idx in results_garch['S&P500'][2].columns):
    fig, ax = plt.subplots(figsize=(8, 3))
    se_series = results_garch['S&P500'][2][target_event_idx]
    ax.plot(se_series.index, se_series.values, marker='o')
    ax.set_title(f'GARCH conditional σ over event window — S&P500, {target_event_date}')
    ax.set_xlabel('Relative day')
    ax.set_ylabel('Conditional σ')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(f"Sanity plot skipped: {target_event_date} not in event list or GARCH dropped it for S&P500.")
```

- [ ] **Step 2: Verify the cell was inserted at the right position**

```bash
python3 <<'EOF'
import json
nb = json.load(open('/Users/khango/work/stock-analysis/src/stock_analyze.ipynb'))
# Old structure: cell 7 = GARCH, cell 8 = CAR Charts
# New structure: cell 7 = GARCH, cell 8 = NEW Cell 8b, cell 9 = CAR Charts
src8 = ''.join(nb['cells'][8]['source'])
assert 'Cell 8b' in src8, f"Cell 8b not at index 8; got: {src8[:80]}"
src9 = ''.join(nb['cells'][9]['source'])
assert 'Cell 9' in src9 and 'CAR Charts' in src9, f"CAR Charts cell expected at index 9; got: {src9[:80]}"
print("Cell 8b inserted at index 8; Cell 9 (CAR Charts) shifted to index 9")
EOF
```

Expected: `Cell 8b inserted at index 8; Cell 9 (CAR Charts) shifted to index 9`.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analyze.ipynb
git commit -m "Cell 8b (new): AAR/CAAR aggregation panel with BMP test + GARCH cond-σ sanity plot"
```

---

## Task 5: Cell 9 — fix chart label and title

**Files:**
- Modify: `src/stock_analyze.ipynb` cell index 9 (was 8 before Task 4)

- [ ] **Step 1: Replace Cell 9 source**

Use `NotebookEdit` on cell index 9:

```python
# Cell 9 — CAR Charts
# smpl_start = date.fromisoformat('2017-08-01')
# smpl_end = date.fromisoformat('2020-12-31')
smpl_start = date.fromisoformat('2024-01-01')
smpl_end = date.fromisoformat('2026-05-09')
smpl_mask = (returns.index >= smpl_start) & (returns.index <= smpl_end)
returns_smpl = returns.loc[smpl_mask]

fig, ax = plt.subplots(figsize=(14, 7))

for col in INDICES:
    ar = returns_smpl[col] - returns_smpl[BENCHMARK]
    car = ar.cumsum()
    ax.plot(car.index, car.values, label=col, linewidth=1.2)

ax.axvspan(
    pd.Timestamp('2025-02-11'), pd.Timestamp('2026-04-02'),
    alpha=0.1, color='gray', label='Trade War Period'
)
ax.axvline(
    pd.Timestamp('2025-04-02'),
    color='red', linestyle='--', linewidth=1.5,
    label='Apr 2, 2025 — Reciprocal Tariffs Announced'
)

ax.set_title('Cumulative Excess Return vs. MSCI ACWI (Daily, AM-style)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative excess return (%)')
ax.legend(loc='best', fontsize=10)
ax.tick_params(axis='x', rotation=65)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Verify the stale label is gone and the new label is in**

```bash
python3 <<'EOF'
import json
nb = json.load(open('/Users/khango/work/stock-analysis/src/stock_analyze.ipynb'))
src = ''.join(nb['cells'][9]['source'])
assert 'May 13, 2019' not in src, "Stale label 'May 13, 2019' still present"
assert 'Apr 2, 2025' in src, "New 2025-04-02 label missing"
assert 'Cumulative Excess Return vs. MSCI ACWI' in src, "New title missing"
print("Cell 9 chart fix OK")
EOF
```

Expected: `Cell 9 chart fix OK`.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analyze.ipynb
git commit -m "Cell 9: fix stale 2019 label, retitle as cumulative excess return vs ACWI"
```

---

## Task 6: Cell 10 — update sub-periods to 2024-2026

**Files:**
- Modify: `src/stock_analyze.ipynb` cell index 11 (after Task 4 insertion)

- [ ] **Step 1: Confirm Cell 10 location**

```bash
python3 -c "import json; nb=json.load(open('src/stock_analyze.ipynb')); print(''.join(nb['cells'][11]['source'])[:80])"
```

Expected: starts with `# Cell 10 — GARCH Sub-Period Analysis`. If a different cell appears, scan to locate it:

```bash
python3 <<'EOF'
import json
nb = json.load(open('src/stock_analyze.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'Cell 10' in src and 'GARCH Sub-Period' in src:
        print(f"Cell 10 at index {i}")
EOF
```

Use the printed index in the next step.

- [ ] **Step 2: Replace Cell 10 source**

Use `NotebookEdit` on the located cell index:

```python
# Cell 10 — GARCH Sub-Period Analysis
SUB_PERIODS = {
    'Pre-Liberation':    (date.fromisoformat('2024-08-07'), date.fromisoformat('2025-04-01')),
    'Liberation-onward': (date.fromisoformat('2025-04-02'), date.fromisoformat('2026-03-20')),
}

print("=" * 60)
print("GARCH SUB-PERIOD ANALYSIS")
print("=" * 60)
print("Full model: AR(1)–AR(9) + GARCH(1,1)")
print("Refined model: significant AR lags only + GARCH(1,1)")

for col in INDICES:
    print(f"\n{'='*60}")
    print(f"INDEX: {col.upper()}")
    print(f"{'='*60}")

    for period_name, (p_start, p_end) in SUB_PERIODS.items():
        print(f"\n  --- {period_name} ({p_start} to {p_end}) ---")
        try:
            full_res, refined_res = fit_garch_subperiod(
                returns, col, p_start, p_end, max_ar_lags=9
            )
            print(f"\n  Full Model Summary:")
            print(full_res.summary().tables[1])
            print(f"\n  Refined Model Summary:")
            print(refined_res.summary().tables[1])
        except Exception as e:
            print(f"  GARCH estimation failed: {e}")
```

- [ ] **Step 3: Verify the new dates are in and old ones are out**

```bash
python3 <<'EOF'
import json
nb = json.load(open('src/stock_analyze.ipynb'))
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'SUB_PERIODS' in src:
        assert '2024-08-07' in src and '2026-03-20' in src, "New sub-periods missing"
        assert "'2017-08-01'" not in src and "'2018-03-01'" not in src, "Old 2017-2020 dates still present"
        print("Cell 10 sub-periods updated OK")
        break
EOF
```

Expected: `Cell 10 sub-periods updated OK`.

- [ ] **Step 4: Commit**

```bash
git add src/stock_analyze.ipynb
git commit -m "Cell 10: replace 2017-2020 sub-periods with 2024-2026 split"
```

---

## Task 7: Cell 11 — extend export with CAAR sheets and dropped-events diagnostic

**Files:**
- Modify: `src/stock_analyze.ipynb` cell index 12 (Cell 11)

- [ ] **Step 1: Confirm Cell 11 location**

```bash
python3 -c "import json; nb=json.load(open('src/stock_analyze.ipynb')); print(''.join(nb['cells'][12]['source'])[:80])"
```

Expected: starts with `# Cell 11 — Export to Excel`. If not, locate via:

```bash
python3 <<'EOF'
import json
nb = json.load(open('src/stock_analyze.ipynb'))
for i, c in enumerate(nb['cells']):
    if 'Cell 11' in ''.join(c.get('source', [])):
        print(f"Cell 11 at index {i}")
EOF
```

- [ ] **Step 2: Replace Cell 11 source**

Use `NotebookEdit` on located index:

```python
# Cell 11 — Export to Excel

# Per-event sheets (existing)
tables = {}
for label, results in [('AM', results_am), ('MM', results_mm), ('G', results_garch)]:
    tables.update(build_all_tables(results, label))

# CAAR panels (new)
for label, (caar_df, tstat_df) in [
    ('AM', (caar_am, tstat_caar_am)),
    ('MM', (caar_mm, tstat_caar_mm)),
    ('G',  (caar_garch, tstat_caar_garch)),
]:
    tables[f'{label}_CAAR'] = add_significance_stars(caar_df, tstat_df)
    tables[f'{label}_CAAR_tstats'] = tstat_df.round(2)

# Dropped-events diagnostic (new)
dropped_rows = []
for label, results in [('AM', results_am), ('MM', results_mm), ('G', results_garch)]:
    for index_name, (_, _, _, dropped) in results.items():
        for event_date, reason in dropped:
            dropped_rows.append({
                'model': label,
                'index': index_name,
                'event_date': event_date,
                'reason': reason,
            })
tables['Dropped_Events'] = pd.DataFrame(dropped_rows) if dropped_rows else pd.DataFrame(
    columns=['model', 'index', 'event_date', 'reason']
)

OUTPUT_PATH = '/content/drive/MyDrive/data/event_study_results.xlsx'
export_to_excel(tables, OUTPUT_PATH)
```

- [ ] **Step 3: Verify the cell mentions all the new sheet groups**

```bash
python3 <<'EOF'
import json
nb = json.load(open('src/stock_analyze.ipynb'))
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'OUTPUT_PATH' in src and 'export_to_excel' in src:
        for needle in ['AM_CAAR', 'MM_CAAR', 'G_CAAR', 'Dropped_Events', 'add_significance_stars']:
            assert needle in src, f"Cell 11 missing reference to {needle}"
        print("Cell 11 export updated OK")
        break
EOF
```

Expected: `Cell 11 export updated OK`.

- [ ] **Step 4: Commit**

```bash
git add src/stock_analyze.ipynb
git commit -m "Cell 11: export CAAR panels and Dropped_Events diagnostic"
```

---

## Task 8: End-to-end smoke validation

This task verifies the entire notebook is internally consistent — cell ordering, helper definitions, downstream references — without requiring a Colab run.

**Files:**
- Create (temp): `/tmp/notebook_e2e_check.py`
- Delete (cleanup): `/tmp/smoke_event_study.py`, `/tmp/notebook_e2e_check.py`

- [ ] **Step 1: Write end-to-end check script**

`/tmp/notebook_e2e_check.py`:

```python
"""End-to-end consistency check on stock_analyze.ipynb after all edits."""
import json, ast

NB = '/Users/khango/work/stock-analysis/src/stock_analyze.ipynb'
nb = json.load(open(NB))

# 1. Cell ordering — find cells by their leading marker
markers = {}
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    for marker in ['Cell 1 —', 'Cell 2 —', 'Cell 3 —', 'Cell 4a —',
                   'Cell 5 —', 'Cell 6 —', 'Cell 7 —', 'Cell 8 —',
                   'Cell 8b —', 'Cell 9 —', 'Cell 10 —', 'Cell 11 —']:
        if marker in src and marker not in markers:
            markers[marker] = i

required_order = ['Cell 1 —', 'Cell 2 —', 'Cell 3 —', 'Cell 4a —', 'Cell 5 —',
                  'Cell 6 —', 'Cell 7 —', 'Cell 8 —', 'Cell 8b —', 'Cell 9 —',
                  'Cell 10 —', 'Cell 11 —']
for m in required_order:
    assert m in markers, f"Missing marker: {m}"
indices = [markers[m] for m in required_order]
assert indices == sorted(indices), f"Cells out of order: {markers}"
print(f"Cell ordering OK: {markers}")

# 2. Cell 1 has scipy import
src1 = ''.join(nb['cells'][markers['Cell 1 —']]['source'])
assert 'from scipy.stats import norm' in src1, "scipy import missing in Cell 1"

# 3. Cell 4a parses; expected helpers present
src4a = ''.join(nb['cells'][markers['Cell 4a —']]['source'])
ast.parse(src4a)
required_fns = {'run_event_study', 'compute_ar_garch', 'bmp_test',
                'build_results_table', 'build_all_tables'}
fns = {n.name for n in ast.walk(ast.parse(src4a)) if isinstance(n, ast.FunctionDef)}
assert required_fns <= fns, f"Missing helpers: {required_fns - fns}"

# 4. AM SE fix present in run_event_study
assert "ar_full.loc[est_start:est_end].std(ddof=1)" in src4a, "Fix #1 (AM SE) missing"

# 5. GARCH cond-vol Series return path
assert "cond_vol = pd.Series" in src4a, "Fix #2 (cond_vol Series) missing"

# 6. 4-tuple return signature
assert "return ar_df, tstat_df, se_df, dropped" in src4a, "Fix #3 (4-tuple return) missing"

# 7. Cells 6/7/8 unpack 4-tuple
for marker in ['Cell 6 —', 'Cell 7 —', 'Cell 8 —']:
    src = ''.join(nb['cells'][markers[marker]]['source'])
    assert 'ar_df, tstat_df, se_df, dropped' in src, f"{marker} does not unpack 4-tuple"

# 8. Cell 8b references bmp_test, CAAR panels, sanity plot
src8b = ''.join(nb['cells'][markers['Cell 8b —']]['source'])
for needle in ['bmp_test', 'CAAR_WINDOWS', 'add_significance_stars',
               'GARCH conditional', 'caar_am', 'caar_mm', 'caar_garch']:
    assert needle in src8b, f"Cell 8b missing: {needle}"

# 9. Cell 9 retitled, no stale label
src9 = ''.join(nb['cells'][markers['Cell 9 —']]['source'])
assert 'May 13, 2019' not in src9, "Stale 2019 label in Cell 9"
assert 'Apr 2, 2025' in src9, "New 2025-04-02 label missing in Cell 9"

# 10. Cell 10 has new sub-periods
src10 = ''.join(nb['cells'][markers['Cell 10 —']]['source'])
assert "'2024-08-07'" in src10 and "'2026-03-20'" in src10, "New sub-periods missing"
assert "'2018-03-01'" not in src10, "Old 2018 sub-period date still present"

# 11. Cell 11 has CAAR + Dropped_Events
src11 = ''.join(nb['cells'][markers['Cell 11 —']]['source'])
for needle in ['AM_CAAR', 'MM_CAAR', 'G_CAAR', 'Dropped_Events']:
    assert needle in src11, f"Cell 11 missing: {needle}"

# 12. Teacher notebook untouched (verify against git)
import subprocess
diff = subprocess.run(['git', 'diff', '--stat', 'HEAD', '--',
                       'src/stock_analyze_teacher.ipynb'],
                      capture_output=True, text=True,
                      cwd='/Users/khango/work/stock-analysis')
assert not diff.stdout.strip(), f"Teacher notebook modified: {diff.stdout}"

print("\nALL END-TO-END CHECKS PASSED")
```

Run:
```bash
python3 /tmp/notebook_e2e_check.py
```

Expected: a sequence of `OK` lines and final `ALL END-TO-END CHECKS PASSED`.

- [ ] **Step 2: Run the synthetic-data smoke test from Task 2 again to confirm helpers still pass**

```bash
python3 /tmp/smoke_event_study.py
```

Expected: `ALL SMOKE TESTS PASSED`.

- [ ] **Step 3: Clean up temporary scripts**

```bash
rm -f /tmp/smoke_event_study.py /tmp/notebook_e2e_check.py
```

- [ ] **Step 4: Final manual check (optional but recommended)**

If the user has Excel closed (no `~$data_full.xlsx` lockfile), test the full pipeline locally by writing a one-shot driver in `/tmp/run_locally.py`:

```python
"""Run the notebook's logic end-to-end against the local data file."""
import json
from datetime import date
import pandas as pd, numpy as np
import statsmodels.api as sm
from arch import arch_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

NB = '/Users/khango/work/stock-analysis/src/stock_analyze.ipynb'
DATA = '/Users/khango/work/stock-analysis/data/data_full.xlsx'
nb = json.load(open(NB))

# Cells to execute: Cell 4a (helpers), Cell 3 (config), Cell 5 (load), Cells 6-8, 8b
# Skip Cell 1 (already imported above) and Cell 2 (Drive mount).

# Inline replacement for Cell 2: load data from local path
sheets = pd.read_excel(DATA, sheet_name=None, usecols=['Date', 'Price', 'Change %'])
returns_dict = {}
for name, df in sheets.items():
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df['Change %'] = df['Change %'].astype(str).str.rstrip('%').replace('nan', None)
    returns_dict[name] = pd.to_numeric(df['Change %'], errors='coerce')
returns = pd.DataFrame(returns_dict)

def exec_cell(idx):
    exec(''.join(nb['cells'][idx]['source']), globals())

exec_cell(3)   # Cell 4a helpers
exec_cell(2)   # Cell 3 configuration

# Replace display() with print() for headless run
import builtins
builtins.display = print

exec_cell(5)   # Cell 6 (AM)
exec_cell(6)   # Cell 7 (MM)
exec_cell(7)   # Cell 8 (GARCH)
exec_cell(8)   # Cell 8b (CAAR)

print("\nLOCAL PIPELINE RAN END-TO-END")
print(f"AM CAAR shape: {caar_am.shape}")
print(f"MM CAAR shape: {caar_mm.shape}")
print(f"GARCH CAAR shape: {caar_garch.shape}")
```

Run:
```bash
python3 /tmp/run_locally.py
```

Expected: `LOCAL PIPELINE RAN END-TO-END` followed by three CAAR shape lines (each `(N_indices, 4)`).

If Excel is open (lockfile present), skip this step; user can verify on Colab.

- [ ] **Step 5: Final commit (no-op if all changes already committed)**

```bash
git status
```

Expected: `working tree clean` if everything was committed in Tasks 1-7. If anything is staged from validation file edits, do not commit it (validation files are temp).

```bash
rm -f /tmp/run_locally.py
```

---

## Self-Review Checklist (run before declaring plan complete)

- [ ] Every spec requirement maps to a task — §6.1 → Task 2, §6.2 → Task 2, §6.3 → Task 2, §6.4 → Task 2, §7 → Task 4, §8 → Task 5, §9 → Task 6, §10 → Task 7, §11 → Task 8.
- [ ] No `TBD`, `TODO`, or `Add appropriate X`. Searched and clean.
- [ ] Function/method names consistent across tasks: `run_event_study` returns `(ar_df, tstat_df, se_df, dropped)` in Task 2; Tasks 3 and 8 unpack the same names. `bmp_test`, `build_caar_panel`, `add_significance_stars` defined in Tasks 2 and 4 with consistent signatures.
- [ ] Cell index references account for the +1 shift after Task 4 inserts Cell 8b.
- [ ] Teacher notebook never appears in any `git add` or `NotebookEdit` `notebook_path`.
- [ ] Each task ends with a verification step before commit.
