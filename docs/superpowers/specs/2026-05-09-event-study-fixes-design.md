# Event-Study Notebook Fixes — Design

**File scope:** `src/stock_analyze.ipynb` only. The teacher notebook (`src/stock_analyze_teacher.ipynb`) is intentionally untouched.

**Date:** 2026-05-09

**Author:** Brainstormed with Claude; decisions made by repo owner.

---

## 1. Goal

Make the event-study results in `src/stock_analyze.ipynb` statistically defensible for a thesis defense by:

1. Fixing three bugs that bias every t-statistic in the current per-event tables.
2. Adding a consolidated AAR/CAAR result panel that the rest of the writeup can cite.
3. Cleaning up stale annotations and silent-skip behavior.

The benchmark, the event list, and the model line-up (AM / MM / GARCH) stay as-is.

## 2. Decisions Locked In (from brainstorm)

| Decision | Choice |
|---|---|
| Scope | **B** — bug fixes + thesis-grade outputs (not full methodological upgrade) |
| Event clustering | **A** — keep all 26 events; document clustering as a known limitation |
| CAAR windows | `[-5,-1]`, `[-1,+1]`, `[0,+5]`, `[-5,+5]` |
| Cross-sectional test | BMP (1991) only |
| GARCH BMP standardizer | Sum-of-variances form: `sqrt(sum(σ_t²))` |
| Significance thresholds | Two-tailed normal: `*` |t|>1.645, `**` |t|>1.96, `***` |t|>2.576 |
| Implementation approach | Approach 2 — surgical Cell 4a fixes + new aggregation cells |
| Cell 9 anchor date | `2025-04-02` ("Liberation Day") |
| Cell 10 sub-periods | `Pre-Liberation: 2024-08-07 → 2025-04-01`, `Liberation-onward: 2025-04-02 → 2026-03-20` |

## 3. Out of Scope

- Editing `stock_analyze_teacher.ipynb`.
- Benchmark substitution (regional indices per market).
- Lengthening GARCH estimation window beyond 61 days.
- Curating or merging events in the `EVENT_DATES` list.
- Currency normalization of returns (kept as local-currency `Change %` per source data).

## 4. Data Sources & Known Data Concerns

### 4.1 Index identification

| Sheet name | Index | Ticker | Source URL | Currency |
|---|---|---|---|---|
| `S&P500` | S&P 500 | `SPX` | https://www.investing.com/indices/us-spx-500-historical-data | USD |
| `SSE` | Shanghai Composite | `SSEC` | https://www.investing.com/indices/shanghai-composite-historical-data | CNY |
| `MSCINA` | MSCI North America | `MINA00000PUS` | https://www.investing.com/indices/msci-north-america-historical-data | USD |
| `MSCIFE` | MSCI Far East | `MIFA00000PUS` | https://www.investing.com/indices/msci-far-east-historical-data | USD |
| `MSCIEU` | MSCI EU | `MIEC00000PEU` | https://www.investing.com/indices/msci-eu-historical-data | **EUR** |
| `MIWD00000PUS` | MSCI ACWI (All-Country World, Developed + Emerging) — benchmark | `MIWD00000PUS` | https://www.investing.com/indices/msci-world-stock-historical-data | USD |

Returns are read directly from the `Change %` column of each sheet (daily local-currency percent return).

### 4.2 Known data concerns (documented, not corrected, per Scope=B)

These do not change the implementation but should appear in the thesis writeup's *Data & Limitations* section.

**(a) Currency mismatch.** SSE returns are in CNY and MSCIEU returns are in EUR; the benchmark `MIWD00000PUS` and all other indices are in USD. Computing `AR = r_index − r_benchmark` across mismatched currencies bundles the equity reaction with the FX reaction. On a tariff-announcement day where USD strengthens, MSCIEU's local-currency return will look "abnormally negative" relative to a USD benchmark even when the equity reaction is identical. The CNY confound is smaller because CNY is managed.

**(b) MSCINA-S&P500 redundancy.** MSCI North America is ≈95% US large/mid-cap with a small Canadian component, so its returns track the S&P 500 closely. Including both indices is defensible (MSCINA = "broader North America", S&P500 = "US blue chip") but a reviewer may ask why. The CAAR panel will show near-identical numbers for both rows.

**(c) Source provenance.** Investing.com is a retail data aggregator. For a thesis the academic gold standard is Bloomberg / Refinitiv / WRDS-CRSP. The daily `Change %` series for major indices from investing.com is generally clean for an event study, but acknowledge the source in the writeup.

*Note: `MIWD00000PUS` is the MSCI ACWI (All-Country World, includes both developed and emerging markets), confirmed with repo owner. The benchmark therefore properly spans the EM channel relevant to `SSE`; no benchmark-coverage bias on SSE's AR.*

## 5. Cell-Level Change Map

| Cell | Title | Action |
|---|---|---|
| 1 | Install & Import | Add `from scipy.stats import norm` if absent. |
| 2 | Mount Drive & Load Data | Untouched. |
| 3 | Configuration | Untouched. |
| 4a | Helper Functions | Edit — fix SE bugs, add per-day GARCH conditional σ, change `run_event_study` return signature, add `bmp_test` helper. |
| 5 | Load Returns | Untouched. |
| 6 | AM model | Edit — adapt to new return tuple. |
| 7 | MM model | Edit — adapt to new return tuple. |
| 8 | GARCH model | Edit — adapt to new return tuple. |
| 8b (new) | AAR/CAAR with BMP | Add — three CAAR panels (one per model) with significance stars + dropped-events note + GARCH cond-σ sanity plot. |
| 9 | CAR Chart | Edit — fix vertical-line label/date, retitle to "Cumulative Excess Return vs. MSCI World". |
| 10 | GARCH Sub-Period Analysis | Edit — replace `SUB_PERIODS` with 2024-2026 split. |
| 11 | Export to Excel | Edit — add CAAR sheets (`AM_CAAR`, `MM_CAAR`, `G_CAAR`, `*_CAAR_tstats`) and `Dropped_Events` diagnostic sheet. |

## 6. Cell 4a — Detailed Helper Changes

### 6.1 Fix #1: AM uses its own estimation-window SE

```python
# Inside run_event_study, AM branch:
if model_type == 'am':
    ar_full = compute_ar_market_adjusted(returns, index_col, benchmark_col)
    se = ar_full.loc[est_start:est_end].std(ddof=1)
```

Replaces the prior `se = ols_se` which used the MM residual SE for AM t-stats.

### 6.2 Fix #2: GARCH per-day conditional volatility

`compute_ar_garch` is rewritten to accept the event-window dates and return per-day conditional σ:

```python
def compute_ar_garch(returns, index_col, benchmark_col, est_start, est_end, event_dates_window):
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

    if cond_vol.std() < 1e-8:
        unconditional_se = float(np.sqrt(np.sum(res.resid ** 2) / (len(res.resid) - len(res.params))))
        cond_vol = pd.Series(unconditional_se, index=event_dates_window)
        print(f"  WARNING: GARCH cond-vol forecast collapsed for {index_col}; falling back to unconditional SE.")

    return ar, cond_vol
```

The forecast covers the 11-day event window because `res.forecast(horizon=L)` produces forecasts starting one day after `est_end`, and the event window starts at `est_end + 1` (since `est_end_pos = event_pos − event_window − 1`).

### 6.3 Fix #3: `run_event_study` returns SE explicitly

Signature change:

```python
def run_event_study(...) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """
    Returns: (ar_df, tstat_df, se_df, dropped)
      - ar_df: relative_day × event_number
      - tstat_df: relative_day × event_number
      - se_df: relative_day × event_number (constant per column for AM/MM, per-day for GARCH)
      - dropped: list of (event_date, reason) for events skipped due to insufficient data
    """
```

Inside the loop, the t-stat construction handles scalar and Series SE:

```python
if isinstance(se, pd.Series):
    se_window = se.loc[ev_dates].values
else:
    se_window = np.full(len(ev_dates), se)

ar_dict[k] = pd.Series(ar_window.values, index=relative_days)
tstat_dict[k] = pd.Series(ar_window.values / se_window, index=relative_days)
se_dict[k] = pd.Series(se_window, index=relative_days)
```

All four existing `continue` branches in `run_event_study` are replaced with `dropped.append((event_date, reason)); continue`, where `reason` is one of:
- `'no_trading_day_on_or_after_event'`
- `'estimation_window_starts_before_data'`
- `'event_window_extends_past_data_tail'`
- `'event_window_partially_missing_returns'`

### 6.4 New aggregation helper — `bmp_test`

`bmp_test` does its own CAAR computation internally (no separate `aggregate_aar` / `aggregate_caar` helpers needed for the chosen output panel).

```python
def bmp_test(ar_df, se_df, window):
    relative_days = list(range(window[0], window[1] + 1))
    L = len(relative_days)

    scar_list = []
    for event in ar_df.columns:
        car = ar_df.loc[relative_days, event].sum()
        # sum-of-variances standardizer (textbook-correct for variance of a sum)
        sigma_squared_sum = (se_df.loc[relative_days, event] ** 2).sum()
        if sigma_squared_sum <= 0 or np.isnan(sigma_squared_sum):
            continue
        scar_list.append(car / np.sqrt(sigma_squared_sum))

    if len(scar_list) < 2:
        return {'caar': np.nan, 'scar_mean': np.nan, 't_stat': np.nan,
                'p_value': np.nan, 'n_events': len(scar_list)}

    scar = np.array(scar_list)
    n = len(scar)
    mean_scar = scar.mean()
    std_scar = scar.std(ddof=1)
    t = mean_scar / (std_scar / np.sqrt(n))
    p = 2 * (1 - norm.cdf(abs(t)))
    caar = ar_df.loc[relative_days].sum().mean()

    return {'caar': caar, 'scar_mean': mean_scar, 't_stat': t,
            'p_value': p, 'n_events': n}
```

`build_all_tables` is updated to read SE from `se_df` directly, removing the back-computation hack.

## 7. Cell 8b (new) — Aggregation Panel

```python
# Cell 8b — Aggregation: AAR/CAAR with BMP cross-sectional test
#
# NOTE on event clustering:
# Several events in this study (notably 2025-04-02 / 04-05 / 04-09 / 04-15) fall within
# overlapping ±5 trading-day windows. Standardized abnormal returns across these events
# are positively correlated, which biases the BMP t-statistic toward rejection of the
# null. CAAR magnitudes are unaffected; only the significance levels should be
# interpreted with this caveat.

CAAR_WINDOWS = [(-5, -1), (-1, 1), (0, 5), (-5, 5)]

def build_caar_panel(results_dict):
    caar_rows = {}
    tstat_rows = {}
    for index_name, (ar_df, _, se_df, _) in results_dict.items():
        caar_row, tstat_row = {}, {}
        for window in CAAR_WINDOWS:
            r = bmp_test(ar_df, se_df, window)
            label = f"CAAR[{window[0]:+d},{window[1]:+d}]"
            caar_row[label] = r['caar']
            tstat_row[label] = r['t_stat']
        caar_rows[index_name] = caar_row
        tstat_rows[index_name] = tstat_row
    return pd.DataFrame(caar_rows).T, pd.DataFrame(tstat_rows).T

def add_significance_stars(caar_df, tstat_df):
    out = caar_df.copy().astype(str)
    for i in caar_df.index:
        for c in caar_df.columns:
            t = abs(tstat_df.loc[i, c])
            stars = ('***' if t > 2.576 else '**' if t > 1.96
                     else '*' if t > 1.645 else '')
            out.loc[i, c] = f"{caar_df.loc[i, c]:+.2f}{stars}"
    return out

caar_am,    tstat_caar_am    = build_caar_panel(results_am)
caar_mm,    tstat_caar_mm    = build_caar_panel(results_mm)
caar_garch, tstat_caar_garch = build_caar_panel(results_garch)

for label, caar, tstat in [('AM', caar_am, tstat_caar_am),
                            ('MM', caar_mm, tstat_caar_mm),
                            ('GARCH', caar_garch, tstat_caar_garch)]:
    print(f"\n--- {label} CAAR panel ---")
    display(add_significance_stars(caar, tstat))
    print(f"BMP t-statistics ({label}):")
    display(tstat.round(2))

# Sanity plot: GARCH conditional volatility for one event (Liberation Day)
sample_event_idx = next(
    (k for k, ev in enumerate(EVENT_DATES, start=1)
     if ev == date.fromisoformat('2025-04-02')), None)
if sample_event_idx is not None and sample_event_idx in results_garch['S&P500'][2].columns:
    fig, ax = plt.subplots(figsize=(8, 3))
    se_series = results_garch['S&P500'][2][sample_event_idx]
    ax.plot(se_series.index, se_series.values, marker='o')
    ax.set_title('GARCH conditional σ over event window — S&P500, 2025-04-02')
    ax.set_xlabel('Relative day')
    ax.set_ylabel('Conditional σ')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

## 8. Cell 9 — Chart Fix

```python
ax.axvline(pd.Timestamp('2025-04-02'), color='red', linestyle='--',
           linewidth=1.5, label='Apr 2, 2025 — Reciprocal Tariffs Announced')
ax.set_title('Cumulative Excess Return vs. MSCI World (Daily, AM-style)', fontsize=14)
ax.set_ylabel('Cumulative excess return (%)')
```

The `axvspan` shading from `2025-02-11` to `2026-04-02` is retained as the "trade war period" annotation.

## 9. Cell 10 — Sub-Period Update

```python
SUB_PERIODS = {
    'Pre-Liberation':    (date.fromisoformat('2024-08-07'), date.fromisoformat('2025-04-01')),
    'Liberation-onward': (date.fromisoformat('2025-04-02'), date.fromisoformat('2026-03-20')),
}
```

`fit_garch_subperiod` itself is unchanged.

## 10. Cell 11 — Export Additions

```python
# Existing per-event sheets (unchanged)
tables = {}
for label, results in [('AM', results_am), ('MM', results_mm), ('G', results_garch)]:
    tables.update(build_all_tables(results, label))

# New: CAAR panels
for label, (caar_df, tstat_df) in [
    ('AM', (caar_am, tstat_caar_am)),
    ('MM', (caar_mm, tstat_caar_mm)),
    ('G',  (caar_garch, tstat_caar_garch)),
]:
    tables[f'{label}_CAAR'] = add_significance_stars(caar_df, tstat_df)
    tables[f'{label}_CAAR_tstats'] = tstat_df.round(2)

# New: dropped-events diagnostic
dropped_rows = []
for label, results in [('AM', results_am), ('MM', results_mm), ('G', results_garch)]:
    for index_name, (_, _, _, dropped) in results.items():
        for event_date, reason in dropped:
            dropped_rows.append({
                'model': label, 'index': index_name,
                'event_date': event_date, 'reason': reason,
            })
tables['Dropped_Events'] = pd.DataFrame(dropped_rows)

export_to_excel(tables, '/content/drive/MyDrive/data/event_study_results.xlsx')
```

## 11. Validation

### 11.1 Bug-fix sanity checks (in-notebook)

- `se_am != se_mm` for at least one (index, event).
- GARCH `cond_vol.std() > 1e-8` over the event-window forecast (else fallback warning printed).
- `se_df.shape == ar_df.shape` for every model.
- `len(dropped) + len(ar_df.columns) == len(EVENT_DATES)` for every (index, model).

### 11.2 Aggregation sanity checks

- `caar_df.loc[idx, 'CAAR[-5,+5]']` ≈ `ar_df.sum().mean()` for every index.
- `bmp_test`'s `n_events ≥ 20` for all combinations (warn otherwise).
- All BMP `t_stat` values are finite.

### 11.3 Cross-model consistency

- AR magnitudes across AM/MM/GARCH for the same (index, event) are within 3× of each other.
- For at least one event, |GARCH t-stat| differs from |MM t-stat| by ≥10% — confirms cond-σ wiring.

### 11.4 Reproducibility

- Notebook runs top-to-bottom without exceptions.
- `event_study_results.xlsx` writes successfully and contains all expected sheets.
- The GARCH cond-σ sanity plot in Cell 8b renders.

### 11.5 Out-of-scope for validation

- Numerical comparison against the teacher notebook (different events).
- BMP statistical correctness under heavy clustering (accepted limitation per Clustering=A).

## 12. Stop Conditions

Implementation is complete when:

1. All Section 11 checks pass.
2. Diff against `stock_analyze_teacher.ipynb` is confined to Cells 3, 4a, 6, 7, 8, 8b (new), 9, 10, 11.
3. No drift in unrelated cells.

## 13. Risks

| Risk | Mitigation |
|---|---|
| `arch_model` forecast API changes between versions | Pin a known-good `arch` version in Cell 1 if behavior diverges (currently `arch>=5.0`). |
| GARCH(1,1) fails to converge on 61-day estimation window for some (index, event) pairs | Existing `try/except` wraps GARCH per index; fallback to unconditional SE printed and recorded. |
| BMP test is biased by clustering | Documented inline in Cell 8b and in the spec; no code-level mitigation per Clustering=A. |
| Late-2026 events skipped due to data-tail | Visible in the new `Dropped_Events` sheet; honest disclosure rather than silent skip. |
| Currency mismatch (MSCIEU in EUR, SSE in CNY, benchmark in USD) bundles FX moves into AR | Documented in §4.2(a); not corrected per Scope=B. |
| MSCINA ≈ S&P500 redundancy may surface as an obvious question at defense | Documented in §4.2(b); have a one-line framing ready ("broader North America" vs "US blue chip"). |
