# ADV-ML Comp Walmart Revenue Forecasting  
### Course Competition – Project README

## Mission Statement
Forecast daily revenue for **10 Walmart stores** plus their **aggregate total** (store 0) over a 92-day horizon (2015-10-01 → 2015-12-31).  
The model must **beat the baseline RMSE ≤ 11 761**, generate coherent forecasts across hierarchy levels, and comply with the Kaggle submission format  
`id, prediction  →  {store_id}_{YYYYMMDD}, value`.

---

## High-Level Workflow

| # | Stage |
|---|-------|
| 1 | **Data audit & hierarchy sanity** |
| 2 | **Univariate EDA per series** |
| 3 | **Trend / seasonality decomposition & stationarity tests** |
| 4 | **Baseline forecasts** (naïve, seasonal-naïve, MA, ETS, Prophet) |
| 5 | **Feature engineering** (lags, rolling stats, Fourier, holiday flags) |
| 6 | **Model family selection & training** (LightGBM, ARIMA/ETS, small NN) |
| 7 | **Time-series CV & hyper-tuning** (rolling window, Tweedie vs RMSE) |
| 8 | **Hierarchical reconciliation** (bottom-up, MinT, hybrid) |
| 9 | **Ensembling / blending** |
| 10| **Final evaluation → CSV packaging** |
| 11| **Post-submission monitoring & iteration** |

---

## Progress Log

### **Step 1 – Data Audit & Hierarchy Sanity** (✅ complete)
| Check | Result |
|-------|--------|
| Shapes & dtypes | `train 18 766×4`, `calendar 162×2`, `forecast 1 012×2` |
| Unique stores | IDs 0–10 (aggregate + 10 stores) |
| Aggregate consistency | `store_id 0` ≈ Σ(1…10) → max diff ≈ \$0.04 |
| Date coverage | Continuous grid 2011-01-29 → 2015-09-30 |
| Zero-sales ratio | < 0.3 % per store; none in aggregate |
| Span per store | Identical min/max dates |
| Merged calendar events | Added `event`, `dow`, `month`, `is_zero_day` → `full_df 18 766×8` |
| Snapshot saved | `data/full_train.pkl` |
| Target transform decision | Use `log1p(revenue)` for modelling |
| Zero-day handling | Keep 0 values + binary flag |

### **Step 2 – Initial EDA** (✔ plots produced)
* **Line facets** → upward trend, weekly saw-tooth, rare zeros, scale gap.  
* **30-day MA (store 0)** → macro growth + yearly holiday bumps.  
* **Weekday box-plot** → Sat/Sun ≈ +25 % vs mid-week.  
* **Monthly stacked area** → revenue roughly doubles over five years.  
* **Histogram of `log1p(revenue)`** → variance stabilised, two-hump (store 0 vs others).  
* **Holiday uplift bar chart** → Super Bowl, Easter, Father’s Day top boosters.  
* **Store × event heat-map** → heterogeneous effects (e.g. Valentine’s negative for some stores).  
* **Outlier box-plot** → <2 % extreme highs; no heavy-tailed problem after log.

### **Decisions Locked-in**
1. **Target**: model `log1p(revenue)`, inverse with `expm1` for submission.  
2. **Zero days**: keep as 0 + `is_zero_day` flag.  
3. **Calendar NaNs**: leave as NaN for tree models, fill `"none"` when one-hotting.  
4. **Canonical dataset**: use `full_df.pkl` for all subsequent steps.

---

## Next Up
* **Step 3 – Trend/seasonality decomposition & stationarity tests**  
  * STL or Prophet components, ADF/KPSS, decide on differencing / Fourier terms.  
* **Step 4 – Baseline forecasts**  
  * Seasonal-naïve, 7-day moving average, ETS(A,A,N), Prophet holiday model.  

_Updated 2025-06-08_
