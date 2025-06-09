# ADV-ML Walmart Revenue Forecasting  
### Course Competition – Project README

## Mission Statement
Forecast daily revenue for **10 Walmart stores** plus the **aggregate total** (store 0) for the 92-day horizon **2015-10-01 → 2015-12-31**.  
The final model must  
* **beat the baseline RMSE ≤ 11 761**,  
* remain coherent across the hierarchy, and  
* output the required Kaggle format `id,prediction` (`id = {store_id}_{YYYYMMDD}`).

---

## High-Level Workflow

| # | Stage |
|---|-------|
| 1 | **Data audit & hierarchy sanity** |
| 2 | **Exploratory data analysis (EDA)** |
| 3 | **Trend / seasonality decomposition & stationarity tests** |
| 4 | **Baseline forecasts** (naïve, seasonal-naïve, MA, ETS/Prophet) |
| 5 | **Feature engineering** (lags, roll-stats, Fourier, holiday flags) |
| 6 | **Model training** (LightGBM, SARIMA/ETS, lightweight NN) |
| 7 | **Time-series CV & hyper-tuning** |
| 8 | **Hierarchical reconciliation** (bottom-up, MinT) |
| 9 | **Ensembling / blending** |
|10 | **Final evaluation → CSV packaging** |
|11 | **Post-submission monitoring** |

---

## Progress Log

### **Step 1 – Data Audit & Hierarchy Sanity**  (✅ complete)

| Check | Result                                                                                                             |
|-------|--------------------------------------------------------------------------------------------------------------------|
| Shapes & dtypes | `train 18 766×4`, `calendar 162×2`, `forecast 1012×2`                                                              |
| Duplicate store-day rows | 0 after aggregation                                                                                                |
| Unique stores | IDs 0–10 (aggregate + 10 branches)                                                                                 |
| Aggregate consistency | `store 0` ≈ Σ(stores 1…10) – max diff ≈ \$0.04                                                                     |
| Date coverage | Continuous grid 2011-01-29 → 2015-09-30                                                                            |
| Zero-sales ratio | \< 0.3 % per store; none in aggregate                                                                              |
| Span per store | Identical min/max dates                                                                                            |
| Calendar merge | Added `event`, **±3-day lead/lag flags** (`event_lead1…lag3`), `dow`, `month`, `is_zero_day` → `full_df 18 766×14` |
| Snapshot saved | `data/full_train.pkl`                                                                                              |
| Target transform | **`log1p(revenue)`**                                                                                               |
| Zero-day handling | Keep 0s + `is_zero_day` flag                                                                                       |

---

### **Step 2 – Exploratory Data Analysis**  (✅ complete)

* **Line facets** – upward trend, weekly saw-tooth, rare outages, scale gap.  
* **30-day MA** – steady growth, yearly Q4 bumps.  
* **Weekday box-plot** – Sat/Sun ≈ +25 % vs mid-week.  
* **Monthly stacked area** – revenue roughly doubles over five years.  
* **`log1p` histogram** – variance stabilised; two humps (store 0 vs others).  
* **Holiday uplift bars** – Super Bowl, Easter, Father’s Day strongest.  
* **Store × event heat-map** – heterogeneous holiday effects.  
* **Outlier box-plot** – \< 2 % extreme highs; no heavy tail post-log.  
* **ACF / PACF** – dominant weekly comb; AR(1)+seasonal AR(1) pattern.  
* **Seasonal sub-series** – mild monthly drift, strong weekend premium.  
* **STL (all stores)** – clear trend + stable 7-day seasonality; outage dips isolated.  
* **ADF & KPSS on STL residuals** – ADF ≪ 0.05, KPSS ≈ 0.10 ⇒ residuals stationary.

### **Decisions Locked-in**

1. **Target** – model on `log1p(revenue)`, invert with `expm1` for submission.  
2. **Zero days** – keep as 0 plus `is_zero_day` feature.  
3. **Calendar NaNs** – leave missing for tree models; fill `"none"` when one-hotting.  
4. **Canonical dataset** – `full_train.pkl` is the single source of truth.  
5. **Base features so far** – `dow`, `month`, `lag_1`, `lag_7`, `lag_14`, `event`, `event_lead/lag1-3`, `time_index`.

---

## Next Steps
* **Step 2 – EDA**  
  * explore if there are more things to find out:
  * Frequency table of event + lead/lag flags
  * Pay-day / month-end indicators
  * Cross-store correlation matrix 
  * Structural-break scan (Prophet changepoints or PELT)
  * ARCH-LM test on STL residuals (store 0)

* **Step 3 – Baseline forecasts**  
  * Seasonal-naïve (lag 7)  
  * 7-day moving average  
  * SARIMA(1,0,0)(1,0,0)[7] on log scale  
  * ETS or Prophet with weekly seasonality + holiday regressors  
  → benchmark RMSE on a rolling 92-day hold-out.

* **Step 4 – Feature engineering**  
  * Longer lags (28, 56), rolling means/stds, Fourier(365/182.5), days-since-outage, promo-window flags.

_Updated 2025-06-08 by Igor_