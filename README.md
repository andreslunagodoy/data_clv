# Auto Insurance Customer Lifetime Value — Regression Analysis

Predicting customer lifetime value (CLV) for an auto-insurance book of business using the [IBM Watson Marketing Customer Value](https://www.kaggle.com/datasets/pankajjsh06/ibm-watson-marketing-customer-value-data) dataset (9,134 customers, 24 columns). This is the regression counterpart of the Telco churn study (`../data_churn`): same three modeling phases, from simple baselines to gradient boosting with Bayesian hyperparameter optimization, but with a continuous, heavily skewed target.

## Results

Test set (1,827 customers). The model predicts `log1p(CLV)`; metrics are reported on the log scale and in dollars.

| Phase | Best Model | RMSE (log) | R² (log) | R² ($) | MAE ($) | Runtime |
|-------|-----------|-----------|----------|--------|---------|---------|
| 1 — Baselines | Random Forest | 0.2007 | 0.909 | 0.676 | 1,395 | ~10s |
| 2 — Tuned sklearn | RF (tuned) | 0.1975 | 0.912 | 0.680 | 1,406 | ~1.6min |
| 3 — XGB/LGB/CatBoost | Stacking ensemble | 0.1970 | 0.912 | 0.686 | 1,415 | ~2.4min |

For reference: a mean-predicting dummy scores R² ≈ 0, and Linear Regression scores R² (log) 0.818 / R² ($) 0.362.

**The biggest gain is trees over linear models, and it is already there in Phase 1.** CLV is close to `monthly premium × a multiplier that depends on the number of policies`, which a linear model cannot express. After that, tuning (Phase 2) and Optuna-tuned boosters (Phase 3) move RMSE by about 0.003 — noise-level for 1,827 test customers — while adding runtime, libraries and complexity.

Runtimes are from a single full run (about 5 minutes end to end, including prediction intervals) and vary with machine load.

## Project Structure

```
clv_auto_insurance.csv       # Raw dataset (9,134 rows)
CLV_DA.ipynb                 # All three phases: baselines → tuned sklearn → XGB/LGB/CatBoost
requirements.txt             # Pinned dependencies (pip install -r requirements.txt)
README.md                    # This file
README_learners.md           # Detailed walkthrough of every technique (regression edition)
```

## Setup

```bash
# Create and activate a virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# Optional: register a Jupyter kernel (VS Code can use .venv directly)
python -m ipykernel install --user --name data_clv --display-name "Python (data_clv)"
```

## Quick Start

Open `CLV_DA.ipynb` — it contains all three phases with timing instrumentation and a complete comparison. Run all cells; expect ~5 minutes total.

## Key Findings

**Top CLV drivers** (consistent across model families, via SHAP):
1. **Premium × number of policies** — the engineered interaction is the single strongest feature
2. **Number of policies** — behaves as three regimes: 1 policy, 2 policies, 3+ policies
3. **Premium per policy** and **monthly premium** — the remaining main drivers
4. Everything else (employment, income, gender, complaints, claims per policy) is an order of magnitude weaker

**The performance ceiling (R² ≈ 0.91 on the log scale) is a data limitation**, not a model limitation. Customers with exactly **two policies** are 25% of the test set but account for **98% of the squared error**: their CLV-per-premium multiplier is bimodal and no feature in the dataset predicts which cluster a customer falls in (cross-validated R² = 0.02). For every other policy count, the typical error is about 2%.

**The dataset looks synthetic.** CLV is almost exactly `premium × ~39` (1 policy) or `premium × ~78` (3+ policies), with only a few percent of spread. Real insurance data is not this clean, so read the absolute accuracy as a property of this dataset rather than a benchmark.

**Regression-specific results:**
- Log-transforming the target cuts dollar MAE by about 12% versus training on raw dollars, but is slightly worse on dollar RMSE / R² (the few very large CLVs dominate those).
- 90% prediction intervals: quantile LightGBM covers 87.9% (narrow, adaptive width); a CV-residual conformal band covers 90.1% but is about 30% wider because it uses one width for everyone.

## Differences from the Churn Study

| Churn (classification) | CLV (regression) |
|------------------------|------------------|
| Class imbalance: `class_weight`, SMOTE | Skewed target: `log1p` transform |
| Accuracy / precision / recall / F1 / ROC-AUC | RMSE / MAE / R² on log and dollar scales |
| Threshold optimization | Target back-transformation and smearing |
| Calibration curves | Prediction intervals (quantile + conformal) |
| Confusion matrices | Residual diagnostics and error by segment |
| Stratified K-fold | Plain K-fold |
