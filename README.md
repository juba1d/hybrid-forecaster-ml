# Hybrid Forecaster ML 📈

A custom, `scikit-learn` compatible time-series forecasting estimator that combines classical statistical modeling with gradient boosting. 

Built as a proof-of-concept for time-series ecosystems like `sktime` and `skforecast`.

## The Problem
Standard tree-based models (like LightGBM or XGBoost) are excellent at capturing complex seasonal patterns, but they **cannot extrapolate trends**. If your test data is higher than your training data, a raw GBM will just predict a flat line. 

## The Solution
This hybrid estimator uses a two-step decomposition approach:
1. **Trend Engine:** Uses `LinearRegression` (OLS) to capture the global deterministic trend.
2. **Residual Engine:** Calculates the "leftover" noise/seasonality (Actual - Trend) and fits a `LGBMRegressor` to capture the non-linear patterns.
3. **Prediction:** Combines both outputs `(Trend + Residuals)` for a robust forecast.

## ⚙️ Installation

Clone the repo and install the dependencies:

```bash
git clone [https://github.com/juba1d/hybrid-forecaster-ml.git](https://github.com/juba1d/hybrid-forecaster-ml.git)
cd hybrid-forecaster-ml
pip install -r requirements.txt
python setup.py install
