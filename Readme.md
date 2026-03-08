# 🏭 Sales Prediction ML — Industrial Production Forecast

Predictive model to forecast monthly production units per industrial line using historical sales and production data.

## 🎯 Objective
Predict next month's production units by line to support planning and inventory decisions.

## 🛠️ Tools & Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib
- NumPy

## ⚙️ Methodology
- Data cleaning and consistency validation across three sources (production, sales, staff)
- Feature engineering: lag variables (prod_lag1, prod_lag2, ventas_lag1), efficiency ratios, defect rate, calendar features (month, quarter, is_peak)
- Temporal train/validation split: train 2024-01 to 2025-06, validation 2025-07 to 2025-09
- TimeSeriesSplit cross-validation (3 folds)
- Linear Regression model with forecast for October 2025

## 📊 Results
| Metric | Value |
|--------|-------|
| MAE    | 81.98 |
| RMSE   | 133.39 |
| R²     | 0.60  |

## 📌 Forecast
Production forecast generated per industrial line for October 2025 using the last available data point per line.
