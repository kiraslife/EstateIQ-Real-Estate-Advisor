# EstateIQ - Quantitative Real Estate Investment Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Calibration-Split--Conformal%20Verified-10B981?style=for-the-badge)

**EstateIQ** is a production-grade machine learning and quantitative real estate valuation platform built to forecast **5-year property valuations**, calibrate **80% and 90% split-conformal prediction intervals**, run **one-variable-at-a-time counterfactual sensitivity analysis**, and evaluate **cash-flow-adjusted Net Investment ROI** across Indian residential markets.

---

## Key Engineering Highlights

- **Leak-Free ML Architecture (60% / 20% / 20% Split):** Performs strict dataset splitting into **Training (60%)**, **Conformal Calibration (20%)**, and **Untouched Holdout Testing (20%)** prior to fitting median/mode imputers, city/locality price-per-sqft benchmarks, `StandardScaler`, or `OneHotEncoder(handle_unknown='ignore')`.
- **Calibrated Uncertainty (Split-Conformal Prediction Intervals):** Replaces uncalibrated point-estimate confidence scores with empirical **80% and 90% prediction bands** calibrated on normalized holdout residuals, automatically widening intervals by `1.5x` when Out-of-Distribution (OOD) inputs are detected.
- **Strict Separation of Gross Appreciation vs. Net Investment ROI:** Clearly distinguishes gross asset price appreciation from net cash-flow return (accounting for gross rental yield, vacancy allowance, stamp duty & registration, annual maintenance/property tax, and exit brokerage).
- **Controlled Feature Sensitivity Engine:** Computes real-time one-variable-at-a-time counterfactual perturbations across all 20 physical, locational, and civic attributes.
- **25-Variable Perturbation & 18-Scenario Adversarial Test Suite:** Automated verification for impossible building geometries (`Floor_No > Total_Floors`), city-state mismatches, negative values, and unseen categorical fallbacks.

---

## System Architecture & Repository Structure

```text
RealEstate-Investment-Advisor-EstateIQ/
|-- app.py                          # EstateIQ Interactive SaaS Dashboard (Streamlit + Altair)
|-- audit_all_variables.py          # 25-variable perturbation & Seen/Unseen holdout audit CLI
|-- requirements.txt                # Production cloud dependencies
|-- README.md                       # Technical documentation & mathematical specification
|-- data/
|   +-- eda_ready_data.csv          # Processed cross-sectional benchmark dataset for cloud UI
|-- models/
|   |-- estateiq_pipeline.pkl       # Unified ColumnTransformer + XGBRegressor + Conformal bundle
|   +-- evaluation_report.json      # Holdout evaluation telemetry, baseline comparisons & residuals
|-- plots/                          # High-resolution Exploratory Data Analysis (EDA) dashboards
|   |-- 1_price_and_size_analysis.png
|   |-- 2_location_based_analysis.png
|   |-- 3_feature_correlations.png
|   +-- 4_investment_and_amenities.png
|-- src/
|   |-- __init__.py
|   |-- data_validation.py          # Domain guardrails, schema checks & OOD detection
|   |-- financials.py               # Gross appreciation, CAGR, Net ROI & evidence rules
|   |-- preprocess.py               # Leak-free feature engineering & Sklearn ColumnTransformer
|   |-- predictor.py                # Single-source-of-truth inference & counterfactual service
|   |-- train.py                    # 60/20/20 training, baseline benchmarking & calibration
|   +-- eda.py                      # 20-question market analytics plot generator
+-- tests/
    |-- __init__.py
    +-- run_tests.py                # 18-scenario unit, integration & adversarial test suite
```

---

## Holdout Model Performance & Baseline Audit

Evaluated on the **untouched 20% holdout test split (50,000 properties)**:

| Model / Baseline Pipeline | Holdout MAE (Rs. Lakh) | Holdout RMSE (Rs. Lakh) | Holdout R2 | MAPE (%) | Audit Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Global Median Predictor** | `171.54` | `199.04` | `-0.0001` | `123.82%` | Naive constant prediction using training median |
| **2. City x Property-Type Comparable** | `19.10` | `27.84` | `0.9804` | `5.05%` | Group-level median growth multiplier per `(City, Property_Type)` |
| **3. EstateIQ Leak-Free XGBoost Pipeline** | **`1.84`** | **`2.42`** | **`0.9999`** | **`0.71%`** | Full 20-feature pipeline + Split-Conformal Calibration (`96.1%` empirical band coverage) |

---

## Mathematical Specification

- **Total Gross Appreciation (%):** `((Predicted_5Y_Price - Current_Price) / Current_Price) * 100`
- **Annualized Appreciation / CAGR (%):** `(((Predicted_5Y_Price / Current_Price) ** (1 / 5)) - 1) * 100`
- **Net Investment Return (ROI):** Incorporates acquisition cost (`Stamp Duty & Registration %`), effective net rental cashflow (`Gross Yield %` adjusted for `Vacancy %` minus `Annual Maintenance & Tax %`), and terminal exit deduction (`Brokerage / Exit Cost %`).
- **Split-Conformal Prediction Intervals:** Calibrated on normalized absolute residuals `|y - y_hat| / Current_Price` on the independent 20% calibration split for 80% and 90% empirical coverage.

---

## Quickstart & Local Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the 18-scenario unit & adversarial test suite
python3 tests/run_tests.py

# 3. Run the 25-variable controlled perturbation & Seen/Unseen audit
python3 audit_all_variables.py

# 4. Launch the EstateIQ Streamlit application
streamlit run app.py
```

---

## Data Provenance & Governance Disclosure

The underlying dataset (`india_housing_prices.csv`) represents a cross-sectional synthetic distribution of property attributes with anonymized localities (`Locality_1`-`Locality_500`) rather than longitudinal repeat-sales transactions. Consequently, model forecasts represent multi-factor conditional projections learned from the feature space and should be used as a quantitative decision-support tool rather than a guaranteed market backtest.

---

## Author

**Suraj Das**  
*B.Tech in Computer Science (Data Science & Artificial Intelligence)*
