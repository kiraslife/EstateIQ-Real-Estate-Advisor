import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.preprocess import engineer_features, compute_multi_factor_target, build_sklearn_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def reg_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-3))) * 100.0)
    return {"mae_lakhs": round(mae, 4), "rmse_lakhs": round(rmse, 4), "r2": round(r2, 4), "mape_pct": round(mape, 4)}

def train_and_evaluate():
    print("[INFO] Loading dataset and performing strict 60/20/20 Train/Calib/Test split BEFORE preprocessing...")
    raw_df = pd.read_csv("data/india_housing_prices.csv").drop_duplicates().reset_index(drop=True)

    df_train_cal, df_test = train_test_split(raw_df, test_size=0.20, random_state=42)
    df_train, df_cal = train_test_split(df_train_cal, test_size=0.25, random_state=42)

    X_train_df, train_stats = engineer_features(df_train, train_city_stats=None)
    X_cal_df, _ = engineer_features(df_cal, train_city_stats=train_stats)
    X_test_df, _ = engineer_features(df_test, train_city_stats=train_stats)

    y_train_reg, _ = compute_multi_factor_target(X_train_df, train_stats)
    y_cal_reg, _ = compute_multi_factor_target(X_cal_df, train_stats)
    y_test_reg, _ = compute_multi_factor_target(X_test_df, train_stats)

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # Baselines on unseen holdout set
    med_pred = np.full(len(y_test_reg), float(y_train_reg.median()))
    base_median = reg_metrics(y_test_reg.values, med_pred)

    tr_ratios = (y_train_reg / np.maximum(X_train_df["Price_in_Lakhs"], 1.0)).groupby([X_train_df["City"], X_train_df["Property_Type"]]).median().to_dict()
    g_ratio = float((y_train_reg / np.maximum(X_train_df["Price_in_Lakhs"], 1.0)).median())
    comp_mult = X_test_df.set_index(["City", "Property_Type"]).index.map(tr_ratios).fillna(g_ratio).values
    base_comp = reg_metrics(y_test_reg.values, X_test_df["Price_in_Lakhs"].values * comp_mult)

    # Fit leak-free Sklearn + XGBoost Pipeline
    reg_pipeline = Pipeline([
        ("preprocessor", build_sklearn_preprocessor()),
        ("regressor", XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1))
    ])
    reg_pipeline.fit(X_train_df[all_features], y_train_reg)

    # Split-Conformal Calibration on independent 20% calibration set
    cal_preds = reg_pipeline.predict(X_cal_df[all_features])
    cal_rel_res = np.abs(y_cal_reg.values - cal_preds) / np.maximum(X_cal_df["Price_in_Lakhs"].values, 1.0)
    q80_raw = float(np.quantile(cal_rel_res, 0.80))
    q90_raw = float(np.quantile(cal_rel_res, 0.90))
    q80_eff = max(q80_raw, 0.035)
    q90_eff = max(q90_raw, 0.050)

    # Holdout Evaluation on untouched 20% test set
    test_preds = reg_pipeline.predict(X_test_df[all_features])
    repaired_metrics = reg_metrics(y_test_reg.values, test_preds)
    test_rel_err = np.abs(y_test_reg.values - test_preds) / np.maximum(X_test_df["Price_in_Lakhs"].values, 1.0)
    cov80_raw = round(float(np.mean(test_rel_err <= q80_raw) * 100.0), 2)
    cov80_eff = round(float(np.mean(test_rel_err <= q80_eff) * 100.0), 2)

    seg_type = []
    for pt, grp in X_test_df.assign(yt=y_test_reg.values, yp=test_preds).groupby("Property_Type"):
        seg_type.append({"Segment": str(pt), "Count": int(len(grp)), **reg_metrics(grp["yt"].values, grp["yp"].values)})

    idx = np.random.RandomState(42).choice(len(y_test_reg), size=min(200, len(y_test_reg)), replace=False)
    plot_samples = pd.DataFrame({
        "Actual_5Y_Lakhs": np.round(y_test_reg.values[idx], 2),
        "Predicted_5Y_Lakhs": np.round(test_preds[idx], 2),
        "Residual_Lakhs": np.round(y_test_reg.values[idx] - test_preds[idx], 2)
    }).to_dict(orient="records")

    conformal_info = {
        "q80_rel_raw": round(q80_raw, 5), "q80_rel_effective": round(q80_eff, 5),
        "q90_rel_effective": round(q90_eff, 5), "empirical_coverage_80_raw_pct": cov80_raw,
        "empirical_coverage_80_eff_pct": cov80_eff,
        "methodology": "Split-Conformal Relative Residual Quantiles calibrated on independent 20% holdout split"
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump({"reg_pipeline": reg_pipeline, "train_stats": train_stats, "all_features": all_features, "conformal": conformal_info}, "models/estateiq_pipeline.pkl")

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M IST"),
        "splits": {"train_records": len(df_train), "calibration_records": len(df_cal), "holdout_test_records": len(df_test)},
        "baselines_on_holdout": {"global_median": base_median, "city_type_comparable": base_comp},
        "legacy_before_repair": {"rmse_lakhs": 0.8485, "r2": 1.0000, "note": "Leaked pre-split target multiplier formula"},
        "repaired_on_holdout": repaired_metrics,
        "conformal": conformal_info,
        "segment_by_type": seg_type,
        "holdout_plot_samples": plot_samples
    }
    with open("models/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    full_feat, _ = engineer_features(raw_df, train_city_stats=train_stats)
    full_y, full_c = compute_multi_factor_target(full_feat, train_stats)
    full_feat["Future_Price_5Y"] = full_y
    full_feat["Good_Investment"] = full_c
    full_feat.to_csv("data/eda_ready_data.csv", index=False)

    print(f"[HOLDOUT RESULTS] Median Baseline MAE: ₹{base_median['mae_lakhs']}L | Comparable Baseline MAE: ₹{base_comp['mae_lakhs']}L")
    print(f"[REPAIRED MODEL]  Holdout MAE: ₹{repaired_metrics['mae_lakhs']}L | RMSE: ₹{repaired_metrics['rmse_lakhs']}L | R²: {repaired_metrics['r2']} | MAPE: {repaired_metrics['mape_pct']}%")
    print(f"[CONFORMAL PI]    80% Raw Coverage: {cov80_raw}% | Effective Coverage: {cov80_eff}%")
    print("✅ STEP 3 COMPLETE: Model Trained & Evaluated on Unseen Holdout Split!")

if __name__ == "__main__":
    train_and_evaluate()
