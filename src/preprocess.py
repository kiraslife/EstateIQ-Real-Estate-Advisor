import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_validation import CURRENT_YEAR

CATEGORICAL_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status"
]

NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
    "Year_Built", "Age_of_Property", "Floor_No", "Total_Floors",
    "Floor_Ratio", "Schools_Numeric", "Hospitals_Numeric",
    "Parking_Numeric", "School_Density_Score", "Infra_Score",
    "City_Train_Median_PPS", "Locality_Train_Median_PPS", "PPS_vs_City_Median_Ratio"
]

def to_num_vec(s, default=0.0):
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(default).astype(float)
    str_s = s.astype(str).str.strip().str.lower()
    mapped = str_s.map({
        "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
        "high": 3.0, "medium": 2.0, "low": 1.0, "none": 0.0,
        "furnished": 2.0, "semi-furnished": 1.0, "unfurnished": 0.0,
        "ready_to_move": 1.0, "under_construction": 0.0
    })
    num_s = pd.to_numeric(s, errors="coerce")
    return mapped.combine_first(num_s).fillna(default).astype(float)

def engineer_features(df, train_city_stats=None):
    out = df.copy()
    out["Price_in_Lakhs"] = to_num_vec(out["Price_in_Lakhs"], 50.0)
    out["Size_in_SqFt"] = np.maximum(to_num_vec(out["Size_in_SqFt"], 1000.0), 150.0)
    out["BHK"] = np.maximum(to_num_vec(out["BHK"], 2.0), 1.0)
    out["Year_Built"] = np.clip(to_num_vec(out["Year_Built"], 2015.0), 1950.0, float(CURRENT_YEAR))
    out["Floor_No"] = np.maximum(to_num_vec(out["Floor_No"], 1.0), 0.0)
    out["Total_Floors"] = np.maximum(to_num_vec(out["Total_Floors"], 5.0), 1.0)

    out["Price_per_SqFt"] = (out["Price_in_Lakhs"] * 100000.0) / out["Size_in_SqFt"]
    out["Age_of_Property"] = float(CURRENT_YEAR) - out["Year_Built"]
    out["Floor_Ratio"] = np.clip(out["Floor_No"] / np.maximum(out["Total_Floors"], 1.0), 0.0, 1.0)

    out["Schools_Numeric"] = to_num_vec(out["Nearby_Schools"], 2.0)
    out["Hospitals_Numeric"] = to_num_vec(out["Nearby_Hospitals"], 2.0)
    out["Parking_Numeric"] = to_num_vec(out["Parking_Space"], 1.0)

    out["School_Density_Score"] = out["Schools_Numeric"] / (out["Size_in_SqFt"] / 1000.0)
    out["Infra_Score"] = (out["Schools_Numeric"] * 0.4) + (out["Hospitals_Numeric"] * 0.4) + (out["Parking_Numeric"] * 0.2)

    if train_city_stats is None:
        city_med_pps = out.groupby("City")["Price_per_SqFt"].median().to_dict()
        loc_med_pps = out.groupby("Locality")["Price_per_SqFt"].median().to_dict()
        global_med_pps = float(out["Price_per_SqFt"].median())
        infra_med = float(out["Infra_Score"].median())
        cs_map = out.groupby("City")["State"].first().to_dict()
        train_city_stats = {
            "city_med_pps": city_med_pps,
            "loc_med_pps": loc_med_pps,
            "global_med_pps": global_med_pps,
            "infra_med": infra_med,
            "known_states": sorted(out["State"].astype(str).unique().tolist()),
            "known_cities": sorted(out["City"].astype(str).unique().tolist()),
            "known_localities": sorted(out["Locality"].astype(str).unique().tolist()),
            "city_state_map": cs_map,
            "size_q01": float(out["Size_in_SqFt"].quantile(0.01)),
            "size_q99": float(out["Size_in_SqFt"].quantile(0.99)),
            "price_q01": float(out["Price_in_Lakhs"].quantile(0.01)),
            "price_q99": float(out["Price_in_Lakhs"].quantile(0.99)),
        }

    city_map = train_city_stats["city_med_pps"]
    loc_map = train_city_stats.get("loc_med_pps", {})
    g_med = train_city_stats["global_med_pps"]
    out["City_Train_Median_PPS"] = out["City"].astype(str).map(city_map).fillna(g_med).astype(float)
    out["Locality_Train_Median_PPS"] = out["Locality"].astype(str).map(loc_map).fillna(out["City_Train_Median_PPS"]).astype(float)
    blended_bench = 0.65 * out["City_Train_Median_PPS"] + 0.35 * out["Locality_Train_Median_PPS"]
    out["PPS_vs_City_Median_Ratio"] = out["Price_per_SqFt"] / np.maximum(blended_bench, 1.0)

    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].fillna("Unknown").astype(str) if c in out.columns else "Unknown"

    return out, train_city_stats

def compute_multi_factor_target(df_feat, train_city_stats):
    base_cagr = 0.065
    prop_adj = df_feat["Property_Type"].astype(str).map({"Villa": 0.012, "Independent House": 0.009, "Apartment": 0.004}).fillna(0.004)
    trans_adj = df_feat["Public_Transport_Accessibility"].astype(str).map({"High": 0.008, "Medium": 0.004, "Low": -0.003}).fillna(0.002)
    infra_norm = np.clip((df_feat["Infra_Score"] - train_city_stats["infra_med"]) / max(train_city_stats["infra_med"], 1.0), -0.5, 0.5)
    infra_adj = infra_norm * 0.012
    age_adj = np.clip((15.0 - df_feat["Age_of_Property"]) * 0.00045, -0.012, 0.007)
    bhk_adj = np.clip((df_feat["BHK"] - 2.0) * 0.0025, -0.003, 0.008)
    floor_adj = (df_feat["Floor_Ratio"] - 0.3) * 0.0035 + np.clip(df_feat["Total_Floors"] / 50.0, 0.0, 0.003)
    sec_adj = np.where(df_feat["Security"].astype(str).str.lower() == "yes", 0.0025, -0.001)
    amen_adj = df_feat["Amenities"].astype(str).map({"Pool": 0.003, "Clubhouse": 0.0025, "Gym": 0.002, "Garden": 0.0015, "Playground": 0.001}).fillna(0.0005)
    face_adj = df_feat["Facing"].astype(str).map({"East": 0.002, "North": 0.0018, "West": 0.0005, "South": 0.0}).fillna(0.0005)
    own_adj = df_feat["Owner_Type"].astype(str).map({"Builder": 0.002, "Owner": 0.001, "Broker": -0.001}).fillna(0.0)
    avail_adj = np.where(df_feat["Availability_Status"].astype(str).str.lower().str.contains("under"), 0.0045, 0.001)
    furn_adj = df_feat["Furnished_Status"].astype(str).map({"Furnished": 0.0025, "Semi-Furnished": 0.0012, "Unfurnished": 0.0}).fillna(0.0)
    overprice_pen = np.clip((1.0 - df_feat["PPS_vs_City_Median_Ratio"]) * 0.012, -0.022, 0.015)

    annual_rate = np.clip(
        base_cagr + prop_adj + trans_adj + infra_adj + age_adj + bhk_adj +
        floor_adj + sec_adj + amen_adj + face_adj + own_adj + avail_adj + furn_adj + overprice_pen,
        0.025, 0.140
    )
    future_5y = np.round(df_feat["Price_in_Lakhs"] * ((1.0 + annual_rate) ** 5), 2)
    good_inv = ((df_feat["PPS_vs_City_Median_Ratio"] <= 1.05) & (df_feat["Infra_Score"] >= train_city_stats["infra_med"] * 0.85) & (annual_rate >= 0.075)).astype(int)
    return pd.Series(future_5y, index=df_feat.index), pd.Series(good_inv, index=df_feat.index)

def build_sklearn_preprocessor():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, NUMERIC_FEATURES), ("cat", cat_pipe, CATEGORICAL_FEATURES)])
