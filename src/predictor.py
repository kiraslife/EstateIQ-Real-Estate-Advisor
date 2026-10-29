import os, json, joblib
import pandas as pd
from src.data_validation import validate_property_input
from src.preprocess import engineer_features
from src.financials import calculate_appreciation, calculate_net_investment_return, evaluate_investment_verdict

class PropertyPredictionService:
    def __init__(self, bundle_path="models/estateiq_pipeline.pkl", report_path="models/evaluation_report.json"):
        self.bundle = joblib.load(bundle_path)
        self.reg_pipeline = self.bundle["reg_pipeline"]
        self.train_stats = self.bundle["train_stats"]
        self.all_features = self.bundle["all_features"]
        self.conformal = self.bundle["conformal"]
        self.eval_report = json.load(open(report_path)) if os.path.exists(report_path) else {}

    def predict_property(self, property_input, cashflow_params=None, horizon_years=5):
        is_valid, errors, warnings = validate_property_input(property_input, self.train_stats)
        if not is_valid:
            return {"is_valid": False, "errors": errors, "warnings": warnings}

        raw_df = pd.DataFrame([property_input])
        feat_df, _ = engineer_features(raw_df, train_city_stats=self.train_stats)
        X_infer = feat_df[self.all_features]

        current_price = float(feat_df["Price_in_Lakhs"].iloc[0])
        pred_5y = max(float(self.reg_pipeline.predict(X_infer)[0]), current_price * 0.5)
        pps = float(feat_df["Price_per_SqFt"].iloc[0])
        city_med_pps = float(feat_df["City_Train_Median_PPS"].iloc[0])

        appreciation = calculate_appreciation(current_price, pred_5y, horizon_years)

        ood_mult = 1.5 if len(warnings) > 0 else 1.0
        hw80 = current_price * float(self.conformal["q80_rel_effective"]) * ood_mult
        hw90 = current_price * float(self.conformal["q90_rel_effective"]) * ood_mult

        uncertainty = {
            "interval_80_lower_lakhs": round(max(0.0, pred_5y - hw80), 2),
            "interval_80_upper_lakhs": round(pred_5y + hw80, 2),
            "interval_90_lower_lakhs": round(max(0.0, pred_5y - hw90), 2),
            "interval_90_upper_lakhs": round(pred_5y + hw90, 2),
            "half_width_80_pct": round((hw80 / max(pred_5y, 1.0)) * 100.0, 2),
            "holdout_coverage_80_pct": self.conformal["empirical_coverage_80_eff_pct"],
            "methodology": self.conformal["methodology"]
        }

        cf = cashflow_params or {}
        net_roi = calculate_net_investment_return(
            current_price, pred_5y, horizon_years,
            cf.get("gross_rental_yield_pct"), cf.get("stamp_duty_registration_pct"),
            cf.get("annual_maintenance_tax_pct"), cf.get("vacancy_rate_pct"), cf.get("exit_brokerage_tax_pct")
        )
        verdict = evaluate_investment_verdict(current_price, pps, city_med_pps, appreciation["annualized_appreciation_pct"], uncertainty["half_width_80_pct"], warnings, net_roi)

        return {
            "is_valid": True, "errors": [], "warnings": warnings,
            "property_summary": {
                "State": str(property_input["State"]), "City": str(property_input["City"]),
                "Locality": str(property_input["Locality"]), "Property_Type": str(property_input["Property_Type"]),
                "BHK": int(float(property_input["BHK"])), "Size_in_SqFt": round(float(property_input["Size_in_SqFt"]), 1),
                "Current_Price_Lakhs": round(current_price, 2), "Price_per_SqFt": round(pps, 0),
                "City_Median_PPS": round(city_med_pps, 0), "Year_Built": int(float(property_input["Year_Built"])),
                "Age_of_Property": int(feat_df["Age_of_Property"].iloc[0]),
                "Floor_No": int(float(property_input["Floor_No"])), "Total_Floors": int(float(property_input["Total_Floors"]))
            },
            "appreciation": appreciation, "uncertainty": uncertainty, "net_roi": net_roi, "verdict": verdict,
            "engineered_features": feat_df[self.all_features].iloc[0].to_dict()
        }

    def compute_feature_sensitivity(self, base_input):
        base_res = self.predict_property(base_input)
        if not base_res.get("is_valid"):
            return []
        base_5y = base_res["appreciation"]["future_price_lakhs"]
        perturbs = [
            ("Type -> Villa", {"Property_Type": "Villa"}),
            ("Type -> Apartment", {"Property_Type": "Apartment"}),
            ("Transit -> High", {"Public_Transport_Accessibility": "High"}),
            ("Transit -> Low", {"Public_Transport_Accessibility": "Low"}),
            ("BHK +1 Bedroom", {"BHK": min(int(float(base_input["BHK"])) + 1, 6)}),
            ("Newer Build (2025)", {"Year_Built": 2025}),
            ("Older Build (1995)", {"Year_Built": 1995}),
            ("High Schools/Hosp (9)", {"Nearby_Schools": 9, "Nearby_Hospitals": 8}),
            ("Low Schools/Hosp (1)", {"Nearby_Schools": 1, "Nearby_Hospitals": 1})
        ]
        out = []
        for label, mod in perturbs:
            cand = {**base_input, **mod}
            r = self.predict_property(cand)
            if r.get("is_valid"):
                f5 = r["appreciation"]["future_price_lakhs"]
                out.append({"Scenario": label, "Predicted_5Y_Lakhs": f5, "Delta_Lakhs": round(f5 - base_5y, 2)})
        return out
