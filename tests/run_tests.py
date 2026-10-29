import sys, os, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.financials import calculate_appreciation
from src.preprocess import engineer_features
from src.predictor import PropertyPredictionService

class TestEstateIQSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = PropertyPredictionService()
        cls.base = {
            "State": "West Bengal", "City": "Durgapur", "Locality": "Locality_246",
            "Property_Type": "Apartment", "BHK": 3, "Size_in_SqFt": 1600.0, "Price_in_Lakhs": 75.0,
            "Year_Built": 2018, "Furnished_Status": "Semi-Furnished", "Floor_No": 3, "Total_Floors": 10,
            "Nearby_Schools": 5, "Nearby_Hospitals": 4, "Public_Transport_Accessibility": "High",
            "Parking_Space": "Yes", "Security": "Yes", "Amenities": "Clubhouse", "Facing": "East",
            "Owner_Type": "Builder", "Availability_Status": "Ready_to_Move"
        }

    def test_01_valid_typical(self):
        r = self.svc.predict_property(self.base)
        self.assertTrue(r["is_valid"])

    def test_02_small_low_cost(self):
        r = self.svc.predict_property({**self.base, "BHK": 1, "Size_in_SqFt": 450.0, "Price_in_Lakhs": 15.0})
        self.assertTrue(r["is_valid"])

    def test_03_premium_property(self):
        r = self.svc.predict_property({**self.base, "Property_Type": "Villa", "BHK": 5, "Size_in_SqFt": 4500.0, "Price_in_Lakhs": 420.0})
        self.assertTrue(r["is_valid"])

    def test_04_05_older_vs_newer(self):
        r_old = self.svc.predict_property({**self.base, "Year_Built": 1985})
        r_new = self.svc.predict_property({**self.base, "Year_Built": 2025})
        self.assertNotEqual(r_old["appreciation"]["future_price_lakhs"], r_new["appreciation"]["future_price_lakhs"])

    def test_06_different_bhk(self):
        r1 = self.svc.predict_property({**self.base, "BHK": 1})
        r4 = self.svc.predict_property({**self.base, "BHK": 4})
        self.assertNotEqual(r1["appreciation"]["future_price_lakhs"], r4["appreciation"]["future_price_lakhs"])

    def test_07_different_cities(self):
        r1 = self.svc.predict_property(self.base)
        r2 = self.svc.predict_property({**self.base, "State": "Tamil Nadu", "City": "Chennai", "Locality": "Locality_84"})
        self.assertNotEqual(r1["engineered_features"]["City_Train_Median_PPS"], r2["engineered_features"]["City_Train_Median_PPS"])

    def test_08_unknown_categories_ood_warning(self):
        r = self.svc.predict_property({**self.base, "State": "UnknownState", "City": "UnknownCity"})
        self.assertTrue(r["is_valid"] and len(r["warnings"]) > 0)

    def test_09_invalid_negative_values_rejected(self):
        for bad in [{"Price_in_Lakhs": -10}, {"Size_in_SqFt": 0}, {"Year_Built": 2035}]:
            self.assertFalse(self.svc.predict_property({**self.base, **bad})["is_valid"])

    def test_10_impossible_floor_combination_rejected(self):
        r = self.svc.predict_property({**self.base, "Floor_No": 15, "Total_Floors": 5})
        self.assertFalse(r["is_valid"])

    def test_11_extreme_valid_inputs_warned(self):
        r = self.svc.predict_property({**self.base, "Size_in_SqFt": 9500.0, "Price_in_Lakhs": 1500.0})
        self.assertTrue(r["is_valid"] and len(r["warnings"]) > 0)

    def test_12_interface_to_model_consistency(self):
        r = self.svc.predict_property({**self.base, "BHK": 5, "Floor_No": 4, "Total_Floors": 8})
        self.assertEqual(r["engineered_features"]["BHK"], 5.0)
        self.assertAlmostEqual(r["engineered_features"]["Floor_Ratio"], 0.5)

    def test_13_appreciation_and_cagr_math(self):
        c = calculate_appreciation(100.0, 146.9328, 5)
        self.assertAlmostEqual(c["total_appreciation_pct"], 46.93, places=2)
        self.assertAlmostEqual(c["annualized_appreciation_pct"], 8.00, places=2)

    def test_14_conformal_uncertainty_bounds(self):
        r = self.svc.predict_property(self.base)
        u, p = r["uncertainty"], r["appreciation"]["future_price_lakhs"]
        self.assertTrue(u["interval_90_lower_lakhs"] < u["interval_80_lower_lakhs"] < p < u["interval_80_upper_lakhs"] < u["interval_90_upper_lakhs"])

    def test_15_insufficient_cashflow_no_fake_verdict(self):
        r = self.svc.predict_property(self.base, cashflow_params=None)
        self.assertFalse(r["net_roi"]["is_available"])
        self.assertNotIn("GOOD INVESTMENT", r["verdict"]["status_label"].upper())

    def test_16_no_train_test_leakage(self):
        df_tr = pd.DataFrame([self.base, {**self.base, "Price_in_Lakhs": 85.0}])
        df_te = pd.DataFrame([{**self.base, "Price_in_Lakhs": 9000.0}])
        _, stats = engineer_features(df_tr, None)
        te_f, _ = engineer_features(df_te, stats)
        self.assertAlmostEqual(te_f["City_Train_Median_PPS"].iloc[0], ((75.0 + 85.0)/2.0 * 100000.0)/1600.0, places=2)

    def test_17_reproducible_predictions(self):
        self.assertEqual(self.svc.predict_property(self.base)["appreciation"]["future_price_lakhs"], self.svc.predict_property(self.base)["appreciation"]["future_price_lakhs"])

    def test_18_mismatched_city_state_rejected(self):
        r = self.svc.predict_property({**self.base, "State": "West Bengal", "City": "Chennai"})
        self.assertFalse(r["is_valid"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
