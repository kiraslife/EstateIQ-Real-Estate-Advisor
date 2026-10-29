import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
MIN_YEAR_BUILT = 1950

def to_numeric_safe(val, default=0.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip().lower()
    mapping = {
        "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
        "high": 3.0, "medium": 2.0, "low": 1.0, "none": 0.0,
        "furnished": 2.0, "semi-furnished": 1.0, "unfurnished": 0.0,
        "ready_to_move": 1.0, "under_construction": 0.0
    }
    if s in mapping:
        return mapping[s]
    try:
        return float(s)
    except Exception:
        return default

def validate_property_input(payload, reference_stats=None):
    errors, warnings = [], []
    req = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No", "Total_Floors"]
    for f in req:
        if f not in payload or payload[f] is None or str(payload[f]).strip() == "":
            errors.append(f"Missing required field: '{f}'.")
    if errors:
        return False, errors, warnings

    try:
        size_sqft = float(payload["Size_in_SqFt"])
        price_lakhs = float(payload["Price_in_Lakhs"])
        bhk = int(float(payload["BHK"]))
        year_built = int(float(payload["Year_Built"]))
        floor_no = int(float(payload["Floor_No"]))
        total_floors = int(float(payload["Total_Floors"]))
    except Exception as exc:
        return False, [f"Invalid numeric format: {exc}"], warnings

    if size_sqft <= 0:
        errors.append("Property size (Size_in_SqFt) must be strictly positive (> 0 sq ft).")
    elif size_sqft < 150:
        errors.append("Property size below 150 sq ft is outside defensible bounds.")
    if price_lakhs <= 0:
        errors.append("Current property price (Price_in_Lakhs) must be strictly positive (> 0 lakh).")
    if bhk < 1 or bhk > 12:
        errors.append("BHK must be between 1 and 12.")
    if year_built > CURRENT_YEAR or year_built < MIN_YEAR_BUILT:
        errors.append(f"Year built ({year_built}) must be between {MIN_YEAR_BUILT} and {CURRENT_YEAR}.")
    if total_floors < 1:
        errors.append("Total floors must be at least 1.")
    if floor_no < 0:
        errors.append("Floor number cannot be negative.")
    if total_floors >= 1 and floor_no > total_floors:
        errors.append(f"Impossible building configuration: Floor number ({floor_no}) cannot exceed Total Floors ({total_floors}).")

    if size_sqft > 0 and price_lakhs > 0:
        pps = (price_lakhs * 100000.0) / size_sqft
        if pps < 500 or pps > 150000:
            warnings.append(f"Extreme Price per SqFt (₹{pps:,.0f}/sq ft). Forecast reliability is reduced.")

    if reference_stats:
        st_val, ct_val, loc_val = str(payload["State"]), str(payload["City"]), str(payload["Locality"])
        known_states = set(reference_stats.get("known_states", []))
        known_cities = set(reference_stats.get("known_cities", []))
        known_locs = set(reference_stats.get("known_localities", []))
        cs_map = reference_stats.get("city_state_map", {})

        if st_val not in known_states:
            warnings.append(f"State '{st_val}' is unseen in training data (Out-of-Distribution).")
        if ct_val not in known_cities:
            warnings.append(f"City '{ct_val}' is unseen in training data (Out-of-Distribution).")
        elif st_val in known_states and ct_val in cs_map and cs_map[ct_val] != st_val:
            errors.append(f"Location mismatch: City '{ct_val}' does not belong to State '{st_val}'.")
        if loc_val not in known_locs:
            warnings.append(f"Locality '{loc_val}' is unseen in training data; using city-level baseline.")

        if size_sqft < reference_stats.get("size_q01", 300) or size_sqft > reference_stats.get("size_q99", 6000):
            warnings.append(f"Property size ({size_sqft:.0f} sq ft) is outside the 1st-99th training percentile.")
        if price_lakhs < reference_stats.get("price_q01", 5.0) or price_lakhs > reference_stats.get("price_q99", 600.0):
            warnings.append(f"Current price (₹{price_lakhs:.2f}L) is outside the 1st-99th training percentile.")

    return (len(errors) == 0), errors, warnings
