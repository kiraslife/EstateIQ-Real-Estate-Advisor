def calculate_appreciation(current_price_lakhs, future_price_lakhs, horizon_years=5):
    if current_price_lakhs <= 0 or horizon_years <= 0:
        raise ValueError("Current price and horizon years must be positive.")
    abs_gain = future_price_lakhs - current_price_lakhs
    total_pct = (abs_gain / current_price_lakhs) * 100.0
    cagr_pct = (((future_price_lakhs / current_price_lakhs) ** (1.0 / horizon_years)) - 1.0) * 100.0
    return {
        "current_price_lakhs": round(float(current_price_lakhs), 2),
        "future_price_lakhs": round(float(future_price_lakhs), 2),
        "absolute_gain_lakhs": round(float(abs_gain), 2),
        "total_appreciation_pct": round(float(total_pct), 2),
        "annualized_appreciation_pct": round(float(cagr_pct), 2),
        "horizon_years": int(horizon_years)
    }

def calculate_net_investment_return(current_price_lakhs, future_price_lakhs, horizon_years=5,
                                    gross_rental_yield_pct=None, stamp_duty_registration_pct=None,
                                    annual_maintenance_tax_pct=None, vacancy_rate_pct=None, exit_brokerage_tax_pct=None):
    req = [gross_rental_yield_pct, stamp_duty_registration_pct, annual_maintenance_tax_pct, vacancy_rate_pct, exit_brokerage_tax_pct]
    if any(p is None for p in req):
        return {
            "is_available": False, "net_roi_pct": None, "annualized_net_roi_pct": None, "net_profit_lakhs": None,
            "reason": "Net ROI not calculated: rental income, stamp duty, maintenance, vacancy, and exit costs were not provided."
        }
    acq_cost = current_price_lakhs * (1.0 + (stamp_duty_registration_pct / 100.0))
    eff_rent = current_price_lakhs * (gross_rental_yield_pct / 100.0) * (1.0 - (vacancy_rate_pct / 100.0))
    hold_cost = current_price_lakhs * (annual_maintenance_tax_pct / 100.0)
    cum_net_rent = (eff_rent - hold_cost) * horizon_years
    exit_cost = future_price_lakhs * (exit_brokerage_tax_pct / 100.0)
    net_term = future_price_lakhs - exit_cost + cum_net_rent
    net_profit = net_term - acq_cost
    net_roi_pct = (net_profit / acq_cost) * 100.0
    net_cagr = (((max(net_term, 0.01) / acq_cost) ** (1.0 / horizon_years)) - 1.0) * 100.0
    return {
        "is_available": True,
        "net_profit_lakhs": round(float(net_profit), 2),
        "net_roi_pct": round(float(net_roi_pct), 2),
        "annualized_net_roi_pct": round(float(net_cagr), 2)
    }

def evaluate_investment_verdict(current_price_lakhs, price_per_sqft, locality_median_pps,
                                annualized_appreciation_pct, uncertainty_half_width_pct, ood_warnings, net_roi_result):
    evaluated = [
        f"Entered Price/SqFt (₹{price_per_sqft:,.0f}) vs. City/Locality Benchmark (₹{locality_median_pps:,.0f})",
        f"Projected 5-Year Annualized Appreciation ({annualized_appreciation_pct:.2f}% p.a.)",
        f"80% Conformal Prediction Interval Half-Width (±{uncertainty_half_width_pct:.1f}%)"
    ]
    omitted = []
    if not net_roi_result.get("is_available", False):
        omitted = ["Rental yield & vacancy allowance", "Stamp duty, registration & exit costs", "Verified repeat-sales historical backtest"]
    else:
        evaluated.append(f"User-supplied Net Investment Return ({net_roi_result['annualized_net_roi_pct']:.2f}% annualized net ROI)")
        omitted = ["Verified historical repeat-sales market index backtest"]

    ratio = price_per_sqft / max(locality_median_pps, 1.0)
    if len(ood_warnings) > 0:
        label, tone = "Forecast Available — Out-of-Distribution Warning", "warning"
        summary = "Inputs lie outside typical training distribution or use unseen categories. Treat projection with caution."
    elif ratio > 1.35:
        label, tone = "Price Above Locality Benchmark — Caution Advised", "warning"
        summary = f"Entered price/sq ft is {(ratio - 1.0)*100:.1f}% above the city median benchmark, compressing upside."
    elif not net_roi_result.get("is_available", False):
        label, tone = "Projected Appreciation Only — Cash Flows Excluded", "neutral"
        summary = "Projection reflects estimated gross property price change only. Enable Cash-Flow inputs to evaluate Net ROI."
    else:
        nc = net_roi_result.get("annualized_net_roi_pct", 0.0)
        if nc >= 7.0 and ratio <= 1.15:
            label, tone = "Positive Net Return Scenario (Under User Assumptions)", "positive"
            summary = f"With supplied rental and transaction assumptions, projected annualized net return is {nc:.2f}%."
        else:
            label, tone = "Moderate / Constrained Net Return Scenario", "warning"
            summary = f"After transaction, holding, and exit costs, projected annualized net return is {nc:.2f}%."

    return {"status_label": label, "badge_tone": tone, "summary": summary, "factors_evaluated": evaluated, "factors_omitted": omitted}
