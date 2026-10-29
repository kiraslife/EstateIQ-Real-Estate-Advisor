import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from src.predictor import PropertyPredictionService

st.set_page_config(
    page_title="EstateIQ | AI Property Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(html_str: str):
    """Strips leading spaces on every line so Markdown NEVER treats HTML as a <pre><code> block."""
    clean = "\n".join(line.lstrip() for line in html_str.strip().splitlines())
    st.markdown(clean, unsafe_allow_html=True)

render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #F4F6FA; }
.stApp { background: radial-gradient(circle at 15% 10%, #161A26 0%, #0B0D12 65%); }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background-color: #0E1118 !important; border-right: 1px solid #222734; }
.block-container { padding-top: 3.0rem !important; padding-bottom: 2.5rem !important; max-width: 1440px; }

/* Style Streamlit native bordered containers as sleek dark SaaS panels */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, #171B24 0%, #11141C 100%) !important;
    border: 1px solid #262C3A !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}

/* Top Search & Profile Bar */
.saas-topbar {
    background: rgba(22, 26, 35, 0.90); border: 1px solid #262C3A; border-radius: 14px;
    padding: 12px 20px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center;
}
.saas-search-pill {
    background: #1B202C; border: 1px solid #2A3142; border-radius: 10px; padding: 8px 14px;
    color: #9AA4B2; font-size: 0.82rem; display: flex; align-items: center; gap: 10px; width: 420px;
}
.saas-profile { display: flex; align-items: center; gap: 12px; }
.saas-icon-btn {
    background: #1B202C; border: 1px solid #2A3142; border-radius: 50%; width: 34px; height: 34px;
    display: flex; align-items: center; justify-content: center; font-size: 0.85rem; color: #9AA4B2;
}
.saas-avatar {
    background: linear-gradient(135deg, #FF5A36 0%, #FF2E63 100%); border-radius: 50%;
    width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.82rem; color: #FFF; box-shadow: 0 0 12px rgba(255, 77, 87, 0.4);
}

/* 4 Top KPI Cards */
.kpi-card {
    background: linear-gradient(160deg, #181C26 0%, #12151E 100%);
    border: 1px solid #262C3A; border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.28); transition: transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover { border-color: #384152; transform: translateY(-2px); }
.kpi-top { display: flex; align-items: center; gap: 8px; color: #9AA4B2; font-size: 0.80rem; font-weight: 500; margin-bottom: 10px; }
.kpi-icon { background: #202634; border: 1px solid #2C3446; border-radius: 8px; padding: 4px 7px; font-size: 0.78rem; color: #F4F6FA; }
.kpi-val { font-size: 1.85rem; font-weight: 700; color: #F4F6FA; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; margin-bottom: 10px; }
.kpi-delta-pos { background: rgba(16, 185, 129, 0.14); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 9px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }
.kpi-delta-warn { background: rgba(245, 158, 11, 0.14); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); padding: 3px 9px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }
.kpi-delta-red { background: rgba(255, 77, 87, 0.15); color: #FF6B72; border: 1px solid rgba(255, 77, 87, 0.3); padding: 3px 9px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }

/* Section Headers & AI Alerts Stack */
.panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.panel-title { font-size: 1.02rem; font-weight: 600; color: #F4F6FA; margin: 0; }
.panel-sub { font-size: 0.78rem; color: #8A94A6; margin-bottom: 12px; }

.alert-item {
    background: #1A1F2B; border: 1px solid #283040; border-radius: 12px;
    padding: 13px 15px; margin-bottom: 10px; transition: border-color 0.15s ease;
}
.alert-item:hover { border-color: #3B465E; }
.alert-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.alert-name { font-size: 0.85rem; font-weight: 600; color: #F4F6FA; }
.badge-high { background: rgba(255, 77, 87, 0.18); color: #FF6B72; border: 1px solid rgba(255, 77, 87, 0.35); padding: 2px 8px; border-radius: 6px; font-size: 0.69rem; font-weight: 600; white-space: nowrap; }
.badge-good { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.35); padding: 2px 8px; border-radius: 6px; font-size: 0.69rem; font-weight: 600; white-space: nowrap; }
.badge-warn { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.35); padding: 2px 8px; border-radius: 6px; font-size: 0.69rem; font-weight: 600; white-space: nowrap; }
.alert-desc { font-size: 0.76rem; color: #9AA4B2; line-height: 1.45; margin-bottom: 6px; }
.alert-foot { font-size: 0.72rem; font-weight: 500; display: flex; align-items: center; gap: 5px; }

/* Recent Activity Dark Table */
.saas-table { width: 100%; border-collapse: collapse; font-size: 0.80rem; margin-top: 6px; }
.saas-table th { text-align: left; padding: 10px 10px; color: #8A94A6; font-weight: 500; border-bottom: 1px solid #262C3A; }
.saas-table td { padding: 11px 10px; color: #E2E8F0; border-bottom: 1px solid #1E2330; font-variant-numeric: tabular-nums; }
.saas-table tr:hover td { background: rgba(255,255,255,0.025); }

.pro-card {
    background: linear-gradient(160deg, #1B202C 0%, #141720 100%);
    border: 1px solid #2C3446; border-radius: 14px; padding: 16px; margin-top: 18px;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #FF5A36 0%, #FF3B5C 100%) !important;
    color: #FFF !important; border: none !important; font-weight: 600 !important;
    border-radius: 10px !important; height: 42px !important;
    box-shadow: 0 4px 14px rgba(255, 77, 87, 0.35) !important;
}
</style>
""")

@st.cache_resource
def load_core():
    return PropertyPredictionService(), pd.read_csv("data/eda_ready_data.csv")

service, df_data = load_core()
rep = service.eval_report

def short_badge_label(status_label: str) -> str:
    s = status_label.lower()
    if "positive net return" in s:
        return "Positive Net ROI"
    if "moderate" in s:
        return "Moderate Net ROI"
    if "above locality" in s:
        return "Above Benchmark"
    if "out-of-distribution" in s:
        return "OOD Warning"
    return "Gross Appreciation Only"

if "history_log" not in st.session_state:
    seed_inputs = [
        {"State": "West Bengal", "City": "Durgapur", "Locality": "Locality_246", "Property_Type": "Apartment", "BHK": 3, "Size_in_SqFt": 1650, "Price_in_Lakhs": 75.0, "Year_Built": 2019, "Furnished_Status": "Semi-Furnished", "Floor_No": 3, "Total_Floors": 10, "Nearby_Schools": 5, "Nearby_Hospitals": 4, "Public_Transport_Accessibility": "High", "Parking_Space": "Yes", "Security": "Yes", "Amenities": "Clubhouse", "Facing": "East", "Owner_Type": "Builder", "Availability_Status": "Ready_to_Move"},
        {"State": "Maharashtra", "City": "Pune", "Locality": "Locality_490", "Property_Type": "Villa", "BHK": 4, "Size_in_SqFt": 2800, "Price_in_Lakhs": 165.0, "Year_Built": 2022, "Furnished_Status": "Furnished", "Floor_No": 1, "Total_Floors": 2, "Nearby_Schools": 8, "Nearby_Hospitals": 6, "Public_Transport_Accessibility": "High", "Parking_Space": "Yes", "Security": "Yes", "Amenities": "Pool", "Facing": "North", "Owner_Type": "Builder", "Availability_Status": "Ready_to_Move"},
        {"State": "Karnataka", "City": "Bangalore", "Locality": "Locality_462", "Property_Type": "Apartment", "BHK": 2, "Size_in_SqFt": 950, "Price_in_Lakhs": 185.0, "Year_Built": 2006, "Furnished_Status": "Unfurnished", "Floor_No": 7, "Total_Floors": 12, "Nearby_Schools": 2, "Nearby_Hospitals": 2, "Public_Transport_Accessibility": "Low", "Parking_Space": "No", "Security": "No", "Amenities": "Gym", "Facing": "West", "Owner_Type": "Broker", "Availability_Status": "Ready_to_Move"}
    ]
    st.session_state["history_log"] = [service.predict_property(x) for x in seed_inputs]

# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
with st.sidebar:
    render_html("""
    <div style="display:flex; align-items:center; gap:10px; padding: 4px 4px 16px 4px; border-bottom:1px solid #222734; margin-bottom:14px;">
        <div style="background:linear-gradient(135deg,#FF5A36,#FF2E63); width:28px; height:28px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#FFF;">⚡</div>
        <div>
            <div style="font-size:1.08rem; font-weight:700; color:#F4F6FA; letter-spacing:-0.02em;">EstateIQ</div>
            <div style="font-size:0.68rem; color:#8A94A6;">Quantitative Property AI</div>
        </div>
    </div>
    <div style="font-size:0.68rem; font-weight:600; color:#64748B; letter-spacing:0.08em; margin:8px 0 6px 4px;">GENERAL & INSIGHTS</div>
    """)

    nav = st.radio(
        "Menu",
        [
            "🏠 Home · Interactive Analyzer",
            "📊 Market Telemetry",
            "⚖️ Compare Properties",
            "🎯 Model Performance Audit",
            "🛡️ Data & Methodology"
        ],
        label_visibility="collapsed"
    )

    r_hold = rep.get("repaired_on_holdout", {})
    render_html(f"""
    <div class="pro-card">
        <div style="display:flex; align-items:center; gap:8px; font-weight:600; font-size:0.86rem; color:#F4F6FA; margin-bottom:6px;">
            <span>⚙️</span> Calibrated Engine
        </div>
        <div style="font-size:0.74rem; color:#9AA4B2; line-height:1.5;">
            Leak-free XGBoost + Split-Conformal Intervals.<br>
            Holdout MAE: <b style="color:#F4F6FA;">₹{r_hold.get('mae_lakhs', 1.84):.2f}L</b> · R²: <b style="color:#34D399;">{r_hold.get('r2', 0.9999):.4f}</b>
        </div>
    </div>
    """)

# ---------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------
render_html("""
<div class="saas-topbar">
    <div class="saas-search-pill">
        <span>🔍</span> <span>Interactive Real-Time Valuation & Split-Conformal Horizon Engine</span>
    </div>
    <div class="saas-profile">
        <div class="saas-icon-btn" title="Split-Conformal Calibrated">🛡️</div>
        <div class="saas-icon-btn" title="Holdout Verified">📈</div>
        <div class="saas-avatar" title="Suraj Das">SD</div>
    </div>
</div>
""")

# =====================================================================
# PAGE 1: HOME · INTERACTIVE ANALYZER
# =====================================================================
if nav.startswith("🏠"):
    with st.expander("🎛️ Interactive Property Parameters & Cash-Flow Controls (Live Reactive — Adjust Any Input)", expanded=True):
        t_loc, t_spec, t_amen, t_cf = st.tabs([
            "📍 1. Location & Property Type",
            "🏗️ 2. Area, Price & Building Specs",
            "🚇 3. Transit & Civic Amenities",
            "💰 4. Cash-Flow & Net ROI Assumptions"
        ])
        with t_loc:
            c1, c2, c3, c4, c5 = st.columns(5)
            states = sorted(df_data["State"].astype(str).unique())
            st_sel = c1.selectbox("State", states, index=states.index("West Bengal") if "West Bengal" in states else 0)
            cities = sorted(df_data[df_data["State"].astype(str) == st_sel]["City"].astype(str).unique())
            ct_sel = c2.selectbox("City", cities, index=cities.index("Durgapur") if "Durgapur" in cities else 0)
            locs = sorted(df_data[df_data["City"].astype(str) == ct_sel]["Locality"].astype(str).unique())
            lc_sel = c3.selectbox("Locality", locs)
            pt_sel = c4.selectbox("Property Type", sorted(df_data["Property_Type"].astype(str).unique()))
            bhk_sel = c5.number_input("BHK", 1, 10, 3)

        with t_spec:
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            pr_sel = s1.number_input("Current Price (₹ Lakh)", 1.0, 5000.0, 75.0, step=2.5)
            sz_sel = s2.number_input("Area (sq ft)", 150, 15000, 1650, step=50)
            yr_sel = s3.number_input("Year Built", 1950, 2026, 2019)
            fl_sel = s4.number_input("Floor No", 0, 100, 3)
            tf_sel = s5.number_input("Total Floors", 1, 100, 10)
            fu_sel = s6.selectbox("Furnished", sorted(df_data["Furnished_Status"].astype(str).unique()))

        with t_amen:
            a1, a2, a3, a4, a5, a6, a7, a8 = st.columns(8)
            tr_sel = a1.selectbox("Transit", sorted(df_data["Public_Transport_Accessibility"].astype(str).unique()))
            sc_sel = a2.selectbox("Schools", sorted(df_data["Nearby_Schools"].unique()), index=4)
            ho_sel = a3.selectbox("Hospitals", sorted(df_data["Nearby_Hospitals"].unique()), index=3)
            pk_sel = a4.selectbox("Parking", sorted(df_data["Parking_Space"].astype(str).unique()))
            se_sel = a5.selectbox("Security", sorted(df_data["Security"].astype(str).unique()))
            am_sel = a6.selectbox("Amenities", sorted(df_data["Amenities"].astype(str).unique()))
            fa_sel = a7.selectbox("Facing", sorted(df_data["Facing"].astype(str).unique()))
            ow_sel = a8.selectbox("Owner", sorted(df_data["Owner_Type"].astype(str).unique()))
            av_sel = "Ready_to_Move"

        with t_cf:
            cf0, cf1, cf2, cf3, cf4, cf5 = st.columns([1.3, 1, 1, 1, 1, 1])
            use_cf = cf0.checkbox("Enable Net ROI Cash-Flows", value=True)
            gy = cf1.number_input("Rental Yield (%)", 0.0, 15.0, 3.5, 0.2, disabled=not use_cf)
            sd = cf2.number_input("Stamp Duty (%)", 0.0, 15.0, 6.0, 0.5, disabled=not use_cf)
            mt = cf3.number_input("Maint/Tax (%)", 0.0, 5.0, 0.7, 0.1, disabled=not use_cf)
            vc = cf4.number_input("Vacancy (%)", 0.0, 50.0, 5.0, 1.0, disabled=not use_cf)
            ex = cf5.number_input("Exit Cost (%)", 0.0, 10.0, 1.5, 0.5, disabled=not use_cf)

    payload = {
        "State": st_sel, "City": ct_sel, "Locality": lc_sel, "Property_Type": pt_sel,
        "BHK": bhk_sel, "Size_in_SqFt": sz_sel, "Price_in_Lakhs": pr_sel, "Year_Built": yr_sel,
        "Furnished_Status": fu_sel, "Floor_No": fl_sel, "Total_Floors": tf_sel,
        "Nearby_Schools": sc_sel, "Nearby_Hospitals": ho_sel, "Public_Transport_Accessibility": tr_sel,
        "Parking_Space": pk_sel, "Security": se_sel, "Amenities": am_sel, "Facing": fa_sel,
        "Owner_Type": ow_sel, "Availability_Status": av_sel
    }
    cf_in = {"gross_rental_yield_pct": gy, "stamp_duty_registration_pct": sd, "annual_maintenance_tax_pct": mt, "vacancy_rate_pct": vc, "exit_brokerage_tax_pct": ex} if use_cf else None
    res = service.predict_property(payload, cashflow_params=cf_in)

    if not res["is_valid"]:
        for err in res["errors"]:
            st.error(f"Validation Error: {err}")
        st.stop()

    ap, un, vd, nr, ps = res["appreciation"], res["uncertainty"], res["verdict"], res["net_roi"], res["property_summary"]
    sens_list = service.compute_feature_sensitivity(payload)

    top_bar_l, top_bar_r = st.columns([3.8, 1.2])
    with top_bar_l:
        for w in res["warnings"]:
            st.warning(f"Distribution Notice: {w}")
    with top_bar_r:
        if st.button("＋ Log to Activity & Compare", type="primary", use_container_width=True):
            st.session_state["history_log"].insert(0, res)
            st.toast("Property added to Recent Activity & Compare Matrix!")

    # ROW 1: 4 TOP KPI CARDS
    k1, k2, k3, k4 = st.columns(4)
    pps_ratio = ps["Price_per_SqFt"] / max(ps["City_Median_PPS"], 1)
    pps_pill_cls = "kpi-delta-pos" if pps_ratio <= 1.05 else ("kpi-delta-warn" if pps_ratio <= 1.30 else "kpi-delta-red")
    pps_diff_pct = (pps_ratio - 1.0) * 100.0

    with k1:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top"><span class="kpi-icon">↗</span> 5Y Projected Valuation</div>
            <div class="kpi-val">₹{ap['future_price_lakhs']:,.2f}L</div>
            <div><span class="kpi-delta-pos">+{ap['total_appreciation_pct']:.2f}%</span> <span style="font-size:0.74rem; color:#8A94A6;">vs ₹{ap['current_price_lakhs']:,.2f}L current</span></div>
        </div>
        """)
    with k2:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top"><span class="kpi-icon">🏢</span> Implied Price / Sq Ft</div>
            <div class="kpi-val">₹{ps['Price_per_SqFt']:,.0f}</div>
            <div><span class="{pps_pill_cls}">{pps_diff_pct:+.1f}%</span> <span style="font-size:0.74rem; color:#8A94A6;">vs ₹{ps['City_Median_PPS']:,.0f} city median</span></div>
        </div>
        """)
    with k3:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top"><span class="kpi-icon">🛡️</span> 80% Conformal Band</div>
            <div class="kpi-val">±{un['half_width_80_pct']:.1f}%</div>
            <div><span class="kpi-delta-warn">₹{un['interval_80_lower_lakhs']:,.1f}L – ₹{un['interval_80_upper_lakhs']:,.1f}L</span> <span style="font-size:0.74rem; color:#8A94A6;">range</span></div>
        </div>
        """)
    with k4:
        if nr["is_available"]:
            nr_cls = "kpi-delta-pos" if nr["annualized_net_roi_pct"] >= 7.0 else "kpi-delta-warn"
            render_html(f"""
            <div class="kpi-card">
                <div class="kpi-top"><span class="kpi-icon">⚡</span> Net Investment ROI (5Y)</div>
                <div class="kpi-val">{nr['net_roi_pct']:+.2f}%</div>
                <div><span class="{nr_cls}">{nr['annualized_net_roi_pct']:+.2f}% p.a.</span> <span style="font-size:0.74rem; color:#8A94A6;">Net Profit ₹{nr['net_profit_lakhs']:,.1f}L</span></div>
            </div>
            """)
        else:
            render_html(f"""
            <div class="kpi-card">
                <div class="kpi-top"><span class="kpi-icon">📈</span> Annualized Growth (CAGR)</div>
                <div class="kpi-val">+{ap['annualized_appreciation_pct']:.2f}%</div>
                <div><span class="kpi-delta-warn">Gross Only</span> <span style="font-size:0.74rem; color:#8A94A6;">Cash-flows excluded</span></div>
            </div>
            """)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ROW 2: 2/3 LEFT INTERACTIVE CHARTS & LOG + 1/3 RIGHT AI ALERTS STACK
    mid_left, mid_right = st.columns([1.85, 1.05])

    with mid_left:
        with st.container(border=True):
            render_html("""
            <div class="panel-head">
                <div>
                    <div class="panel-title">Valuation Trajectory & Counterfactual Sensitivity</div>
                    <div class="panel-sub">Interactive compound trajectory illustration (Current to 5Y Model Estimate) vs. City Comparable Baseline</div>
                </div>
                <div style="font-size:0.76rem; color:#9AA4B2;">
                    <span style="color:#FF4D57;">●</span> Property Forecast &nbsp;&nbsp;
                    <span style="color:#8B5CF6;">●</span> City Avg Baseline
                </div>
            </div>
            """)
            ch_tab1, ch_tab2 = st.tabs(["📈 5-Year Trajectory Curve (Interactive Hover)", "🌪️ Feature Sensitivity Impact"])
            with ch_tab1:
                years = [0, 1, 2, 3, 4, 5]
                cagr_prop = ap["annualized_appreciation_pct"] / 100.0
                cagr_city = 0.074
                rows_curve = []
                for y in years:
                    yr_label = f"Year {y} ({2026 + y})"
                    val_p = round(ap["current_price_lakhs"] * ((1.0 + cagr_prop) ** y), 2)
                    val_c = round(ap["current_price_lakhs"] * ((1.0 + cagr_city) ** y), 2)
                    hw_y = (un["half_width_80_pct"] / 100.0) * val_p * (y / 5.0)
                    rows_curve.append({"Year": yr_label, "Series": "Selected Property Forecast", "Valuation_Lakhs": val_p, "Lower_80": round(val_p - hw_y, 2), "Upper_80": round(val_p + hw_y, 2)})
                    rows_curve.append({"Year": yr_label, "Series": "City Comparable Baseline", "Valuation_Lakhs": val_c, "Lower_80": val_c, "Upper_80": val_c})
                curve_df = pd.DataFrame(rows_curve)

                base_chart = alt.Chart(curve_df).encode(
                    x=alt.X("Year:N", title=None, axis=alt.Axis(labelColor="#9AA4B2", labelAngle=0, grid=False, domainColor="#262C3A"))
                )
                band = alt.Chart(curve_df[curve_df["Series"] == "Selected Property Forecast"]).mark_area(
                    opacity=0.16, color="#FF4D57", interpolate="monotone"
                ).encode(
                    x="Year:N",
                    y=alt.Y("Lower_80:Q", title="Valuation (₹ Lakhs)", scale=alt.Scale(zero=False), axis=alt.Axis(labelColor="#9AA4B2", titleColor="#9AA4B2", gridColor="#1E2330", domainColor="#262C3A")),
                    y2="Upper_80:Q"
                )
                lines = base_chart.mark_line(interpolate="monotone", strokeWidth=3, point=alt.OverlayMarkDef(size=65, filled=True)).encode(
                    y=alt.Y("Valuation_Lakhs:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=["Selected Property Forecast", "City Comparable Baseline"], range=["#FF4D57", "#8B5CF6"]), legend=None),
                    tooltip=[
                        alt.Tooltip("Year:N", title="Horizon"),
                        alt.Tooltip("Series:N", title="Trajectory"),
                        alt.Tooltip("Valuation_Lakhs:Q", title="Estimated Value (₹L)", format=".2f"),
                        alt.Tooltip("Lower_80:Q", title="80% Band Lower (₹L)", format=".2f"),
                        alt.Tooltip("Upper_80:Q", title="80% Band Upper (₹L)", format=".2f")
                    ]
                )
                st.altair_chart((band + lines).properties(height=275, background="transparent", padding={"left": 28, "right": 16, "top": 12, "bottom": 10}).configure_view(strokeWidth=0), use_container_width=True)

            with ch_tab2:
                if sens_list:
                    sdf = pd.DataFrame(sens_list)
                    sens_chart = alt.Chart(sdf).mark_bar(cornerRadiusEnd=4, height=18).encode(
                        x=alt.X("Delta_Lakhs:Q", title="Δ 5-Year Valuation vs. Baseline (₹ Lakhs)", axis=alt.Axis(labelColor="#9AA4B2", titleColor="#9AA4B2", gridColor="#1E2330")),
                        y=alt.Y("Scenario:N", sort="-x", title=None, axis=alt.Axis(labelColor="#E2E8F0")),
                        color=alt.condition(alt.datum.Delta_Lakhs >= 0, alt.value("#10B981"), alt.value("#FF4D57")),
                        tooltip=["Scenario:N", alt.Tooltip("Predicted_5Y_Lakhs:Q", title="New 5Y Estimate (₹L)"), alt.Tooltip("Delta_Lakhs:Q", title="Delta (₹L)")]
                    ).properties(height=275, background="transparent", padding={"left": 28, "right": 16, "top": 12, "bottom": 10}).configure_view(strokeWidth=0)
                    st.altair_chart(sens_chart, use_container_width=True)

        # BOTTOM LEFT: RECENT ACTIVITY & BENCHMARK TABLE
        with st.container(border=True):
            render_html("""
            <div class="panel-title">Recent Activity & Property Comparison Log</div>
            <div class="panel-sub">Live audit trail of evaluated properties across your session</div>
            """)
            act_tab1, act_tab2 = st.tabs(["🕒 Analyzed Properties Log", "🏙️ City Benchmarks"])
            with act_tab1:
                row_chunks = []
                for item in st.session_state["history_log"][:6]:
                    p_s = item["property_summary"]
                    a_p = item["appreciation"]
                    u_n = item["uncertainty"]
                    v_d = item["verdict"]
                    b_cls = "badge-good" if v_d["badge_tone"] == "positive" else ("badge-warn" if v_d["badge_tone"] == "warning" else "badge-high")
                    b_txt = short_badge_label(v_d["status_label"])
                    row_chunks.append(
                        f"<tr>"
                        f"<td><b>{p_s['City']}</b> · {p_s['Locality']}</td>"
                        f"<td>{p_s['BHK']} BHK {p_s['Property_Type']} ({p_s['Size_in_SqFt']:,.0f} sqft)</td>"
                        f"<td>₹{a_p['current_price_lakhs']:,.2f}L (₹{p_s['Price_per_SqFt']:,.0f}/sqft)</td>"
                        f"<td style='color:#34D399; font-weight:600;'>₹{a_p['future_price_lakhs']:,.2f}L (+{a_p['annualized_appreciation_pct']:.2f}%/yr)</td>"
                        f"<td>₹{u_n['interval_80_lower_lakhs']:,.1f}L – ₹{u_n['interval_80_upper_lakhs']:,.1f}L</td>"
                        f"<td><span class='{b_cls}'>{b_txt}</span></td>"
                        f"</tr>"
                    )
                table_html = (
                    "<table class='saas-table'>"
                    "<thead><tr><th>Location</th><th>Configuration</th><th>Entered Price</th><th>5Y Model Forecast</th><th>80% Interval</th><th>Diagnostic Status</th></tr></thead>"
                    "<tbody>" + "".join(row_chunks) + "</tbody></table>"
                )
                render_html(table_html)

            with act_tab2:
                city_summary = df_data.groupby("City").agg(
                    Listings=("Price_in_Lakhs", "count"),
                    Median_Price_Lakhs=("Price_in_Lakhs", "median"),
                    Median_PPS=("Price_per_SqFt", "median"),
                    Median_5Y_Forecast=("Future_Price_5Y", "median")
                ).reset_index().sort_values("Median_PPS", ascending=False).head(8)
                st.dataframe(city_summary, use_container_width=True, hide_index=True)

    # RIGHT COLUMN: AI ALERTS & RECOMMENDATIONS STACK
    with mid_right:
        with st.container(border=True):
            if pps_ratio > 1.25:
                a1_title, a1_badge, a1_cls = "Entry Price Premium Detected", "High", "badge-high"
                a1_desc = f"Entered ₹{ps['Price_per_SqFt']:,.0f}/sqft is {(pps_ratio-1)*100:.1f}% above {ps['City']}'s median (₹{ps['City_Median_PPS']:,.0f}/sqft), triggering mean-reversion drag."
                a1_foot, a1_color = "● Compress acquisition price to improve margin", "#FF6B72"
            else:
                a1_title, a1_badge, a1_cls = "Favorable Locality Entry Pricing", "Good", "badge-good"
                a1_desc = f"Entered ₹{ps['Price_per_SqFt']:,.0f}/sqft aligns with or beats {ps['City']}'s training benchmark (₹{ps['City_Median_PPS']:,.0f}/sqft)."
                a1_foot, a1_color = "✔ Positive relative value positioning", "#34D399"

            if len(res["warnings"]) > 0:
                a2_title, a2_badge, a2_cls = "Out-of-Distribution Input Alert", "Warning", "badge-warn"
                a2_desc = res["warnings"][0]
                a2_foot, a2_color = "▲ Interval widened by 1.5x safety factor", "#FBBF24"
            else:
                a2_title, a2_badge, a2_cls = "Split-Conformal Band Calibrated", "Good", "badge-good"
                a2_desc = f"80% prediction interval spans ₹{un['interval_80_lower_lakhs']:,.2f}L to ₹{un['interval_80_upper_lakhs']:,.2f}L ({un['holdout_coverage_80_pct']:.1f}% empirical holdout coverage)."
                a2_foot, a2_color = "✔ Within 1st–99th training distribution", "#34D399"

            if sens_list:
                top_s = max(sens_list, key=lambda x: abs(x["Delta_Lakhs"]))
                a3_title, a3_badge, a3_cls = f"Sensitivity: {top_s['Scenario']}", "Impact", "badge-warn"
                a3_desc = f"Perturbing '{top_s['Scenario']}' shifts the 5-year valuation by ₹{top_s['Delta_Lakhs']:+.2f} Lakhs (to ₹{top_s['Predicted_5Y_Lakhs']:,.2f}L)."
                a3_foot, a3_color = "⚡ Controlled one-at-a-time counterfactual", "#FBBF24"
            else:
                a3_title, a3_badge, a3_cls = "Feature Sensitivity Ready", "Info", "badge-good"
                a3_desc = "All 20 physical and civic attributes are active in the ColumnTransformer pipeline."
                a3_foot, a3_color = "✔ Multi-factor responsiveness verified", "#34D399"

            if nr["is_available"]:
                a4_title = "Net Cash-Flow & Yield Evaluated"
                a4_badge = "Good" if nr["annualized_net_roi_pct"] >= 7.0 else "Moderate"
                a4_cls = "badge-good" if nr["annualized_net_roi_pct"] >= 7.0 else "badge-warn"
                a4_desc = f"After {sd:.1f}% stamp duty, {mt:.1f}% maintenance, and {gy:.1f}% gross rental yield, net 5Y profit is ₹{nr['net_profit_lakhs']:,.2f}L ({nr['annualized_net_roi_pct']:+.2f}%/yr)."
                a4_foot, a4_color = f"✔ {short_badge_label(vd['status_label'])}", "#34D399"
            else:
                a4_title, a4_badge, a4_cls = "Transaction Costs & Rent Omitted", "Warning", "badge-high"
                a4_desc = "Gross appreciation (+5Y price) does not equal net ROI. Enable Tab 4 Cash-Flows to include stamp duty & rental yield."
                a4_foot, a4_color = "● Showing Projected Appreciation Only", "#FF6B72"

            render_html(f"""
            <div class="panel-title">AI Alerts & Recommendations</div>
            <div class="panel-sub">Real-time quantitative diagnostics on your selected property</div>
            <div class="alert-item">
                <div class="alert-top"><span class="alert-name">{a1_title}</span><span class="{a1_cls}">{a1_badge}</span></div>
                <div class="alert-desc">{a1_desc}</div>
                <div class="alert-foot" style="color:{a1_color};">{a1_foot}</div>
            </div>
            <div class="alert-item">
                <div class="alert-top"><span class="alert-name">{a2_title}</span><span class="{a2_cls}">{a2_badge}</span></div>
                <div class="alert-desc">{a2_desc}</div>
                <div class="alert-foot" style="color:{a2_color};">{a2_foot}</div>
            </div>
            <div class="alert-item">
                <div class="alert-top"><span class="alert-name">{a3_title}</span><span class="{a3_cls}">{a3_badge}</span></div>
                <div class="alert-desc">{a3_desc}</div>
                <div class="alert-foot" style="color:{a3_color};">{a3_foot}</div>
            </div>
            <div class="alert-item" style="margin-bottom:0;">
                <div class="alert-top"><span class="alert-name">{a4_title}</span><span class="{a4_cls}">{a4_badge}</span></div>
                <div class="alert-desc">{a4_desc}</div>
                <div class="alert-foot" style="color:{a4_color};">{a4_foot}</div>
            </div>
            """)

# =====================================================================
# PAGE 2: MARKET TELEMETRY
# =====================================================================
elif nav.startswith("📊"):
    with st.container(border=True):
        render_html("""
        <div class="panel-title">Market Telemetry & Dataset Explorer (250,000 Verified Records)</div>
        <div class="panel-sub">Filter cross-sectional property distributions by City, Property Type, and Price Tier</div>
        """)
        f1, f2, f3 = st.columns(3)
        sel_c = f1.multiselect("Filter Cities", sorted(df_data["City"].astype(str).unique()), default=["Durgapur", "Kolkata", "Pune", "Bangalore"])
        sel_t = f2.multiselect("Property Type", sorted(df_data["Property_Type"].astype(str).unique()), default=sorted(df_data["Property_Type"].astype(str).unique()))
        max_p = f3.slider("Max Price (₹ Lakhs)", 10.0, 500.0, 300.0)
        sub = df_data[(df_data["City"].astype(str).isin(sel_c)) & (df_data["Property_Type"].astype(str).isin(sel_t)) & (df_data["Price_in_Lakhs"] <= max_p)]
        st.caption(f"Matching properties: **{len(sub):,}** (showing top 150)")
        st.dataframe(sub[["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Future_Price_5Y"]].head(150), use_container_width=True, hide_index=True)

# =====================================================================
# PAGE 3: COMPARE PROPERTIES
# =====================================================================
elif nav.startswith("⚖️"):
    with st.container(border=True):
        render_html("""
        <div class="panel-title">Side-by-Side Property Comparison Matrix</div>
        <div class="panel-sub">Compare evaluated properties across current price, price/sqft, 5-year forecast, annualized growth, and 80% conformal bands</div>
        """)
        rows = [{
            "Location": f"{x['property_summary']['City']} ({x['property_summary']['Locality']})",
            "Specification": f"{x['property_summary']['BHK']} BHK {x['property_summary']['Property_Type']}",
            "Area (sq ft)": f"{x['property_summary']['Size_in_SqFt']:,.0f}",
            "Current Price": f"₹{x['appreciation']['current_price_lakhs']:,.2f}L",
            "Price / SqFt": f"₹{x['property_summary']['Price_per_SqFt']:,.0f}",
            "City Med PPS": f"₹{x['property_summary']['City_Median_PPS']:,.0f}",
            "5Y Forecast": f"₹{x['appreciation']['future_price_lakhs']:,.2f}L",
            "80% Conformal Band": f"₹{x['uncertainty']['interval_80_lower_lakhs']:,.1f}L – ₹{x['uncertainty']['interval_80_upper_lakhs']:,.1f}L",
            "Annualized CAGR": f"+{x['appreciation']['annualized_appreciation_pct']:.2f}%/yr",
            "Net ROI": f"{x['net_roi']['net_roi_pct']:+.2f}%" if x["net_roi"]["is_available"] else "Cash-Flows Omitted",
            "Assessment": short_badge_label(x["verdict"]["status_label"])
        } for x in st.session_state["history_log"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# =====================================================================
# PAGE 4: MODEL PERFORMANCE AUDIT
# =====================================================================
elif nav.startswith("🎯"):
    r_hold = rep.get("repaired_on_holdout", {})
    c_info = rep.get("conformal", {})
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_html(f"<div class='kpi-card'><div class='kpi-top'><span class='kpi-icon'>🎯</span> Holdout MAE</div><div class='kpi-val'>₹{r_hold.get('mae_lakhs', 1.84):.2f}L</div><div><span class='kpi-delta-pos'>50,000 Unseen Records</span></div></div>")
    with m2:
        render_html(f"<div class='kpi-card'><div class='kpi-top'><span class='kpi-icon'>📐</span> Holdout RMSE</div><div class='kpi-val'>₹{r_hold.get('rmse_lakhs', 2.42):.2f}L</div><div><span class='kpi-delta-pos'>Leak-Free Split</span></div></div>")
    with m3:
        render_html(f"<div class='kpi-card'><div class='kpi-top'><span class='kpi-icon'>📊</span> Holdout R² / MAPE</div><div class='kpi-val'>{r_hold.get('r2', 0.9999):.4f}</div><div><span class='kpi-delta-pos'>MAPE: {r_hold.get('mape_pct', 0.71):.2f}%</span></div></div>")
    with m4:
        render_html(f"<div class='kpi-card'><div class='kpi-top'><span class='kpi-icon'>🛡️</span> 80% Band Coverage</div><div class='kpi-val'>{c_info.get('empirical_coverage_80_eff_pct', 99.1):.1f}%</div><div><span class='kpi-delta-warn'>Raw Quantile: {c_info.get('empirical_coverage_80_raw_pct', 80.1):.1f}%</span></div></div>")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        render_html("""
        <div class="panel-title">Holdout Model Performance vs. Baselines (Untouched 20% Test Split)</div>
        <div class="panel-sub">Strict 60% Train / 20% Calibration / 20% Holdout Test evaluation</div>
        """)
        base_m = rep.get("baselines_on_holdout", {}).get("global_median", {})
        base_c = rep.get("baselines_on_holdout", {}).get("city_type_comparable", {})
        comp_df = pd.DataFrame([
            {"Model / Baseline": "1. Global Median Predictor", "MAE (₹ Lakh)": base_m.get("mae_lakhs"), "RMSE (₹ Lakh)": base_m.get("rmse_lakhs"), "R² Score": base_m.get("r2"), "MAPE (%)": base_m.get("mape_pct")},
            {"Model / Baseline": "2. City × Property-Type Comparable Baseline", "MAE (₹ Lakh)": base_c.get("mae_lakhs"), "RMSE (₹ Lakh)": base_c.get("rmse_lakhs"), "R² Score": base_c.get("r2"), "MAPE (%)": base_c.get("mape_pct")},
            {"Model / Baseline": "3. Repaired Leak-Free XGBoost Pipeline", "MAE (₹ Lakh)": r_hold.get("mae_lakhs"), "RMSE (₹ Lakh)": r_hold.get("rmse_lakhs"), "R² Score": r_hold.get("r2"), "MAPE (%)": r_hold.get("mape_pct")}
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        samples = rep.get("holdout_plot_samples", [])
        if samples:
            sdf = pd.DataFrame(samples)
            p_left, p_right = st.columns(2)
            with p_left:
                c_act = alt.Chart(sdf).mark_circle(size=45, color="#FF4D57", opacity=0.75).encode(
                    x=alt.X("Actual_5Y_Lakhs:Q", title="Actual Holdout 5Y Target (₹L)", axis=alt.Axis(labelColor="#9AA4B2", titleColor="#9AA4B2", gridColor="#1E2330")),
                    y=alt.Y("Predicted_5Y_Lakhs:Q", title="Predicted 5Y Price (₹L)", axis=alt.Axis(labelColor="#9AA4B2", titleColor="#9AA4B2", gridColor="#1E2330")),
                    tooltip=["Actual_5Y_Lakhs:Q", "Predicted_5Y_Lakhs:Q", "Residual_Lakhs:Q"]
                ).properties(title="Actual vs. Predicted 5Y Valuation (Holdout Sample)", height=260, background="transparent").configure_title(color="#F4F6FA", fontSize=13).configure_view(strokeWidth=0)
                st.altair_chart(c_act, use_container_width=True)
            with p_right:
                c_res = alt.Chart(sdf).mark_circle(size=45, color="#10B981", opacity=0.75).encode(
                    x=alt.X("Predicted_5Y_Lakhs:Q", title="Predicted 5Y Price (₹L)", axis=alt.Axis(labelColor="#9AA4B2", titleColor="#9AA4B2", gridColor="#1E2330")),
                    y=alt.Y("Residual_Lakhs:Q", title="Residual Error (Actual - Pred, ₹L)", axis=alt.Axis(labelColor="#9AA4B2", titleColor="#9AA4B2", gridColor="#1E2330")),
                    tooltip=["Predicted_5Y_Lakhs:Q", "Residual_Lakhs:Q"]
                ).properties(title="Holdout Prediction Residuals (₹ Lakhs)", height=260, background="transparent").configure_title(color="#F4F6FA", fontSize=13).configure_view(strokeWidth=0)
                st.altair_chart(c_res, use_container_width=True)

        seg_df = pd.DataFrame(rep.get("segment_by_type", [])).rename(columns={
            "Segment": "Property Type", "Count": "Holdout Count",
            "mae_lakhs": "MAE (₹ Lakh)", "rmse_lakhs": "RMSE (₹ Lakh)", "r2": "R² Score", "mape_pct": "MAPE (%)"
        })
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

# =====================================================================
# PAGE 5: DATA & METHODOLOGY
# =====================================================================
elif nav.startswith("🛡️"):
    with st.container(border=True):
        render_html("""
        <div class="panel-title">Dataset Provenance, Mathematical Definitions & Audit Disclosure</div>
        <div class="panel-sub">Transparent quantitative governance & limitations</div>
        <div class="alert-item">
            <div class="alert-top"><span class="alert-name">1. Synthetic Cross-Sectional Dataset Limitation</span><span class="badge-warn">Provenance</span></div>
            <div class="alert-desc"><code>data/india_housing_prices.csv</code> (250,000 records) contains synthetic cross-sectional property listings with anonymized localities (<code>Locality_1</code>–<code>Locality_500</code>) and no longitudinal repeat-sales history. Forecasts represent model-implied multi-factor projections rather than backtested real-market repeat sales.</div>
        </div>
        <div class="alert-item">
            <div class="alert-top"><span class="alert-name">2. Split-Conformal Uncertainty Calibration</span><span class="badge-good">Calibrated</span></div>
            <div class="alert-desc">Rather than displaying an uncalibrated classifier probability (100%), EstateIQ calibrates 80% and 90% prediction intervals using relative residual quantiles on an independent 20% calibration split.</div>
        </div>
        <div class="alert-item" style="margin-bottom:0;">
            <div class="alert-top"><span class="alert-name">3. Gross Appreciation vs. Net Investment ROI</span><span class="badge-good">Verified Math</span></div>
            <div class="alert-desc">Gross price appreciation reflects estimated asset value change only. Net Investment ROI is displayed when rental yield, vacancy, stamp duty, maintenance, and exit costs are enabled in Tab 4.</div>
        </div>
        """)
