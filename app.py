# app.py
# Main application file for the Veridex ESG Intelligence Platform
# Handles login, navigation, and all dashboard pages

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from esg_scorer import (
    score_label,
    compute_custom_esg,
    greenwashing_flag,
    sector_benchmark,
    financial_health_score,
    get_recommendations,
)
from ml_engine import train_random_forest, predict_esg, forecast_metric
from pdf_generator import generate_report

# Page configuration - this must be the very first Streamlit command
st.set_page_config(
    page_title="Veridex ESG Platform",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0f1e;
        color: #e8eaf0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1424;
        border-right: 1px solid #1e2a3a;
    }

    /* Cards */
    .veridex-card {
        background-color: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Metric cards */
    .metric-card {
        background-color: #111827;
        border: 1px solid #00a693;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }

    .metric-label {
        font-size: 13px;
        color: #8892a4;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
    }

    .metric-sub {
        font-size: 12px;
        color: #00a693;
        margin-top: 4px;
    }

    /* Section headings */
    .section-heading {
        font-size: 18px;
        font-weight: 600;
        color: #00a693;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e2a3a;
    }

    /* ESG rating badges */
    .badge-excellent {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    .badge-good {
        background-color: #1e3a5f;
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    .badge-fair {
        background-color: #451a03;
        color: #fb923c;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    .badge-poor {
        background-color: #450a0a;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    /* Login box */
    .login-container {
        max-width: 420px;
        margin: 80px auto;
        background-color: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 16px;
        padding: 40px;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Buttons */
    .stButton > button {
        background-color: #00a693;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #00c4ae;
        color: white;
    }

    /* Plotly chart background */
    .js-plotly-plot {
        border-radius: 10px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
    }

    /* Selectbox and slider labels */
    label {
        color: #8892a4 !important;
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)
# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """
    Load and clean the CSV file.
    @st.cache_data means Streamlit only loads this once,
    not every time the user clicks something. Much faster.
    """
    df = pd.read_csv("data/ESG_Final_Validated.csv")

    # Convert Year to integer
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Convert all financial and ESG columns to numeric
    numeric_cols = [
        "Revenue", "Net_Income", "Operating_Income", "Total_Assets",
        "Total_Debt", "Operating_Cash_Flow", "Capital_Expenditure",
        "Market_Cap", "Stock_Price_Year_End", "Total_Emissions",
        "Scope_1_Emissions", "Scope_2_Emissions", "Scope_3_Emissions",
        "Renewable_Energy_Percentage", "Employee_Count",
        "Overall_ESG_Score", "Environmental_Score", "Social_Score",
        "Governance_Score", "Emissions_Intensity", "Gender_Diversity_Percentage",
        "Board_Diversity_Percentage", "Independent_Board_Percentage",
        "Audit_Committee_Independence_Percentage", "ESG_Linked_Compensation_Yes_No",
        "Workplace_Injury_Rate", "Employee_Turnover_Percentage",
        "Training_Hours_Per_Employee", "Sustainability_Investment",
        "Regulatory_Fines", "Carbon_Reduction_Target_Percentage",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fmt_billions(value):
    """
    Format large numbers into readable format.
    Example: 289000000000 becomes $289.00B
    This is used throughout the dashboard for financial figures.
    """
    try:
        v = float(value)
        if np.isnan(v):
            return "N/A"
        if abs(v) >= 1e12:
            return f"${v/1e12:.2f}T"
        elif abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        elif abs(v) >= 1e6:
            return f"${v/1e6:.2f}M"
        else:
            return f"${v:,.0f}"
    except Exception:
        return "N/A"


def esg_badge(score):
    """Return a colour coded HTML badge based on ESG score."""
    try:
        s = float(score)
        if s >= 85:
            return '<span class="badge-excellent">Excellent</span>'
        elif s >= 70:
            return '<span class="badge-good">Good</span>'
        elif s >= 55:
            return '<span class="badge-fair">Fair</span>'
        else:
            return '<span class="badge-poor">Poor</span>'
    except Exception:
        return '<span class="badge-fair">N/A</span>'


# ── Login system ──────────────────────────────────────────────────────────────

# User accounts - in a real production system these would be in a database
# For this prototype, we define them here directly
USERS = {
    "admin": {"password": "veridex2024", "role": "Admin"},
    "analyst": {"password": "analyst123", "role": "Analyst"},
    "demo": {"password": "demo2024", "role": "Viewer"},
}


def login_page():
    """
    Render the login page.
    Uses Streamlit session state to remember if the user is logged in.
    Session state persists across reruns of the app within the same session.
    """
    st.markdown("""
        <div style='text-align: center; padding: 40px 0 10px 0;'>
            <h1 style='color: #00a693; font-size: 42px; font-weight: 800;'>Veridex</h1>
            <p style='color: #8892a4; font-size: 16px;'>ESG Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div class='veridex-card'>", unsafe_allow_html=True)
        st.markdown("### Sign in to your account")
        st.markdown("<br>", unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Sign In"):
            if username in USERS and USERS[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = USERS[username]["role"]
                st.rerun()
            else:
                st.error("Incorrect username or password. Please try again.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <p style='color: #8892a4; font-size: 12px; text-align: center;'>
            Demo accounts: admin / veridex2024 &nbsp;|&nbsp; analyst / analyst123
            </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def logout():
    """Clear session state and return to login page."""
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""
    st.rerun()


# ── Session state initialisation ─────────────────────────────────────────────
# This runs every time the app loads
# If the user is not logged in, show login page and stop

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()
    # ── Sidebar navigation ────────────────────────────────────────────────────────
df = load_data()

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px 0 20px 0;'>
            <h2 style='color: #00a693; font-weight: 800; font-size: 24px;'>Veridex</h2>
            <p style='color: #8892a4; font-size: 12px;'>ESG Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style='background-color: #0d1f2d; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;'>
            <p style='color: #8892a4; font-size: 11px; margin: 0;'>Signed in as</p>
            <p style='color: #ffffff; font-size: 14px; font-weight: 600; margin: 0;'>
                {st.session_state['username']}
            </p>
            <p style='color: #00a693; font-size: 11px; margin: 0;'>
                {st.session_state['role']}
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Navigation")

    page = st.radio(
        "Go to",
        [
            "Overview",
            "Company Dashboard",
            "Compare Companies",
            "ESG Analysis",
            "ML Predictor",
            "Forecasting",
            "Scenario Simulator",
            "PDF Report",
            "Data Quality",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Global filters applied across all pages
    st.markdown("#### Filters")

    companies = sorted(df["Company"].dropna().unique().tolist())
    company = st.selectbox("Select Company", companies)

    years = sorted(df["Year"].dropna().unique().astype(int).tolist())
    if years:
        yr_min, yr_max = st.select_slider(
            "Year Range",
            options=years,
            value=(years[0], years[-1])
        )
    else:
        yr_min, yr_max = None, None

    st.divider()

    # ESG weight sliders - this is your transparency feature
    st.markdown("#### Custom ESG Weights")
    st.caption("Adjust how Environmental, Social and Governance scores are weighted")

    wE = st.slider("Environmental", 0.0, 1.0, 0.4, 0.05)
    wS = st.slider("Social", 0.0, 1.0, 0.3, 0.05)
    wG = st.slider("Governance", 0.0, 1.0, 0.3, 0.05)

    st.divider()

    if st.button("Sign Out"):
        logout()


# Filtered dataframe for selected company and year range
df_company = df[
    (df["Company"] == company) &
    (df["Year"].between(yr_min, yr_max))
].copy().sort_values("Year")

# Latest row for the selected company
latest = df_company.dropna(subset=["Year"]).iloc[-1] if not df_company.empty else None
# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("<h1 style='color: #ffffff;'>Platform Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892a4;'>A summary of all companies tracked on the Veridex platform.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top stats row ─────────────────────────────────────────────────────────
    total_companies = df["Company"].nunique()
    avg_esg = df.groupby("Company")["Overall_ESG_Score"].mean().mean()
    top_company = df.groupby("Company")["Overall_ESG_Score"].mean().idxmax()
    total_sectors = df["Sector"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Companies Tracked</div>
                <div class='metric-value'>{total_companies}</div>
                <div class='metric-sub'>Across all sectors</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Platform Avg ESG Score</div>
                <div class='metric-value'>{avg_esg:.1f}</div>
                <div class='metric-sub'>All companies, all years</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Top ESG Performer</div>
                <div class='metric-value' style='font-size: 18px;'>{top_company}</div>
                <div class='metric-sub'>Highest average score</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Sectors Covered</div>
                <div class='metric-value'>{total_sectors}</div>
                <div class='metric-sub'>Industry diversity</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ESG scores by sector ──────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='section-heading'>Average ESG Score by Sector</div>", unsafe_allow_html=True)
        sector_avg = df.groupby("Sector")["Overall_ESG_Score"].mean().reset_index()
        sector_avg.columns = ["Sector", "Average ESG Score"]
        sector_avg = sector_avg.sort_values("Average ESG Score", ascending=True)

        fig = px.bar(
            sector_avg,
            x="Average ESG Score",
            y="Sector",
            orientation="h",
            color="Average ESG Score",
            color_continuous_scale=[[0, "#f87171"], [0.5, "#fb923c"], [1, "#34d399"]],
        )
        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-heading'>ESG Score Trend (Platform Average)</div>", unsafe_allow_html=True)
        trend = df.groupby("Year")["Overall_ESG_Score"].mean().reset_index()
        trend.columns = ["Year", "Average ESG Score"]
        trend["Year"] = trend["Year"].astype(str)

        fig2 = px.line(
            trend,
            x="Year",
            y="Average ESG Score",
            markers=True,
        )
        fig2.update_traces(line_color="#00a693", marker_color="#00a693")
        fig2.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── ESG leaderboard ───────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>ESG Leaderboard</div>", unsafe_allow_html=True)

    leaderboard = (
        df.groupby(["Company", "Sector"])[
            ["Overall_ESG_Score", "Environmental_Score", "Social_Score", "Governance_Score"]
        ]
        .mean()
        .round(1)
        .reset_index()
        .sort_values("Overall_ESG_Score", ascending=False)
    )

    leaderboard["Rating"] = leaderboard["Overall_ESG_Score"].apply(score_label)

    st.dataframe(
        leaderboard.rename(columns={
            "Overall_ESG_Score": "Overall ESG",
            "Environmental_Score": "Environmental",
            "Social_Score": "Social",
            "Governance_Score": "Governance",
        }),
        use_container_width=True,
        hide_index=True,
    )
    # ── Page: Company Dashboard ───────────────────────────────────────────────────
elif page == "Company Dashboard":
    st.markdown("<h1 style='color: #ffffff;'>Company Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #8892a4;'>Detailed ESG and financial analysis for {company}</p>", unsafe_allow_html=True)

    if df_company.empty or latest is None:
        st.warning("No data available for this selection.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key metrics row ───────────────────────────────────────────────────────
    overall_esg = latest.get("Overall_ESG_Score", np.nan)
    revenue = latest.get("Revenue", np.nan)
    net_income = latest.get("Net_Income", np.nan)
    market_cap = latest.get("Market_Cap", np.nan)
    fin_health = financial_health_score(latest)
    rating = score_label(float(overall_esg)) if not pd.isna(overall_esg) else "N/A"

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Overall ESG Score</div>
                <div class='metric-value'>{overall_esg:.1f}</div>
                <div class='metric-sub'>{rating}</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Revenue</div>
                <div class='metric-value' style='font-size: 20px;'>{fmt_billions(revenue)}</div>
                <div class='metric-sub'>Latest year</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Net Income</div>
                <div class='metric-value' style='font-size: 20px;'>{fmt_billions(net_income)}</div>
                <div class='metric-sub'>Latest year</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Market Cap</div>
                <div class='metric-value' style='font-size: 20px;'>{fmt_billions(market_cap)}</div>
                <div class='metric-sub'>Latest year</div>
            </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Financial Health</div>
                <div class='metric-value' style='font-size: 18px;'>{fin_health}</div>
                <div class='metric-sub'>Based on profit margin</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ESG trend and breakdown ───────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='section-heading'>ESG Score Trend</div>", unsafe_allow_html=True)

        esg_cols = [c for c in ["Overall_ESG_Score", "Environmental_Score",
                                 "Social_Score", "Governance_Score"] if c in df_company.columns]
        esg_trend = df_company[["Year"] + esg_cols].copy()
        esg_trend["Year"] = esg_trend["Year"].astype(str)

        fig = px.line(
            esg_trend,
            x="Year",
            y=esg_cols,
            markers=True,
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_map={
                "Overall_ESG_Score": "#00a693",
                "Environmental_Score": "#34d399",
                "Social_Score": "#60a5fa",
                "Governance_Score": "#a78bfa",
            }
        )
        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-heading'>ESG Component Breakdown (Latest Year)</div>", unsafe_allow_html=True)

        env = pd.to_numeric(latest.get("Environmental_Score", 0), errors="coerce") or 0
        soc = pd.to_numeric(latest.get("Social_Score", 0), errors="coerce") or 0
        gov = pd.to_numeric(latest.get("Governance_Score", 0), errors="coerce") or 0

        fig2 = go.Figure()
        categories = ["Environmental", "Social", "Governance"]
        values = [env, soc, gov]
        colors = ["#34d399", "#60a5fa", "#a78bfa"]

        for i, (cat, val, col) in enumerate(zip(categories, values, colors)):
            fig2.add_trace(go.Bar(
                x=[val],
                y=[cat],
                orientation="h",
                marker_color=col,
                name=cat,
                text=f"{val:.1f}",
                textposition="inside",
            ))

        fig2.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(range=[0, 100], gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            barmode="overlay",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Financial trends ──────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Financial Performance</div>", unsafe_allow_html=True)

    left2, right2 = st.columns(2)

    with left2:
        fin_cols = [c for c in ["Revenue", "Net_Income", "Operating_Cash_Flow"]
                    if c in df_company.columns]
        fin_data = df_company[["Year"] + fin_cols].copy()
        fin_data["Year"] = fin_data["Year"].astype(str)

        # Convert to billions for readability
        for col in fin_cols:
            fin_data[col] = fin_data[col] / 1e9

        fig3 = px.line(
            fin_data,
            x="Year",
            y=fin_cols,
            markers=True,
            labels={"value": "Value ($B)", "variable": "Metric"},
        )
        fig3.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with right2:
        em_cols = [c for c in ["Total_Emissions", "Scope_1_Emissions",
                                "Scope_2_Emissions", "Scope_3_Emissions"]
                   if c in df_company.columns]
        em_data = df_company[["Year"] + em_cols].copy()
        em_data["Year"] = em_data["Year"].astype(str)

        fig4 = px.line(
            em_data,
            x="Year",
            y=em_cols,
            markers=True,
            labels={"value": "tCO2e", "variable": "Metric"},
        )
        fig4.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Greenwashing check ────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Greenwashing Check</div>", unsafe_allow_html=True)
    gw_flag = greenwashing_flag(latest)

    if gw_flag == "Potential Greenwashing":
        st.error(f"{company}: {gw_flag} — High ESG score but high emissions intensity detected.")
    elif gw_flag == "Watch":
        st.warning(f"{company}: {gw_flag} — Minor inconsistency between ESG score and emissions.")
    else:
        st.success(f"{company}: {gw_flag} — ESG score is consistent with emissions data.")

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Improvement Recommendations</div>", unsafe_allow_html=True)
    tips = get_recommendations(latest)
    for tip in tips:
        st.markdown(f"- {tip}")

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("View raw data for this company"):
        st.dataframe(df_company, use_container_width=True, hide_index=True)
        # ── Page: Compare Companies ───────────────────────────────────────────────────
elif page == "Compare Companies":
    st.markdown("<h1 style='color: #ffffff;'>Compare Companies</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892a4;'>Compare ESG and financial performance across multiple companies side by side.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    selected = st.multiselect(
        "Select companies to compare",
        companies,
        default=companies[:4]
    )

    if not selected:
        st.warning("Please select at least one company to compare.")
        st.stop()

    df_comp = df[
        (df["Company"].isin(selected)) &
        (df["Year"].between(yr_min, yr_max))
    ].copy()

    # ── ESG score over time ───────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Overall ESG Score Over Time</div>", unsafe_allow_html=True)

    esg_time = df_comp[["Company", "Year", "Overall_ESG_Score"]].copy()
    esg_time["Year"] = esg_time["Year"].astype(str)

    fig = px.line(
        esg_time,
        x="Year",
        y="Overall_ESG_Score",
        color="Company",
        markers=True,
        labels={"Overall_ESG_Score": "ESG Score"},
    )
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font_color="#e8eaf0",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#1e2a3a"),
        yaxis=dict(gridcolor="#1e2a3a"),
        legend=dict(bgcolor="#111827"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── ESG component comparison ──────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='section-heading'>Environmental Score Comparison</div>", unsafe_allow_html=True)

        latest_year = int(df_comp["Year"].max())
        snap = df_comp[df_comp["Year"] == latest_year].copy()

        fig2 = px.bar(
            snap.sort_values("Environmental_Score", ascending=True),
            x="Environmental_Score",
            y="Company",
            orientation="h",
            color="Environmental_Score",
            color_continuous_scale=[[0, "#f87171"], [0.5, "#fb923c"], [1, "#34d399"]],
            labels={"Environmental_Score": "Environmental Score"},
        )
        fig2.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a", range=[0, 100]),
            yaxis=dict(gridcolor="#1e2a3a"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("<div class='section-heading'>Governance Score Comparison</div>", unsafe_allow_html=True)

        fig3 = px.bar(
            snap.sort_values("Governance_Score", ascending=True),
            x="Governance_Score",
            y="Company",
            orientation="h",
            color="Governance_Score",
            color_continuous_scale=[[0, "#f87171"], [0.5, "#fb923c"], [1, "#34d399"]],
            labels={"Governance_Score": "Governance Score"},
        )
        fig3.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a", range=[0, 100]),
            yaxis=dict(gridcolor="#1e2a3a"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Revenue comparison ────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Revenue Comparison (Latest Year)</div>", unsafe_allow_html=True)

    snap_rev = snap.copy()
    snap_rev["Revenue_B"] = snap_rev["Revenue"] / 1e9

    fig4 = px.bar(
        snap_rev.sort_values("Revenue_B", ascending=False),
        x="Company",
        y="Revenue_B",
        color="Company",
        labels={"Revenue_B": "Revenue ($B)"},
    )
    fig4.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font_color="#e8eaf0",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#1e2a3a"),
        yaxis=dict(gridcolor="#1e2a3a"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Full Comparison Table (Latest Year)</div>", unsafe_allow_html=True)

    table_cols = [c for c in [
        "Company", "Sector", "Overall_ESG_Score",
        "Environmental_Score", "Social_Score", "Governance_Score",
        "Revenue", "Net_Income", "Market_Cap",
        "Renewable_Energy_Percentage", "Total_Emissions",
    ] if c in snap.columns]

    display = snap[table_cols].copy()

    for col in ["Revenue", "Net_Income", "Market_Cap"]:
        if col in display.columns:
            display[col] = display[col].apply(fmt_billions)

    for col in ["Overall_ESG_Score", "Environmental_Score",
                "Social_Score", "Governance_Score"]:
        if col in display.columns:
            display[col] = display[col].round(1)

    if "Renewable_Energy_Percentage" in display.columns:
        display["Renewable_Energy_Percentage"] = display["Renewable_Energy_Percentage"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )

    st.dataframe(
        display.rename(columns={
            "Overall_ESG_Score": "Overall ESG",
            "Environmental_Score": "Environmental",
            "Social_Score": "Social",
            "Governance_Score": "Governance",
            "Renewable_Energy_Percentage": "Renewable %",
            "Total_Emissions": "Emissions (tCO2e)",
        }),
        use_container_width=True,
        hide_index=True,
    )
    # ── Page: ESG Analysis ────────────────────────────────────────────────────────
elif page == "ESG Analysis":
    st.markdown("<h1 style='color: #ffffff;'>ESG Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #8892a4;'>Deep dive into ESG performance for {company}</p>", unsafe_allow_html=True)

    if df_company.empty or latest is None:
        st.warning("No data available for this selection.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Custom weighted ESG score ─────────────────────────────────────────────
    custom_score = compute_custom_esg(latest, wE, wS, wG)
    official_score = pd.to_numeric(latest.get("Overall_ESG_Score", np.nan), errors="coerce")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Official ESG Score</div>
                <div class='metric-value'>{official_score:.1f}</div>
                <div class='metric-sub'>{score_label(official_score)}</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Custom Weighted Score</div>
                <div class='metric-value'>{custom_score:.1f}</div>
                <div class='metric-sub'>E:{wE:.0%} S:{wS:.0%} G:{wG:.0%}</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        env = pd.to_numeric(latest.get("Environmental_Score", np.nan), errors="coerce")
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Environmental Score</div>
                <div class='metric-value' style='color: #34d399;'>{env:.1f}</div>
                <div class='metric-sub'>Weight: {wE:.0%}</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        soc = pd.to_numeric(latest.get("Social_Score", np.nan), errors="coerce")
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Social Score</div>
                <div class='metric-value' style='color: #60a5fa;'>{soc:.1f}</div>
                <div class='metric-sub'>Weight: {wS:.0%}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='section-heading'>ESG Radar Chart</div>", unsafe_allow_html=True)
        st.caption("Visual overview of all ESG dimensions for the latest year")

        radar_metrics = [
            "Environmental_Score", "Social_Score", "Governance_Score",
            "Renewable_Energy_Percentage", "Gender_Diversity_Percentage",
            "Independent_Board_Percentage",
        ]
        radar_labels = [
            "Environmental", "Social", "Governance",
            "Renewable Energy", "Gender Diversity", "Board Independence",
        ]

        radar_vals = []
        for col in radar_metrics:
            val = pd.to_numeric(latest.get(col, 0), errors="coerce")
            radar_vals.append(float(val) if not pd.isna(val) else 0)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(0, 166, 147, 0.2)",
            line=dict(color="#00a693", width=2),
            name=company,
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="#111827",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor="#1e2a3a",
                    color="#8892a4",
                ),
                angularaxis=dict(color="#8892a4"),
            ),
            paper_bgcolor="#111827",
            font_color="#e8eaf0",
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-heading'>ESG vs Emissions Intensity</div>", unsafe_allow_html=True)
        st.caption("Checks if ESG score is consistent with actual emissions performance")

        latest_year = int(df["Year"].max())
        scatter_data = df[df["Year"] == latest_year].copy()

        fig2 = px.scatter(
            scatter_data,
            x="Emissions_Intensity",
            y="Overall_ESG_Score",
            color="Sector",
            hover_name="Company",
            size="Market_Cap",
            size_max=40,
            labels={
                "Emissions_Intensity": "Emissions Intensity",
                "Overall_ESG_Score": "Overall ESG Score",
            },
        )
        fig2.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Key ESG metrics table ─────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Key ESG Metrics Breakdown</div>", unsafe_allow_html=True)

    metrics_display = {
        "Renewable Energy (%)": latest.get("Renewable_Energy_Percentage"),
        "Total Emissions (tCO2e)": latest.get("Total_Emissions"),
        "Emissions Intensity": latest.get("Emissions_Intensity"),
        "Gender Diversity (%)": latest.get("Gender_Diversity_Percentage"),
        "Board Diversity (%)": latest.get("Board_Diversity_Percentage"),
        "Independent Board (%)": latest.get("Independent_Board_Percentage"),
        "Workplace Injury Rate": latest.get("Workplace_Injury_Rate"),
        "Employee Turnover (%)": latest.get("Employee_Turnover_Percentage"),
        "Training Hours Per Employee": latest.get("Training_Hours_Per_Employee"),
        "ESG Linked Compensation": latest.get("ESG_Linked_Compensation_Yes_No"),
        "Sustainability Investment": latest.get("Sustainability_Investment"),
        "Regulatory Fines": latest.get("Regulatory_Fines"),
        "Carbon Reduction Target (%)": latest.get("Carbon_Reduction_Target_Percentage"),
    }

    metrics_df = pd.DataFrame(
        metrics_display.items(),
        columns=["Metric", "Value"]
    )
    metrics_df["Value"] = metrics_df["Value"].apply(
        lambda x: round(float(x), 2) if pd.notna(x) and str(x).replace(".", "").replace("-", "").isdigit() else (
            "Yes" if str(x) == "1.0" or str(x) == "1" else (
            "No" if str(x) == "0.0" or str(x) == "0" else (
            "N/A" if pd.isna(x) else x)))
    )

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Sector benchmark ──────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Sector Benchmark</div>", unsafe_allow_html=True)
    st.caption(f"How {company} compares to the average in the {latest.get('Sector', 'N/A')} sector")

    bench = sector_benchmark(df)
    company_sector = str(latest.get("Sector", ""))
    sector_row = bench[bench["Sector"] == company_sector]

    if not sector_row.empty:
        sector_avg_esg = sector_row["Overall_ESG_Score"].values[0]
        diff = float(official_score) - float(sector_avg_esg)
        direction = "above" if diff >= 0 else "below"

        st.markdown(f"""
            <div class='veridex-card'>
                <p style='color: #8892a4;'>Sector average ESG score ({company_sector}):
                    <strong style='color: #ffffff;'>{sector_avg_esg:.1f}</strong>
                </p>
                <p style='color: #8892a4;'>{company} is
                    <strong style='color: {"#34d399" if diff >= 0 else "#f87171"};'>
                        {abs(diff):.1f} points {direction}
                    </strong>
                    the sector average.
                </p>
            </div>
        """, unsafe_allow_html=True)

        fig3 = px.bar(
            bench.sort_values("Overall_ESG_Score", ascending=True),
            x="Overall_ESG_Score",
            y="Sector",
            orientation="h",
            color="Overall_ESG_Score",
            color_continuous_scale=[[0, "#f87171"], [0.5, "#fb923c"], [1, "#34d399"]],
            labels={"Overall_ESG_Score": "Average ESG Score"},
        )
        fig3.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
        )
        st.plotly_chart(fig3, use_container_width=True)
        # ── Page: ML Predictor ────────────────────────────────────────────────────────
elif page == "ML Predictor":
    st.markdown("<h1 style='color: #ffffff;'>ESG Score Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892a4;'>Machine learning model that predicts ESG scores based on company characteristics. Built using Random Forest regression trained on the Veridex dataset.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Train the model ───────────────────────────────────────────────────────
    with st.spinner("Training Random Forest model on dataset..."):
        model, metrics, importance_df, X_test, y_test = train_random_forest(df)

    if model is None:
        st.error("Not enough data to train the model. Need at least 20 rows.")
        st.stop()

    # ── Model performance metrics ─────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Model Performance</div>", unsafe_allow_html=True)
    st.caption("Evaluated on 20% held-out test data the model has never seen before")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>R2 Score</div>
                <div class='metric-value'>{metrics['R2']}</div>
                <div class='metric-sub'>1.0 = perfect prediction</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Mean Absolute Error</div>
                <div class='metric-value'>{metrics['MAE']}</div>
                <div class='metric-sub'>Avg points off per prediction</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Training Samples</div>
                <div class='metric-value'>{metrics['Training samples']}</div>
                <div class='metric-sub'>Rows used to train</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Test Samples</div>
                <div class='metric-value'>{metrics['Test samples']}</div>
                <div class='metric-sub'>Rows used to evaluate</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance ────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='section-heading'>Feature Importance</div>", unsafe_allow_html=True)
        st.caption("Which factors the model relies on most to predict ESG scores")

        fig = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#1e3a5f"], [1, "#00a693"]],
        )
        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a", categoryorder="total ascending"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-heading'>Actual vs Predicted ESG Scores</div>", unsafe_allow_html=True)
        st.caption("How closely the model predictions match real ESG scores on test data")

        y_pred = model.predict(X_test)
        scatter_df = pd.DataFrame({
            "Actual ESG Score": y_test.values,
            "Predicted ESG Score": y_pred.round(2),
        })

        fig2 = px.scatter(
            scatter_df,
            x="Actual ESG Score",
            y="Predicted ESG Score",
        )

        # Perfect prediction line
        min_val = float(scatter_df["Actual ESG Score"].min())
        max_val = float(scatter_df["Actual ESG Score"].max())
        fig2.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#f87171", dash="dash", width=1),
            name="Perfect Prediction",
        ))

        fig2.update_traces(
            marker=dict(color="#00a693", size=8),
            selector=dict(mode="markers"),
        )
        fig2.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Predict for a specific company ────────────────────────────────────────
    st.markdown("<div class='section-heading'>Predict ESG Score for a Company</div>", unsafe_allow_html=True)
    st.caption("Select a company and the model will predict its ESG score from its characteristics")

    pred_company = st.selectbox("Select company to predict", companies, key="pred_company")
    pred_data = df[df["Company"] == pred_company].copy()

    if not pred_data.empty:
        pred_latest = pred_data.sort_values("Year").iloc[-1]
        feature_cols = importance_df["Feature"].tolist()

        input_dict = {}
        for col in feature_cols:
            val = pd.to_numeric(pred_latest.get(col, 0), errors="coerce")
            input_dict[col] = float(val) if not pd.isna(val) else 0.0

        predicted_score = predict_esg(model, input_dict, feature_cols)
        actual_score = pd.to_numeric(pred_latest.get("Overall_ESG_Score", np.nan), errors="coerce")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Predicted ESG Score</div>
                    <div class='metric-value' style='color: #00a693;'>{predicted_score}</div>
                    <div class='metric-sub'>Random Forest prediction</div>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Actual ESG Score</div>
                    <div class='metric-value'>{actual_score:.1f}</div>
                    <div class='metric-sub'>From dataset</div>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            diff = abs(predicted_score - float(actual_score))
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Prediction Difference</div>
                    <div class='metric-value' style='color: {"#34d399" if diff < 5 else "#fb923c"};'>{diff:.2f}</div>
                    <div class='metric-sub'>Points off actual score</div>
                </div>
            """, unsafe_allow_html=True)

    # ── Manual prediction tool ────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Manual ESG Predictor</div>", unsafe_allow_html=True)
    st.caption("Manually enter company characteristics to predict an ESG score. Useful for hypothetical companies or future scenarios.")

    with st.expander("Open manual predictor"):
        col1, col2, col3 = st.columns(3)

        with col1:
            m_renewable = st.slider("Renewable Energy %", 0.0, 100.0, 50.0)
            m_gender = st.slider("Gender Diversity %", 0.0, 100.0, 40.0)
            m_board_ind = st.slider("Independent Board %", 0.0, 100.0, 60.0)
            m_emissions = st.number_input("Total Emissions (tCO2e)", value=50000000)

        with col2:
            m_injury = st.slider("Workplace Injury Rate", 0.0, 10.0, 1.0)
            m_turnover = st.slider("Employee Turnover %", 0.0, 50.0, 15.0)
            m_training = st.slider("Training Hours Per Employee", 0.0, 100.0, 30.0)
            m_esg_comp = st.selectbox("ESG Linked Compensation", [1, 0],
                                       format_func=lambda x: "Yes" if x == 1 else "No")

        with col3:
            m_revenue = st.number_input("Revenue ($)", value=50000000000)
            m_net_income = st.number_input("Net Income ($)", value=5000000000)
            m_sustainability = st.number_input("Sustainability Investment ($)", value=1000000000)
            m_audit = st.slider("Audit Committee Independence %", 0.0, 100.0, 70.0)

        if st.button("Predict ESG Score"):
            manual_input = {
                "Renewable_Energy_Percentage": m_renewable,
                "Gender_Diversity_Percentage": m_gender,
                "Independent_Board_Percentage": m_board_ind,
                "Total_Emissions": m_emissions,
                "Workplace_Injury_Rate": m_injury,
                "Employee_Turnover_Percentage": m_turnover,
                "Training_Hours_Per_Employee": m_training,
                "ESG_Linked_Compensation_Yes_No": m_esg_comp,
                "Revenue": m_revenue,
                "Net_Income": m_net_income,
                "Sustainability_Investment": m_sustainability,
                "Audit_Committee_Independence_Percentage": m_audit,
                "Emissions_Intensity": 0,
                "Board_Diversity_Percentage": m_gender,
                "Sector_Encoded": 0,
            }

            manual_pred = predict_esg(model, manual_input, feature_cols)

            st.markdown(f"""
                <div class='veridex-card' style='text-align: center; margin-top: 16px;'>
                    <p style='color: #8892a4; font-size: 14px;'>Predicted ESG Score</p>
                    <p style='color: #00a693; font-size: 48px; font-weight: 800;'>{manual_pred}</p>
                    <p style='color: #8892a4; font-size: 13px;'>{score_label(manual_pred)}</p>
                </div>
            """, unsafe_allow_html=True)
            # ── Page: Forecasting ─────────────────────────────────────────────────────────
elif page == "Forecasting":
    st.markdown("<h1 style='color: #ffffff;'>Forecasting</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #8892a4;'>Linear regression forecasts for {company} based on historical trends.</p>", unsafe_allow_html=True)

    if df_company.empty or latest is None:
        st.warning("No data available for this selection.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast settings
    steps = st.slider("How many years to forecast", 1, 5, 3)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ESG Score Forecast ────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>ESG Score Forecast</div>", unsafe_allow_html=True)

    esg_forecast = forecast_metric(df_company, "Overall_ESG_Score", steps)

    if not esg_forecast.empty:
        # Combine historical and forecast data for the chart
        hist = df_company[["Year", "Overall_ESG_Score"]].dropna().copy()
        hist["Year"] = hist["Year"].astype(str)
        hist["Type"] = "Historical"
        hist = hist.rename(columns={"Overall_ESG_Score": "Value"})

        fut = esg_forecast[["Year", "Predicted"]].copy()
        fut["Year"] = fut["Year"].astype(str)
        fut["Type"] = "Forecast"
        fut = fut.rename(columns={"Predicted": "Value"})

        combined = pd.concat([hist, fut], ignore_index=True)

        fig = px.line(
            combined,
            x="Year",
            y="Value",
            color="Type",
            markers=True,
            labels={"Value": "ESG Score"},
            color_discrete_map={
                "Historical": "#00a693",
                "Forecast": "#f59e0b",
            },
        )
        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show forecast values as metric cards
        st.markdown("<div class='section-heading'>Forecasted ESG Scores</div>", unsafe_allow_html=True)
        cols = st.columns(len(esg_forecast))
        for i, (_, row) in enumerate(esg_forecast.iterrows()):
            with cols[i]:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Year {int(row['Year'])}</div>
                        <div class='metric-value'>{row['Predicted']:.1f}</div>
                        <div class='metric-sub'>{score_label(row['Predicted'])}</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Not enough data points to forecast ESG score. Need at least 3 years.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue Forecast ──────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Revenue Forecast</div>", unsafe_allow_html=True)

    rev_forecast = forecast_metric(df_company, "Revenue", steps)

    if not rev_forecast.empty:
        hist_rev = df_company[["Year", "Revenue"]].dropna().copy()
        hist_rev["Year"] = hist_rev["Year"].astype(str)
        hist_rev["Type"] = "Historical"
        hist_rev["Revenue"] = hist_rev["Revenue"] / 1e9
        hist_rev = hist_rev.rename(columns={"Revenue": "Value"})

        fut_rev = rev_forecast[["Year", "Predicted"]].copy()
        fut_rev["Year"] = fut_rev["Year"].astype(str)
        fut_rev["Type"] = "Forecast"
        fut_rev["Predicted"] = fut_rev["Predicted"] / 1e9
        fut_rev = fut_rev.rename(columns={"Predicted": "Value"})

        combined_rev = pd.concat([hist_rev, fut_rev], ignore_index=True)

        fig2 = px.line(
            combined_rev,
            x="Year",
            y="Value",
            color="Type",
            markers=True,
            labels={"Value": "Revenue ($B)"},
            color_discrete_map={
                "Historical": "#60a5fa",
                "Forecast": "#f59e0b",
            },
        )
        fig2.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Not enough data points to forecast revenue.")

    # ── Emissions Forecast ────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Emissions Forecast</div>", unsafe_allow_html=True)

    em_forecast = forecast_metric(df_company, "Total_Emissions", steps)

    if not em_forecast.empty:
        hist_em = df_company[["Year", "Total_Emissions"]].dropna().copy()
        hist_em["Year"] = hist_em["Year"].astype(str)
        hist_em["Type"] = "Historical"
        hist_em = hist_em.rename(columns={"Total_Emissions": "Value"})

        fut_em = em_forecast[["Year", "Predicted"]].copy()
        fut_em["Year"] = fut_em["Year"].astype(str)
        fut_em["Type"] = "Forecast"
        fut_em = fut_em.rename(columns={"Predicted": "Value"})

        combined_em = pd.concat([hist_em, fut_em], ignore_index=True)

        fig3 = px.line(
            combined_em,
            x="Year",
            y="Value",
            color="Type",
            markers=True,
            labels={"Value": "Total Emissions (tCO2e)"},
            color_discrete_map={
                "Historical": "#f87171",
                "Forecast": "#f59e0b",
            },
        )
        fig3.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Not enough data points to forecast emissions.")

    # ── Methodology note ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class='veridex-card'>
            <p style='color: #8892a4; font-size: 13px;'>
                <strong style='color: #ffffff;'>Forecasting Methodology:</strong>
                Forecasts are generated using Linear Regression trained on historical data for the selected company.
                The model identifies the underlying trend in each metric and extrapolates it forward.
                With 5 years of historical data, short term forecasts of 1 to 3 years are most reliable.
                Forecasts beyond 3 years should be treated as indicative only.
                This approach prioritises explainability and transparency over complexity.
            </p>
        </div>
    """, unsafe_allow_html=True)
    # ── Page: Scenario Simulator ──────────────────────────────────────────────────
elif page == "Scenario Simulator":
    st.markdown("<h1 style='color: #ffffff;'>Scenario Simulator</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #8892a4;'>Simulate how changes in company behaviour affect ESG scores. Adjust the sliders to model different improvement scenarios for {company}.</p>", unsafe_allow_html=True)

    if df_company.empty or latest is None:
        st.warning("No data available for this selection.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # Train model for predictions
    with st.spinner("Loading prediction model..."):
        model, metrics, importance_df, X_test, y_test = train_random_forest(df)

    if model is None:
        st.error("Not enough data to run scenario simulation.")
        st.stop()

    feature_cols = importance_df["Feature"].tolist()

    # Current values from latest data
    current_renewable = float(pd.to_numeric(latest.get("Renewable_Energy_Percentage", 50), errors="coerce") or 50)
    current_gender = float(pd.to_numeric(latest.get("Gender_Diversity_Percentage", 30), errors="coerce") or 30)
    current_board = float(pd.to_numeric(latest.get("Independent_Board_Percentage", 60), errors="coerce") or 60)
    current_injury = float(pd.to_numeric(latest.get("Workplace_Injury_Rate", 1.5), errors="coerce") or 1.5)
    current_turnover = float(pd.to_numeric(latest.get("Employee_Turnover_Percentage", 15), errors="coerce") or 15)
    current_training = float(pd.to_numeric(latest.get("Training_Hours_Per_Employee", 30), errors="coerce") or 30)
    current_esg_comp = float(pd.to_numeric(latest.get("ESG_Linked_Compensation_Yes_No", 0), errors="coerce") or 0)
    current_sustainability = float(pd.to_numeric(latest.get("Sustainability_Investment", 1e9), errors="coerce") or 1e9)
    current_emissions = float(pd.to_numeric(latest.get("Total_Emissions", 50000000), errors="coerce") or 50000000)
    current_esg = float(pd.to_numeric(latest.get("Overall_ESG_Score", 60), errors="coerce") or 60)

    # ── Scenario sliders ──────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Adjust Company Characteristics</div>", unsafe_allow_html=True)
    st.caption("Move the sliders to simulate different scenarios. The predicted ESG score updates automatically.")

    left, right = st.columns(2)

    with left:
        st.markdown("**Environmental Factors**")
        sim_renewable = st.slider(
            "Renewable Energy %",
            0.0, 100.0, current_renewable,
            help="Percentage of energy from renewable sources"
        )
        sim_emissions = st.slider(
            "Total Emissions (millions tCO2e)",
            0.0, 500.0, min(current_emissions / 1e6, 500.0),
            help="Total greenhouse gas emissions in millions of tonnes"
        )
        sim_sustainability = st.slider(
            "Sustainability Investment ($B)",
            0.0, 20.0, min(current_sustainability / 1e9, 20.0),
            help="Amount invested in sustainability initiatives"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Social Factors**")
        sim_gender = st.slider(
            "Gender Diversity %",
            0.0, 100.0, current_gender,
            help="Percentage of female employees"
        )
        sim_injury = st.slider(
            "Workplace Injury Rate",
            0.0, 10.0, current_injury,
            help="Number of injuries per 100 employees"
        )
        sim_turnover = st.slider(
            "Employee Turnover %",
            0.0, 50.0, current_turnover,
            help="Percentage of employees leaving per year"
        )
        sim_training = st.slider(
            "Training Hours Per Employee",
            0.0, 100.0, current_training,
            help="Average training hours per employee per year"
        )

    with right:
        st.markdown("**Governance Factors**")
        sim_board = st.slider(
            "Independent Board %",
            0.0, 100.0, current_board,
            help="Percentage of independent board members"
        )
        sim_esg_comp = st.selectbox(
            "ESG Linked Compensation",
            [0, 1],
            index=int(current_esg_comp),
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether executive pay is linked to ESG targets"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Build input for prediction
        sim_input = {}
        for col in feature_cols:
            val = pd.to_numeric(latest.get(col, 0), errors="coerce")
            sim_input[col] = float(val) if not pd.isna(val) else 0.0

        # Override with slider values
        sim_input["Renewable_Energy_Percentage"] = sim_renewable
        sim_input["Total_Emissions"] = sim_emissions * 1e6
        sim_input["Sustainability_Investment"] = sim_sustainability * 1e9
        sim_input["Gender_Diversity_Percentage"] = sim_gender
        sim_input["Workplace_Injury_Rate"] = sim_injury
        sim_input["Employee_Turnover_Percentage"] = sim_turnover
        sim_input["Training_Hours_Per_Employee"] = sim_training
        sim_input["Independent_Board_Percentage"] = sim_board
        sim_input["ESG_Linked_Compensation_Yes_No"] = float(sim_esg_comp)

        # Predict new ESG score
        simulated_score = predict_esg(model, sim_input, feature_cols)
        score_change = simulated_score - current_esg
        change_direction = "improvement" if score_change >= 0 else "decline"
        change_color = "#34d399" if score_change >= 0 else "#f87171"

        # ── Results display ───────────────────────────────────────────────────
        st.markdown("<div class='section-heading'>Simulation Result</div>", unsafe_allow_html=True)

        st.markdown(f"""
            <div class='veridex-card' style='text-align: center;'>
                <p style='color: #8892a4; font-size: 14px; margin-bottom: 4px;'>Simulated ESG Score</p>
                <p style='color: #00a693; font-size: 52px; font-weight: 800; margin: 0;'>{simulated_score:.1f}</p>
                <p style='color: {change_color}; font-size: 18px; font-weight: 600;'>
                    {"+{:.1f}".format(score_change) if score_change >= 0 else "{:.1f}".format(score_change)}
                    points {change_direction} from current score
                </p>
                <p style='color: #8892a4; font-size: 13px;'>Current score: {current_esg:.1f} | Rating: {score_label(simulated_score)}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual comparison bar
        st.markdown("**Current vs Simulated Score**")
        compare_df = pd.DataFrame({
            "Scenario": ["Current Score", "Simulated Score"],
            "ESG Score": [current_esg, simulated_score],
        })

        fig = px.bar(
            compare_df,
            x="Scenario",
            y="ESG Score",
            color="Scenario",
            color_discrete_map={
                "Current Score": "#8892a4",
                "Simulated Score": "#00a693",
            },
            text="ESG Score",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a", range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── What changed summary ──────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>What You Changed</div>", unsafe_allow_html=True)

    changes = []
    if abs(sim_renewable - current_renewable) > 0.1:
        changes.append(f"Renewable Energy: {current_renewable:.1f}% to {sim_renewable:.1f}%")
    if abs(sim_emissions - current_emissions / 1e6) > 0.1:
        changes.append(f"Emissions: {current_emissions/1e6:.1f}M to {sim_emissions:.1f}M tCO2e")
    if abs(sim_gender - current_gender) > 0.1:
        changes.append(f"Gender Diversity: {current_gender:.1f}% to {sim_gender:.1f}%")
    if abs(sim_board - current_board) > 0.1:
        changes.append(f"Independent Board: {current_board:.1f}% to {sim_board:.1f}%")
    if abs(sim_injury - current_injury) > 0.01:
        changes.append(f"Workplace Injury Rate: {current_injury:.2f} to {sim_injury:.2f}")
    if abs(sim_turnover - current_turnover) > 0.1:
        changes.append(f"Employee Turnover: {current_turnover:.1f}% to {sim_turnover:.1f}%")
    if abs(sim_training - current_training) > 0.1:
        changes.append(f"Training Hours: {current_training:.1f} to {sim_training:.1f} hrs")
    if sim_esg_comp != current_esg_comp:
        changes.append(f"ESG Linked Compensation: {'No' if current_esg_comp == 0 else 'Yes'} to {'Yes' if sim_esg_comp == 1 else 'No'}")

    if changes:
        for change in changes:
            st.markdown(f"- {change}")
    else:
        st.info("No changes made yet. Move the sliders to simulate a scenario.")

    # ── Methodology note ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class='veridex-card'>
            <p style='color: #8892a4; font-size: 13px;'>
                <strong style='color: #ffffff;'>How the Simulator Works:</strong>
                The scenario simulator uses the trained Random Forest model to predict
                what a company's ESG score would be if its characteristics were different.
                When you move a slider, the new value is fed into the model alongside
                the company's other real data points, and a new ESG score is predicted instantly.
                This allows analysts and companies to model the ESG impact of strategic decisions
                before committing to them.
            </p>
        </div>
    """, unsafe_allow_html=True)
    # ── Page: PDF Report ──────────────────────────────────────────────────────────
elif page == "PDF Report":
    st.markdown("<h1 style='color: #ffffff;'>PDF Report Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892a4;'>Generate a professional ESG report for any company. Download it as a PDF instantly.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    report_company = st.selectbox("Select company to generate report for", companies)

    # Show a preview of what will be in the report
    preview_data = df[df["Company"] == report_company].copy()
    if not preview_data.empty:
        preview_latest = preview_data.sort_values("Year").iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        esg_val = pd.to_numeric(preview_latest.get("Overall_ESG_Score", np.nan), errors="coerce")
        env_val = pd.to_numeric(preview_latest.get("Environmental_Score", np.nan), errors="coerce")
        soc_val = pd.to_numeric(preview_latest.get("Social_Score", np.nan), errors="coerce")
        gov_val = pd.to_numeric(preview_latest.get("Governance_Score", np.nan), errors="coerce")

        with c1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Overall ESG</div>
                    <div class='metric-value'>{esg_val:.1f}</div>
                    <div class='metric-sub'>{score_label(esg_val)}</div>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Environmental</div>
                    <div class='metric-value' style='color: #34d399;'>{env_val:.1f}</div>
                    <div class='metric-sub'>E Score</div>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Social</div>
                    <div class='metric-value' style='color: #60a5fa;'>{soc_val:.1f}</div>
                    <div class='metric-sub'>S Score</div>
                </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Governance</div>
                    <div class='metric-value' style='color: #a78bfa;'>{gov_val:.1f}</div>
                    <div class='metric-sub'>G Score</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
            <div class='veridex-card'>
                <p style='color: #ffffff; font-weight: 600; margin-bottom: 8px;'>Report will include:</p>
                <p style='color: #8892a4; font-size: 13px;'>
                    Company overview and ESG score summary |
                    Environmental performance breakdown |
                    Social performance breakdown |
                    Governance performance breakdown |
                    Financial summary with formatted figures |
                    Improvement recommendations |
                    Veridex scoring methodology note |
                    Professional branding and disclaimer
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Generate PDF Report"):
            with st.spinner("Generating report..."):
                pdf_bytes = generate_report(df, report_company)

            if pdf_bytes:
                st.success("Report generated successfully.")
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"Veridex_{report_company}_ESG_Report_{date.today().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("Could not generate report. Please check the data for this company.")


# ── Page: Data Quality ────────────────────────────────────────────────────────
elif page == "Data Quality":
    st.markdown("<h1 style='color: #ffffff;'>Data Quality</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892a4;'>Audit of the Veridex dataset — completeness, consistency, and coverage.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset overview ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Rows</div>
                <div class='metric-value'>{len(df)}</div>
                <div class='metric-sub'>Company-year records</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Columns</div>
                <div class='metric-value'>{len(df.columns)}</div>
                <div class='metric-sub'>Features tracked</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Companies</div>
                <div class='metric-value'>{df["Company"].nunique()}</div>
                <div class='metric-sub'>Unique organisations</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        overall_completeness = (1 - df.isna().mean().mean()) * 100
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Overall Completeness</div>
                <div class='metric-value'>{overall_completeness:.1f}%</div>
                <div class='metric-sub'>Non-missing values</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Missing values chart ──────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Missing Values by Column</div>", unsafe_allow_html=True)

    missing = df.isna().mean().sort_values(ascending=False) * 100
    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing %"]
    missing_df = missing_df[missing_df["Missing %"] > 0]

    if not missing_df.empty:
        fig = px.bar(
            missing_df,
            x="Missing %",
            y="Column",
            orientation="h",
            color="Missing %",
            color_continuous_scale=[[0, "#34d399"], [0.5, "#fb923c"], [1, "#f87171"]],
        )
        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the dataset.")

    # ── Completeness by company ───────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Data Completeness by Company</div>", unsafe_allow_html=True)

    key_fields = [c for c in [
        "Revenue", "Net_Income", "Market_Cap", "Overall_ESG_Score",
        "Environmental_Score", "Social_Score", "Governance_Score",
        "Total_Emissions", "Renewable_Energy_Percentage",
        "Gender_Diversity_Percentage", "Independent_Board_Percentage",
    ] if c in df.columns]

    completeness = (
        df.groupby("Company")[key_fields]
        .apply(lambda g: (g.notna().mean() * 100).round(1))
    )
    st.dataframe(completeness, use_container_width=True)

    # ── Emissions consistency check ───────────────────────────────────────────
    st.markdown("<div class='section-heading'>Emissions Consistency Check</div>", unsafe_allow_html=True)
    st.caption("Verifies that Scope 1 + Scope 2 + Scope 3 emissions are consistent with Total Emissions")

    scope_cols = ["Scope_1_Emissions", "Scope_2_Emissions", "Scope_3_Emissions"]
    if all(c in df.columns for c in scope_cols) and "Total_Emissions" in df.columns:
        check = df[["Company", "Year", "Total_Emissions"]].copy()
        check["Sum_of_Scopes"] = (
            df["Scope_1_Emissions"].fillna(0) +
            df["Scope_2_Emissions"].fillna(0) +
            df["Scope_3_Emissions"].fillna(0)
        )
        check["Difference"] = (check["Total_Emissions"] - check["Sum_of_Scopes"]).round(0)
        check["Year"] = check["Year"].astype(str)
        st.dataframe(
            check.sort_values("Difference", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Scope emissions columns not found in dataset.")

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("View full raw dataset"):
        st.dataframe(df, use_container_width=True, hide_index=True)