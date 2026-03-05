# app.py
# Veridex ESG Intelligence Platform - Main Dashboard
# Streamlit 1.54.0 compatible

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json as _json, os as _os, hashlib as _hashlib

from ml_engine import train_random_forest, predict_esg, forecast_metric, get_esg_rating
from esg_scorer import (
    calculate_weighted_esg,
    get_esg_rating as score_rating,
    check_greenwashing,
    get_sector_average,
    get_recommendations,
    get_fallback_recommendations,
)
from pdf_generator import generate_report

# ════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Veridex — ESG Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════
# STYLING  — tested against Streamlit 1.54.0
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── hide chrome ── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── app background ── */
.stApp { background-color: #030b18 !important; }
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 100% !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #061022 !important;
    border-right: 1px solid #0f2040 !important;
}
section[data-testid="stSidebar"] > div {
    background-color: #061022 !important;
}

/* ── SLIDERS  (1.54.0 uses data-baseweb) ── */
div[data-baseweb="slider"] {
    padding-top: 6px !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background-color: #00d4aa !important;
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 4px rgba(0,212,170,0.2) !important;
}
div[data-baseweb="slider"] div[data-testid="stSlider"] > div > div > div:first-child {
    background: #00d4aa !important;
}
/* track fill */
div[data-baseweb="slider"] > div > div:first-child > div:first-child {
    background: linear-gradient(to right, #00d4aa, #00d4aa) !important;
}

/* ── METRIC CARDS ── */
div[data-testid="stMetric"] {
    background: #091528 !important;
    border: 1px solid #0f2040 !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="stMetricLabel"] p {
    color: #94a3b8 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricDelta"] {
    color: #00d4aa !important;
    font-size: 12px !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #00d4aa !important;
    color: #030b18 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #00f5c4 !important; }
.stButton > button p { color: #030b18 !important; font-weight: 700 !important; }

/* ── SELECTBOX ── */
div[data-baseweb="select"] > div {
    background: #091528 !important;
    border-color: #0f2040 !important;
    border-radius: 8px !important;
    color: #e2eeff !important;
}
div[data-baseweb="select"] span {
    color: #e2eeff !important;
}
div[data-baseweb="popover"] {
    background: #091528 !important;
    border: 1px solid #0f2040 !important;
}
div[data-baseweb="menu"] {
    background: #091528 !important;
}
div[data-baseweb="menu"] li {
    color: #e2eeff !important;
}
div[data-baseweb="menu"] li:hover {
    background: #0f2040 !important;
}

/* ── MULTISELECT ── */
span[data-baseweb="tag"] {
    background-color: rgba(0,212,170,0.15) !important;
    border: 1px solid rgba(0,212,170,0.3) !important;
}
span[data-baseweb="tag"] span { color: #00d4aa !important; }

/* ── TEXT INPUT ── */
div[data-baseweb="input"] input {
    background: #091528 !important;
    border-color: #0f2040 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* ── TYPOGRAPHY ── */
h1 { color: #ffffff !important; font-size: 2rem !important; font-weight: 800 !important; font-family: 'Syne', sans-serif !important; }
h2 { color: #00d4aa !important; font-size: 1.1rem !important; font-family: 'DM Sans', sans-serif !important; }
h3 { color: #ffffff !important; font-family: 'DM Sans', sans-serif !important; }
p  { color: #cbd5e1 !important; font-family: 'DM Sans', sans-serif !important; }
li { color: #cbd5e1 !important; font-family: 'DM Sans', sans-serif !important; }
hr { border-color: #0f2040 !important; }
label { color: #94a3b8 !important; font-family: 'DM Sans', sans-serif !important; }
.stMarkdown p { color: #cbd5e1 !important; }

/* ── DATAFRAME ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #0f2040 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── EXPANDER ── */
div[data-testid="stExpander"] {
    border: 1px solid #0f2040 !important;
    border-radius: 8px !important;
    background: #091528 !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# USER MANAGEMENT
# ════════════════════════════════════════════════════════
def _hash(pwd):
    return _hashlib.sha256(pwd.encode()).hexdigest()

def load_users():
    if not _os.path.exists("users.json"):
        default = {
            "admin@veridex.com": {
                "password": _hash("Admin@2024"),
                "role": "Admin",
                "name": "Platform Admin",
                "email": "admin@veridex.com"
            }
        }
        with open("users.json", "w") as f:
            _json.dump(default, f, indent=2)
        return default
    with open("users.json", "r") as f:
        return _json.load(f)

def save_users(users):
    with open("users.json", "w") as f:
        _json.dump(users, f, indent=2)

USERS = load_users()

ROLE_PAGES = {
    "Admin":   ["Overview", "Company Analysis", "Compare Companies", "ESG Analysis",
                "Predictive Analytics", "ESG Outlook", "Strategy Simulator",
                "ESG Reports", "Data Governance", "Admin Panel"],
    "Analyst": ["Overview", "Company Analysis", "Compare Companies", "ESG Analysis",
                "Predictive Analytics", "ESG Outlook", "Strategy Simulator",
                "ESG Reports", "Data Governance"],
    "Viewer":  ["Overview", "Company Analysis", "Compare Companies"],
}

# ════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════
for key, val in [("logged_in", False), ("user", None), ("role", None),
                 ("name", None), ("page", "Overview")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ════════════════════════════════════════════════════════
# FALLBACK LOGIN (if accessed directly, not via home.py)
# ════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style='text-align:center;margin-top:60px;margin-bottom:8px'>
            <span style='font-size:38px;font-weight:800;color:#00d4aa;font-family:Syne,sans-serif'>Veridex</span>
        </div>
        <div style='text-align:center;color:#64748b;font-size:13px;margin-bottom:32px'>
            ESG Intelligence Platform
        </div>
        """, unsafe_allow_html=True)

        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Sign In", use_container_width=True):
            email = email.strip().lower()
            if email in USERS and USERS[email]["password"] == _hash(password):
                st.session_state.logged_in = True
                st.session_state.user      = email
                st.session_state.role      = USERS[email]["role"]
                st.session_state.name      = USERS[email]["name"]
                st.rerun()
            else:
                st.error("Incorrect email or password.")
    st.stop()

# ════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("data/ESG_Final_Validated.csv")
    sectors    = df["Sector"].unique()
    sector_map = {s: i for i, s in enumerate(sectors)}
    df["Sector_Encoded"] = df["Sector"].map(sector_map)
    if df["ESG_Linked_Compensation_Yes_No"].dtype == object:
        df["ESG_Linked_Compensation_Yes_No"] = df["ESG_Linked_Compensation_Yes_No"].map(
            {"Yes": 1, "No": 0}
        ).fillna(0)
    return df, sector_map

df, sector_map = load_data()

# ════════════════════════════════════════════════════════
# TRAIN ML MODEL
# ════════════════════════════════════════════════════════
@st.cache_resource
def get_model(df):
    return train_random_forest(df)

model, ml_metrics, importance_df, X_test, y_test, feature_cols = get_model(df)

# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════
def chart_style():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,21,40,0.6)",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#0f2040", linecolor="#0f2040"),
        yaxis=dict(gridcolor="#0f2040", linecolor="#0f2040"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

def esg_colour(score):
    if score >= 80:   return "#00d4aa"
    elif score >= 65: return "#0ea5e9"
    elif score >= 50: return "#f59e0b"
    else:             return "#ef4444"

def fmt_money(val):
    try:
        val = float(val)
        if val >= 1_000_000_000_000: return f"${val/1_000_000_000_000:.1f}T"
        elif val >= 1_000_000_000:   return f"${val/1_000_000_000:.1f}B"
        elif val >= 1_000_000:       return f"${val/1_000_000:.1f}M"
        else:                        return f"${val:.1f}"
    except:
        return "N/A"

def clean_year_axis(fig):
    fig.update_xaxes(type="category")
    return fig

def to_year_str(series):
    return series.astype(int).astype(str)

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 6px'>
        <span style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#00d4aa'>Veridex</span><br>
        <span style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em'>ESG Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    role_colours = {"Admin": "#8b5cf6", "Analyst": "#0ea5e9", "Viewer": "#f59e0b"}
    rc = role_colours.get(st.session_state.role, "#64748b")
    st.markdown(f"""
    <div style='background:#091528;border:1px solid #0f2040;border-radius:10px;
                padding:10px 14px;margin-bottom:14px'>
        <div style='font-size:13px;font-weight:600;color:#ffffff'>{st.session_state.name}</div>
        <div style='font-size:11px;color:{rc};margin-top:3px'>{st.session_state.role}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;color:#64748b;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-bottom:8px'>Navigation</div>", unsafe_allow_html=True)

    page_icons = {
        "Overview":             "◈  ",
        "Company Analysis":     "◉  ",
        "Compare Companies":    "⇄  ",
        "ESG Analysis":         "◎  ",
        "Predictive Analytics": "◆  ",
        "ESG Outlook":          "↗  ",
        "Strategy Simulator":   "⊕  ",
        "ESG Reports":          "▤  ",
        "Data Governance":      "◫  ",
        "Admin Panel":          "⚙  ",
    }

    for p in ROLE_PAGES[st.session_state.role]:
        icon = page_icons.get(p, "•  ")
        if st.button(f"{icon}{p}", key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()

    st.divider()
    st.markdown("<div style='font-size:10px;color:#64748b;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-bottom:8px'>Filters</div>", unsafe_allow_html=True)

    companies        = sorted(df["Company"].unique().tolist())
    selected_company = st.selectbox("Select Company", companies)
    years            = sorted(df["Year"].unique().tolist())
    year_range       = st.select_slider("Year Range", options=years, value=(min(years), max(years)))

    st.divider()
    if st.button("Sign Out", use_container_width=True):
        for key in ["logged_in", "user", "role", "name"]:
            st.session_state[key] = None
        st.session_state.logged_in = False
        st.rerun()

# ════════════════════════════════════════════════════════
# FILTERED DATA
# ════════════════════════════════════════════════════════
filtered_df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
company_df  = filtered_df[filtered_df["Company"] == selected_company].sort_values("Year")
latest_data = company_df.iloc[-1].to_dict() if not company_df.empty else {}
page        = st.session_state.page

# ════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Platform Overview")
    st.markdown("A snapshot of all companies tracked on the Veridex platform")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies Tracked",    df["Company"].nunique())
    c2.metric("Platform Average ESG", round(df["Overall_ESG_Score"].mean(), 1))
    c3.metric("Top ESG Performer",    df.groupby("Company")["Overall_ESG_Score"].mean().idxmax())
    c4.metric("Sectors Covered",      df["Sector"].nunique())

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average ESG Score by Sector")
        st.caption("Which industries lead on ESG — higher is better")
        sector_avg = df.groupby("Sector")["Overall_ESG_Score"].mean().reset_index()
        sector_avg.columns = ["Sector", "Average ESG Score"]
        sector_avg = sector_avg.sort_values("Average ESG Score", ascending=True)
        sector_avg["colour"] = sector_avg["Average ESG Score"].apply(esg_colour)
        fig = px.bar(sector_avg, x="Average ESG Score", y="Sector",
                     orientation="h", color="colour",
                     color_discrete_map="identity", text="Average ESG Score")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(**chart_style(), showlegend=False)
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Platform ESG Trend Over Time")
        st.caption("Average ESG score across all companies by year")
        trend = df.groupby("Year")["Overall_ESG_Score"].mean().reset_index()
        trend.columns = ["Year", "Average ESG Score"]
        trend["Year"] = to_year_str(trend["Year"])
        fig2 = px.line(trend, x="Year", y="Average ESG Score", markers=True, line_shape="spline")
        fig2.update_traces(line_color="#00d4aa", marker_color="#00d4aa", marker_size=8)
        fig2 = clean_year_axis(fig2)
        fig2.update_layout(**chart_style())
        fig2.update_yaxes(range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("ESG Leaderboard")
    st.caption("All companies ranked by average ESG score")
    leaderboard = df.groupby(["Company", "Sector"]).agg(
        ESG_Score=("Overall_ESG_Score", "mean"),
        Environmental=("Environmental_Score", "mean"),
        Social=("Social_Score", "mean"),
        Governance=("Governance_Score", "mean"),
    ).round(1).reset_index().sort_values("ESG_Score", ascending=False).reset_index(drop=True)
    leaderboard.index += 1
    leaderboard["Rating"] = leaderboard["ESG_Score"].apply(get_esg_rating)
    st.dataframe(leaderboard, use_container_width=True)

# ════════════════════════════════════════════════════════
# COMPANY ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "Company Analysis":
    st.title("Company Analysis")
    st.markdown(f"Detailed ESG and financial breakdown for **{selected_company}**")
    st.divider()

    if company_df.empty:
        st.warning("No data found for this company in the selected year range.")
        st.stop()

    esg  = latest_data.get("Overall_ESG_Score", 0)
    env  = latest_data.get("Environmental_Score", 0)
    soc  = latest_data.get("Social_Score", 0)
    gov  = latest_data.get("Governance_Score", 0)
    rev  = latest_data.get("Revenue", 0)
    ni   = latest_data.get("Net_Income", 0)
    mcap = latest_data.get("Market_Cap", 0)

    margin = (ni / rev * 100) if rev > 0 else 0
    fin = "Strong" if margin > 15 else ("Profitable" if margin > 0 else "Loss-Making")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall ESG Score", f"{esg:.1f}", get_esg_rating(esg))
    c2.metric("Revenue",           fmt_money(rev))
    c3.metric("Net Income",        fmt_money(ni))
    c4.metric("Market Cap",        fmt_money(mcap) if mcap and mcap > 0 else "N/A")
    c5.metric("Financial Health",  fin, f"{margin:.1f}% margin")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ESG Score Trend")
        melt = company_df[["Year","Environmental_Score","Social_Score",
                            "Governance_Score","Overall_ESG_Score"]].copy()
        melt["Year"] = to_year_str(melt["Year"])
        melt = melt.melt("Year", var_name="Metric", value_name="Score")
        melt["Metric"] = melt["Metric"].str.replace("_Score","").str.replace("_"," ")
        fig = px.line(melt, x="Year", y="Score", color="Metric", markers=True, line_shape="spline")
        fig = clean_year_axis(fig)
        fig.update_layout(**chart_style())
        fig.update_yaxes(range=[0,100], title="Score (0-100)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ESG Pillar Breakdown")
        pillars = pd.DataFrame({
            "Pillar": ["Environmental","Social","Governance"],
            "Score":  [env, soc, gov],
            "colour": [esg_colour(env), esg_colour(soc), esg_colour(gov)]
        })
        fig2 = px.bar(pillars, x="Pillar", y="Score", color="colour",
                      color_discrete_map="identity", text="Score")
        fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig2.update_layout(**chart_style(), showlegend=False)
        fig2.update_yaxes(range=[0,100])
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Financial Performance")
        fin_df = company_df[["Year","Revenue","Net_Income"]].copy()
        fin_df["Year"]       = to_year_str(fin_df["Year"])
        fin_df["Revenue"]    = fin_df["Revenue"]    / 1_000_000_000
        fin_df["Net_Income"] = fin_df["Net_Income"] / 1_000_000_000
        fin_melt = fin_df.melt("Year", var_name="Metric", value_name="Value ($B)")
        fin_melt["Metric"] = fin_melt["Metric"].str.replace("_"," ")
        fig3 = px.line(fin_melt, x="Year", y="Value ($B)", color="Metric", markers=True)
        fig3 = clean_year_axis(fig3)
        fig3.update_layout(**chart_style())
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Emissions Overview")
        has_emissions = company_df["Total_Emissions"].sum() > 0
        if has_emissions:
            em_df = company_df[["Year","Total_Emissions"]].copy()
            em_df["Year"] = to_year_str(em_df["Year"])
            fig4 = px.line(em_df, x="Year", y="Total_Emissions", markers=True)
            fig4.update_traces(line_color="#ef4444", marker_color="#ef4444")
            fig4 = clean_year_axis(fig4)
            fig4.update_layout(**chart_style())
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Emissions data not available for this company.")

    st.divider()
    st.subheader("Greenwashing Check")
    verdict, colour = check_greenwashing(latest_data)
    st.markdown(f"""
    <div style='background:#091528;border:1px solid {colour}44;border-radius:8px;
                padding:14px 18px;color:{colour};font-size:14px'>{verdict}</div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Improvement Recommendations")
    sector     = latest_data.get("Sector","Unknown")
    sector_avg = get_sector_average(df, sector)
    with st.spinner("Generating recommendations..."):
        recs = get_recommendations(selected_company, latest_data, sector, sector_avg)
    for i, rec in enumerate(recs, 1):
        st.markdown(f"""
        <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;
                    background:#091528;border:1px solid #0f2040;border-radius:8px;padding:14px'>
            <div style='width:24px;height:24px;border-radius:50%;background:#00d4aa;color:#030b18;
                        font-weight:800;font-size:12px;display:flex;align-items:center;
                        justify-content:center;flex-shrink:0'>{i}</div>
            <div style='color:#cbd5e1;font-size:13px;line-height:1.6'>{rec}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("View raw data"):
        st.dataframe(company_df.sort_values("Year", ascending=False), use_container_width=True)

# ════════════════════════════════════════════════════════
# COMPARE COMPANIES
# ════════════════════════════════════════════════════════
elif page == "Compare Companies":
    st.title("Compare Companies")
    st.divider()

    all_companies = sorted(df["Company"].unique().tolist())
    default       = all_companies[:4] if len(all_companies) >= 4 else all_companies
    selected      = st.multiselect("Select companies to compare", all_companies, default=default)

    if len(selected) < 2:
        st.info("Please select at least 2 companies.")
        st.stop()

    compare_df     = filtered_df[filtered_df["Company"].isin(selected)]
    latest_compare = compare_df.groupby("Company").last().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overall ESG Over Time")
        trend_df = compare_df[["Company","Year","Overall_ESG_Score"]].copy()
        trend_df["Year"] = to_year_str(trend_df["Year"])
        fig = px.line(trend_df.sort_values("Year"), x="Year", y="Overall_ESG_Score",
                      color="Company", markers=True, line_shape="spline")
        fig = clean_year_axis(fig)
        fig.update_layout(**chart_style())
        fig.update_yaxes(range=[0,100], title="ESG Score")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Environmental Score")
        fig2 = px.bar(latest_compare.sort_values("Environmental_Score"),
                      x="Environmental_Score", y="Company", orientation="h",
                      text="Environmental_Score", color="Environmental_Score",
                      color_continuous_scale=["#ef4444","#f59e0b","#00d4aa"])
        fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig2.update_layout(**chart_style(), showlegend=False, coloraxis_showscale=False)
        fig2.update_xaxes(range=[0,100])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Social Score")
        fig3 = px.bar(latest_compare.sort_values("Social_Score"),
                      x="Social_Score", y="Company", orientation="h",
                      text="Social_Score", color="Social_Score",
                      color_continuous_scale=["#ef4444","#f59e0b","#00d4aa"])
        fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig3.update_layout(**chart_style(), showlegend=False, coloraxis_showscale=False)
        fig3.update_xaxes(range=[0,100])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Governance Score")
        fig4 = px.bar(latest_compare.sort_values("Governance_Score"),
                      x="Governance_Score", y="Company", orientation="h",
                      text="Governance_Score", color="Governance_Score",
                      color_continuous_scale=["#ef4444","#f59e0b","#00d4aa"])
        fig4.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig4.update_layout(**chart_style(), showlegend=False, coloraxis_showscale=False)
        fig4.update_xaxes(range=[0,100])
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Full Comparison Table")
    table = compare_df.groupby("Company").last().reset_index()[
        ["Company","Sector","Overall_ESG_Score","Environmental_Score",
         "Social_Score","Governance_Score","Revenue","Net_Income"]
    ].round(1).sort_values("Overall_ESG_Score", ascending=False)
    table["Rating"] = table["Overall_ESG_Score"].apply(get_esg_rating)
    st.dataframe(table, use_container_width=True)

# ════════════════════════════════════════════════════════
# ESG ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "ESG Analysis":
    st.title("ESG Analysis")
    st.markdown(f"Deep dive ESG breakdown for **{selected_company}**")
    st.divider()

    if company_df.empty:
        st.warning("No data available.")
        st.stop()

    esg        = latest_data.get("Overall_ESG_Score", 0)
    env        = latest_data.get("Environmental_Score", 0)
    soc        = latest_data.get("Social_Score", 0)
    gov        = latest_data.get("Governance_Score", 0)
    sector     = latest_data.get("Sector", "Unknown")
    sector_avg = get_sector_average(df, sector) or 0

    st.markdown("**Adjust ESG Pillar Weights**")
    wc1, wc2, wc3 = st.columns(3)
    with wc1: env_w = st.slider("Environmental", 0.0, 1.0, 0.4, 0.05, key="env_w")
    with wc2: soc_w = st.slider("Social",        0.0, 1.0, 0.3, 0.05, key="soc_w")
    with wc3: gov_w = st.slider("Governance",    0.0, 1.0, 0.3, 0.05, key="gov_w")

    custom_esg = calculate_weighted_esg(env, soc, gov, env_w, soc_w, gov_w)
    vs_sector  = round(custom_esg - sector_avg, 1)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Official ESG Score",  f"{esg:.1f}",        get_esg_rating(esg))
    c2.metric("Your Weighted Score", f"{custom_esg:.1f}", f"E:{env_w} S:{soc_w} G:{gov_w}")
    c3.metric("Sector Average",      f"{sector_avg:.1f}", sector)
    c4.metric("vs Sector",           f"{vs_sector:+.1f}", "Above avg" if vs_sector >= 0 else "Below avg")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ESG Radar Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[env, soc, gov, env],
            theta=["Environmental","Social","Governance","Environmental"],
            fill="toself", fillcolor="rgba(0,212,170,0.15)",
            line_color="#00d4aa", name=selected_company
        ))
        fig.add_trace(go.Scatterpolar(
            r=[sector_avg]*4,
            theta=["Environmental","Social","Governance","Environmental"],
            line_color="#64748b", line_dash="dash", name="Sector Average"
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], gridcolor="#0f2040", linecolor="#0f2040"),
                angularaxis=dict(gridcolor="#0f2040", linecolor="#0f2040"),
                bgcolor="rgba(9,21,40,0.6)"
            ),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=40,r=40,t=40,b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ESG Score vs Emissions")
        latest_all = df.groupby("Company").last().reset_index()
        latest_all["Highlight"] = latest_all["Company"].apply(
            lambda c: selected_company if c == selected_company else "Other"
        )
        fig2 = px.scatter(latest_all, x="Total_Emissions", y="Overall_ESG_Score",
                          color="Highlight",
                          color_discrete_map={selected_company: "#00d4aa", "Other": "#334155"},
                          hover_name="Company")
        fig2.update_traces(marker_size=10)
        fig2.update_layout(**chart_style())
        fig2.update_yaxes(range=[0,100])
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Renewable Energy vs ESG Score")
    scatter2 = df.groupby("Company").last().reset_index()
    fig3 = px.scatter(scatter2, x="Renewable_Energy_Percentage", y="Overall_ESG_Score",
                      color="Sector", hover_name="Company", trendline="ols")
    fig3.update_layout(**chart_style())
    fig3.update_yaxes(range=[0,100])
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════
# PREDICTIVE ANALYTICS
# ════════════════════════════════════════════════════════
elif page == "Predictive Analytics":
    st.title("Predictive Analytics")
    st.markdown("ML model predicts ESG scores from 16 company metrics. Adjust sliders and the score updates instantly.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy (R²)", ml_metrics["r2"])
    c2.metric("Average Error",       f"{ml_metrics['mae']} pts")
    c3.metric("Training Samples",    ml_metrics["train_samples"])
    c4.metric("Test Samples",        ml_metrics["test_samples"])

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What Drives ESG Scores?")
        top10 = importance_df.head(10).copy()
        top10["Feature"] = top10["Feature"].str.replace("_"," ")
        fig = px.bar(top10.sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", text="Importance", color="Importance",
                     color_continuous_scale=["#334155","#00d4aa"])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(**chart_style(), coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Actual vs Predicted")
        y_pred     = model.predict(X_test)
        scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig2 = px.scatter(scatter_df, x="Actual", y="Predicted",
                          trendline="ols", trendline_color_override="#ef4444")
        fig2.update_traces(marker_color="#00d4aa", marker_size=8, selector=dict(mode="markers"))
        mn = min(scatter_df["Actual"].min(), scatter_df["Predicted"].min()) - 5
        mx = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max()) + 5
        fig2.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                       line=dict(color="#64748b", dash="dash"))
        fig2.update_layout(**chart_style())
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader(f"Predict ESG Score for {selected_company}")

    if not company_df.empty:
        latest = company_df.iloc[-1]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Environmental**")
            renewable = st.slider("Renewable Energy %",          0.0, 100.0, float(latest.get("Renewable_Energy_Percentage",30)),           1.0, key="p_ren")
            emissions = st.slider("Total Emissions (M tCO2e)",   0.0, 500.0, min(float(latest.get("Total_Emissions",50)),500.0),            1.0, key="p_em")
            carbon    = st.slider("Carbon Reduction Target %",   0.0, 100.0, float(latest.get("Carbon_Reduction_Target_Percentage",20)),    1.0, key="p_car")
            st.markdown("**Social**")
            diversity = st.slider("Gender Diversity %",          0.0, 100.0, float(latest.get("Gender_Diversity_Percentage",30)),           1.0, key="p_div")
            training  = st.slider("Training Hours/Employee",     0.0, 100.0, min(float(latest.get("Training_Hours_Per_Employee",30)),100.0),1.0, key="p_tr")
            injury    = st.slider("Workplace Injury Rate",       0.0,  10.0, min(float(latest.get("Workplace_Injury_Rate",1.0)),10.0),      0.1, key="p_inj")

        with col2:
            st.markdown("**Governance**")
            board_ind = st.slider("Independent Board %",         0.0, 100.0, float(latest.get("Independent_Board_Percentage",70)),          1.0, key="p_bi")
            board_div = st.slider("Board Diversity %",           0.0, 100.0, float(latest.get("Board_Diversity_Percentage",30)),            1.0, key="p_bd")
            esg_comp  = st.selectbox("ESG Linked Compensation",  [0,1], format_func=lambda x:"Yes" if x==1 else "No", key="p_ec")
            st.markdown("**Financial**")
            sust      = st.slider("Sustainability Investment $B",0.0,  20.0, min(float(latest.get("Sustainability_Investment",1.0)),20.0),  0.1, key="p_si")
            turnover  = st.slider("Employee Turnover %",         0.0,  50.0, float(latest.get("Employee_Turnover_Percentage",15)),          0.5, key="p_to")

            input_dict = {
                "Renewable_Energy_Percentage":             renewable,
                "Total_Emissions":                         emissions,
                "Emissions_Intensity":                     emissions / max(float(latest.get("Revenue",1)),0.01),
                "Gender_Diversity_Percentage":             diversity,
                "Board_Diversity_Percentage":              board_div,
                "Employee_Turnover_Percentage":            turnover,
                "Training_Hours_Per_Employee":             training,
                "Workplace_Injury_Rate":                   injury,
                "Independent_Board_Percentage":            board_ind,
                "Audit_Committee_Independence_Percentage": float(latest.get("Audit_Committee_Independence_Percentage",80)),
                "ESG_Linked_Compensation_Yes_No":          esg_comp,
                "Carbon_Reduction_Target_Percentage":      carbon,
                "Sustainability_Investment":               sust,
                "Community_Investment":                    float(latest.get("Community_Investment",0.5)),
                "Revenue":                                 float(latest.get("Revenue",10)),
                "Net_Income":                              float(latest.get("Net_Income",1)),
            }

            predicted = predict_esg(model, input_dict, feature_cols)
            colour    = esg_colour(predicted)
            rating    = get_esg_rating(predicted)

            st.markdown(f"""
            <div style='background:#091528;border:2px solid {colour}55;border-radius:12px;
                        padding:24px;text-align:center;margin-top:16px'>
                <div style='font-size:11px;color:#94a3b8;text-transform:uppercase;
                            letter-spacing:0.1em;margin-bottom:8px'>Predicted ESG Score</div>
                <div style='font-size:56px;font-weight:800;color:{colour};line-height:1'>{predicted}</div>
                <div style='font-size:16px;color:{colour};margin-top:8px;font-weight:600'>{rating}</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# ESG OUTLOOK
# ════════════════════════════════════════════════════════
elif page == "ESG Outlook":
    st.title("ESG Outlook")
    st.markdown(f"Forward looking projections for **{selected_company}**")
    st.divider()

    if company_df.empty:
        st.warning("No data available.")
        st.stop()

    years_ahead = st.slider("Years to forecast", 1, 5, 3, key="fc_years")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ESG Score Forecast")
        esg_fc = forecast_metric(company_df, "Overall_ESG_Score", years_ahead)
        if esg_fc is not None:
            esg_fc["Year"] = to_year_str(esg_fc["Year"])
            fig = px.line(esg_fc, x="Year", y="Value", color="Type", markers=True, line_shape="spline",
                          color_discrete_map={"Historical":"#00d4aa","Forecast":"#f59e0b"})
            fig.update_traces(line_dash="dash", selector=dict(name="Forecast"))
            fig = clean_year_axis(fig)
            fig.update_layout(**chart_style())
            fig.update_yaxes(range=[0,100])
            st.plotly_chart(fig, use_container_width=True)

            forecast_only = esg_fc[esg_fc["Type"]=="Forecast"]
            cols = st.columns(len(forecast_only))
            for i, (_,row) in enumerate(forecast_only.iterrows()):
                c = esg_colour(row["Value"])
                cols[i].markdown(f"""
                <div style='background:#091528;border:1px solid {c}33;border-radius:10px;
                            padding:14px;text-align:center'>
                    <div style='font-size:10px;color:#94a3b8'>{row["Year"]}</div>
                    <div style='font-size:26px;font-weight:800;color:{c}'>{row["Value"]:.1f}</div>
                    <div style='font-size:11px;color:{c}'>{get_esg_rating(row["Value"])}</div>
                </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Revenue Forecast ($B)")
        rev_fc = forecast_metric(company_df, "Revenue", years_ahead)
        if rev_fc is not None:
            rev_fc["Value"] = rev_fc["Value"] / 1_000_000_000
            rev_fc["Year"]  = to_year_str(rev_fc["Year"])
            fig2 = px.line(rev_fc, x="Year", y="Value", color="Type", markers=True,
                           color_discrete_map={"Historical":"#0ea5e9","Forecast":"#f59e0b"})
            fig2.update_traces(line_dash="dash", selector=dict(name="Forecast"))
            fig2 = clean_year_axis(fig2)
            fig2.update_layout(**chart_style())
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    if company_df["Total_Emissions"].sum() > 0:
        st.subheader("Emissions Forecast")
        em_fc = forecast_metric(company_df, "Total_Emissions", years_ahead)
        if em_fc is not None:
            em_fc["Year"] = to_year_str(em_fc["Year"])
            fig3 = px.line(em_fc, x="Year", y="Value", color="Type", markers=True,
                           color_discrete_map={"Historical":"#ef4444","Forecast":"#f59e0b"})
            fig3.update_traces(line_dash="dash", selector=dict(name="Forecast"))
            fig3 = clean_year_axis(fig3)
            fig3.update_layout(**chart_style())
            st.plotly_chart(fig3, use_container_width=True)

    st.info("Forecasts use Linear Regression. Short-term projections are more reliable.")

# ════════════════════════════════════════════════════════
# STRATEGY SIMULATOR
# ════════════════════════════════════════════════════════
elif page == "Strategy Simulator":
    st.title("Strategy Simulator")
    st.markdown(f"Model ESG impact of strategic decisions for **{selected_company}**")
    st.divider()

    if company_df.empty:
        st.warning("No data available.")
        st.stop()

    current_esg = latest_data.get("Overall_ESG_Score", 0)
    latest      = company_df.iloc[-1]

    st.info(f"Current ESG score: **{current_esg:.1f}** — adjust sliders to simulate strategies")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("**Environmental**")
        sim_ren  = st.slider("Renewable Energy %",          0.0, 100.0, float(latest.get("Renewable_Energy_Percentage",30)),           1.0, key="s_ren")
        sim_em   = st.slider("Total Emissions (M tCO2e)",   0.0, 500.0, min(float(latest.get("Total_Emissions",50)),500.0),            1.0, key="s_em")
        sim_car  = st.slider("Carbon Reduction Target %",   0.0, 100.0, float(latest.get("Carbon_Reduction_Target_Percentage",20)),    1.0, key="s_car")
        sim_sust = st.slider("Sustainability Invest ($B)",  0.0,  20.0, min(float(latest.get("Sustainability_Investment",1.0)),20.0),  0.1, key="s_sust")
        st.markdown("**Social**")
        sim_div  = st.slider("Gender Diversity %",          0.0, 100.0, float(latest.get("Gender_Diversity_Percentage",30)),           1.0, key="s_div")
        sim_tr   = st.slider("Training Hours/Employee",     0.0, 100.0, min(float(latest.get("Training_Hours_Per_Employee",30)),100.0),1.0, key="s_tr")
        sim_inj  = st.slider("Workplace Injury Rate",       0.0,  10.0, min(float(latest.get("Workplace_Injury_Rate",1.0)),10.0),      0.1, key="s_inj")
        st.markdown("**Governance**")
        sim_bi   = st.slider("Independent Board %",         0.0, 100.0, float(latest.get("Independent_Board_Percentage",70)),          1.0, key="s_bi")
        sim_ec   = st.selectbox("ESG Linked Compensation",  [0,1], format_func=lambda x:"Yes" if x==1 else "No", key="s_ec")

    with col2:
        sim_input = {
            "Renewable_Energy_Percentage":             sim_ren,
            "Total_Emissions":                         sim_em,
            "Emissions_Intensity":                     sim_em / max(float(latest.get("Revenue",1)),0.01),
            "Gender_Diversity_Percentage":             sim_div,
            "Board_Diversity_Percentage":              float(latest.get("Board_Diversity_Percentage",30)),
            "Employee_Turnover_Percentage":            float(latest.get("Employee_Turnover_Percentage",15)),
            "Training_Hours_Per_Employee":             sim_tr,
            "Workplace_Injury_Rate":                   sim_inj,
            "Independent_Board_Percentage":            sim_bi,
            "Audit_Committee_Independence_Percentage": float(latest.get("Audit_Committee_Independence_Percentage",80)),
            "ESG_Linked_Compensation_Yes_No":          sim_ec,
            "Carbon_Reduction_Target_Percentage":      sim_car,
            "Sustainability_Investment":               sim_sust,
            "Community_Investment":                    float(latest.get("Community_Investment",0.5)),
            "Revenue":                                 float(latest.get("Revenue",10)),
            "Net_Income":                              float(latest.get("Net_Income",1)),
        }

        simulated = predict_esg(model, sim_input, feature_cols)
        change    = round(simulated - current_esg, 1)
        cc        = "#00d4aa" if change >= 0 else "#ef4444"
        sc        = esg_colour(simulated)

        st.markdown(f"""
        <div style='background:#091528;border:1px solid #0f2040;border-radius:12px;
                    padding:24px;text-align:center;margin-bottom:16px'>
            <div style='display:flex;justify-content:center;align-items:center;gap:24px'>
                <div>
                    <div style='font-size:11px;color:#94a3b8;margin-bottom:4px'>CURRENT</div>
                    <div style='font-size:40px;font-weight:800;color:#64748b'>{current_esg:.1f}</div>
                </div>
                <div style='font-size:28px;color:{cc}'>→</div>
                <div>
                    <div style='font-size:11px;color:#94a3b8;margin-bottom:4px'>SIMULATED</div>
                    <div style='font-size:40px;font-weight:800;color:{sc}'>{simulated}</div>
                </div>
            </div>
            <div style='font-size:18px;font-weight:700;color:{cc};margin-top:12px'>
                {"+" if change>=0 else ""}{change} points {"improvement" if change>=0 else "decline"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Current: {current_esg:.1f}**")
        st.progress(min(current_esg/100, 1.0))
        st.markdown(f"**Simulated: {simulated}**")
        st.progress(min(simulated/100, 1.0))

# ════════════════════════════════════════════════════════
# ESG REPORTS
# ════════════════════════════════════════════════════════
elif page == "ESG Reports":
    st.title("ESG Reports")
    st.markdown("Generate a professional PDF report for any company and year")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        report_company = st.selectbox("Select company", companies, key="rep_co")
    with col2:
        available_years = sorted(df["Year"].unique().tolist())
        report_year     = st.selectbox("Select year", available_years, index=len(available_years)-1, key="rep_yr")

    if report_company:
        rep_df = df[(df["Company"]==report_company) & (df["Year"]==report_year)]
        if rep_df.empty:
            rep_df = df[df["Company"]==report_company]
        if not rep_df.empty:
            latest_rep = rep_df.iloc[-1]
            esg = latest_rep.get("Overall_ESG_Score",0)
            env = latest_rep.get("Environmental_Score",0)
            soc = latest_rep.get("Social_Score",0)
            gov = latest_rep.get("Governance_Score",0)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Overall ESG",   f"{esg:.1f}", get_esg_rating(esg))
            c2.metric("Environmental", f"{env:.1f}")
            c3.metric("Social",        f"{soc:.1f}")
            c4.metric("Governance",    f"{gov:.1f}")

            if st.button(f"Generate PDF for {report_company} ({report_year})"):
                with st.spinner("Building report..."):
                    try:
                        pdf_bytes = generate_report(df, report_company)
                        st.download_button(
                            label="Download ESG Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"Veridex_{report_company}_ESG_{report_year}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.success("Report generated!")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════
# DATA GOVERNANCE
# ════════════════════════════════════════════════════════
elif page == "Data Governance":
    st.title("Data Governance")
    st.divider()

    total_rows   = len(df)
    total_cols   = len(df.columns)
    missing      = df.isnull().sum().sum()
    completeness = round((1 - missing/(total_rows*total_cols))*100, 1)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records",     total_rows)
    c2.metric("Total Columns",     total_cols)
    c3.metric("Companies",         df["Company"].nunique())
    c4.metric("Data Completeness", f"{completeness}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values by Column")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column","Missing"]
        missing_df = missing_df[missing_df["Missing"]>0].sort_values("Missing", ascending=True)
        if missing_df.empty:
            st.success("No missing values!")
        else:
            fig = px.bar(missing_df, x="Missing", y="Column", orientation="h")
            fig.update_traces(marker_color="#f59e0b")
            fig.update_layout(**chart_style())
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Data Coverage by Company")
        coverage = df.groupby("Company")["Year"].count().reset_index()
        coverage.columns = ["Company","Years of Data"]
        coverage = coverage.sort_values("Years of Data", ascending=True)
        fig2 = px.bar(coverage, x="Years of Data", y="Company", orientation="h")
        fig2.update_traces(marker_color="#00d4aa")
        fig2.update_layout(**chart_style())
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View full raw dataset"):
        st.dataframe(df, use_container_width=True)

# ════════════════════════════════════════════════════════
# ADMIN PANEL
# ════════════════════════════════════════════════════════
elif page == "Admin Panel":
    if st.session_state.role != "Admin":
        st.error("Access Denied.")
        st.stop()

    st.title("Admin Panel")
    st.divider()

    USERS = load_users()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Users",     len(USERS))
    c2.metric("Companies",       df["Company"].nunique())
    c3.metric("Total Records",   len(df))
    c4.metric("ML Model Status", "Active", f"R²={ml_metrics['r2']}")

    st.divider()
    st.subheader("User Management")

    for email, info in list(USERS.items()):
        is_admin = info["role"] == "Admin"
        c1,c2,c3,c4 = st.columns([2.5,1.5,1.5,0.8])
        with c1:
            rc = {"Admin":"#8b5cf6","Analyst":"#0ea5e9","Viewer":"#f59e0b"}.get(info["role"],"#64748b")
            st.markdown(f"""
            <div style='padding:8px 0'>
                <div style='font-size:13px;font-weight:600;color:#ffffff'>{info['name']}</div>
                <div style='font-size:11px;color:#64748b'>{email}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='padding:14px 0;font-size:12px;color:{rc};font-weight:600'>{info['role']}</div>", unsafe_allow_html=True)
        with c3:
            if not is_admin:
                new_role = st.selectbox("Role", ["Viewer","Analyst"],
                    index=0 if info["role"]=="Viewer" else 1,
                    key=f"role_{email}", label_visibility="collapsed")
                if new_role != info["role"]:
                    USERS[email]["role"] = new_role
                    save_users(USERS)
                    st.success(f"Updated {info['name']} to {new_role}")
                    st.rerun()
        with c4:
            if not is_admin:
                if st.button("Delete", key=f"del_{email}"):
                    del USERS[email]
                    save_users(USERS)
                    st.success(f"Deleted {info['name']}")
                    st.rerun()
        st.markdown("<hr style='border-color:#0f2040;margin:2px 0'>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Role Permissions")
    for role, pages in ROLE_PAGES.items():
        colour = {"Admin":"#8b5cf6","Analyst":"#0ea5e9","Viewer":"#f59e0b"}[role]
        st.markdown(f"""
        <div style='background:#091528;border:1px solid #0f2040;border-radius:8px;
                    padding:14px;margin-bottom:10px'>
            <div style='color:{colour};font-weight:700;margin-bottom:6px'>{role}</div>
            <div style='color:#94a3b8;font-size:12px'>{" · ".join(pages)}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("ML Model Performance")
    st.dataframe(pd.DataFrame({
        "Metric": ["R² Score","Mean Absolute Error","Training Samples","Test Samples"],
        "Value":  [ml_metrics["r2"], ml_metrics["mae"],
                   ml_metrics["train_samples"], ml_metrics["test_samples"]]
    }), use_container_width=True)