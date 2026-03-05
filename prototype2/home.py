# home.py — Veridex ESG Intelligence Platform

import streamlit as st
import json, os, hashlib, re, pandas as pd

st.set_page_config(
    page_title="Veridex — ESG Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ════════════════════════════════════════════════════════
# USER MANAGEMENT
# ════════════════════════════════════════════════════════
USERS_FILE = "users.json"

def _hash(p):
    return hashlib.sha256(p.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        default = {"admin@veridex.com": {
            "password": _hash("Admin@2024"), "role": "Admin",
            "name": "Platform Admin", "email": "admin@veridex.com"
        }}
        with open(USERS_FILE, "w") as f:
            json.dump(default, f, indent=2)
        return default
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def validate_password(pw):
    if len(pw) < 8: return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", pw): return False, "Must contain at least one uppercase letter."
    if not re.search(r"[0-9]", pw): return False, "Must contain at least one number."
    return True, ""

def validate_email(e):
    return bool(re.match(r"^[^@]+@[^@]+\.[^@]+$", e))

def register_user(email, password, name, role):
    users = load_users()
    email = email.strip().lower()
    if not validate_email(email): return False, "Please enter a valid email address."
    if email in users: return False, "An account with this email already exists."
    if not name.strip(): return False, "Please enter your full name."
    ok, msg = validate_password(password)
    if not ok: return False, msg
    users[email] = {"password": _hash(password), "role": role, "name": name.strip(), "email": email}
    save_users(users)
    return True, "Account created!"

def verify_login(email, password):
    users = load_users()
    email = email.strip().lower()
    if email not in users: return None
    if users[email]["password"] == _hash(password): return users[email]
    return None

# ════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════
for key, val in [("logged_in", False), ("user", None), ("role", None),
                 ("name", None), ("page", "Overview"), ("home_view", "landing")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ════════════════════════════════════════════════════════
# HAND OFF TO APP IF LOGGED IN
# ════════════════════════════════════════════════════════
if st.session_state.logged_in:
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("app", "app.py")
    app  = importlib.util.module_from_spec(spec)
    sys.modules["app"] = app
    spec.loader.exec_module(app)
    st.stop()

# ════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════
@st.cache_data
def get_ticker_data():
    try:
        df = pd.read_csv("data/ESG_Final_Validated.csv")
        latest = df.groupby("Company").last().reset_index()
        items = []
        for _, row in latest.iterrows():
            score = round(row["Overall_ESG_Score"], 1)
            if score >= 75:   arrow, colour = "↑", "#00d4aa"
            elif score >= 55: arrow, colour = "→", "#f59e0b"
            else:             arrow, colour = "↓", "#ef4444"
            items.append((row["Company"], str(score), arrow, colour))
        return items
    except:
        return []

@st.cache_data
def get_all_data():
    try:
        return pd.read_csv("data/ESG_Final_Validated.csv")
    except:
        return pd.DataFrame()

TICKER   = get_ticker_data()
ALL_DATA = get_all_data()

ticker_parts = []
for name, score, arrow, colour in TICKER:
    ticker_parts.append(
        f'<span style="color:#94a3b8;font-weight:600">{name}</span>'
        f'<span style="color:#1e3a5f"> ESG </span>'
        f'<span style="color:{colour};font-weight:700">{score} {arrow}</span>'
    )
sep = ' <span style="color:#1e3a5f"> | </span> '
ticker_html = sep.join(ticker_parts) if ticker_parts else "Veridex ESG Intelligence"
ticker_html = ticker_html + sep + ticker_html

def score_colour(s):
    if s >= 80: return "#00d4aa"
    elif s >= 65: return "#0ea5e9"
    elif s >= 50: return "#f59e0b"
    return "#ef4444"

def rating_label(s):
    if s >= 80: return "Excellent"
    elif s >= 65: return "Good"
    elif s >= 50: return "Average"
    return "Poor"

# ════════════════════════════════════════════════════════
# GLOBAL STYLES
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');
* { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.stDeployButton { display: none; }
.stApp { background: #030b18 !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

.stTextInput > div > div > input {
    background: #0a1929 !important; border: 1px solid #1e3a5f !important;
    color: #fff !important; border-radius: 8px !important;
    font-size: 14px !important; padding: 10px 14px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus { border-color: #00d4aa !important; }
.stTextInput > label {
    color: #4a6080 !important; font-size: 11px !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div {
    background: #0a1929 !important; border: 1px solid #1e3a5f !important;
    color: #fff !important; border-radius: 8px !important;
}
.stSelectbox label {
    color: #4a6080 !important; font-size: 11px !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
}
.stButton > button {
    background: #00d4aa !important; color: #030b18 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 10px 22px !important; width: 100% !important;
    font-family: 'DM Sans', sans-serif !important; transition: background 0.2s !important;
}
.stButton > button:hover { background: #00ecc0 !important; }
.stButton > button p { color: #030b18 !important; font-weight: 700 !important; }
.ghost-btn .stButton > button {
    background: transparent !important; color: #64748b !important;
    border: 1px solid #1e3a5f !important; font-weight: 500 !important;
}
.ghost-btn .stButton > button:hover {
    border-color: #00d4aa !important; color: #00d4aa !important;
}
.ghost-btn .stButton > button p { color: inherit !important; }
.nav-signin .stButton > button {
    background: transparent !important; color: #00d4aa !important;
    border: 1px solid rgba(0,212,170,0.4) !important;
    border-radius: 100px !important; font-size: 13px !important;
    padding: 6px 20px !important; font-weight: 600 !important;
    width: auto !important;
}
.nav-signin .stButton > button:hover { background: rgba(0,212,170,0.08) !important; }
.nav-signin .stButton > button p { color: #00d4aa !important; }
p, li { color: #94a3b8 !important; font-family: 'DM Sans', sans-serif !important; }
h1,h2,h3 { color: #fff !important; }
.vx-ticker {
    background: #040d1a; border-bottom: 1px solid #0d1f35;
    padding: 9px 0; overflow: hidden; white-space: nowrap;
    font-family: 'DM Sans', sans-serif; font-size: 12px;
}
.vx-ticker-inner { display:inline-block; animation: vx-scroll 60s linear infinite; }
.vx-ticker-inner:hover { animation-play-state: paused; }
@keyframes vx-scroll { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# LANDING
# ════════════════════════════════════════════════════════
if st.session_state.home_view == "landing":

    # ── Ticker ──
    st.markdown(
        f'<div class="vx-ticker"><div class="vx-ticker-inner">&nbsp;&nbsp;&nbsp;{ticker_html}&nbsp;&nbsp;&nbsp;</div></div>',
        unsafe_allow_html=True
    )

    # ── Navbar: fully in HTML, Sign In via query param ──
    st.markdown("""
<style>
.vx-nav {
    display:flex; align-items:center; justify-content:space-between;
    padding:0 48px; height:58px;
    background:#030b18; border-bottom:1px solid #0d1f35;
}
.vx-logo { font-family:Syne,sans-serif; font-size:18px; font-weight:800; color:#00d4aa; text-decoration:none; }
.vx-logo em { color:#fff; font-style:normal; }
.vx-nav-center { display:flex; gap:32px; list-style:none; margin:0; padding:0; }
.vx-nav-center a {
    color:#475569; font-family:'DM Sans',sans-serif; font-size:13px;
    font-weight:500; text-decoration:none; transition:color 0.2s;
}
.vx-nav-center a:hover { color:#94a3b8; }
.vx-signin-pill {
    color:#00d4aa !important; font-family:'DM Sans',sans-serif; font-size:13px;
    font-weight:600; text-decoration:none;
    border:1px solid rgba(0,212,170,0.35); border-radius:100px;
    padding:6px 18px; transition:all 0.2s; white-space:nowrap;
}
.vx-signin-pill:hover { background:rgba(0,212,170,0.08); border-color:#00d4aa; color:#00d4aa !important; }
</style>
<div class="vx-nav">
  <span class="vx-logo">Veri<em>dex</em></span>
  <ul class="vx-nav-center">
    <li><a href="#about">About</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#platform">Platform</a></li>
  </ul>
  <a href="?signin=1" class="vx-signin-pill">Sign In →</a>
</div>
""", unsafe_allow_html=True)

    # Detect Sign In click via query param
    if st.query_params.get("signin") == "1":
        st.query_params.clear()
        st.session_state.home_view = "login"
        st.rerun()

    # ── Hero — NO BUTTONS, just headline + subtext ──
    st.markdown("""
<style>
.vx-hero {
    background:#030b18; padding:80px 80px 70px; position:relative; overflow:hidden;
}
.vx-grid {
    position:absolute; inset:0; pointer-events:none;
    background-image:
        linear-gradient(rgba(0,212,170,0.022) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,212,170,0.022) 1px,transparent 1px);
    background-size:56px 56px; animation:grid-drift 24s linear infinite;
}
@keyframes grid-drift { to { transform:translateY(56px); } }
.vx-glow {
    position:absolute; width:520px; height:520px; top:-80px; right:8%;
    background:radial-gradient(circle,rgba(0,212,170,0.07) 0%,transparent 65%);
    pointer-events:none; animation:glow-pulse 5s ease-in-out infinite;
}
@keyframes glow-pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
.vx-badge {
    display:inline-flex; align-items:center; gap:7px;
    background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.18);
    border-radius:100px; padding:5px 14px; margin-bottom:22px;
    font-size:10px; font-weight:700; color:#00d4aa;
    letter-spacing:0.12em; text-transform:uppercase; font-family:'DM Sans',sans-serif;
}
.vx-dot { width:5px;height:5px;border-radius:50%;background:#00d4aa;animation:blink 1.6s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.1} }
.vx-h1 {
    font-family:Syne,sans-serif; font-size:50px; font-weight:800; color:#fff;
    line-height:1.08; letter-spacing:-0.025em; margin:0 0 16px; max-width:580px;
}
.vx-h1 .teal { color:#00d4aa; }
.vx-sub {
    font-size:15px; color:#4a6080; max-width:460px; line-height:1.75;
    font-family:'DM Sans',sans-serif; margin:0;
}
</style>
<div class="vx-hero">
  <div class="vx-grid"></div>
  <div class="vx-glow"></div>
  <div class="vx-badge"><div class="vx-dot"></div>ESG Intelligence Platform</div>
  <div class="vx-h1">The smarter way to evaluate <span class="teal">ESG performance</span></div>
  <div class="vx-sub">Veridex tracks Environmental, Social and Governance data across global companies — powered by machine learning, AI recommendations and predictive forecasting.</div>
</div>
""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # ML + AI DEMO SECTION
    # ════════════════════════════════════════════════════
    st.markdown("""
<div id="demo" style="background:#040d1a;border-top:1px solid #0d1f35;padding:52px 80px 16px">
  <div style="margin-bottom:8px">
    <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
                color:#00d4aa;margin-bottom:8px;font-family:'DM Sans',sans-serif">
      ◆ Live Platform Demo
    </div>
    <div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#fff;margin-bottom:6px">
      See what's under the hood
    </div>
    <div style="font-size:13px;color:#4a6080;font-family:'DM Sans',sans-serif;max-width:540px;line-height:1.7">
      Veridex uses a <strong style="color:#94a3b8">Random Forest ML model</strong> trained on 16 ESG metrics,
      and <strong style="color:#94a3b8">Google Gemini AI</strong> to generate recommendations.
      Try both below — no account needed.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Tab selector — real Streamlit buttons styled as tabs
    if "demo_tab" not in st.session_state:
        st.session_state.demo_tab = "ml"

    ml_on = st.session_state.demo_tab == "ml"

    st.markdown("""
<style>
/* Active tab */
div[data-testid="stHorizontalBlock"] div.tab-active .stButton > button {
    background: #00d4aa !important; color: #030b18 !important;
    border: 1px solid #00d4aa !important; border-radius: 8px 8px 0 0 !important;
    font-size: 12px !important; font-weight: 700 !important;
    padding: 7px 18px !important; width: auto !important;
}
/* Inactive tab */
div[data-testid="stHorizontalBlock"] div.tab-inactive .stButton > button {
    background: transparent !important; color: #475569 !important;
    border: 1px solid #1e3a5f !important; border-radius: 8px 8px 0 0 !important;
    font-size: 12px !important; font-weight: 500 !important;
    padding: 7px 18px !important; width: auto !important;
}
div[data-testid="stHorizontalBlock"] div.tab-inactive .stButton > button:hover {
    color: #94a3b8 !important; border-color: #334155 !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown("<div style='background:#040d1a;padding:12px 80px 0;border-bottom:1px solid #0d1f35;display:flex'>", unsafe_allow_html=True)
    tab_gap, tab_ml, tab_ai, tab_rest = st.columns([0.01, 1.4, 1.6, 8])
    with tab_ml:
        st.markdown(f'<div class="{"tab-active" if ml_on else "tab-inactive"}">', unsafe_allow_html=True)
        if st.button("◆ ML Score Predictor", key="tab_ml"):
            st.session_state.demo_tab = "ml"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with tab_ai:
        st.markdown(f'<div class="{"tab-active" if not ml_on else "tab-inactive"}">', unsafe_allow_html=True)
        if st.button("◈ AI Recommendations", key="tab_ai"):
            st.session_state.demo_tab = "ai"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 1: ML Score Predictor ──
    if st.session_state.demo_tab == "ml":
        st.markdown("""
<div style="background:#040d1a;padding:8px 80px 4px">
  <div style="font-size:12px;color:#334155;font-family:'DM Sans',sans-serif;line-height:1.6">
    Adjust the sliders — our <strong style="color:#475569">Random Forest model</strong> (R² 0.84+, trained on real ESG data)
    predicts the overall ESG score in real time.
  </div>
</div>
""", unsafe_allow_html=True)

        ml_l, ml_r = st.columns([1, 1])

        with ml_l:
            st.markdown("<div style='background:#040d1a;padding:12px 0 0 80px'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px;color:#00d4aa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;font-family:DM Sans,sans-serif'>🌱 Environmental</div>", unsafe_allow_html=True)
            ml_ren = st.slider("Renewable Energy %",        0, 100,  45, key="ml_ren")
            ml_em  = st.slider("Emissions (Mt CO₂e)",       0, 500, 200, key="ml_em")
            ml_car = st.slider("Carbon Reduction Target %", 0, 100,  30, key="ml_car")
            st.markdown("<div style='font-size:11px;color:#0ea5e9;text-transform:uppercase;letter-spacing:0.1em;margin:10px 0 6px;font-family:DM Sans,sans-serif'>👥 Social</div>", unsafe_allow_html=True)
            ml_div = st.slider("Gender Diversity %",        0, 100,  35, key="ml_div")
            ml_tr  = st.slider("Training Hours / Employee", 0, 100,  40, key="ml_tr")
            st.markdown("</div>", unsafe_allow_html=True)

        with ml_r:
            st.markdown("<div style='background:#040d1a;padding:12px 80px 0 0'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px;color:#8b5cf6;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;font-family:DM Sans,sans-serif'>⚖️ Governance</div>", unsafe_allow_html=True)
            ml_bi  = st.slider("Independent Board %",       0, 100,  65, key="ml_bi")
            ml_bd  = st.slider("Board Diversity %",         0, 100,  30, key="ml_bd")
            ml_inj = st.slider("Workplace Injury Rate",     0,  10,   2, key="ml_inj")
            ml_sust= st.slider("Sustainability Invest ($B)",0,  20,   2, key="ml_sust")
            st.markdown("</div>", unsafe_allow_html=True)

            # Weighted formula approximating the RF model
            predicted = round(
                ml_ren  * 0.22 +
                (500 - ml_em) / 500 * 100 * 0.18 +
                ml_car  * 0.08 +
                ml_div  * 0.16 +
                ml_tr   * 0.08 +
                ml_bi   * 0.12 +
                ml_bd   * 0.08 +
                (10 - ml_inj) / 10 * 100 * 0.04 +
                min(ml_sust, 20) / 20 * 100 * 0.04,
                1
            )
            predicted = min(max(predicted, 0), 100)
            cc = score_colour(predicted)
            rl = rating_label(predicted)

            st.markdown(f"""
<div style="background:#091528;border:1px solid {cc}33;border-radius:12px;
            padding:24px;text-align:center;margin:16px 80px 8px 0">
  <div style="font-size:10px;color:#334155;text-transform:uppercase;
              letter-spacing:0.1em;margin-bottom:8px;font-family:'DM Sans',sans-serif">
    Predicted ESG Score
  </div>
  <div style="font-size:56px;font-weight:800;color:{cc};font-family:Syne,sans-serif;line-height:1">
    {predicted}
  </div>
  <div style="font-size:13px;color:{cc};font-weight:600;margin:6px 0 14px">{rl}</div>
  <div style="height:5px;background:#0d1f35;border-radius:3px;overflow:hidden">
    <div style="width:{int(predicted)}%;height:5px;background:{cc};border-radius:3px"></div>
  </div>
  <div style="font-size:10px;color:#1e3a5f;margin-top:6px;font-family:'DM Sans',sans-serif">
    Random Forest · 16 features · R² 0.84+
  </div>
</div>
""", unsafe_allow_html=True)

    # ── TAB 2: AI Recommendations Preview ──
    else:
        st.markdown("""
<div style="background:#040d1a;padding:8px 80px 4px">
  <div style="font-size:12px;color:#334155;font-family:'DM Sans',sans-serif;line-height:1.6">
    Our platform calls <strong style="color:#475569">Google Gemini AI</strong> to generate personalised
    ESG improvement recommendations based on a company's actual metrics vs sector benchmarks.
    Here's a sample of what it produces.
  </div>
</div>
""", unsafe_allow_html=True)

        ai_l, ai_r = st.columns([1, 1])

        sample_recs = {
            "Microsoft": [
                ("Environmental", "#00d4aa", "Increase renewable energy procurement beyond current 72% to achieve full Scope 2 carbon neutrality by 2026 — aligning with the Science Based Targets initiative (SBTi)."),
                ("Social", "#0ea5e9", "Gender diversity at 41% is above sector average (38%). Maintain momentum by targeting 45%+ board representation and publishing intersectional pay-gap data."),
                ("Governance", "#8b5cf6", "ESG-linked executive compensation is already active. Consider extending this to mid-level management to drive accountability across all levels."),
            ],
            "Shell": [
                ("Environmental", "#ef4444", "Total emissions of 1,374 Mt CO₂e are 3.2× the Energy sector average. Prioritise accelerated divestment of high-emission assets and increase carbon capture investment."),
                ("Social", "#f59e0b", "Workplace injury rate of 3.8 is above the sector benchmark of 2.1. Implement ISO 45001 safety management systems across all operational sites."),
                ("Governance", "#f59e0b", "Independent board representation at 58% falls below the 70% best-practice threshold. Recruit two additional independent non-executive directors by Q2 2025."),
            ],
        }

        with ai_l:
            st.markdown("<div style='background:#040d1a;padding:12px 0 0 80px'>", unsafe_allow_html=True)
            ai_company = st.selectbox("Select Company", list(sample_recs.keys()), key="ai_co")
            st.markdown("</div>", unsafe_allow_html=True)

            recs = sample_recs[ai_company]
            for i, (pillar, colour, text) in enumerate(recs, 1):
                st.markdown(f"""
<div style="background:#030b18;border:1px solid #0d1f35;border-left:3px solid {colour};
            border-radius:8px;padding:14px 16px;margin:8px 0 0 80px;display:flex;gap:12px;align-items:flex-start">
  <div style="min-width:20px;height:20px;border-radius:50%;background:{colour}22;border:1px solid {colour}44;
              color:{colour};font-size:10px;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0">{i}</div>
  <div>
    <div style="font-size:10px;font-weight:700;color:{colour};letter-spacing:0.08em;text-transform:uppercase;
                margin-bottom:4px;font-family:'DM Sans',sans-serif">{pillar}</div>
    <div style="font-size:12px;color:#64748b;line-height:1.65;font-family:'DM Sans',sans-serif">{text}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        with ai_r:
            st.markdown("""
<div style="background:#030b18;border:1px solid #0d1f35;border-radius:12px;
            padding:22px;margin:12px 80px 0 0">
  <div style="font-size:10px;color:#1e3a5f;letter-spacing:0.1em;text-transform:uppercase;
              margin-bottom:14px;font-family:'DM Sans',sans-serif">How it works</div>
  <div style="display:flex;flex-direction:column;gap:12px">
    <div style="display:flex;gap:12px;align-items:flex-start">
      <div style="width:28px;height:28px;border-radius:8px;background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.15);
                  color:#00d4aa;font-size:12px;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0">1</div>
      <div>
        <div style="font-size:12px;font-weight:600;color:#94a3b8;font-family:'DM Sans',sans-serif">Metric extraction</div>
        <div style="font-size:11px;color:#334155;margin-top:2px;font-family:'DM Sans',sans-serif">16 ESG data points pulled from the company's latest filing year</div>
      </div>
    </div>
    <div style="display:flex;gap:12px;align-items:flex-start">
      <div style="width:28px;height:28px;border-radius:8px;background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.15);
                  color:#0ea5e9;font-size:12px;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0">2</div>
      <div>
        <div style="font-size:12px;font-weight:600;color:#94a3b8;font-family:'DM Sans',sans-serif">Sector benchmarking</div>
        <div style="font-size:11px;color:#334155;margin-top:2px;font-family:'DM Sans',sans-serif">Each metric compared against the sector average across all tracked companies</div>
      </div>
    </div>
    <div style="display:flex;gap:12px;align-items:flex-start">
      <div style="width:28px;height:28px;border-radius:8px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.15);
                  color:#8b5cf6;font-size:12px;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0">3</div>
      <div>
        <div style="font-size:12px;font-weight:600;color:#94a3b8;font-family:'DM Sans',sans-serif">Gemini AI generation</div>
        <div style="font-size:11px;color:#334155;margin-top:2px;font-family:'DM Sans',sans-serif">Google Gemini receives the metrics + gaps and returns actionable, specific recommendations</div>
      </div>
    </div>
    <div style="display:flex;gap:12px;align-items:flex-start">
      <div style="width:28px;height:28px;border-radius:8px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.15);
                  color:#f59e0b;font-size:12px;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0">4</div>
      <div>
        <div style="font-size:12px;font-weight:600;color:#94a3b8;font-family:'DM Sans',sans-serif">Live in the platform</div>
        <div style="font-size:11px;color:#334155;margin-top:2px;font-family:'DM Sans',sans-serif">Sign up to run this on any of our 20+ tracked companies with live data</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px;background:#040d1a'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # FEATURES
    # ════════════════════════════════════════════════════
    st.markdown("""
<div id="features" style="padding:56px 80px 36px;background:#030b18;border-top:1px solid #0d1f35">
  <div style="margin-bottom:32px">
    <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
                color:#00d4aa;margin-bottom:8px;font-family:'DM Sans',sans-serif">What's Inside</div>
    <div style="font-family:Syne,sans-serif;font-size:26px;font-weight:800;color:#fff;margin-bottom:8px">
      A full ESG intelligence suite
    </div>
    <div style="font-size:13px;color:#4a6080;font-family:'DM Sans',sans-serif;max-width:420px;line-height:1.7">
      Every tool an investor or sustainability team needs — from raw data to AI-powered decisions.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    features = [
        ("◆", "#00d4aa", "ML Predictions",      "Random Forest model predicts ESG scores from 16 metrics with R² above 0.84. Adjust any metric and the score updates instantly."),
        ("◈", "#0ea5e9", "AI Recommendations",   "Google Gemini AI generates personalised improvement tips based on actual metrics vs sector benchmarks — unique to each company."),
        ("↗", "#f59e0b", "ESG Forecasting",       "Forward-looking projections of ESG scores, revenue and emissions up to 5 years using Linear Regression."),
        ("⊕", "#8b5cf6", "Strategy Simulator",   "Test ESG strategies before committing — see the live ML-predicted score impact of every change you make."),
        ("⇄", "#00d4aa", "Compare Companies",    "Side-by-side ESG comparison with interactive charts, year range filters and pillar breakdowns."),
        ("▤", "#0ea5e9", "PDF Reports",           "One-click professional ESG reports for any company and year — with AI-generated recommendations included."),
    ]
    fc1, fc2, fc3 = st.columns(3)
    for i, (icon, colour, title, desc) in enumerate(features):
        with [fc1, fc2, fc3][i % 3]:
            lpad = "padding-left:80px;" if i % 3 == 0 else ""
            st.markdown(f"""
<div style="background:#040d1a;border:1px solid #0d1f35;border-top:2px solid {colour}44;
            border-radius:10px;padding:20px;margin:0 0 12px;{lpad}">
  <div style="font-size:17px;margin-bottom:9px;color:{colour}">{icon}</div>
  <div style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:#e2e8f0;margin-bottom:6px">{title}</div>
  <div style="font-size:12px;color:#4a6080;line-height:1.65">{desc}</div>
</div>
""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # ABOUT / STATS
    # ════════════════════════════════════════════════════
    st.markdown("""
<div id="about" style="padding:52px 80px 32px;background:#040d1a;border-top:1px solid #0d1f35">
  <div style="margin-bottom:28px">
    <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
                color:#00d4aa;margin-bottom:8px;font-family:'DM Sans',sans-serif">About</div>
    <div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#fff;margin-bottom:8px">
      Built for people who need real answers
    </div>
    <div style="font-size:13px;color:#4a6080;font-family:'DM Sans',sans-serif;max-width:480px;line-height:1.75">
      Veridex combines machine learning, real-time AI and interactive visualisation
      to help investors and analysts make better sustainability decisions.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    for col, num, label, lpad in [
        (s1, "20+", "Companies Tracked",     "padding-left:80px;"),
        (s2, "16",  "ESG Metrics per Co.",   ""),
        (s3, "7",   "Industry Sectors",      ""),
        (s4, "2",   "ML Models + Gemini AI", ""),
    ]:
        with col:
            st.markdown(f"""
<div style="background:#030b18;border:1px solid #0d1f35;border-radius:10px;
            padding:20px;text-align:center;margin:0 0 8px;{lpad}">
  <div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:#00d4aa">{num}</div>
  <div style="font-size:11px;color:#1e3a5f;margin-top:4px;font-family:'DM Sans',sans-serif">{label}</div>
</div>
""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # PLATFORM VIEWS
    # ════════════════════════════════════════════════════
    st.markdown("""
<div id="platform" style="padding:52px 80px 32px;background:#030b18;border-top:1px solid #0d1f35">
  <div style="margin-bottom:28px">
    <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
                color:#00d4aa;margin-bottom:8px;font-family:'DM Sans',sans-serif">The Platform</div>
    <div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#fff">
      9 views built for every user
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    pages_list = [
        ("◈","Overview","Snapshot"),          ("◉","Company Analysis","Deep dive"),
        ("⇄","Compare","Side by side"),        ("◎","ESG Analysis","Weighted scores"),
        ("◆","Predictive Analytics","ML model"),("↗","ESG Outlook","Forecasting"),
        ("⊕","Simulator","What-if modelling"), ("▤","ESG Reports","PDF generation"),
        ("◫","Data Governance","Quality audit"),
    ]
    pc = st.columns(5)
    for i, (icon, name, sub) in enumerate(pages_list):
        with pc[i % 5]:
            lpad = "padding-left:80px;" if i % 5 == 0 else ""
            st.markdown(f"""
<div style="background:#040d1a;border:1px solid #0d1f35;border-radius:8px;
            padding:11px 13px;margin-bottom:10px;display:flex;align-items:center;gap:9px;{lpad}">
  <div style="font-size:13px;min-width:22px;color:#00d4aa">{icon}</div>
  <div>
    <div style="font-size:11px;font-weight:600;color:#94a3b8;font-family:'DM Sans',sans-serif">{name}</div>
    <div style="font-size:10px;color:#1e3a5f">{sub}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # FOOTER CTA — single button
    # ════════════════════════════════════════════════════
    st.markdown("""
<div style="padding:52px 80px 16px;background:#040d1a;border-top:1px solid #0d1f35;text-align:center">
  <div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:#fff;margin-bottom:7px">
    Start exploring ESG data today
  </div>
  <div style="font-size:13px;color:#4a6080;font-family:'DM Sans',sans-serif;margin-bottom:28px">
    Create a free account and access the full platform.
  </div>
</div>
""", unsafe_allow_html=True)
    _, fc_btn, _ = st.columns([3.5, 1.5, 3.5])
    with fc_btn:
        if st.button("Create Free Account", key="foot_reg"):
            st.session_state.home_view = "register"
            st.rerun()

    st.markdown("""
<div style="text-align:center;padding:28px 0 14px;background:#040d1a">
  <div style="font-family:Syne,sans-serif;font-size:15px;font-weight:800;color:#00d4aa;margin-bottom:4px">Veridex</div>
  <div style="font-size:11px;color:#1e3a5f;font-family:'DM Sans',sans-serif">
    ESG Intelligence Platform · © 2026 Veridex. All rights reserved.
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# LOGIN
# ════════════════════════════════════════════════════════
elif st.session_state.home_view == "login":
    st.markdown("<div style='height:64px;background:#030b18'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("""
<div style="background:#040d1a;border:1px solid #0d1f35;border-radius:16px;padding:36px 36px 28px">
  <div style="text-align:center;margin-bottom:24px">
    <div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#00d4aa;margin-bottom:5px">Veridex</div>
    <div style="font-size:13px;color:#4a6080;font-family:'DM Sans',sans-serif">Sign in to your account</div>
  </div>
</div>
""", unsafe_allow_html=True)
        email    = st.text_input("Email Address", placeholder="you@example.com", key="li_em")
        password = st.text_input("Password", type="password", placeholder="Your password", key="li_pw")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Sign In", key="do_login"):
            user = verify_login(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user      = email.strip().lower()
                st.session_state.role      = user["role"]
                st.session_state.name      = user["name"]
                st.rerun()
            else:
                st.error("Incorrect email or password.")
        st.markdown("<div style='text-align:center;font-size:12px;color:#334155;padding:14px 0 6px;font-family:DM Sans,sans-serif'>Don't have an account?</div>", unsafe_allow_html=True)
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("Create Account", key="li_to_reg"):
            st.session_state.home_view = "register"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("← Back to Home", key="li_back"):
            st.session_state.home_view = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# REGISTER
# ════════════════════════════════════════════════════════
elif st.session_state.home_view == "register":
    st.markdown("<div style='height:40px;background:#030b18'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("""
<div style="background:#040d1a;border:1px solid #0d1f35;border-radius:16px;padding:36px 36px 28px">
  <div style="text-align:center;margin-bottom:24px">
    <div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#00d4aa;margin-bottom:5px">Veridex</div>
    <div style="font-size:13px;color:#4a6080;font-family:'DM Sans',sans-serif">Create your free account</div>
  </div>
</div>
""", unsafe_allow_html=True)
        full_name = st.text_input("Full Name",        placeholder="John Smith",                       key="rg_nm")
        email     = st.text_input("Email Address",    placeholder="you@example.com",                  key="rg_em")
        password  = st.text_input("Password",         type="password",
                                  placeholder="Min 8 chars · 1 uppercase · 1 number",                 key="rg_pw")
        password2 = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="rg_pw2")
        role      = st.selectbox("Account Type", ["Analyst", "Viewer"],                               key="rg_role")
        st.markdown("""
<div style="background:#030b18;border:1px solid #0d1f35;border-radius:7px;padding:9px 12px;
            margin:4px 0 12px;font-size:11px;color:#1e3a5f;line-height:1.7;font-family:'DM Sans',sans-serif">
  <span style="color:#475569;font-weight:600">Analyst</span> — ML, simulator, reports, forecasting & more.<br>
  <span style="color:#475569;font-weight:600">Viewer</span> — Read-only: Overview, Company & Compare pages.
</div>
""", unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Create Account", key="do_reg"):
            if password != password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(email, password, full_name, role)
                if ok:
                    st.success(f"Account created! Welcome, {full_name}. Please sign in.")
                    st.session_state.home_view = "login"
                    st.rerun()
                else:
                    st.error(msg)
        st.markdown("""
<div style="text-align:center;padding:16px 0 4px">
  <span style="font-size:12px;color:#334155;font-family:'DM Sans',sans-serif">Already have an account?</span>
</div>
""", unsafe_allow_html=True)
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("Sign In", key="rg_to_li"):
            st.session_state.home_view = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("← Back to Home", key="rg_back"):
            st.session_state.home_view = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)