import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CogniBurn · Burnout Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS — dark editorial, sharp typography, real depth
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── ROOT ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2028;
    --border-hi: #2e3140;
    --text:      #e8eaf0;
    --muted:     #7c8099;
    --accent:    #ff5a36;
    --accent2:   #ffb84d;
    --accent3:   #38d9a9;
    --danger:    #ff4c6a;
    --warn:      #ffbb33;
    --safe:      #34d399;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --radius:    10px;
    --shadow:    0 4px 24px rgba(0,0,0,.45);
}

/* ── APP SHELL ── */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 60% at 70% -10%, rgba(255,90,54,.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 0% 80%, rgba(56,217,169,.05) 0%, transparent 55%);
    font-family: var(--font-body);
    color: var(--text);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}

/* hide default streamlit elements */
#MainMenu, footer{ visibility: hidden; }
[data-testid="stToolbar"] {
    display: block;
}

/* ── TYPOGRAPHY ── */
h1 {
    font-family: var(--font-head) !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-family: var(--font-head) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
}
h3 {
    font-family: var(--font-head) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.01em !important;
}
p, li, label, .stMarkdown {
    font-family: var(--font-body) !important;
    color: var(--muted) !important;
}

/* ── METRIC CARDS ── */
[data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -0.03em !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
}

/* ── CUSTOM CARD COMPONENTS ── */
.burn-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.burn-card:hover { border-color: var(--border-hi); }
.burn-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    opacity: 0;
    transition: opacity .2s;
}
.burn-card:hover::before { opacity: 1; }

.stat-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    background: rgba(255,255,255,.05);
    color: var(--muted);
    border: 1px solid var(--border);
    margin-bottom: 8px;
}

.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
    display: block;
}

.big-stat {
    font-family: var(--font-head);
    font-size: 3rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.04em;
    line-height: 1;
}

.sub-stat {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 4px;
}

.divider {
    height: 1px;
    background: var(--border);
    margin: 24px 0;
}

.tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.tag-red   { background: rgba(255,76,106,.12); color: #ff4c6a; border: 1px solid rgba(255,76,106,.25); }
.tag-amber { background: rgba(255,187,51,.12); color: #ffbb33; border: 1px solid rgba(255,187,51,.25); }
.tag-green { background: rgba(52,211,153,.12); color: #34d399; border: 1px solid rgba(52,211,153,.25); }
.tag-blue  { background: rgba(99,179,237,.12); color: #63b3ed; border: 1px solid rgba(99,179,237,.25); }

/* ── PREDICTION RESULT ── */
.result-hero {
    border-radius: 16px;
    padding: 40px 32px;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.result-hero .result-level {
    font-family: var(--font-head);
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1;
}
.result-hero .result-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: .7;
    margin-top: 8px;
}
.result-hero .result-score {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    margin-top: 12px;
    opacity: .85;
}

/* ── INSIGHT ROW ── */
.insight-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: border-color .15s;
}
.insight-row:hover { border-color: var(--border-hi); }
.insight-title {
    font-family: var(--font-head);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
}
.insight-body {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.5;
}

/* ── SIDEBAR NAV ── */
.nav-logo {
    padding: 28px 24px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}
.nav-logo .logo-mark {
    font-family: var(--font-head);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
}
.nav-logo .logo-sub {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── BUTTONS ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01em !important;
    padding: 14px 32px !important;
    transition: opacity .2s, transform .15s !important;
    box-shadow: 0 0 0 0 rgba(255,90,54,0) !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}

/* ── FORM INPUTS ── */
div[data-baseweb="slider"] div:first-child {
    background: #2a2e3b !important;
    height: 4px !important;
    border-radius: 6px !important;
}



/* Thumb (small & clean) */
div[data-baseweb="slider"] [role="slider"] {
    background: #ffffff !important;
    border: 2px solid #ff5a36 !important;
    height: 12px !important;
    width: 12px !important;
    box-shadow: none !important;
}

/* Value label */
.stSlider [data-testid="stThumbValue"] {
    background: transparent !important;
    color: #ff5a36 !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
}

.stSelectbox > label, .stSlider > label, .stMultiSelect > label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    color: var(--muted) !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    letter-spacing: 0.01em !important;
    transition: all .15s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] iframe { border-radius: 8px !important; }

/* ── RADIO (nav) ── */
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--font-body) !important;
    color: var(--muted) !important;
    font-size: 0.9rem !important;
    padding: 8px 0 !important;
    transition: color .15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--text) !important; }

/* ── HORIZONTAL RULE ── */
hr { border-color: var(--border) !important; }

/* ── SELECT SLIDER ── */
.stSelectSlider > label {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PLOTLY THEME  (dark, matches the app)
# ─────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    plot_bgcolor  = "#111318",
    paper_bgcolor = "#111318",
    font          = dict(family="DM Sans, sans-serif", color="#7c8099", size=12),
    title_font    = dict(family="Syne, sans-serif", color="#e8eaf0", size=16, weight=700),
    xaxis         = dict(gridcolor="#1e2028", zerolinecolor="#1e2028",
                         tickfont=dict(family="JetBrains Mono", size=10, color="#7c8099")),
    yaxis         = dict(gridcolor="#1e2028", zerolinecolor="#1e2028",
                         tickfont=dict(family="JetBrains Mono", size=10, color="#7c8099")),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7c8099", size=11)),
    margin        = dict(l=40, r=24, t=52, b=40),
    colorway      = ["#ff5a36", "#ffb84d", "#38d9a9", "#63b3ed", "#9f7aea"],
)

PALETTE = dict(low="#34d399", mid="#ffbb33", high="#ff4c6a")
BURNOUT_COLORS = [PALETTE["low"], PALETTE["mid"], PALETTE["high"]]


def plot_cfg(fig, height=380, title="", showlegend=True):
    cfg = dict(PLOT_LAYOUT)
    cfg["height"] = height
    if title:
        cfg["title"] = dict(text=title, font=dict(family="Syne,sans-serif",
                            color="#e8eaf0", size=15), x=0, xanchor="left", pad=dict(l=4, b=12))
    fig.update_layout(**cfg, showlegend=showlegend)
    return fig


# ─────────────────────────────────────────────────────────────────
# DATA / MODEL LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("models/burnout_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except Exception:
        return None, None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("datasets/burnout_final_dataset.csv",index_col=0)
    except Exception:
        return None


model, scaler = load_model()
df = load_data()

LEVEL_MAP   = {"Very Low": 0, "Low": 0.25, "Neutral": 0.5, "High": 0.75, "Very High": 1}
QUALITY_MAP = {"Awful": 0, "Bad": 0.25, "Neutral": 0.5, "Good": 0.75, "Excellent": 1}

FEATURE_LABELS = [
    "Study Time", "Social Media", "Sleep Quality", "Headache Freq",
    "Study Load", "Extracurricular", "Screen Time", "Sleep Hours",
    "Physical Activity", "Exam Anxiety", "Stress Level", "Mental Strain",
    "Recovery Score", "Performance Pressure", "Lifestyle Stress"
]


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="nav-logo">
        <div class="logo-mark">CogniBurn</div>
        <div class="logo-sub">Burnout Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["Overview", "EDA & Analytics", "Feature Relationships", "Model Insights", "Risk Predictor"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    status_model = "● Live" if model  is not None else "● Offline"
    status_data  = "● Loaded" if df   is not None else "● Missing"
    color_m = "#34d399" if model is not None else "#ff4c6a"
    color_d = "#34d399" if df    is not None else "#ff4c6a"
    n_rows  = len(df) if df is not None else 0
    n_cols  = len(df.columns) if df is not None else 0

    st.markdown(f"""
    <div style='padding:0 8px;'>
        <div style='margin-bottom:12px;'>
            <span class='section-label'>System Status</span>
        </div>
        <div style='display:flex;flex-direction:column;gap:8px;'>
            <div class='burn-card' style='padding:12px 16px;'>
                <span style='font-family:var(--font-mono);font-size:.65rem;
                             color:var(--muted);letter-spacing:.1em;text-transform:uppercase;'>Model</span>
                <div style='color:{color_m};font-family:var(--font-mono);font-size:.82rem;
                            font-weight:600;margin-top:4px;'>{status_model}</div>
            </div>
            <div class='burn-card' style='padding:12px 16px;'>
                <span style='font-family:var(--font-mono);font-size:.65rem;
                             color:var(--muted);letter-spacing:.1em;text-transform:uppercase;'>Dataset</span>
                <div style='color:{color_d};font-family:var(--font-mono);font-size:.82rem;
                            font-weight:600;margin-top:4px;'>{status_data}</div>
                <div style='color:var(--muted);font-family:var(--font-mono);font-size:.62rem;
                            margin-top:2px;'>{n_rows:,} rows · {n_cols} cols</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='padding:0 8px;'>
        <span style='font-family:var(--font-mono);font-size:.62rem;color:var(--muted);
                     letter-spacing:.08em;'>{datetime.now().strftime('%d %b %Y  ·  %H:%M')}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPER — page header
# ─────────────────────────────────────────────────────────────────
def page_header(label: str, title: str, subtitle: str = ""):
    st.markdown(f"<span class='section-label'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='font-size:.95rem;color:var(--muted);margin-top:4px;margin-bottom:0;'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "Overview":
    page_header("01 / OVERVIEW", "Burnout Intelligence", 
                "Population-level snapshot of academic stress, sleep quality, and burnout risk distribution.")

    if df is None:
        st.error("Dataset not found — place `burnout_final_dataset.csv` inside `datasets/`.")
        st.stop()

    # ── KPI ROW ──────────────────────────────────────────────
    col = "burnout_level"

    # convert everything to string
    levels = df[col].astype(str).str.strip().str.lower()

    # 🔥 map numeric to labels
    levels = levels.replace({
        "0": "low",
        "1": "medium",
        "2": "high"
    })

    # now count
    low_risk  = (levels == "low").sum()
    mid_risk  = (levels == "medium").sum()
    high_risk = (levels == "high").sum()
    avg_stress = df['stress_level'].mean() if 'stress_level' in df.columns else 0
    avg_sleep  = df['sleep_hours'].mean()  if 'sleep_hours'  in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)

    # TOTAL
    with c1:
        st.markdown(f"""
        <div class='burn-card'>
            <span class='section-label'>Total Records</span>
            <div class='big-stat'>{len(df):,}</div>
            <div class='sub-stat'>dataset size</div>
        </div>""", unsafe_allow_html=True)

    # LOW
    with c2:
        st.markdown(f"""
        <div class='burn-card'>
            <span class='section-label'>Low Risk</span>
            <div class='big-stat' style='color:#34d399;'>{low_risk}</div>
            <div class='sub-stat'>{low_risk/len(df)*100:.1f}% of population</div>
        </div>""", unsafe_allow_html=True)

    # MEDIUM
    with c3:
        st.markdown(f"""
        <div class='burn-card'>
            <span class='section-label'>Medium Risk</span>
            <div class='big-stat' style='color:#ffbb33;'>{mid_risk}</div>
            <div class='sub-stat'>{mid_risk/len(df)*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # HIGH
    with c4:
        st.markdown(f"""
        <div class='burn-card'>
            <span class='section-label'>High Risk</span>
            <div class='big-stat' style='color:#ff4c6a;'>{high_risk}</div>
            <div class='sub-stat'>{high_risk/len(df)*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)


    with c1:
        st.metric("Avg Stress Score", f"{avg_stress:.2f}")

    with c2:
        st.metric("Avg Sleep Hours", f"{avg_sleep:.1f} h")

    # ── CHARTS ROW ───────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        if 'burnout_level' in df.columns:

            counts = df['burnout_level'].value_counts().sort_index()

        # 🔥 FIX: ALWAYS HANDLE BOTH CASES SAFELY
            if isinstance(counts.index[0], str):
                x_vals = counts.index.tolist()
            else:
                label_map = {0: "Low", 1: "Medium", 2: "High"}
                x_vals = [label_map.get(i, str(i)) for i in counts.index]

            fig = go.Figure(go.Bar(
                x=x_vals,
                y=counts.values,
                marker_color=BURNOUT_COLORS,
                text=counts.values,
                textposition='outside'
            ))

            fig = plot_cfg(fig, 360, "Burnout Level Distribution", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if 'burnout_level' in df.columns:

        # 🔥 FIX: SAME LOGIC AS BAR CHART
            if isinstance(counts.index[0], str):
                pie_labels = counts.index.tolist()
            else:
                label_map = {0: "Low", 1: "Medium", 2: "High"}
                pie_labels = [label_map.get(i, str(i)) for i in counts.index]

            fig = go.Figure(go.Pie(
                labels=pie_labels,
                values=counts.values,
                marker_colors=BURNOUT_COLORS,
                hole=0.6,
                textinfo='percent',
                textfont=dict(family="JetBrains Mono", size=10),
                hovertemplate="<b>%{label}</b><br>%{value} cases<br>%{percent}<extra></extra>",
                pull=[0.02, 0.02, 0.06]
            ))

            fig = plot_cfg(fig, 360, "Risk Split", showlegend=True)
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                annotations=[dict(text=f"<b>{len(df):,}</b>", x=0.5, y=0.5,
                              font_size=22, font_family="Syne", font_color="#e8eaf0",
                              showarrow=False)]
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── TREND PANELS ─────────────────────────────────────────
    st.markdown("<h3 style='margin-bottom:16px;'>Feature Snapshot</h3>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["💤  Sleep", "📚  Study & Activity", "😰  Stress Indicators"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if 'sleep_hours' in df.columns:
                fig = px.histogram(df, x='sleep_hours', nbins=35,
                                   color_discrete_sequence=["#63b3ed"])
                fig.update_traces(marker_line_width=0)
                fig = plot_cfg(fig, 280, "Sleep Hours — Distribution", showlegend=False)
                fig.update_layout(xaxis_title="Hours", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'sleep_quality' in df.columns:
                fig = go.Figure(go.Violin(
                    y=df['sleep_quality'], box_visible=True, meanline_visible=True,
                    fillcolor="rgba(99,179,237,.18)", line_color="#63b3ed", opacity=1,
                    hoverinfo='y', showlegend=False
                ))
                fig = plot_cfg(fig, 280, "Sleep Quality — Spread", showlegend=False)
                fig.update_layout(yaxis_title="Quality Score")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if 'study_time' in df.columns:
                fig = px.histogram(df, x='study_time', nbins=30,
                                   color_discrete_sequence=["#38d9a9"])
                fig.update_traces(marker_line_width=0)
                fig = plot_cfg(fig, 280, "Daily Study Time (hrs)", showlegend=False)
                fig.update_layout(xaxis_title="Hours", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'physical_activity' in df.columns:
                fig = px.histogram(df, x='physical_activity', nbins=30,
                                   color_discrete_sequence=["#9f7aea"])
                fig.update_traces(marker_line_width=0)
                fig = plot_cfg(fig, 280, "Physical Activity (hrs/week)", showlegend=False)
                fig.update_layout(xaxis_title="Hours/week", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            if 'stress_level' in df.columns:
                fig = go.Figure(go.Violin(
                    y=df['stress_level'], box_visible=True, meanline_visible=True,
                    fillcolor="rgba(255,76,106,.18)", line_color="#ff4c6a", opacity=1,
                    hoverinfo='y', showlegend=False
                ))
                fig = plot_cfg(fig, 280, "Stress Level — Spread", showlegend=False)
                fig.update_layout(yaxis_title="Stress Score")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'exam_anxiety' in df.columns:
                fig = go.Figure(go.Violin(
                    y=df['exam_anxiety'], box_visible=True, meanline_visible=True,
                    fillcolor="rgba(255,187,51,.18)", line_color="#ffbb33", opacity=1,
                    hoverinfo='y', showlegend=False
                ))
                fig = plot_cfg(fig, 280, "Exam Anxiety — Spread", showlegend=False)
                fig.update_layout(yaxis_title="Anxiety Score")
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — EDA & ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "EDA & Analytics":
    page_header("02 / EDA", "Exploratory Data Analysis",
                "Statistical summaries, correlation structures, and distribution profiles.")

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    num_cols = df.select_dtypes(include='number').columns.tolist()

    # ── STATS TABLE ──────────────────────────────────────────
    st.markdown("<h3>Statistical Summary</h3>", unsafe_allow_html=True)
    summary = df[num_cols].describe().T.rename(columns={"50%": "median"})
    summary['range'] = summary['max'] - summary['min']
    summary['cv']    = (summary['std'] / summary['mean'].replace(0, np.nan)).round(3)
    summary = summary.round(3)
    st.dataframe(summary, use_container_width=True, height=360)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── CORRELATION HEATMAP ──────────────────────────────────
    st.markdown("<h3>Correlation Matrix</h3>", unsafe_allow_html=True)
    corr = df[num_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,"#3b82f6"],[0.5,"#111318"],[1,"#ff4c6a"]],
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=9, family="JetBrains Mono", color="#e8eaf0"),
        showscale=True,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>"
    ))
    fig = plot_cfg(fig, 640, "Pearson Correlations — All Numeric Features")
    fig.update_layout(
        xaxis=dict(tickangle=-40, tickfont=dict(size=10, family="JetBrains Mono", color="#7c8099")),
        yaxis=dict(tickfont=dict(size=10, family="JetBrains Mono", color="#7c8099")),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── TOP CORR PAIRS ──────────────────────────────────────
    st.markdown("<h3>Strongest Feature Pairs</h3>", unsafe_allow_html=True)

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                "Feature A": corr.columns[i],
                "Feature B": corr.columns[j],
                "Correlation": round(corr.iloc[i, j], 4)
            })
    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False).head(12)

    fig = go.Figure(go.Bar(
        x=pairs_df["Correlation"],
        y=[f"{r['Feature A']}  ↔  {r['Feature B']}" for _, r in pairs_df.iterrows()],
        orientation='h',
        marker=dict(
            color=pairs_df["Correlation"],
            colorscale=[[0,"#3b82f6"],[0.5,"#1e2028"],[1,"#ff4c6a"]],
            cmid=0, cmin=-1, cmax=1,
            line_width=0
        ),
        text=pairs_df["Correlation"].map(lambda x: f"{x:+.3f}"),
        textposition='outside',
        textfont=dict(family="JetBrains Mono", size=10, color="#7c8099"),
        hovertemplate="r = %{x:.3f}<extra></extra>"
    ))
    fig = plot_cfg(fig, 480, "Top 12 Correlated Feature Pairs", showlegend=False)
    fig.update_layout(yaxis_title="", xaxis=dict(range=[-1.1, 1.1]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── RAW DATA ────────────────────────────────────────────
    with st.expander("📋  Raw Dataset"):
        st.dataframe(df, use_container_width=True, height=420)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE RELATIONSHIPS
# ═══════════════════════════════════════════════════════════════
elif page == "Feature Relationships":
    page_header("03 / DEEP DIVE", "Feature Relationships",
                "Drill into how individual features interact with burnout levels.")

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    num_cols = df.select_dtypes(include='number').columns.tolist()

    # ── BURNOUT-STRATIFIED BOX / VIOLIN ──────────────────────
    if 'burnout_level' in df.columns:
        st.markdown("<h3>Feature Distribution by Burnout Level</h3>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if 'sleep_quality' in df.columns:
                fig = px.box(df, x='burnout_level', y='sleep_quality', color='burnout_level',
                             color_discrete_sequence=BURNOUT_COLORS, points="outliers")
                fig = plot_cfg(fig, 340, "Sleep Quality × Burnout", showlegend=False)
                fig.update_layout(xaxis_title="Burnout Level (0=Low, 1=Med, 2=High)",
                                  yaxis_title="Sleep Quality")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'study_time' in df.columns:
                fig = px.violin(df, x='burnout_level', y='study_time', color='burnout_level',
                                color_discrete_sequence=BURNOUT_COLORS, box=True, points=False)
                fig = plot_cfg(fig, 340, "Study Time × Burnout", showlegend=False)
                fig.update_layout(xaxis_title="Burnout Level", yaxis_title="Study Time (hrs)")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3-D scatter
        st.markdown("<h3>3-D Multi-Axis Explorer</h3>", unsafe_allow_html=True)
        avail = [c for c in ['stress_level','sleep_hours','study_time',
                              'physical_activity','social_media_usage'] if c in df.columns]

        if len(avail) >= 3:
            cc1, cc2, cc3 = st.columns(3)
            with cc1: x_ = st.selectbox("X Axis", avail, index=0)
            with cc2: y_ = st.selectbox("Y Axis", avail, index=min(1, len(avail)-1))
            with cc3: z_ = st.selectbox("Z Axis", avail, index=min(2, len(avail)-1))

            fig = px.scatter_3d(
                df, x=x_, y=y_, z=z_, color='burnout_level',
                color_discrete_sequence=BURNOUT_COLORS, opacity=0.65,
                labels={'burnout_level': 'Level'}
            )
            fig.update_traces(marker_size=2.5)
            fig.update_layout(
                height=580,
                paper_bgcolor="#111318",
                scene=dict(
                    bgcolor="#111318",
                    xaxis=dict(backgroundcolor="#0a0c10", gridcolor="#1e2028",
                               tickfont=dict(family="JetBrains Mono",size=9,color="#7c8099")),
                    yaxis=dict(backgroundcolor="#0a0c10", gridcolor="#1e2028",
                               tickfont=dict(family="JetBrains Mono",size=9,color="#7c8099")),
                    zaxis=dict(backgroundcolor="#0a0c10", gridcolor="#1e2028",
                               tickfont=dict(family="JetBrains Mono",size=9,color="#7c8099")),
                ),
                font=dict(family="DM Sans", color="#7c8099"),
                margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── PAIRWISE SCATTER ─────────────────────────────────────
    st.markdown("<h3>Custom Scatter — Pick Any Two Features</h3>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca: feat_a = st.selectbox("Feature A", num_cols, index=0, key='fa')
    with cb: feat_b = st.selectbox("Feature B", num_cols, index=min(1, len(num_cols)-1), key='fb')

    if feat_a and feat_b:
        fig = px.scatter(
            df, x=feat_a, y=feat_b,
            color='burnout_level' if 'burnout_level' in df.columns else None,
            color_discrete_sequence=BURNOUT_COLORS, opacity=0.5, trendline="ols",
            trendline_color_override="#e8eaf0"
        )
        fig.update_traces(selector=dict(mode="markers"), marker_size=4)
        fig = plot_cfg(fig, 420, f"{feat_a}  vs  {feat_b}")
        fig.update_layout(xaxis_title=feat_a, yaxis_title=feat_b)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── MULTI-VIOLIN COMPARISON ──────────────────────────────
    st.markdown("<h3>Multi-Feature Distribution Comparison</h3>", unsafe_allow_html=True)

    selected = st.multiselect(
        "Select features to compare",
        num_cols,
        default=num_cols[:min(5, len(num_cols))]
    )
    def hex_to_rgba(hex_color, alpha=0.15):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if selected:
        fig = go.Figure()
        palette = ["#ff5a36","#ffb84d","#38d9a9","#63b3ed","#9f7aea","#f687b3","#fc8181"]
        for idx, feat in enumerate(selected):
            clr = palette[idx % len(palette)]
            fig.add_trace(go.Violin(
                y=df[feat], name=feat, box_visible=True, meanline_visible=True,
                fillcolor=hex_to_rgba(clr, 0.15), line_color=clr, opacity=1,
                points=False, hoverinfo='y+name'
            ))
        fig = plot_cfg(fig, 420, "Feature Distribution Comparison")
        fig.update_layout(yaxis_title="Normalised Value")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════
elif page == "Model Insights":
    page_header("04 / MODEL", "Model Architecture & Insights",
                "Feature importances, coefficient maps, and class-level decision logic.")

    if model is None:
        st.error("Model not loaded — place `burnout_model.pkl` and `scaler.pkl` inside `models/`.")
        st.stop()

    # ── FEATURE IMPORTANCE ───────────────────────────────────
    importance = np.mean(np.abs(model.coef_), axis=0)
    imp_df = pd.DataFrame({"Feature": FEATURE_LABELS, "Importance": importance}) \
               .sort_values("Importance", ascending=True)

    st.markdown("<h3>Feature Importance — Mean |Coefficient|</h3>", unsafe_allow_html=True)

    # colour bars by relative importance
    norm_imp = (imp_df["Importance"] - imp_df["Importance"].min()) / \
               (imp_df["Importance"].max() - imp_df["Importance"].min() + 1e-9)
    bar_colors = [f"rgba(255,90,54,{0.35 + 0.65*v:.2f})" for v in norm_imp]

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"], y=imp_df["Feature"],
        orientation='h',
        marker=dict(color=bar_colors, line_width=0),
        text=imp_df["Importance"].map(lambda v: f"{v:.4f}"),
        textposition='outside',
        textfont=dict(family="JetBrains Mono", size=9, color="#7c8099"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    fig = plot_cfg(fig, 520, "Feature Importance", showlegend=False)
    fig.update_layout(xaxis_title="Mean |Coefficient|", yaxis_title="",
                      yaxis=dict(tickfont=dict(family="DM Sans", size=11, color="#e8eaf0")))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── TOP / BOTTOM CARDS ──────────────────────────────────
    col_top, col_bot = st.columns(2)

    with col_top:
        st.markdown("<h3>Top 5 Drivers</h3>", unsafe_allow_html=True)
        top5 = imp_df.tail(5).sort_values("Importance", ascending=False)
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            pct = row["Importance"] / imp_df["Importance"].max() * 100
            st.markdown(f"""
            <div class='insight-row'>
                <div style='display:flex;align-items:center;gap:8px;'>
                    <span class='tag tag-red'>#{rank}</span>
                    <span class='insight-title'>{row["Feature"]}</span>
                </div>
                <div style='margin-top:8px;background:var(--border);border-radius:4px;height:4px;width:100%;'>
                    <div style='background:#ff5a36;height:4px;border-radius:4px;width:{pct:.1f}%;'></div>
                </div>
                <span style='font-family:var(--font-mono);font-size:.65rem;color:var(--muted);'>{row["Importance"]:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_bot:
        st.markdown("<h3>Least Impactful</h3>", unsafe_allow_html=True)
        bot5 = imp_df.head(5)
        for _, row in bot5.iterrows():
            pct = row["Importance"] / imp_df["Importance"].max() * 100
            st.markdown(f"""
            <div class='insight-row'>
                <div class='insight-title' style='color:var(--muted);'>{row["Feature"]}</div>
                <div style='margin-top:8px;background:var(--border);border-radius:4px;height:4px;width:100%;'>
                    <div style='background:var(--border-hi);height:4px;border-radius:4px;width:{pct:.1f}%;'></div>
                </div>
                <span style='font-family:var(--font-mono);font-size:.65rem;color:var(--muted);'>{row["Importance"]:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── COEFFICIENT HEATMAP ──────────────────────────────────
    st.markdown("<h3>Model Coefficients — Per Class</h3>", unsafe_allow_html=True)

    coef_df = pd.DataFrame(model.coef_, columns=FEATURE_LABELS, index=["Low", "Medium", "High"])
    fig = go.Figure(go.Heatmap(
        z=coef_df.values, x=coef_df.columns, y=coef_df.index,
        colorscale=[[0,"#3b82f6"],[0.5,"#111318"],[1,"#ff4c6a"]],
        text=coef_df.round(3).values,
        texttemplate="%{text}",
        textfont=dict(size=9, family="JetBrains Mono", color="#e8eaf0"),
        showscale=True,
        hovertemplate="<b>%{y}</b> / %{x}<br>coef = %{z:.4f}<extra></extra>"
    ))
    fig = plot_cfg(fig, 280, "")
    fig.update_layout(
        xaxis=dict(tickangle=-35, tickfont=dict(size=9, family="DM Sans", color="#7c8099")),
        yaxis=dict(tickfont=dict(size=11, family="Syne", color="#e8eaf0"))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── MODEL META ──────────────────────────────────────────
    st.markdown("<h3>Model Metadata</h3>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    meta = [
        ("Algorithm", "Logistic Regression"),
        ("Classes", "3  (Low · Med · High)"),
        ("Features", str(len(FEATURE_LABELS))),
        ("Solver", getattr(model, 'solver', 'lbfgs')),
    ]
    cols = [m1, m2, m3, m4]
    for col, (k, v) in zip(cols, meta):
        with col:
            st.markdown(f"""
            <div class='burn-card' style='padding:16px 20px;'>
                <span class='section-label'>{k}</span>
                <div style='font-family:var(--font-head);font-size:1.05rem;
                            font-weight:700;color:var(--text);margin-top:6px;'>{v}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif page == "Risk Predictor":

    page_header("05 / PREDICT", "Burnout Risk Predictor",
                "A quick check on how your lifestyle impacts burnout risk.")

    if model is None or scaler is None:
        st.error("Model not loaded.")
        st.stop()

    with st.form("predict_form"):

        st.subheader("📊 Lifestyle Inputs")

        c1, c2, c3 = st.columns(3)

        with c1:
            study_time = st.slider("Study Time (hrs/day)", 0.0, 12.0, 4.0)
            social_media = st.slider("Social Media (hrs/day)", 0.0, 10.0, 2.0)
            screen_time = st.slider("Screen Time (hrs/day)", 0.0, 16.0, 6.0)

        with c2:
            sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
            sleep_quality = st.selectbox("Sleep Quality", list(QUALITY_MAP.keys()))
            recovery = st.selectbox("Recovery Ability", list(QUALITY_MAP.keys()))

        with c3:
            physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 20.0, 3.0)
            headache = st.selectbox("Headache Frequency", list(QUALITY_MAP.keys()))

        st.subheader("🧠 Mental State")

        c4, c5, c6 = st.columns(3)

        with c4:
            study_load = st.selectbox("Study Load", list(LEVEL_MAP.keys()))
            exam_anxiety = st.selectbox("Exam Anxiety", list(LEVEL_MAP.keys()))
            performance = st.selectbox("Performance Pressure", list(LEVEL_MAP.keys()))

        with c5:
            stress = st.selectbox("Stress Level", list(LEVEL_MAP.keys()))
            mental = st.selectbox("Mental Strain", list(LEVEL_MAP.keys()))
            lifestyle = st.selectbox("Lifestyle Stress", list(LEVEL_MAP.keys()))

        with c6:
            extracurricular = st.selectbox("Extracurricular", list(LEVEL_MAP.keys()))

        submit = st.form_submit_button("🔍 Predict Burnout")

    if submit:

        input_data = np.array([[
            study_time / 12,
            social_media / 10,
            QUALITY_MAP[sleep_quality],
            QUALITY_MAP[headache],
            LEVEL_MAP[study_load],
            LEVEL_MAP[extracurricular],
            screen_time / 16,
            sleep_hours / 12,
            physical_activity / 20,
            LEVEL_MAP[exam_anxiety],
            LEVEL_MAP[stress],
            LEVEL_MAP[mental],
            QUALITY_MAP[recovery],
            LEVEL_MAP[performance],
            LEVEL_MAP[lifestyle]
        ]])

        scaled = scaler.transform(input_data)
        probs = model.predict_proba(scaled)[0]
        pred = int(np.argmax(probs))

        labels = ["Low", "Medium", "High"]
        confidence = probs[pred] * 100
        high_risk = probs[2] * 100

        # 🎨 Color logic
        def get_risk_color(risk):
            if risk < 30:
                return "#34d399"   # green
            elif risk < 70:
                return "#ffbb33"   # yellow
            else:
                return "#ff4c6a"   # red

        risk_color = get_risk_color(high_risk)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

# 🔥 HERO CARD
        st.markdown(f"""
        <div style="
        background: linear-gradient(135deg, {risk_color}22, transparent);
        border: 1px solid {risk_color};
        border-radius: 16px;
        padding: 28px;
        margin-top: 10px;
        ">
        <div style="font-size:0.8rem; letter-spacing:0.1em; color:#7c8099;">
        BURNOUT ASSESSMENT
        </div>

        <div style="font-size:2.5rem; font-weight:800; color:{risk_color}; margin-top:8px;">
        {labels[pred]}
        </div>

        <div style="margin-top:10px; color:#9aa0b5;">
        Confidence: <b>{confidence:.1f}%</b> &nbsp;&nbsp;|&nbsp;&nbsp;
        High Risk: <b style="color:{risk_color};">{high_risk:.1f}%</b>
        </div>
        </div>
        """, unsafe_allow_html=True)


# ⚡ GAUGE (REAL RISK)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=high_risk,
            number={'suffix': "%", 'font': {'size': 40}},
    
            gauge={
                'axis': {'range': [0, 100]},
        
                'bar': {'color': risk_color},
        
                'steps': [
                    {'range': [0, 30], 'color': '#34d399'},
                    {'range': [30, 70], 'color': '#ffbb33'},
                    {'range': [70, 100], 'color': '#ff4c6a'},
                ],
        
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': high_risk
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor="#111318",
            font={'color': "#e8eaf0"},
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)


# 🧠 AI INSIGHT
        if high_risk < 30:
            msg = "You are in a healthy range. Maintain your current lifestyle."
        elif high_risk < 70:
            msg = "Moderate burnout signals detected. Improve sleep & reduce stress."
        else:
            msg = "High burnout risk detected. Immediate intervention recommended."

        st.markdown(f"""
        <div style="
        background:#111318;
        border:1px solid #2e3140;
        border-radius:10px;
        padding:16px;
        margin-top:12px;
        ">
        <b style="color:{risk_color};">AI Insight:</b><br>
        <span style="color:#9aa0b5;">{msg}</span>
        </div>
        """, unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:24px 0 12px;border-top:1px solid #1e2028;'>
    <span style='font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem;color:#e8eaf0;
                 letter-spacing:-.02em;'>CogniBurn</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:.65rem;color:#7c8099;
                 letter-spacing:.1em;margin-left:12px;'>BURNOUT INTELLIGENCE PLATFORM</span>
    <div style='font-family:DM Sans,sans-serif;font-size:.75rem;color:#3a3f55;margin-top:8px;'>
        For research &amp; educational purposes only · Not clinical advice
    </div>
</div>
""", unsafe_allow_html=True)