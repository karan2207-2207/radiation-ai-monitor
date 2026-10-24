# Radiation Leak Detection Dashboard
# Optional dependency: plotly (falls back to Streamlit's native chart if missing).

import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PROJECT_TITLE = "IoT-Based Radiation Leak Detection with AI Alert System"
PROJECT_SUBTITLE = "Intelligent Monitoring • Machine Learning • Alert Management"
MODEL_FILENAME = "radiation_ml_model.pkl"
LABELS = {0: "NORMAL", 1: "ELEVATED"}

COLOR_NORMAL = "#22c55e"
COLOR_ELEVATED = "#f97316"
COLOR_NEUTRAL = "#64748b"
COLOR_ACCENT = "#22d3ee"

st.set_page_config(
    page_title="Radiation Monitoring Dashboard",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #11213f 0%, rgba(11,17,32,0) 60%), #0b1120;
    color: #e6edf7;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}
.stApp p, .stApp label, .stApp span, .stApp li { color: inherit; }

/* Header */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.25rem;
    flex-wrap: wrap;
    padding: 1.4rem 1.6rem;
    border: 1px solid #1f2d4a;
    border-radius: 18px;
    background: linear-gradient(135deg, #111b33 0%, #0f1a2e 100%);
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    margin-bottom: 1.6rem;
}
.app-header-left { display: flex; align-items: center; gap: 1.1rem; min-width: 260px; }
.app-logo {
    width: 58px; height: 58px; flex: 0 0 58px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 14px;
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.35);
}
.app-title { font-size: 1.45rem; font-weight: 700; line-height: 1.25; color: #f1f5fb; margin: 0; }
.app-subtitle { font-size: 0.9rem; color: #8ea3c4; margin-top: 0.25rem; letter-spacing: 0.02em; }
.status-chip {
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.5rem 0.95rem;
    border-radius: 999px;
    border: 1px solid #2a3b5f;
    background: #0e1730;
    font-size: 0.74rem; font-weight: 600; letter-spacing: 0.1em; color: #a9bbd8;
}
.status-chip .dot { width: 8px; height: 8px; border-radius: 50%; background: #22d3ee; }

/* Section headings */
.section-head { margin: 2rem 0 0.9rem 0; }
.section-head .eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
    color: #22d3ee; text-transform: uppercase;
}
.section-head h3 { margin: 0.15rem 0 0 0; padding: 0; font-size: 1.2rem; font-weight: 650; color: #f1f5fb; }
.section-head .desc { font-size: 0.85rem; color: #8ea3c4; margin-top: 0.2rem; }

/* Bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .panel-marker) {
    border: 1px solid #1f2d4a;
    background: #111b33;
}

/* Metric cards */
.metric-card {
    position: relative;
    height: 100%;
    padding: 1.25rem 1.35rem 1.2rem 1.35rem;
    border-radius: 16px;
    border: 1px solid #1f2d4a;
    background: #111b33;
    box-shadow: 0 6px 20px rgba(0,0,0,0.28);
    overflow: hidden;
}
.metric-card::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: var(--accent, #64748b);
}
.metric-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #8ea3c4; }
.metric-value { font-size: 2.2rem; font-weight: 750; line-height: 1.15; margin-top: 0.55rem; color: var(--accent-text, #f1f5fb); }
.metric-sub { font-size: 0.84rem; color: #8ea3c4; margin-top: 0.35rem; }

/* Alert panel */
.alert-panel {
    padding: 1.2rem 1.3rem;
    border-radius: 14px;
    border: 1px solid var(--accent, #64748b);
    background: var(--accent-bg, rgba(100,116,139,0.10));
}
.alert-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.3rem 0.8rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em;
    color: var(--accent, #64748b);
    border: 1px solid var(--accent, #64748b);
    background: rgba(0,0,0,0.18);
}
.alert-badge .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent, #64748b); }
.alert-title { font-size: 1.05rem; font-weight: 650; color: #f1f5fb; margin-top: 0.85rem; }
.alert-message { font-size: 0.9rem; color: #b3c3de; margin-top: 0.25rem; }
.alert-meta { font-size: 0.78rem; color: #7f93b5; margin-top: 0.8rem; }

/* Panel helper text */
.helper-text { font-size: 0.85rem; color: #8ea3c4; margin: 0.1rem 0 0.4rem 0; }
.panel-title { font-size: 1rem; font-weight: 650; color: #f1f5fb; margin: 0 0 0.15rem 0; }

/* Slider label */
div[data-testid="stSlider"] label p { color: #d5e0f2 !important; font-weight: 600; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid #2a3b5f;
    background: #14213f;
    color: #e6edf7;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease, transform 0.05s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #22d3ee;
    background: #19294d;
    color: #ffffff;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: none;
    border-color: #22d3ee;
    box-shadow: 0 0 0 3px rgba(34,211,238,0.35);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%);
    border: 1px solid #38bdf8;
    color: #ffffff;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0ea5c6 0%, #3b74f0 100%);
    border-color: #7dd3fc;
    box-shadow: 0 6px 18px rgba(34,211,238,0.25);
}
.stDownloadButton > button:disabled, .stButton > button:disabled { opacity: 0.45; }

/* Empty state */
.empty-state {
    padding: 2rem 1.2rem;
    text-align: center;
    border: 1px dashed #2a3b5f;
    border-radius: 14px;
    background: rgba(17,27,51,0.6);
    color: #8ea3c4;
    font-size: 0.92rem;
}
.empty-state strong { color: #d5e0f2; display: block; margin-bottom: 0.3rem; font-size: 1rem; }

/* System information */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 0.75rem;
}
.info-item {
    padding: 0.8rem 1rem;
    border: 1px solid #1f2d4a;
    border-radius: 12px;
    background: #0f1830;
}
.info-key { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #7f93b5; }
.info-val { font-size: 0.92rem; color: #e6edf7; margin-top: 0.2rem; word-break: break-word; }
.info-note {
    margin-top: 0.9rem;
    font-size: 0.82rem;
    color: #8ea3c4;
    padding: 0.8rem 1rem;
    border-left: 3px solid #2a3b5f;
    background: rgba(15,24,48,0.7);
    border-radius: 0 10px 10px 0;
}

@media (max-width: 768px) {
    .app-title { font-size: 1.15rem; }
    .metric-value { font-size: 1.8rem; }
    .app-header { padding: 1.1rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def trefoil_svg() -> str:
    """Return an inline SVG of a radiation trefoil symbol."""
    cx, cy, r_in, r_out = 24, 24, 6.5, 19

    def pt(radius: float, angle_deg: float):
        a = math.radians(angle_deg)
        return cx + radius * math.cos(a), cy + radius * math.sin(a)

    paths = []
    for center in (-90, 30, 150):
        a1, a2 = center - 30, center + 30
        x1, y1 = pt(r_in, a1)
        x2, y2 = pt(r_out, a1)
        x3, y3 = pt(r_out, a2)
        x4, y4 = pt(r_in, a2)
        paths.append(
            f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f} "
            f"A{r_out},{r_out} 0 0 1 {x3:.2f},{y3:.2f} "
            f"L{x4:.2f},{y4:.2f} A{r_in},{r_in} 0 0 0 {x1:.2f},{y1:.2f} Z"
        )
    body = "".join(f'<path d="{d}" fill="#22d3ee" fill-opacity="0.9"/>' for d in paths)
    return (
        '<svg width="34" height="34" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">'
        f'{body}<circle cx="{cx}" cy="{cy}" r="3.4" fill="#22d3ee"/></svg>'
    )


def section_header(eyebrow: str, title: str, description: str = "") -> None:
    desc_html = f'<div class="desc">{description}</div>' if description else ""
    st.markdown(
        f'<div class="section-head"><div class="eyebrow">{eyebrow}</div>'
        f"<h3>{title}</h3>{desc_html}</div>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str, accent: str, accent_text: str = "#f1f5fb") -> str:
    return (
        f'<div class="metric-card" style="--accent:{accent};--accent-text:{accent_text};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>'
    )


def status_color(label: str) -> str:
    if label == "NORMAL":
        return COLOR_NORMAL
    if label == "ELEVATED":
        return COLOR_ELEVATED
    return COLOR_NEUTRAL


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def stretch_call(func, *args, **kwargs):
    """Call a Streamlit element with full-width sizing across Streamlit versions."""
    try:
        return func(*args, width="stretch", **kwargs)
    except Exception:
        return func(*args, use_container_width=True, **kwargs)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def resolve_model_path() -> Path:
    local = Path(__file__).resolve().parent / MODEL_FILENAME
    return local if local.exists() else Path(MODEL_FILENAME)


@st.cache_resource(show_spinner=False)
def load_model(path_str: str):
    return joblib.load(path_str)


model_path = resolve_model_path()

if not model_path.exists():
    st.error(f"Model file missing. Add {MODEL_FILENAME} to the project folder.")
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as exc:
    st.error(f"The model file could not be loaded: {exc}")
    st.stop()

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


def analyze_reading() -> None:
    """Run one model prediction for the currently selected reading."""
    reading_value = int(st.session_state.get("reading_input", 20))
    sample = pd.DataFrame({"reading": [reading_value]})
    try:
        prediction = int(model.predict(sample)[0])
        probabilities = list(model.predict_proba(sample)[0])
        classes = [int(c) for c in getattr(model, "classes_", [])]
        if prediction in classes:
            idx = classes.index(prediction)
        else:
            idx = max(range(len(probabilities)), key=probabilities.__getitem__)
        confidence = float(probabilities[idx]) * 100
    except Exception as exc:
        st.session_state.analysis_error = str(exc)
        return

    st.session_state.analysis_error = None
    st.session_state.history.append(
        {
            "Sample": len(st.session_state.history) + 1,
            "Reading": reading_value,
            "AI Classification": LABELS.get(prediction, str(prediction)),
            "Model Confidence (%)": round(confidence, 2),
        }
    )


def clear_history() -> None:
    st.session_state.history = []
    st.session_state.analysis_error = None


history = st.session_state.history
latest = history[-1] if history else None
history_df = (
    pd.DataFrame(history)
    if history
    else pd.DataFrame(columns=["Sample", "Reading", "AI Classification", "Model Confidence (%)"])
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="app-header">'
    '<div class="app-header-left">'
    f'<div class="app-logo">{trefoil_svg()}</div>'
    "<div>"
    f'<div class="app-title">{PROJECT_TITLE}</div>'
    f'<div class="app-subtitle">{PROJECT_SUBTITLE}</div>'
    "</div></div>"
    '<div class="status-chip"><span class="dot"></span>MONITORING INTERFACE</div>'
    "</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Current reading + alert management
# ----------------------------------------------------------------------------
section_header(
    "Overview",
    "Monitoring Dashboard",
    "Select a reading, analyze it with the classification model, and review the resulting status.",
)

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<span class="panel-marker"></span>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Current Reading</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="helper-text">This value is the input used by the model for classification.</div>',
            unsafe_allow_html=True,
        )
        st.slider(
            "Radiation Reading (Arbitrary Units)",
            min_value=0,
            max_value=150,
            value=20,
            step=1,
            key="reading_input",
        )
        btn_col, clear_col, _ = st.columns([2, 2, 3])
        with btn_col:
            st.button(
                "Analyze Reading",
                type="primary",
                on_click=analyze_reading,
                key="analyze_button",
            )
        with clear_col:
            st.button(
                "Clear History",
                on_click=clear_history,
                disabled=not history,
                key="clear_button",
            )
        if st.session_state.analysis_error:
            st.error(f"Analysis could not be completed: {st.session_state.analysis_error}")

with right_col:
    with st.container(border=True):
        st.markdown('<span class="panel-marker"></span>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Alert Management</div>', unsafe_allow_html=True)
        if latest is None:
            accent = COLOR_NEUTRAL
            badge = "AWAITING ANALYSIS"
            title = "System Status"
            message = "No reading has been analyzed yet."
            meta = "Use Analyze Reading to generate a classification."
        else:
            label = latest["AI Classification"]
            accent = status_color(label)
            badge = label
            title = "System Status"
            message = "No alert triggered." if label == "NORMAL" else "Elevated reading classification."
            meta = (
                f'Sample #{latest["Sample"]} • Reading {latest["Reading"]} AU • '
                "Status derived from the model classification."
            )
        st.markdown(
            f'<div class="alert-panel" style="--accent:{accent};--accent-bg:{hex_to_rgba(accent, 0.10)};">'
            f'<span class="alert-badge"><span class="dot"></span>{badge}</span>'
            f'<div class="alert-title">{title}</div>'
            f'<div class="alert-message">{message}</div>'
            f'<div class="alert-meta">{meta}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# AI analysis cards
# ----------------------------------------------------------------------------
section_header("AI Analysis", "Classification Summary")

c1, c2, c3 = st.columns(3, gap="medium")

if latest is None:
    card_reading = metric_card("Current Reading", "—", "Awaiting analysis", COLOR_NEUTRAL, "#8ea3c4")
    card_class = metric_card("AI Classification", "—", "No classification available yet", COLOR_NEUTRAL, "#8ea3c4")
    card_conf = metric_card("Model Confidence", "—", "Predicted-class probability", COLOR_NEUTRAL, "#8ea3c4")
else:
    label = latest["AI Classification"]
    accent = status_color(label)
    card_reading = metric_card(
        "Current Reading",
        f'{latest["Reading"]}',
        "Arbitrary Units",
        COLOR_ACCENT,
    )
    sub_text = "No alert triggered." if label == "NORMAL" else "Elevated reading classification."
    card_class = metric_card("AI Classification", label, sub_text, accent, accent)
    card_conf = metric_card(
        "Model Confidence",
        f'{latest["Model Confidence (%)"]:.2f}%',
        "Predicted-class probability from the model",
        COLOR_ACCENT,
    )

c1.markdown(card_reading, unsafe_allow_html=True)
c2.markdown(card_class, unsafe_allow_html=True)
c3.markdown(card_conf, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Reading analytics
# ----------------------------------------------------------------------------
section_header(
    "Reading Analytics",
    "Reading Trend",
    "Plotted from the readings analyzed during the current session.",
)


def build_chart(df: pd.DataFrame):
    colors = [status_color(v) for v in df["AI Classification"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Sample"],
            y=df["Reading"],
            mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=2.5, shape="linear"),
            marker=dict(size=10, color=colors, line=dict(color="#0b1120", width=2)),
            customdata=df[["AI Classification", "Model Confidence (%)"]].to_numpy(),
            hovertemplate=(
                "<b>Sample %{x}</b><br>"
                "Reading: %{y} AU<br>"
                "Classification: %{customdata[0]}<br>"
                "Model Confidence: %{customdata[1]:.2f}%"
                "<extra></extra>"
            ),
            name="Reading",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1830",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#c5d3ea", size=13),
        showlegend=False,
        hoverlabel=dict(bgcolor="#14213f", bordercolor="#2a3b5f", font=dict(color="#e6edf7")),
        xaxis=dict(
            title="Sample Number",
            gridcolor="rgba(142,163,196,0.12)",
            zeroline=False,
            dtick=1 if len(df) <= 20 else None,
            tickmode="linear" if len(df) <= 20 else "auto",
            linecolor="#2a3b5f",
        ),
        yaxis=dict(
            title="Reading (Arbitrary Units)",
            gridcolor="rgba(142,163,196,0.12)",
            zeroline=False,
            range=[0, max(150, int(df["Reading"].max()) + 10)],
            linecolor="#2a3b5f",
        ),
    )
    return fig


with st.container(border=True):
    st.markdown('<span class="panel-marker"></span>', unsafe_allow_html=True)
    if history_df.empty:
        st.markdown(
            '<div class="empty-state"><strong>No readings to display</strong>'
            "Analyze a reading to start building the trend chart.</div>",
            unsafe_allow_html=True,
        )
    elif go is not None:
        stretch_call(st.plotly_chart, build_chart(history_df), config={"displaylogo": False})
    else:
        st.line_chart(history_df.set_index("Sample")["Reading"])

# ----------------------------------------------------------------------------
# Monitoring history + export
# ----------------------------------------------------------------------------
section_header(
    "Monitoring History",
    "Session Readings",
    "Entries are added only when a reading is analyzed.",
)

with st.container(border=True):
    st.markdown('<span class="panel-marker"></span>', unsafe_allow_html=True)
    if history_df.empty:
        st.markdown(
            '<div class="empty-state"><strong>No history available</strong>'
            "Analyze a reading to populate the monitoring history.</div>",
            unsafe_allow_html=True,
        )
    else:
        stretch_call(
            st.dataframe,
            history_df,
            hide_index=True,
            column_config={
                "Sample": st.column_config.NumberColumn("Sample", format="%d"),
                "Reading": st.column_config.NumberColumn("Reading (AU)", format="%d"),
                "AI Classification": st.column_config.TextColumn("AI Classification"),
                "Model Confidence (%)": st.column_config.NumberColumn(
                    "Model Confidence (%)", format="%.2f"
                ),
            },
        )

    export_csv = history_df.to_csv(index=False)
    st.download_button(
        "Export Report (CSV)",
        data=export_csv,
        file_name="radiation_monitoring_report.csv",
        mime="text/csv",
        disabled=history_df.empty,
        key="export_button",
    )

# ----------------------------------------------------------------------------
# System information
# ----------------------------------------------------------------------------
section_header("System Information", "Configuration Details")

info_items = [
    ("Project", PROJECT_TITLE),
    ("Classification Engine", "Random Forest Classifier"),
    ("Input Feature", "Reading"),
    ("Output Classes", "NORMAL / ELEVATED"),
    ("Model File", MODEL_FILENAME),
    ("Reading Unit", "Arbitrary Units"),
    ("Hardware Integration", "Not yet connected in the current application"),
]
info_html = "".join(
    f'<div class="info-item"><div class="info-key">{k}</div><div class="info-val">{v}</div></div>'
    for k, v in info_items
)
st.markdown(
    f'<div class="info-grid">{info_html}</div>'
    '<div class="info-note">Current readings are generated for software testing and are not '
    "measurements from a physical sensor. Classifications are produced by the machine-learning model "
    "and have not been validated for real-world radiation detection. Actual radiation measurement "
    "requires a suitable physical radiation sensor and validation.</div>",
    unsafe_allow_html=True,
)
