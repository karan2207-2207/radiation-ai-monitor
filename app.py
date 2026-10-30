import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import requests

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
COLOR_ELEVATED = "#ef4444"
COLOR_NEUTRAL = "#71717a"

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: #09090b; color: #fafafa; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

/* Top bar */
.topbar {
    display: flex; align-items: flex-end; justify-content: space-between;
    flex-wrap: wrap; gap: 0.5rem;
    padding-bottom: 1.2rem; margin-bottom: 0.5rem;
    border-bottom: 1px solid #27272a;
}
.topbar h1 { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; margin: 0; padding: 0; color: #fafafa; }
.topbar p { font-size: 0.9rem; color: #a1a1aa; margin: 0.3rem 0 0 0; }
.topbar .meta { font-size: 0.8rem; color: #71717a; }

/* Section headings */
.sec { margin: 2.2rem 0 0.9rem 0; }
.sec h2 { font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; padding: 0; color: #fafafa; }
.sec p { font-size: 0.85rem; color: #a1a1aa; margin: 0.2rem 0 0 0; }

/* Containers */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }

.card-title { font-size: 0.95rem; font-weight: 600; color: #fafafa; margin: 0; }
.card-desc { font-size: 0.82rem; color: #a1a1aa; margin: 0.15rem 0 0.6rem 0; }

/* Stat cards */
.stat {
    border: 1px solid #27272a; border-radius: 10px; background: #0c0c0f;
    padding: 1.1rem 1.2rem; height: 100%;
}
.stat .k { font-size: 0.8rem; color: #a1a1aa; }
.stat .v { font-size: 1.9rem; font-weight: 600; letter-spacing: -0.02em; margin-top: 0.35rem; line-height: 1.2; }
.stat .s { font-size: 0.8rem; color: #71717a; margin-top: 0.25rem; }

/* Status block */
.status {
    border: 1px solid #27272a; border-radius: 10px; padding: 1rem 1.1rem;
    background: #0c0c0f;
}
.badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
    padding: 0.2rem 0.65rem; border-radius: 6px;
    border: 1px solid var(--c); color: var(--c);
}
.badge i { width: 7px; height: 7px; border-radius: 50%; background: var(--c); display: inline-block; }
.status .msg { font-size: 1rem; font-weight: 500; margin-top: 0.8rem; color: #fafafa; }
.status .sub { font-size: 0.82rem; color: #a1a1aa; margin-top: 0.2rem; }

/* Slider thumb */
div[data-testid="stSlider"] [role="slider"] { background-color: #fafafa; border-color: #fafafa; }
div[data-testid="stSlider"] label p { color: #e4e4e7; font-weight: 500; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px; border: 1px solid #27272a; background: #18181b;
    color: #fafafa; font-weight: 500; padding: 0.5rem 1.1rem;
    transition: background 0.12s ease, border-color 0.12s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #27272a; border-color: #3f3f46; color: #fafafa;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: none; box-shadow: 0 0 0 2px #09090b, 0 0 0 4px #71717a;
}
.stButton > button[kind="primary"] { background: #fafafa; border-color: #fafafa; color: #09090b; font-weight: 600; }
.stButton > button[kind="primary"]:hover { background: #e4e4e7; border-color: #e4e4e7; color: #09090b; }
.stButton > button:disabled, .stDownloadButton > button:disabled { opacity: 0.4; }

/* Tables */
.tbl-wrap { border: 1px solid #27272a; border-radius: 10px; overflow: auto; max-height: 380px; background: #0c0c0f; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
table.tbl th {
    text-align: left; font-weight: 500; font-size: 0.78rem; color: #a1a1aa;
    padding: 0.7rem 1rem; border-bottom: 1px solid #27272a;
    position: sticky; top: 0; background: #0c0c0f;
}
table.tbl td { padding: 0.65rem 1rem; border-bottom: 1px solid #1c1c1f; color: #e4e4e7; }
table.tbl tr:last-child td { border-bottom: none; }
table.tbl tr.active td { background: #141417; }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 0.5rem; }

/* Empty state */
.empty {
    border: 1px dashed #27272a; border-radius: 10px; padding: 2rem 1rem;
    text-align: center; color: #a1a1aa; font-size: 0.88rem;
}
.empty b { display: block; color: #e4e4e7; font-weight: 500; margin-bottom: 0.2rem; }

/* Hardware + info grids */
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.75rem; }
.item { border: 1px solid #27272a; border-radius: 10px; background: #0c0c0f; padding: 1rem 1.1rem; }
.item .n { font-size: 0.92rem; font-weight: 600; color: #fafafa; }
.item .r { font-size: 0.82rem; color: #a1a1aa; margin-top: 0.25rem; line-height: 1.45; }
.kv .k { font-size: 0.76rem; color: #71717a; }
.kv .v { font-size: 0.9rem; color: #e4e4e7; margin-top: 0.15rem; word-break: break-word; }

@media (max-width: 768px) {
    .topbar h1 { font-size: 1.2rem; }
    .stat .v { font-size: 1.6rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value))


def section(title: str, description: str = "") -> None:
    desc = f"<p>{esc(description)}</p>" if description else ""
    st.markdown(f'<div class="sec"><h2>{esc(title)}</h2>{desc}</div>', unsafe_allow_html=True)


def status_color(label: str) -> str:
    return {"NORMAL": COLOR_NORMAL, "ELEVATED": COLOR_ELEVATED}.get(label, COLOR_NEUTRAL)


def stat_card(label: str, value: str, sub: str, color: str = "#fafafa") -> str:
    return (
        f'<div class="stat"><div class="k">{esc(label)}</div>'
        f'<div class="v" style="color:{color}">{esc(value)}</div>'
        f'<div class="s">{esc(sub)}</div></div>'
    )


def stretch_call(func, *args, **kwargs):
    """Full-width sizing that works across Streamlit versions."""
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
def publish_to_thingspeak(reading, prediction):
    try:
        write_key = st.secrets["thingspeak"]["write_api_key"]

        response = requests.post(
            "https://api.thingspeak.com/update",
            data={
                "api_key": write_key,
                "field2": reading,
                "field3": prediction,
            },
            timeout=10,
        )

        if response.text.strip() == "0":
            return False, "ThingSpeak update failed."

        return True, "Hardware update sent successfully."

    except Exception as exc:
        return False, str(exc)


def analyze_reading() -> None:
    """Run one model prediction for the selected reading and store it."""
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

sync_ok, sync_message = publish_to_thingspeak(
    reading_value,
    prediction
)

st.session_state.hardware_sync_ok = sync_ok
st.session_state.hardware_sync_message = sync_message


def clear_history() -> None:
    st.session_state.history = []
    st.session_state.analysis_error = None


history = st.session_state.history
latest = history[-1] if history else None
COLUMNS = ["Sample", "Reading", "AI Classification", "Model Confidence (%)"]
history_df = pd.DataFrame(history, columns=COLUMNS) if history else pd.DataFrame(columns=COLUMNS)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="topbar"><div>'
    f"<h1>{esc(PROJECT_TITLE)}</h1>"
    f"<p>{esc(PROJECT_SUBTITLE)}</p>"
    "</div>"
    '<div class="meta">ESP32 • LED • Buzzer • OLED Display</div></div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Reading input + system status
# ----------------------------------------------------------------------------
section("Monitoring Dashboard", "Select a reading and analyze it with the classification model.")

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<p class="card-title">Current Reading</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="card-desc">This value is the input used by the model for classification.</p>',
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
        b1, b2, _ = st.columns([2, 2, 3])
        with b1:
            st.button("Analyze Reading", type="primary", on_click=analyze_reading, key="analyze_button")
        with b2:
            st.button("Clear History", on_click=clear_history, disabled=not history, key="clear_button")
        if st.session_state.analysis_error:
            st.error(f"Analysis could not be completed: {st.session_state.analysis_error}")
        if st.session_state.get("hardware_sync_ok") is True:
            st.success("Reading sent to hardware successfully.")

        elif st.session_state.get("hardware_sync_ok") is False:
            st.warning(
                "Hardware sync failed: "
                + st.session_state.hardware_sync_message
            )
        

with right_col:
    with st.container(border=True):
        st.markdown('<p class="card-title">System Status</p>', unsafe_allow_html=True)
        if latest is None:
            color, badge = COLOR_NEUTRAL, "AWAITING ANALYSIS"
            msg, sub = "No reading analyzed yet.", "Use Analyze Reading to generate a classification."
        else:
            label = latest["AI Classification"]
            color, badge = status_color(label), label
            msg = "No alert triggered." if label == "NORMAL" else "Elevated reading classification."
            sub = f'Sample #{latest["Sample"]} • Reading {latest["Reading"]} AU'
        st.markdown(
            f'<div class="status" style="margin-top:0.6rem"><span class="badge" style="--c:{color}">'
            f"<i></i>{esc(badge)}</span>"
            f'<div class="msg">{esc(msg)}</div><div class="sub">{esc(sub)}</div></div>',
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Summary cards
# ----------------------------------------------------------------------------
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="medium")

if latest is None:
    c1.markdown(stat_card("Current Reading", "—", "Awaiting analysis", "#52525b"), unsafe_allow_html=True)
    c2.markdown(stat_card("AI Classification", "—", "No classification yet", "#52525b"), unsafe_allow_html=True)
    c3.markdown(stat_card("Model Confidence", "—", "Predicted-class probability", "#52525b"), unsafe_allow_html=True)
else:
    label = latest["AI Classification"]
    c1.markdown(stat_card("Current Reading", str(latest["Reading"]), "Arbitrary Units"), unsafe_allow_html=True)
    c2.markdown(
        stat_card(
            "AI Classification",
            label,
            "No alert triggered." if label == "NORMAL" else "Elevated reading classification.",
            status_color(label),
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        stat_card("Model Confidence", f'{latest["Model Confidence (%)"]:.2f}%', "Predicted-class probability"),
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Reading analytics
# ----------------------------------------------------------------------------
section("Reading Analytics", "Readings analyzed during the current session.")


def build_chart(df: pd.DataFrame):
    colors = [status_color(v) for v in df["AI Classification"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Sample"],
            y=df["Reading"],
            mode="lines+markers",
            line=dict(color="#a1a1aa", width=2),
            marker=dict(size=9, color=colors, line=dict(color="#09090b", width=2)),
            customdata=df[["AI Classification", "Model Confidence (%)"]].to_numpy(),
            hovertemplate=(
                "<b>Sample %{x}</b><br>"
                "Reading: %{y} AU<br>"
                "Classification: %{customdata[0]}<br>"
                "Model Confidence: %{customdata[1]:.2f}%"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#a1a1aa", size=12),
        showlegend=False,
        hoverlabel=dict(bgcolor="#18181b", bordercolor="#3f3f46", font=dict(color="#fafafa")),
        xaxis=dict(
            title="Sample Number",
            gridcolor="#1f1f23",
            zeroline=False,
            linecolor="#27272a",
            dtick=1 if len(df) <= 20 else None,
        ),
        yaxis=dict(
            title="Reading (Arbitrary Units)",
            gridcolor="#1f1f23",
            zeroline=False,
            linecolor="#27272a",
            range=[0, max(150, int(df["Reading"].max()) + 10)],
        ),
    )
    return fig


with st.container(border=True):
    if history_df.empty:
        st.markdown(
            '<div class="empty"><b>No readings to display</b>Analyze a reading to start the trend chart.</div>',
            unsafe_allow_html=True,
        )
    elif go is not None:
        stretch_call(st.plotly_chart, build_chart(history_df), config={"displaylogo": False})
    else:
        st.line_chart(history_df.set_index("Sample")["Reading"])

# ----------------------------------------------------------------------------
# Monitoring history + export
# ----------------------------------------------------------------------------
section("Monitoring History", "Entries are added only when a reading is analyzed.")

if history_df.empty:
    st.markdown(
        '<div class="empty"><b>No history available</b>'
        "Analyze a reading to populate the monitoring history.</div>",
        unsafe_allow_html=True,
    )
else:
    rows = []
    last_sample = int(history_df["Sample"].iloc[-1])
    for _, r in history_df.iloc[::-1].iterrows():
        c = status_color(r["AI Classification"])
        cls = "active" if int(r["Sample"]) == last_sample else ""
        rows.append(
            f'<tr class="{cls}"><td>{int(r["Sample"])}</td><td>{int(r["Reading"])}</td>'
            f'<td><span class="dot" style="background:{c}"></span>{esc(r["AI Classification"])}</td>'
            f'<td>{float(r["Model Confidence (%)"]):.2f}</td></tr>'
        )
    st.markdown(
        '<div class="tbl-wrap"><table class="tbl"><thead><tr>'
        "<th>Sample</th><th>Reading (AU)</th><th>AI Classification</th><th>Model Confidence (%)</th>"
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
st.download_button(
    "Export Report (CSV)",
    data=history_df.to_csv(index=False),
    file_name="radiation_monitoring_report.csv",
    mime="text/csv",
    disabled=history_df.empty,
    key="export_button",
)

# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
section("Hardware", "Components used in the alert system.")

hardware = [
    ("ESP32", "Microcontroller that runs the device firmware and drives the alert outputs."),
    ("Status LEDs", "Green and red LEDs for at-a-glance NORMAL and ELEVATED indication."),
    ("Buzzer", "Audible alert for an ELEVATED classification."),
    ("0.96\" OLED Display", "Compact on-device display for the current reading and status."),
]
st.markdown(
    '<div class="grid">'
    + "".join(
        f'<div class="item"><div class="n">{esc(n)}</div><div class="r">{esc(d)}</div></div>'
        for n, d in hardware
    )
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
active = latest["AI Classification"] if latest else None
output_rows = [
    ("NORMAL", "Green LED on", "Off", "Shows NORMAL"),
    ("ELEVATED", "Red LED on", "On", "Shows ELEVATED"),
]
body = ""
for name, led, buzzer, oled in output_rows:
    c = status_color(name)
    cls = "active" if name == active else ""
    body += (
        f'<tr class="{cls}"><td><span class="dot" style="background:{c}"></span>{name}</td>'
        f"<td>{led}</td><td>{buzzer}</td><td>{oled}</td></tr>"
    )
st.markdown(
    '<div class="tbl-wrap"><table class="tbl"><thead><tr>'
    "<th>Classification</th><th>LED</th><th>Buzzer</th><th>OLED Display</th>"
    f"</tr></thead><tbody>{body}</tbody></table></div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# System information
# ----------------------------------------------------------------------------
section("System Information")

info_items = [
    ("Project", PROJECT_TITLE),
    ("Classification Engine", "Random Forest Classifier"),
    ("Input Feature", "Reading"),
    ("Output Classes", "NORMAL / ELEVATED"),
    ("Model File", MODEL_FILENAME),
    ("Reading Unit", "Arbitrary Units"),
    ("Controller", "ESP32"),
]
st.markdown(
    '<div class="grid">'
    + "".join(
        f'<div class="item kv"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
        for k, v in info_items
    )
    + "</div>",
    unsafe_allow_html=True,
)

