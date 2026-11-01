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
