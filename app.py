
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Radiation AI Monitor",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0b1120;
}
[data-testid="stMetric"] {
    background: #151f32;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("📡 Radiation AI Monitor")
st.caption("IoT-Based Radiation Leak Detection with AI Alert System")
st.markdown("****")
st.warning(
    "DEMO MODE: This application uses synthetic readings. "
    "It is not a real radiation detector or safety monitoring system."
)

@st.cache_resource
def load_model():
    return joblib.load("radiation_ml_model.pkl")

model_path = Path("radiation_ml_model.pkl")

if not model_path.exists():
    st.error("Model file missing. Add radiation_ml_model.pkl to the project folder.")
    st.stop()

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

reading = st.slider(
    "Select a simulated reading",
    min_value=0,
    max_value=150,
    value=20
)

if st.button("Run AI Prediction", type="primary"):
    sample = pd.DataFrame({"reading": [reading]})
    prediction = int(model.predict(sample)[0])
    probability = float(max(model.predict_proba(sample)[0])) * 100

    status = "ELEVATED" if prediction == 1 else "NORMAL"

    st.session_state.history.append({
        "Reading": reading,
        "Prediction": status,
        "Predicted-class probability (%)": round(probability, 2)
    })

    st.rerun()

if st.session_state.history:
    latest = st.session_state.history[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Reading", latest["Reading"])
    col2.metric("AI Classification", latest["Prediction"])
    col3.metric("Predicted-class probability", f'{latest["Predicted-class probability (%)"]}%')

    if latest["Prediction"] == "ELEVATED":
        st.error("Elevated simulated reading — demo alert!")
    else:
        st.success("Normal simulated reading — no demo alert.")

    history_df = pd.DataFrame(st.session_state.history)

    st.subheader("Monitoring History")
    st.dataframe(history_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history_df["Reading"], marker="o")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Simulated reading")
    ax.set_title("Simulated Monitoring Trend")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Reading History (CSV)",
        data=history_df.to_csv(index=False),
        file_name="radiation_monitoring_history.csv",
        mime="text/csv"
    )

st.divider()
st.markdown("### About the Project")
st.write(
    "This prototype demonstrates a machine-learning workflow using "
    "synthetic readings. Predictions are not validated for actual "
    "radiation detection. A suitable radiation sensor and validated "
    "detection pipeline are required for real-world use."
)
st.caption("Project Prototype")
