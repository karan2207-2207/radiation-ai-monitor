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
    ("NORMAL", "Green LED on", "Shows NORMAL"),
    ("ELEVATED", "Red LED on", "Shows ELEVATED"),
]
body = ""
for name, led, oled in output_rows:
    c = status_color(name)
    cls = "active" if name == active else ""
    body += (
        f'<tr class="{cls}"><td><span class="dot" style="background:{c}"></span>{name}</td>'
        f"<td>{led}</td><td>{oled}</td></tr>"
    )
st.markdown(
    '<div class="tbl-wrap"><table class="tbl"><thead><tr>'
    "<th>Classification</th><th>LED</th><th>OLED Display</th>"
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

