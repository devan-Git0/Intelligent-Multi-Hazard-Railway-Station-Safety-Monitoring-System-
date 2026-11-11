"""Full-width railway safety operations dashboard (Streamlit).

Run in a second terminal while main.py is running:
    streamlit run dashboard.py
"""

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from database import IncidentDatabase


LEVEL_COLORS = {
    "CRITICAL": "#c93636",
    "HIGH": "#e77832",
    "MEDIUM": "#d39a18",
    "LOW": "#55758f",
}
LEVEL_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

st.set_page_config(
    page_title="Railway Safety Operations",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --navy: #102a43;
        --ink: #183247;
        --muted: #62798b;
        --line: #dce6ed;
        --surface: #ffffff;
        --canvas: #f2f6f9;
        --teal: #087f8c;
      }
      html, body, [data-testid="stAppViewContainer"] {
        background: var(--canvas);
        color: var(--ink);
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stMainBlockContainer"] {
        width: 100%;
        max-width: none;
        padding: 1.25rem 2.25rem 3rem;
      }
      [data-testid="stSidebar"] {
        background: #102a43;
        border-right: 1px solid #1d405a;
      }
      [data-testid="stSidebar"] * { color: #e9f1f6; }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #c1d2df;
      }
      .ops-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1.55rem 1.8rem;
        margin: 0 0 1.35rem;
        color: white;
        border-radius: 18px;
        background: linear-gradient(115deg, #102a43 0%, #174d68 65%, #087f8c 100%);
        box-shadow: 0 12px 28px rgba(16, 42, 67, .15);
      }
      .ops-title { margin: 0; font-size: 1.7rem; font-weight: 750; letter-spacing: -.025em; }
      .ops-subtitle { margin: .35rem 0 0; color: #d2e3ec; font-size: .94rem; }
      .ops-status {
        display: inline-flex;
        align-items: center;
        gap: .55rem;
        padding: .55rem .8rem;
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 999px;
        background: rgba(255,255,255,.1);
        color: #f4fbfd;
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .06em;
        white-space: nowrap;
      }
      .ops-dot { width: .55rem; height: .55rem; border-radius: 50%; background: #56d6a2; box-shadow: 0 0 0 4px rgba(86,214,162,.16); }
      h2, h3 { color: var(--navy); }
      [data-testid="stMetric"] {
        min-height: 112px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
        box-shadow: 0 3px 12px rgba(16,42,67,.035);
      }
      [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 650; }
      [data-testid="stMetricValue"] { color: var(--navy); }
      [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 14px !important;
        background: var(--surface);
      }
      [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: white;
      }
      [data-testid="stAlert"] { border-radius: 12px; }
      div[data-testid="stSelectbox"] > label,
      div[data-testid="stMultiSelect"] > label,
      div[data-testid="stSlider"] > label { font-weight: 650; }
      @media (max-width: 800px) {
        [data-testid="stMainBlockContainer"] { padding: .8rem .85rem 2rem; }
        .ops-header { align-items: flex-start; flex-direction: column; padding: 1.2rem; }
        .ops-title { font-size: 1.35rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Dashboard controls")
    st.caption("Choose what appears in the incident table.")
    limit = st.slider("Incidents to load", 20, 500, 100, step=20)
    levels = st.multiselect(
        "Severity levels",
        LEVEL_ORDER,
        default=LEVEL_ORDER,
    )
    st.divider()
    st.markdown("### Live updates")
    auto = st.checkbox("Auto-refresh", value=True)
    interval = st.slider("Refresh interval", 1, 30, 3, format="%d sec")
    st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

st.markdown(
    """
    <div class="ops-header">
      <div>
        <p class="ops-title">Railway Safety Operations</p>
        <p class="ops-subtitle">Live incident monitoring · Platform and track safety</p>
      </div>
      <div class="ops-status"><span class="ops-dot"></span> MONITORING</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if os.path.exists(config.DB_PATH):
    db = IncidentDatabase(config.DB_PATH)
    rows = db.recent_incidents(limit)
    totals = db.count_by_type()
    db.close()
else:
    rows = []
    totals = {}

df_all = pd.DataFrame(rows)
if rows:
    df_all["time"] = df_all["timestamp"].apply(
        lambda value: datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    )
    latest = rows[0]
    latest_time = datetime.fromtimestamp(latest["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    alert_text = (
        f"{latest['hazard_type'].replace('_', ' ').title()} detected · "
        f"{latest['severity_level']} ({latest['severity_score']:.0f}/100) · "
        f"{latest['details'] or 'No additional details'} · {latest_time}"
    )
    if latest["severity_level"] in {"CRITICAL", "HIGH"}:
        st.error(f"LATEST ALERT  |  {alert_text}")
    elif latest["severity_level"] == "MEDIUM":
        st.warning(f"LATEST ALERT  |  {alert_text}")
    else:
        st.info(f"LATEST INCIDENT  |  {alert_text}")
else:
    st.info(f"No incidents recorded yet. Start the safety monitor to populate {config.DB_PATH}.")

df = df_all.copy()
if rows:
    df = df[df["severity_level"].isin(levels)]

critical_count = int((df["severity_level"] == "CRITICAL").sum()) if rows else 0
high_count = int((df["severity_level"] == "HIGH").sum()) if rows else 0
last_incident = df_all["time"].iloc[0] if rows else "—"

k1, k2, k3, k4 = st.columns(4)
k1.metric("All-time incidents", sum(totals.values()))
k2.metric("Critical · loaded", critical_count)
k3.metric("High · loaded", high_count)
k4.metric("Latest incident", last_incident)

incidents_tab, analytics_tab = st.tabs(["Recent incidents", "Analytics & snapshots"])
with incidents_tab:
    st.subheader("Incident log")
    if rows and not df.empty:
        show = df[[
            "time", "hazard_type", "track_id", "severity_score",
            "severity_level", "details",
        ]].rename(columns={
            "time": "Time",
            "hazard_type": "Hazard",
            "track_id": "Track ID",
            "severity_score": "Score",
            "severity_level": "Severity",
            "details": "Details",
        })
        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%d"
                ),
                "Track ID": st.column_config.NumberColumn("Track ID", format="%d"),
            },
        )
    elif rows:
        st.caption("No incidents match the selected severity levels.")
    else:
        st.caption("New incidents will appear here as the monitor records them.")

with analytics_tab:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Incidents by hazard")
        if totals:
            chart_data = pd.Series(totals, name="Incidents").sort_values(ascending=False)
            st.bar_chart(chart_data, color="#087f8c", horizontal=True)
        else:
            st.caption("Hazard totals will appear when incidents are recorded.")
    with right:
        st.subheader("Incident snapshot")
        if rows:
            with_img = df[df["snapshot_path"].fillna("").apply(
                lambda path: bool(path) and os.path.exists(path)
            )]
        else:
            with_img = pd.DataFrame()
        if not with_img.empty:
            labels = [
                f"{row.time} · {row.hazard_type.replace('_', ' ')} · {row.severity_level}"
                for row in with_img.itertuples()
            ]
            choice = st.selectbox("Select snapshot", range(len(labels)), format_func=lambda i: labels[i])
            st.image(with_img.iloc[choice]["snapshot_path"], width="stretch")
        else:
            st.caption("No snapshots are available for the selected incidents.")

if auto:
    time.sleep(interval)
    st.rerun()
