import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# Set page configuration
st.set_page_config(
    page_title="Cyclone Preheater Anomaly Detection",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

SENSORS = [
    "Cyclone_Inlet_Gas_Temp",
    "Cyclone_Gas_Outlet_Temp",
    "Cyclone_Outlet_Gas_draft",
    "Cyclone_cone_draft",
    "Cyclone_Inlet_Draft",
    "Cyclone_Material_Temp"
]

COLOR_MAP = {
    "Draft loss while hot": "#E63946",
    "Process anomaly (model)": "#F4A261",
    "Data fault: frozen values": "#457B9D",
    "Data fault: sensor error": "#6C757D"
}

@st.cache_data(show_spinner=False)
def load_and_clean_data(file_source):
    if isinstance(file_source, str):
        df = pd.read_csv(file_source)
    else:
        df = pd.read_csv(file_source)
        
    # Fast vectorized datetime parsing
    parsed_time = pd.to_datetime(df["time"], format="%d-%m-%Y %H:%M", errors="coerce")
    nat_mask = parsed_time.isna()
    if nat_mask.any():
        parsed_time[nat_mask] = pd.to_datetime(df["time"][nat_mask], errors="coerce", dayfirst=True, format="mixed")
    df["time"] = parsed_time

    for c in SENSORS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    df = (df.dropna(subset=["time"])
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True))
    return df

@st.cache_data(show_spinner=False)
def run_anomaly_pipeline(df, ml_quantile=0.997, shift_quantile=0.997):
    df = df.copy()
    
    # 1. Data Quality Checks
    df["sensor_error"] = df[SENSORS].isna().any(axis=1)
    change = df[SENSORS].diff().abs().max(axis=1)
    df["all_sensors_unchanged"] = change.eq(0)
    df["frozen_values"] = (
        df["all_sensors_unchanged"].rolling(12, min_periods=12).sum().ge(12)
    )
    
    # 2. Process Features
    df["dP_inlet_cone"] = df["Cyclone_Inlet_Draft"] - df["Cyclone_cone_draft"]
    df["dP_cone_outlet"] = df["Cyclone_cone_draft"] - df["Cyclone_Outlet_Gas_draft"]
    df["dT_gas"] = df["Cyclone_Inlet_Gas_Temp"] - df["Cyclone_Gas_Outlet_Temp"]
    
    for c in SENSORS:
        df[f"{c}_diff"] = df[c].diff()
        
    # 3. Domain Rule: Draft Loss While Hot
    hot_threshold = df["Cyclone_Inlet_Gas_Temp"].quantile(0.75)
    draft_zero_threshold = df["Cyclone_Inlet_Draft"].quantile(0.85)
    previous_draft = df["Cyclone_Inlet_Draft"].rolling(12, min_periods=6).median().shift(1)
    normal_draft_median = df["Cyclone_Inlet_Draft"].quantile(0.50)
    
    df["draft_loss_while_hot"] = (
        (df["Cyclone_Inlet_Gas_Temp"] >= hot_threshold)
        & (df["Cyclone_Inlet_Draft"] >= draft_zero_threshold)
        & (previous_draft < normal_draft_median)
    )
    
    # 4. Fast Isolation Forest
    features = (
        SENSORS
        + ["dP_inlet_cone", "dP_cone_outlet", "dT_gas"]
        + [f"{c}_diff" for c in SENSORS]
    )
    
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.interpolate(limit=3).ffill().bfill()
    q005 = X.quantile(0.005)
    q995 = X.quantile(0.995)
    X = X.clip(q005, q995, axis=1)
    X_scaled = RobustScaler().fit_transform(X)
    
    model = IsolationForest(
        n_estimators=100,
        max_samples=1024,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    # Subsampled fast fitting
    fit_sample_size = min(50000, len(X_scaled))
    rng = np.random.RandomState(42)
    fit_idx = rng.choice(len(X_scaled), size=fit_sample_size, replace=False)
    model.fit(X_scaled[fit_idx])
    
    df["ml_score"] = -model.decision_function(X_scaled)
    ml_threshold = df["ml_score"].quantile(ml_quantile)
    df["ml_anomaly"] = df["ml_score"] >= ml_threshold
    
    # 5. Fast Sustained Shift (Vectorized)
    shift_scores = np.zeros(len(df))
    for c in SENSORS:
        short = df[c].rolling(12, min_periods=8).mean().to_numpy()
        long = df[c].rolling(72, min_periods=36).mean().to_numpy()
        std = df[c].rolling(72, min_periods=36).std().to_numpy()
        score = np.abs(short - long) / (std + 1e-6)
        shift_scores = np.maximum(shift_scores, np.nan_to_num(score, 0))
        
    df["shift_score"] = shift_scores
    shift_threshold = df["shift_score"].quantile(shift_quantile)
    df["sustained_shift"] = df["shift_score"] >= shift_threshold
    
    df["process_anomaly"] = (
        (df["ml_anomaly"] | df["sustained_shift"])
        & ~df["sensor_error"]
        & ~df["frozen_values"]
    )
    
    # 6. Fast Group into Events
    def get_events(mask, label, evidence, min_points):
        vals = mask.fillna(False).to_numpy()
        times = df["time"].to_numpy()
        ml_scores = df["ml_score"].to_numpy()
        events = []
        i = 0
        while i < len(vals):
            if not vals[i]:
                i += 1
                continue
            start = i
            end = i
            while (
                end + 1 < len(vals)
                and vals[end + 1]
                and (times[end + 1] - times[end]) <= np.timedelta64(10, "m")
            ):
                end += 1
            if end - start + 1 >= min_points:
                st_time = pd.Timestamp(times[start])
                et_time = pd.Timestamp(times[end])
                dur = (et_time - st_time).total_seconds() / 60 + 5
                score = np.nanmax(ml_scores[start:end+1])
                events.append({
                    "Anomaly Class": label,
                    "Start Time": st_time,
                    "End Time": et_time,
                    "Duration (minutes)": round(dur, 1),
                    "Evidence": evidence,
                    "Anomaly Score": round(float(score), 4)
                })
            i = end + 1
        return events

    events = []
    events += get_events(df["draft_loss_while_hot"], "Draft loss while hot", "Hot inlet gas with inlet draft collapse", 2)
    events += get_events(df["process_anomaly"], "Process anomaly (model)", "Multivariate Isolation Forest / rolling shift", 3)
    events += get_events(df["frozen_values"], "Data fault: frozen values", "All 6 sensors unchanged for >= 1 hour", 12)
    events += get_events(df["sensor_error"], "Data fault: sensor error", "Missing / invalid telemetry values", 2)

    results = pd.DataFrame(events)
    if len(results):
        results = results.sort_values("Start Time").reset_index(drop=True)
        results.insert(0, "Event ID", range(1, len(results) + 1))
        
        summary = (
            results.groupby("Anomaly Class")
            .agg(
                Number_of_Events=("Event ID", "count"),
                Total_Duration_Hours=("Duration (minutes)", lambda s: round(s.sum() / 60, 2)),
                First_Detected=("Start Time", "min"),
                Last_Detected=("End Time", "max")
            )
            .reset_index()
        )
    else:
        results = pd.DataFrame(columns=[
            "Event ID", "Anomaly Class", "Start Time", "End Time",
            "Duration (minutes)", "Evidence", "Anomaly Score"
        ])
        summary = pd.DataFrame(columns=[
            "Anomaly Class", "Number_of_Events", "Total_Duration_Hours",
            "First_Detected", "Last_Detected"
        ])

    return df, results, summary

# ==================== UI LAYOUT ====================

st.title("🌪️ Cyclone Preheater Anomaly Detection")
st.markdown("Automated detection of process upsets, draft loss, and sensor errors from industrial telemetry data.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

dataset_source = st.sidebar.radio(
    "Select Data Source:",
    ("Bundled Dataset (Data (1).csv)", "Upload Custom CSV"),
    index=0
)

uploaded_file = None
data_path = "Data (1).csv"

if dataset_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file in the sidebar to begin.")
        st.stop()
else:
    if not os.path.exists(data_path):
        st.sidebar.warning("`Data (1).csv` not found in root. Please upload a file.")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is None:
            st.stop()

st.sidebar.subheader("Sensitivity Settings")
ml_sensitivity = st.sidebar.slider(
    "ML Anomaly Threshold Quantile",
    min_value=0.990,
    max_value=0.999,
    value=0.997,
    step=0.001,
    help="Higher quantile means fewer, more extreme anomaly detections."
)

with st.spinner("Processing telemetry data and running ML pipeline..."):
    source = uploaded_file if uploaded_file is not None else data_path
    raw_df = load_and_clean_data(source)
    df, results, summary = run_anomaly_pipeline(raw_df, ml_quantile=ml_sensitivity)

# Top Metrics Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Telemetry Rows", f"{len(df):,}")
col2.metric("Total Anomaly Events", f"{len(results):,}")
total_hours = summary["Total_Duration_Hours"].sum() if not summary.empty else 0
col3.metric("Total Duration (Hours)", f"{total_hours:.1f} hrs")
top_class = summary.sort_values("Number_of_Events", ascending=False).iloc[0]["Anomaly Class"] if not summary.empty else "N/A"
col4.metric("Top Anomaly Class", top_class)

st.markdown("---")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Summary & Distribution",
    "📋 Event Log & Export",
    "📈 Time-Series Plots",
    "ℹ️ Sensors & Rules Guide"
])

# TAB 1: SUMMARY
with tab1:
    st.subheader("Summary by Anomaly Class")
    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Event Counts per Class")
            st.bar_chart(summary.set_index("Anomaly Class")["Number_of_Events"])
        with col_c2:
            st.markdown("#### Total Duration (Hours) per Class")
            st.bar_chart(summary.set_index("Anomaly Class")["Total_Duration_Hours"])
    else:
        st.info("No anomalies detected with current sensitivity.")

# TAB 2: EVENT LOG
with tab2:
    st.subheader("Detailed Anomaly Events")
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        classes = ["All"] + list(results["Anomaly Class"].unique())
        selected_class = st.selectbox("Filter by Class:", classes)
        
    filtered_results = results if selected_class == "All" else results[results["Anomaly Class"] == selected_class]
    
    st.dataframe(filtered_results, use_container_width=True, hide_index=True)
    
    csv_buffer = io.StringIO()
    filtered_results.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Filtered Events (CSV)",
        data=csv_buffer.getvalue(),
        file_name="anomaly_events_export.csv",
        mime="text/csv"
    )

# TAB 3: PLOTS
with tab3:
    st.subheader("Visual Analysis")
    
    plot_type = st.radio(
        "Select Plot View:",
        ("Overview Timeline", "Single Event Zoom-In"),
        horizontal=True
    )
    
    if plot_type == "Overview Timeline":
        st.markdown("**Overview Timeline (Subsampled for smooth rendering)**")
        # Subsample for fast plotting
        sample_df = df.iloc[::10]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        
        axes[0].plot(sample_df["time"], sample_df["Cyclone_Inlet_Gas_Temp"], color="#1D3557", lw=0.8, label="Inlet Gas Temp (°C)")
        axes[0].plot(sample_df["time"], sample_df["Cyclone_Gas_Outlet_Temp"], color="#457B9D", lw=0.8, label="Outlet Gas Temp (°C)")
        axes[0].set_ylabel("Temp (°C)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(sample_df["time"], sample_df["Cyclone_Inlet_Draft"], color="#E63946", lw=0.8, label="Inlet Draft")
        axes[1].plot(sample_df["time"], sample_df["Cyclone_cone_draft"], color="#F4A261", lw=0.8, label="Cone Draft")
        axes[1].set_ylabel("Draft (mmWC)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(sample_df["time"], sample_df["dP_inlet_cone"], color="#2A9D8F", lw=0.8, label="dP (Inlet - Cone)")
        axes[2].set_ylabel("dP (mmWC)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)
        
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)
        
    else:
        if len(results) > 0:
            event_id = st.selectbox(
                "Select Event to Inspect:",
                results["Event ID"].tolist(),
                format_func=lambda x: f"Event {x}: {results.loc[results['Event ID']==x, 'Anomaly Class'].values[0]} ({results.loc[results['Event ID']==x, 'Start Time'].values[0]})"
            )
            
            ev = results[results["Event ID"] == event_id].iloc[0]
            st_t = pd.to_datetime(ev["Start Time"]) - pd.Timedelta(hours=4)
            et_t = pd.to_datetime(ev["End Time"]) + pd.Timedelta(hours=4)
            
            window = df[(df["time"] >= st_t) & (df["time"] <= et_t)]
            
            if len(window) > 0:
                fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
                
                axes[0].plot(window["time"], window["Cyclone_Inlet_Gas_Temp"], label="Inlet Gas Temp", color="#1D3557")
                axes[0].axvspan(ev["Start Time"], ev["End Time"], color=COLOR_MAP.get(ev["Anomaly Class"], "red"), alpha=0.3, label="Anomaly Window")
                axes[0].set_ylabel("Temp (°C)")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(window["time"], window["Cyclone_Inlet_Draft"], label="Inlet Draft", color="#E63946")
                axes[1].plot(window["time"], window["Cyclone_cone_draft"], label="Cone Draft", color="#F4A261")
                axes[1].axvspan(ev["Start Time"], ev["End Time"], color=COLOR_MAP.get(ev["Anomaly Class"], "red"), alpha=0.3)
                axes[1].set_ylabel("Draft")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                axes[2].plot(window["time"], window["Cyclone_Material_Temp"], label="Material Temp", color="#2A9D8F")
                axes[2].axvspan(ev["Start Time"], ev["End Time"], color=COLOR_MAP.get(ev["Anomaly Class"], "red"), alpha=0.3)
                axes[2].set_ylabel("Material Temp (°C)")
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                fig.autofmt_xdate()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No events available to display.")

# TAB 4: ABOUT
with tab4:
    st.subheader("Sensors & Anomaly Rules Reference")
    st.markdown(r"""
    ### 📡 Telemetry Sensors
    - **`Cyclone_Inlet_Gas_Temp`**: Temperature of gas entering the preheater cyclone.
    - **`Cyclone_Gas_Outlet_Temp`**: Temperature of gas leaving the cyclone.
    - **`Cyclone_Outlet_Gas_draft`**: Static draft pressure at cyclone gas outlet.
    - **`Cyclone_cone_draft`**: Draft pressure at the cyclone cone section.
    - **`Cyclone_Inlet_Draft`**: Draft pressure at the inlet duct.
    - **`Cyclone_Material_Temp`**: Temperature of raw meal material.

    ### ⚠️ Anomaly Classification
    1. **Draft loss while hot**: High temperature accompanied by loss of negative suction draft.
    2. **Process anomaly (model)**: Isolation Forest multivariate outlier and rolling sustained shifts.
    3. **Data fault (frozen values)**: Zero fluctuation across all 6 channels for $\ge 60$ minutes.
    4. **Data fault (sensor error)**: Missing readings, communication dropouts, or non-numeric error codes.
    """)
