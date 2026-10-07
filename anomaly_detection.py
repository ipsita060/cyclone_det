import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# Graceful tabulate import with fallback table formatter
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers="keys", tablefmt="fancy_grid", showindex=False, **kwargs):
        if not data:
            return ""
        if isinstance(data, list) and data and isinstance(data[0], dict):
            cols = list(data[0].keys())
            rows = [[str(r.get(c, "")) for c in cols] for r in data]
        elif isinstance(data, pd.DataFrame):
            cols = list(data.columns)
            rows = [[str(val) for val in row] for row in data.values]
        else:
            return str(data)
        col_w = [max(len(str(c)), max((len(r[i]) for r in rows), default=0)) for i, c in enumerate(cols)]
        sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
        hdr = "| " + " | ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cols)) + " |"
        body = ["| " + " | ".join(r[i].ljust(col_w[i]) for i, c in enumerate(cols)) + " |" for r in rows]
        return "\n".join([sep, hdr, sep] + body + [sep])

def resolve_input_file():
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            return sys.argv[1]
        else:
            print(f"Error: Specified input file '{sys.argv[1]}' not found.")
            sys.exit(1)
            
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "Data (1).csv",
        "data.csv",
        "Data.csv",
        os.path.join(script_dir, "Data (1).csv"),
        os.path.join(script_dir, "data.csv"),
        os.path.join(script_dir, "Data.csv"),
        os.path.join(script_dir, "..", "Data (1).csv"),
        os.path.join(script_dir, "..", "data.csv")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
            
    # Search working dir & script dir
    for folder in [os.getcwd(), script_dir]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(".csv") and not f.startswith("anomaly_"):
                    return os.path.abspath(os.path.join(folder, f))
                    
    print("Error: Could not locate dataset 'Data (1).csv'.")
    print("Usage: python anomaly_detection.py [path/to/Data.csv]")
    sys.exit(1)

INPUT_FILE = resolve_input_file()
OUTPUT_DIR = "cyclone_output"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

SENSORS = [
    "Cyclone_Inlet_Gas_Temp",
    "Cyclone_Gas_Outlet_Temp",
    "Cyclone_Outlet_Gas_draft",
    "Cyclone_cone_draft",
    "Cyclone_Inlet_Draft",
    "Cyclone_Material_Temp"
]

# ========== 1. LOAD + CLEAN ==========
print(f"Loading data from: {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)

# Fast vectorized datetime parsing with fallback
parsed_time = pd.to_datetime(df["time"], format="%d-%m-%Y %H:%M", errors="coerce")
nat_mask = parsed_time.isna()
if nat_mask.any():
    parsed_time[nat_mask] = pd.to_datetime(df["time"][nat_mask], errors="coerce", dayfirst=True, format="mixed")
df["time"] = parsed_time

for c in SENSORS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = (df.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True))

print(f"  Loaded {len(df)} valid timestamp rows")

# ========== 2. DATA QUALITY ==========
df["sensor_error"] = df[SENSORS].isna().any(axis=1)

change = df[SENSORS].diff().abs().max(axis=1)
df["all_sensors_unchanged"] = change.eq(0)
df["frozen_values"] = (
    df["all_sensors_unchanged"].rolling(12, min_periods=12).sum().ge(12)
)

# ========== 3. PROCESS FEATURES ==========
df["dP_inlet_cone"] = df["Cyclone_Inlet_Draft"] - df["Cyclone_cone_draft"]
df["dP_cone_outlet"] = df["Cyclone_cone_draft"] - df["Cyclone_Outlet_Gas_draft"]
df["dT_gas"] = df["Cyclone_Inlet_Gas_Temp"] - df["Cyclone_Gas_Outlet_Temp"]

for c in SENSORS:
    df[f"{c}_diff"] = df[c].diff()

# ========== 4. DOMAIN RULE: DRAFT LOSS WHILE HOT ==========
hot_threshold = df["Cyclone_Inlet_Gas_Temp"].quantile(0.75)
draft_zero_threshold = df["Cyclone_Inlet_Draft"].quantile(0.85)
previous_draft = (
    df["Cyclone_Inlet_Draft"].rolling(12, min_periods=6).median().shift(1)
)
normal_draft_median = df["Cyclone_Inlet_Draft"].quantile(0.50)

df["draft_loss_while_hot"] = (
    (df["Cyclone_Inlet_Gas_Temp"] >= hot_threshold)
    & (df["Cyclone_Inlet_Draft"] >= draft_zero_threshold)
    & (previous_draft < normal_draft_median)
)

# ========== 5. ISOLATION FOREST ==========
print("Training Isolation Forest model (accelerated)...")
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
# Fast training on representative sample
fit_sample_size = min(50000, len(X_scaled))
rng = np.random.RandomState(42)
fit_idx = rng.choice(len(X_scaled), size=fit_sample_size, replace=False)
model.fit(X_scaled[fit_idx])

df["ml_score"] = -model.decision_function(X_scaled)
ml_threshold = df["ml_score"].quantile(.997)
df["ml_anomaly"] = df["ml_score"] >= ml_threshold

# ========== 6. SUSTAINED SHIFT ==========
shift_scores = np.zeros(len(df))
for c in SENSORS:
    short = df[c].rolling(12, min_periods=8).mean().to_numpy()
    long = df[c].rolling(72, min_periods=36).mean().to_numpy()
    std = df[c].rolling(72, min_periods=36).std().to_numpy()
    score = np.abs(short - long) / (std + 1e-6)
    shift_scores = np.maximum(shift_scores, np.nan_to_num(score, 0))

df["shift_score"] = shift_scores
shift_threshold = df["shift_score"].quantile(.997)
df["sustained_shift"] = df["shift_score"] >= shift_threshold

df["process_anomaly"] = (
    (df["ml_anomaly"] | df["sustained_shift"])
    & ~df["sensor_error"]
    & ~df["frozen_values"]
)

# ========== 7. GROUP POINTS INTO TIME PERIODS ==========
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
            st = pd.Timestamp(times[start])
            et = pd.Timestamp(times[end])
            duration = (et - st).total_seconds() / 60 + 5
            score = np.nanmax(ml_scores[start:end+1])

            events.append({
                "Anomaly Class": label,
                "Start Time": st,
                "End Time": et,
                "Duration (minutes)": round(duration, 1),
                "Evidence": evidence,
                "Anomaly Score": round(float(score), 4)
            })

        i = end + 1

    return events

events = []

events += get_events(
    df["draft_loss_while_hot"],
    "Draft loss while hot",
    "Hot inlet gas with inlet draft collapse",
    2
)

events += get_events(
    df["process_anomaly"],
    "Process anomaly (model)",
    "Unusual multivariate behaviour detected by Isolation Forest / rolling shift",
    3
)

events += get_events(
    df["frozen_values"],
    "Data fault: frozen values",
    "All six sensors unchanged for at least one hour",
    12
)

events += get_events(
    df["sensor_error"],
    "Data fault: sensor error",
    "One or more sensor values missing or invalid",
    2
)

results = pd.DataFrame(events)

if len(results):
    results = results.sort_values("Start Time").reset_index(drop=True)
    results.insert(0, "Event ID", range(1, len(results) + 1))
else:
    results = pd.DataFrame(columns=[
        "Event ID", "Anomaly Class", "Start Time", "End Time",
        "Duration (minutes)", "Evidence", "Anomaly Score"
    ])

# ========== 8. SAVE CSVs ==========
results.to_csv(os.path.join(OUTPUT_DIR, "anomaly_results.csv"), index=False)

summary = (
    results.groupby("Anomaly Class")
    .agg(
        Number_of_Events=("Event ID", "count"),
        Total_Duration_Hours=(
            "Duration (minutes)", lambda x: round(x.sum()/60, 2)
        ),
        First_Detected=("Start Time", "min"),
        Last_Detected=("End Time", "max")
    )
    .reset_index()
)
summary.to_csv(os.path.join(OUTPUT_DIR, "anomaly_summary.csv"), index=False)

# ========== 9. PRINT TABLES ==========
print(f"\nRows: {len(df)} | Events detected: {len(results)}")

print("\n### Summary by Anomaly Class ###")
print(tabulate(summary.to_dict(orient="records"), headers="keys", tablefmt="fancy_grid", showindex=False, numalign="right", stralign="left"))

print("\n### All Detected Anomaly Periods ###")
disp = results.copy()
disp["Start Time"] = pd.to_datetime(disp["Start Time"]).dt.strftime("%Y-%m-%d %H:%M")
disp["End Time"]   = pd.to_datetime(disp["End Time"]).dt.strftime("%Y-%m-%d %H:%M")
print(tabulate(disp.to_dict(orient="records"), headers="keys", tablefmt="fancy_grid", showindex=False, numalign="right", stralign="left"))

# ========== 10. GRAPH 1: OVERALL TIMELINE ==========
print("\nGenerating graphs...")

COLORS = {
    "Draft loss while hot": "#FF8C00",
    "Process anomaly (model)": "#DC143C",
    "Data fault: frozen values": "#8A2BE2",
    "Data fault: sensor error": "#9370DB",
}

fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

axes[0].plot(df["time"], df["Cyclone_Inlet_Gas_Temp"], lw=0.5, color="steelblue")
axes[0].set_ylabel("Inlet Gas Temp (°C)", fontsize=9)
axes[0].set_title("Cyclone Preheater — Abnormal Operating Periods", fontsize=14, fontweight="bold", pad=12)

axes[1].plot(df["time"], df["Cyclone_cone_draft"], lw=0.5, color="steelblue")
axes[1].set_ylabel("Cone Draft (mmWC)", fontsize=9)

axes[2].plot(df["time"], df["dP_inlet_cone"], lw=0.5, color="steelblue")
axes[2].set_ylabel("ΔP Inlet–Cone (mmWC)", fontsize=9)
axes[2].set_xlabel("Time")

seen = set()
for _, e in results.iterrows():
    color = COLORS.get(e["Anomaly Class"], "grey")
    for ax in axes:
        lbl = e["Anomaly Class"] if e["Anomaly Class"] not in seen else None
        ax.axvspan(e["Start Time"], e["End Time"], color=color, alpha=0.5, lw=0, label=lbl)
    seen.add(e["Anomaly Class"])

for ax in axes:
    ax.grid(alpha=0.2)
axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_timeline.png"), dpi=110)
plt.close()
print("  → anomaly_timeline.png")

# ========== 11. GRAPH 2: CLASS DISTRIBUTION ==========
fig, ax = plt.subplots(figsize=(11, 6))
counts = results["Anomaly Class"].value_counts()
bars = ax.bar(counts.index, counts.values, color=[COLORS.get(c, "grey") for c in counts.index], edgecolor="white", linewidth=0.8)

ax.set_title("Detected Abnormal Periods by Class", fontsize=15, fontweight="bold")
ax.set_xlabel("Anomaly Class")
ax.set_ylabel("Number of Periods")
ax.tick_params(axis="x", rotation=15)

for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
            str(int(b.get_height())), ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_class_distribution.png"), dpi=110)
plt.close()
print("  → anomaly_class_distribution.png")

# ========== 12. GRAPH 3: MONTHLY DISTRIBUTION ==========
if len(results):
    res_copy = results.copy()
    res_copy["month"] = pd.to_datetime(res_copy["Start Time"]).dt.to_period("M").astype(str)
    monthly = res_copy.groupby(["month", "Anomaly Class"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(15, 6))
    monthly.plot(kind="bar", stacked=True, ax=ax,
                 color=[COLORS.get(c, "grey") for c in monthly.columns],
                 edgecolor="white", linewidth=0.5)
    ax.set_title("Monthly Distribution of Abnormal Periods", fontsize=15, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Periods")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_anomaly_distribution.png"), dpi=110)
    plt.close()
    print("  → monthly_anomaly_distribution.png")

# ========== 13. GRAPH 4: TOP 15 EVENTS (DETAILED) ==========
top_events = results.sort_values("Anomaly Score", ascending=False).head(15)

for _, e in top_events.iterrows():
    start = e["Start Time"] - pd.Timedelta(hours=2)
    end = e["End Time"] + pd.Timedelta(hours=2)
    s = df[(df["time"] >= start) & (df["time"] <= end)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(s["time"], s["Cyclone_Inlet_Gas_Temp"], lw=0.9, label="Inlet Gas Temp", color="#2196F3")
    axes[0].plot(s["time"], s["Cyclone_Gas_Outlet_Temp"], lw=0.9, label="Gas Outlet Temp", color="#FF9800")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(s["time"], s["Cyclone_Inlet_Draft"], lw=0.9, label="Inlet Draft", color="#4CAF50")
    axes[1].plot(s["time"], s["Cyclone_cone_draft"], lw=0.9, label="Cone Draft", color="#E91E63")
    axes[1].plot(s["time"], s["Cyclone_Outlet_Gas_draft"], lw=0.9, label="Outlet Gas Draft", color="#9C27B0")
    axes[1].set_ylabel("Draft (mmWC)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    axes[2].plot(s["time"], s["Cyclone_Material_Temp"], lw=0.9, label="Material Temp", color="#795548")
    axes[2].set_ylabel("Material Temp (°C)")
    axes[2].set_xlabel("Time")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)

    color = COLORS.get(e["Anomaly Class"], "grey")
    for ax in axes:
        ax.axvspan(e["Start Time"], e["End Time"], color=color, alpha=0.25)

    fig.suptitle(
        f"Event {int(e['Event ID'])} — {e['Anomaly Class']}\n"
        f"{e['Start Time']} → {e['End Time']}  |  Score: {e['Anomaly Score']}",
        fontsize=13, fontweight="bold"
    )

    safe = (e["Anomaly Class"].lower()
            .replace(" ", "_").replace(":", "").replace("(", "").replace(")", ""))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"event_{int(e['Event ID']):03d}_{safe}.png"), dpi=110)
    plt.close()

print(f"  → {len(top_events)} detailed event plots in {PLOT_DIR}/")

print("\n✅ DONE")
print(f"  CSV:    {os.path.abspath(os.path.join(OUTPUT_DIR, 'anomaly_results.csv'))}")
print(f"  Summary:{os.path.abspath(os.path.join(OUTPUT_DIR, 'anomaly_summary.csv'))}")
print(f"  Graphs: {os.path.abspath(OUTPUT_DIR)}/")
