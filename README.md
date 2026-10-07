# 🌪️ Cyclone Preheater Anomaly Detection

[![Anomaly Detection Pipeline](https://github.com/ipsita060/cyclone_det/actions/workflows/anomaly_detection_pipeline.yml/badge.svg)](https://github.com/ipsita060/cyclone_det/actions/workflows/anomaly_detection_pipeline.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Ipsita Jash  
**Target:** Industrial Cyclone Preheater Operation Monitoring

---

## 📌 Project Overview

This project detects abnormal operating periods in a cement plant cyclone preheater using a combination of **domain-specific engineering rules** and an **Isolation Forest machine learning model**.

It analyzes ~3.5 years of 5-minute telemetry data (2017–2020, ~600,000+ data points) to detect process upsets and sensor data quality issues, exporting structured CSV tables and high-resolution diagnostic plots.

---

## 🔍 Anomaly Classes Detected

| # | Anomaly Class | Detection Mechanism | Operational Impact |
|---|---|---|---|
| **1** | **Draft Loss While Hot** | Rule-based (Inlet draft collapsing near 0 while gas is hot) | Indicates gas blockage or fan failure |
| **2** | **Process Anomaly (ML)** | Isolation Forest (200 trees, RobustScaler) + Rolling sustained shift | Abnormal multivariate interaction across pressure & temperatures |
| **3** | **Data Fault: Frozen Values** | Rule-based (All 6 sensors unchanged for $\ge$ 1 hour) | PLC / telemetry communication freeze |
| **4** | **Data Fault: Sensor Error** | Rule-based (Missing values, communication dropouts, "Comm Fail") | Sensor hardware failure or line disconnection |

---

## 🌐 Live Web App Deployment (Streamlit & Render)

### Option A: Streamlit Community Cloud (Recommended — Free & Instant)
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account (`ipsita060`).
2. Click **"New app"**.
3. Select:
   - **Repository:** `ipsita060/cyclone_det`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**. Your app will be live with a public URL in ~1 minute.

---


---

## 💻 Local Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ipsita060/cyclone_det.git
cd cyclone_det
```

### 2. Create and activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Detection
```bash
# Run with default dataset
python anomaly_detection.py

# Or pass a custom dataset path
python anomaly_detection.py "path/to/Data (1).csv"
```

---

## 📊 Generated Output Files

All outputs are saved to the `cyclone_output/` folder:

```text
cyclone_output/
├── anomaly_results.csv               # Complete event log (start/end, duration, score, evidence)
├── anomaly_summary.csv               # Aggregated statistics per anomaly class
├── anomaly_timeline.png              # 3-panel continuous overview with color-coded anomalies
├── anomaly_class_distribution.png    # Event counts by anomaly class
├── monthly_anomaly_distribution.png  # Month-by-month anomaly breakdown
└── plots/                            # Zoom-in diagnostic plots for top 15 severe events
    ├── event_001_Draft_loss_while_hot.png
    ├── event_002_Process_anomaly_(model).png
    └── ...
```

---

## 📂 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── anomaly_detection_pipeline.yml  # GitHub Actions automated workflow
├── .gitignore                              # Git ignore rules for venv, cache & outputs
├── anomaly_detection.py                    # Main analysis & ML detection pipeline
├── cyclone_anomaly_detection.py            # Reference / modular pipeline
├── requirements.txt                        # Python dependencies
├── Data (1).csv                            # Input telemetry dataset
└── README.md                               # Project documentation
```
