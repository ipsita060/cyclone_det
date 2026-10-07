# 🌪️ Cyclone Preheater Anomaly Detection

[![Anomaly Detection Pipeline](https://github.com/USERNAME/REPO_NAME/actions/workflows/anomaly_detection_pipeline.yml/badge.svg)](https://github.com/USERNAME/REPO_NAME/actions/workflows/anomaly_detection_pipeline.yml)
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

## 🚀 GitHub Actions Deployment & Automated Runs

This repository is configured with **GitHub Actions** to automatically run anomaly detection, render diagnostic charts, and publish downloadable artifacts on every commit or on-demand.

### 1. Automated Runs on Push / PR
Every push to `main` (or pull request) triggers the workflow:
1. Provisions an Ubuntu Python environment.
2. Installs dependencies from `requirements.txt`.
3. Runs `anomaly_detection.py` on the dataset.
4. Publishes a **Markdown summary table** directly into the GitHub Actions run summary.
5. Bundles all output CSVs and PNG charts into a downloadable artifact (`cyclone-anomaly-detection-output`).

### 2. Manual Trigger via GitHub UI (`workflow_dispatch`)
1. Navigate to the **Actions** tab in your GitHub repository.
2. Select **Cyclone Preheater Anomaly Detection Pipeline** from the left sidebar.
3. Click **Run workflow** dropdown and select the branch.
4. Click **Run workflow**.

---

## 💻 Local Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
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
