# Homelab AI Monitor

A homelab resource monitor built as a portfolio project demonstrating AI/ML operations
and engineering applied to real infrastructure data.  It collects real hardware metrics
from the host running the collector, detects anomalies in the metrics with machine
learning, and provides visualizations of the results in an interactive Streamlit dashboard.

## Features
- Collects CPU, memory, GPU, and disk metrics every 60 seconds via a systemd service
- Stores metrics to CSV for analysis
- Detects anomalies using IsolationForest with z-score comparison
- Visualizes results in an interactive Streamlit dashboard with tunable sensitivity

## Stack
- Python, pandas, scikit-learn, matplotlib, Streamlit
- psutil / pynvml for hardware metrics
- Docker + JupyterLab for analysis environment
- systemd for persistent background collection

## Tested Hardware
- AMD Ryzen 9850x3D / 32GB RAM / NVIDIA RTX 3070
- Ubuntu 24.04 bare metal

## Project Structure
```
homelab-ai/
├── models/
|   ├── .gitkeep
|   ├── ram_growth_predictor.json  # Metadata for the model training
|   └── ram_growth_predictor.pkl   # Linear Regression model for RAM trend
├── monitor/
│   ├── collector/
│   │   ├── collector.py          # Hardware metrics collection script
│   │   └── requirements.txt      # Python dependencies
│   ├── notebooks/
│   │   ├── 01_eda.ipynb          # Exploratory data analysis
│   │   ├── 02_anomaly_detection.ipynb  # IsolationForest vs z-score comparison
|   |   ├── 03_ram_growth_predictor.ipynb   # Exploration of RAM data trend, model selection, and training
|   |   └── status.ipynb          # Outputs the hardware and checks the CUDA config
│   ├── dashboard/
│   │   └── app.py                # Streamlit dashboard
│   └── data/
│       └── metrics.csv           # Generated at runtime, gitignored
├── docker/
│   └── jupyter/
│       ├── Dockerfile
│       └── docker-compose.yml
├── .venv/                        # gitignored
├── .gitignore
├── .editorconfig
├── Makefile
└── README.md
```

## Running it
### Start the collector
```bash
systemctl --user start homelab-collector
```
### Start the dashboard
```bash
make dashboard
```

### Start Jupyter
```bash
make start
```

### Stop Jupyter
```bash
make stop
```

### Check lab status
```bash
make status
```