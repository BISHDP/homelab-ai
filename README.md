# Homelab AI Monitor

A homelab resource monitor built as an initial foray into learning AI/ML operations
and engineering.  It collects real hardware metrics from the host running the collector,
detects anomalies in the metrics with machine learning, and provides visualizations of
the results in an interactive Streamlit dashboard.

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
├── monitor/
│   ├── collector/
│   │   ├── collector.py          # Hardware metrics collection script
│   │   └── requirements.txt      # Python dependencies
│   ├── notebooks/
│   │   ├── 01_eda.ipynb          # Exploratory data analysis
│   │   ├── 02_anomaly_detection.ipynb  # IsolationForest vs z-score comparison
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
source .venv/bin/activate
streamlit run monitor/dashboard/app.py
```
or
```bash
make dashboard
```

### Start Jupyter
```bash
cd docker/jupyter && docker compose up -d
```
or
```bash
make start
```

### Stop Jupyter
```bash
cd docker/jupyter && docker compose down
```
or
```bash
make stop
```

### Check lab status
```bash
make status
```