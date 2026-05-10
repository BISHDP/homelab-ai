import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from scipy import stats
import os

st.set_page_config(page_title="Homelab Monitor", layout="wide")
st.title("Homelab Resource Monitor")

DATA_PATH = os.path.expanduser("~/projects/homelab-ai/monitor/data/metrics.csv")

@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    df['disk_read_delta'] = df['disk_read_mb'].diff().clip(lower=0)
    df['disk_write_delta'] = df['disk_write_mb'].diff().clip(lower=0)
    df['mem_util_percent'] = (df['mem_used_gb'] / (df['mem_used_gb'] + df['mem_available_gb'])) * 100
    df = df.dropna()
    return df

df = load_data()

# Sidebar
st.sidebar.header("Controls")
hours = st.sidebar.slider("hours of data to diplay", min_value=1, max_value=72, value=24)
contamination = st.sidebar.slider("Anomaly sensitivity", min_value=0.01, max_value=0.10, value=0.03, step=0.01)

cutoff = df.index.max() - pd.Timedelta(hours=hours)
df_view = df[df.index >= cutoff]

st.sidebar.markdown(f"**Rows in view:** {len(df_view)}")
st.sidebar.markdown(f"**Total rows collected:** {len(df)}")

# Run IsolationForest on viewed data
features= ['cpu_percent', 'mem_util_percent', 'gpu_util_percent',
           'gpu_mem_used_gb', 'disk_read_delta', 'disk_write_delta']

x = df_view[features].copy()
iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
df_view = df_view.copy()
df_view['anomaly'] = iso.fit_predict(x) == -1

# Charts
metrics = [
    ('cpu_percent', 'CPU %', 'steelblue'),
    ('gpu_util_percent', 'GPU %', 'darkorange'),
    ('mem_util_percent', 'Memory %', 'seagreen'),
    ('gpu_mem_used_gb', 'GPU Memory (GB)', 'mediumpurple')
]

for metric, label, color in metrics:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df_view.index, df_view[metric], color=color, linewidth=0.8, alpha=0.7)
    ax.scatter(df_view.index[df_view['anomaly']], 
               df_view[metric][df_view['anomaly']],
               color='red', s=20, zorder=5, label='Anomaly')
    ax.set_ylabel(label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Summary row
col1, col2, col3, col4 = st.columns(4)

col1.metric("Current CPU %", f"{df_view['cpu_percent'].iloc[-1]:.1f}%")
col2.metric("Current Memory %", f"{df_view['mem_util_percent'].iloc[-1]:.1f}%")
col3.metric("Current GPU %", f"{df_view['gpu_util_percent'].iloc[-1]:.1f}%")
col4.metric("Anomalies in view", f"{df_view['anomaly'].sum()}")