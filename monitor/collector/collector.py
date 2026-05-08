import psutil
import pynvml
import pandas as pd
from datetime import datetime
import time
import os

# Initialize NVIDIA ML
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

DATA_DIR = os.path.expanduser("~/projects/homelab-ai/monitor/data")
CSV_PATH = os.path.join(DATA_DIR, "metrics.csv")

def collect_metrics():
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
    
    # Memory
    mem = psutil.virtual_memory()
    mem_used_gb = mem.used / (1024 ** 3)
    mem_available_gb = mem.available / (1024 ** 3)
    
    # Disk IO
    disk_io = psutil.disk_io_counters()
    disk_read_mb = disk_io.read_bytes / (1024 ** 2)
    disk_write_mb = disk_io.write_bytes / (1024 ** 2)
    
    # GPU
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_mem_used_gb = gpu_mem.used / (1024 ** 3)
    gpu_mem_total_gb = gpu_mem.total / (1024 ** 3)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": cpu_percent,
        "mem_used_gb": round(mem_used_gb, 2),
        "mem_available_gb": round(mem_available_gb, 2),
        "disk_read_mb": round(disk_read_mb, 2),
        "disk_write_mb": round(disk_write_mb, 2),
        "gpu_util_percent": gpu_util.gpu,
        "gpu_mem_used_gb": round(gpu_mem_used_gb, 2),
        "gpu_mem_total_gb": round(gpu_mem_total_gb, 2)
    }

def write_metrics(metrics):
    df = pd.DataFrame([metrics])
    file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    df.to_csv(CSV_PATH, mode='a', header=not file_exists, index=False)

if __name__ == "__main__":
    print(f"Starting collector, writing to {CSV_PATH}")
    while True:
        try:
            metrics = collect_metrics()
            write_metrics(metrics)
            print(f"{metrics['timestamp']} - CPU: {metrics['cpu_percent']}% | "
                  f"MEM: {metrics['mem_used_gb']}GB | "
                  f"GPU: {metrics['gpu_util_percent']}%")
            time.sleep(60)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)