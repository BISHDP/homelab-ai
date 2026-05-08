FROM python:3.12-slim

RUN pip install --no-cache-dir \
    jupyterlab \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    psutil \
    nvidia-ml-py

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]