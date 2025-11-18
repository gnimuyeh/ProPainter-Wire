FROM runpod/pytorch:2.8.0-cu1281-torch241-ubuntu2404-devel

# Install system tools (needed for wget/unzip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (Tsinghua mirror = fast in China)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy opencv-python-headless scikit-image imageio-ffmpeg \
    einops pyyaml requests matplotlib runpod av tqdm

# Download and install BaiduPCS-Go into /workspace
RUN wget -q https://github.com/qjfoidnh/BaiduPCS-Go/releases/latest/download/BaiduPCS-Go-v3.9.7-linux-amd64.zip && \
    unzip -q BaiduPCS-Go-v3.9.7-linux-amd64.zip && \
    mv BaiduPCS-Go-v3.9.7-linux-amd64/BaiduPCS-Go /workspace/BaiduPCS-Go && \
    chmod +x /workspace/BaiduPCS-Go && \
    rm -rf BaiduPCS-Go-v3.9.7-linux-amd64.zip BaiduPCS-Go-v3.9.7-linux-amd64

# Set working directory
WORKDIR /workspace

# Copy your code (handler.py, propainter_batch.py, etc.)
COPY . .

# RunPod Serverless ignores CMD, but it's good practice
CMD ["python3", "-u", "runpod_handler.py"]
