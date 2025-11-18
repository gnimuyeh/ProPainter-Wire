# This tag is guaranteed to exist right now
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg wget unzip aria2 && \
    rm -rf /var/lib/apt/lists/*

# Python packages (Tsinghua mirror = fast)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy opencv-python-headless scikit-image imageio-ffmpeg \
    einops pyyaml requests matplotlib runpod av tqdm

# Install BaiduPCS-Go into /workspace
RUN wget -q https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/v4.0.0/BaiduPCS-Go-v4.0.0-linux-amd64.zip && \
    unzip -q BaiduPCS-Go-v4.0.0-linux-amd64.zip && \
    mv BaiduPCS-Go-v4.0.0-linux-amd64/BaiduPCS-Go /workspace/BaiduPCS-Go && \
    chmod +x /workspace/BaiduPCS-Go && \
    rm -rf BaiduPCS-Go-v4.0.0-linux-amd64.zip BaiduPCS-Go-v4.0.0-linux-amd64

WORKDIR /workspace
COPY . .

CMD ["python3", "-u", "runpod_handler.py"]
