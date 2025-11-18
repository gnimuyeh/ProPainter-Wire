# Use RunPod's PyTorch base image for optimal compatibility
FROM runpod/pytorch:2.8.0-cu1281-torch241-ubuntu2404-devel

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy opencv-python-headless scikit-image imageio-ffmpeg \
    einops pyyaml requests matplotlib runpod av

WORKDIR /workspace

COPY . .

RUN chmod +x BaiduPCS-Go

CMD ["python3", "-u", "runpod_handler.py"]
