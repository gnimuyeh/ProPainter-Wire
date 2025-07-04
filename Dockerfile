# Use RunPod's PyTorch base image for optimal compatibility
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy opencv-python-headless scikit-image imageio-ffmpeg \
    einops pyyaml requests matplotlib runpod

# Set working directory to the handler script's folder inside the mounted volume
WORKDIR /runpod-volume/ProPainter-Wire

CMD ["python3", "-u", "runpod_handler.py"]
