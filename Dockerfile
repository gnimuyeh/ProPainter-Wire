# Use RunPod's PyTorch base image for optimal compatibility
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install FFmpeg runtime and dev libraries for PyAV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev

# Install Python dependencies, including PyAV
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy opencv-python-headless scikit-image imageio-ffmpeg \
    einops pyyaml requests matplotlib runpod av

WORKDIR /workspace

COPY . /workspace

CMD ["python3", "-u", "runpod_handler.py"]
