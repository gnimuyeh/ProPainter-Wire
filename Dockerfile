# Use RunPod's PyTorch base image for optimal compatibility
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy opencv-python-headless scikit-image imageio-ffmpeg \
    einops pyyaml requests matplotlib runpod av

WORKDIR /workspace

COPY . /workspace

CMD ["python3", "-u", "runpod_handler.py"]
